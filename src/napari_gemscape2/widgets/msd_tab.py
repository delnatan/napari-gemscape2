"""The diffusion widget's MSD tab: diffusionkit's classic MSD analysis,
beside the posteriors as the comparison people know, and on its own tab
because it is a separate analysis -- it needs only tracks, not a posterior
run, and makes its own assumption (no blur model: it runs with the
exposure treated as 0, and says so wherever it shows).

- **Per track** -- each track's time-averaged MSD, D fitted over its first
  lags (through the origin of the SD-corrected curve) and alpha by log-log
  over a wider window (`diffusion.MSDWindow`): the D_msd / K_msd / alpha_msd
  columns. Run here with its own button; a batch (`gemscape2 diffusion`)
  always runs it alongside the posteriors.
- **Track** -- the selected track's MSD as `joint_plot.plot_track_msd`
  draws it: linear and log-log, each fit over its window.
- **Ensemble** -- the ensemble-averaged MSD over the tracks the filters
  pass, one row per region class (`EnsembleMSDWindow`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import polars as pl
from napari.qt.threading import thread_worker
from qtkit import CollapsibleSection, flow_row, note_label, status_label, style_status_label
from qtkit.plot import PlotWindow
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from napari_gemscape2.diffusion import (
    MSD_ALPHA_LAG_FRACTION,
    MSD_ALPHA_MAX_LAG,
    MSD_LAG_FRACTION,
    MSD_MAX_LAG,
    MSD_MIN_LAG,
    MSDWindow,
    ensemble_msd_blur_free,
    ensemble_msd_panels,
    group_experiments,
    msd_fits_blur_free,
    msd_track_panel,
    msd_track_table,
)
from napari_gemscape2.joint_plot import plot_ensemble_msd, plot_track_msd
from napari_gemscape2.widgets.msd_widgets import EnsembleWindowControls, fit_window_to_figure, labelled, percent_box

if TYPE_CHECKING:
    from napari_gemscape2.widgets.diffusion_panel import DiffusionAnalysisWidget


@thread_worker(start_thread=False)
def _msd_worker(tracks: pl.DataFrame, dt_s: float, min_frames: int, window: MSDWindow, progress) -> pl.DataFrame:
    """Every track's MSD fits (`diffusion.msd_fits_blur_free`), off the GUI thread."""
    return msd_fits_blur_free(tracks, dt_s, min_frames, window, progress)


@thread_worker(start_thread=False)
def _ensemble_msd_worker(experiments: list, min_frames: int, max_lag: int):
    """`diffusion.ensemble_msd_blur_free` off the GUI thread: every track's
    MSD, then the bootstrap over tracks -- a second or so per thousand."""
    return ensemble_msd_blur_free(experiments, min_frames, max_lag)


_MSD_HELP = (
    "<b>Classic MSD analysis</b>, a comparison with the posteriors. The MSD estimators have no "
    "blur model, so they run with the exposure treated as 0, which biases D low when the "
    "exposure is a good share of the frame interval. The offset subtracted is the "
    "localization error the tracks' own SDs imply."
    "<br><br><b>Per track</b>: each track's time-averaged MSD. D is fitted over its first lags "
    "(the best measured), &alpha; by log-log over a wider window, since a slope in log-log "
    "needs a span of lags to mean much &mdash; three lags are half a decade of &tau;. The windows "
    "are a number of lags, or a share of each track's longest lag (so long tracks fit more of "
    "their curve), each capped at the track's own length. One track's &alpha; scatters widely: "
    "its lags share their displacements and the late ones rest on a pair or two. The "
    "<i>Track</i> plot shows the selected track's fits on linear and log-log axes, and a linear "
    "fit with a free intercept whose localization SD is a check on the SDs."
    "<br><br><b>Ensemble</b>: the tracks' MSDs averaged, each squared displacement counting "
    "once, with &plusmn;1 SEM over tracks and intervals from resampling whole tracks. D and "
    "&alpha; are fitted over separate shares of the curve: D over its first lags (the "
    "25&ndash;40% rule), &alpha; over a wider span (by default all of it). The late lags rest on "
    "the long tracks alone, which are the slow particles more often than not (fast ones leave "
    "the focus sooner); the &alpha; panel says how many reach its last lag, and the Map tab can "
    "color the tracks by <i>track_length</i>. An ensemble average describes the population only "
    "if its particles share one motion: a mix of slow and fast tracks averages to one curve, "
    "which the posteriors' deconvolved distribution would show as two."
)


class MSDTab(QWidget):
    """Per-track MSD fits with their own Run, and the Track and Ensemble
    plots. The fits' window is `diffusion.MSDWindow`, set here."""

    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host
        self._track_window: Optional[PlotWindow] = None
        self._ensemble_window: Optional[EnsembleMSDWindow] = None
        # The window the shown fits ran with, and the one a running fit asked for.
        self._ran: Optional[MSDWindow] = None
        self._pending: Optional[MSDWindow] = None

        intro = note_label(
            "Classic MSD analysis: needs only tracks. A comparison with the posteriors — "
            "the exposure is treated as 0 (no blur model)."
        )

        self._mode = QComboBox()
        self._mode.addItems(["lags", "% of each track"])
        self._mode.setToolTip(
            "The fits' windows: the same number of lags for every track (each\n"
            "capped at the track's own longest lag), or a share of each track's\n"
            "longest lag, so long tracks fit more of their curve."
        )
        self._d_lags = _lag_box(
            MSD_MAX_LAG,
            "How many of each track's first lags D is fitted over (through the\n"
            f"origin of the SD-corrected MSD). At least {MSD_MIN_LAG}; few is better:\n"
            "the first lags are the best measured.",
        )
        self._alpha_lags = _lag_box(
            MSD_ALPHA_MAX_LAG,
            "How many of each track's first lags α is fitted over, by log-log.\n"
            "Wider than D's: a slope needs a span of lags -- 10 lags is about a\n"
            "decade of τ, 3 only half of one.",
        )
        self._d_percent = percent_box(
            MSD_LAG_FRACTION,
            f"The share of each track's longest lag D is fitted over: the usual\n25-40% rule, never under {MSD_MIN_LAG} lags.",
        )
        self._alpha_percent = percent_box(
            MSD_ALPHA_LAG_FRACTION,
            "The share of each track's longest lag α is fitted over, by log-log.\n"
            "Short of the last lags, where one track's MSD rests on a pair or two.",
        )
        self._mode.currentIndexChanged.connect(self._on_mode)
        for box in (self._d_lags, self._alpha_lags, self._d_percent, self._alpha_percent):
            box.valueChanged.connect(lambda _v: self._on_window_changed())
        window_row = flow_row(
            labelled("window in", self._mode),
            labelled("· D over the first", self._d_lags, self._d_percent),
            labelled("· α over the first", self._alpha_lags, self._alpha_percent),
        )

        self._run_button = QPushButton("Run MSD fits")
        self._run_button.setToolTip(
            "Fit every track's MSD (or only the filtered ones, with the footer's\n"
            "switch): adds the D_msd, K_msd and α_msd columns. Seconds per\n"
            "thousand tracks. Save analysis writes them with the posteriors."
        )
        self._run_button.clicked.connect(self._run)
        self._status = status_label("")
        self._window_status = status_label("")

        self._track_button = QPushButton("Track")
        self._track_button.setToolTip(
            "The selected track's MSD on linear (D) and log-log (α) axes, each fit\n"
            "over the window above. Needs no run. Stays open and follows the selection."
        )
        self._track_button.clicked.connect(self._show_track)
        self._ensemble_button = QPushButton("Ensemble")
        self._ensemble_button.setToolTip(
            "The ensemble-averaged MSD over the tracks the filters pass, one row\n"
            "per region class, with its linear (D) and log-log (α) fits, each over\n"
            "its own window. Needs no run."
        )
        self._ensemble_button.clicked.connect(self._show_ensemble)
        plot_row = flow_row(QLabel("plot:"), self._track_button, self._ensemble_button)

        help_text = note_label(_MSD_HELP)
        help_text.setTextFormat(Qt.TextFormat.RichText)
        self._help = CollapsibleSection("Reading the MSD", help_text, expanded=False)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(intro)
        layout.addWidget(window_row)
        layout.addWidget(self._window_status)
        layout.addWidget(self._run_button)
        layout.addWidget(self._status)
        layout.addWidget(plot_row)
        layout.addWidget(self._help)
        layout.addStretch()
        self.setLayout(layout)
        self._on_mode()
        self.refresh_inputs()

    # -- the window --

    def window(self) -> MSDWindow:
        """The window the controls ask for."""
        if self._mode.currentIndex() == 1:
            return MSDWindow(
                self._d_lags.value(), self._alpha_lags.value(),
                self._d_percent.value() / 100, self._alpha_percent.value() / 100,
            )
        return MSDWindow(self._d_lags.value(), self._alpha_lags.value(), alpha_lag_fraction=self._alpha_percent.value() / 100)

    def set_window(self, window: MSDWindow) -> None:
        for box, value in (
            (self._d_lags, window.max_lag),
            (self._alpha_lags, window.alpha_max_lag),
            (self._d_percent, round(100 * (window.lag_fraction or MSD_LAG_FRACTION))),
            (self._alpha_percent, round(100 * window.alpha_lag_fraction)),
        ):
            box.setValue(value)
        self._mode.setCurrentIndex(1 if window.by_fraction else 0)

    def _on_mode(self) -> None:
        share = self._mode.currentIndex() == 1
        for box in (self._d_lags, self._alpha_lags):
            box.setVisible(not share)
        for box in (self._d_percent, self._alpha_percent):
            box.setVisible(share)
        self._on_window_changed()

    def _on_window_changed(self) -> None:
        """Say when the shown columns ran on another window; redraw the open Track plot."""
        ran = self._ran
        if ran is not None and ran.text() != self.window().text():
            self._window_status.setText(f"columns shown: {ran.text()} — Run to use this window")
            style_status_label(self._window_status, "caution")
        else:
            self._window_status.setText("")
            style_status_label(self._window_status)
        if self._track_window is not None and self._track_window.isVisible():
            self._show_track()

    # -- state the host drives --

    @property
    def min_frames(self) -> int:
        return self.host.min_frames

    def reset(self) -> None:
        """New tracks: the fits shown belong to the old ones."""
        self._ran = None
        self._status.setText("")
        style_status_label(self._status)
        self._on_window_changed()
        if self._ensemble_window is not None:
            self._ensemble_window.invalidate()
        self.refresh()
        self.refresh_inputs()

    def refresh_inputs(self) -> None:
        has_tracks = self.host.has_tracks
        for button in (self._run_button, self._track_button, self._ensemble_button):
            button.setEnabled(has_tracks)

    def refresh(self) -> None:
        """Follow the filters (and new tracks) while the Ensemble window is open."""
        if self._ensemble_window is not None and self._ensemble_window.isVisible():
            self._ensemble_window.refresh()

    def on_track_selected(self) -> None:
        if self._track_window is not None and self._track_window.isVisible():
            self._show_track()

    def restore(self, msd: pl.DataFrame, window: MSDWindow) -> None:
        """A saved analysis's MSD columns, reopened with the window they ran on."""
        self.set_window(window)
        self._adopt(msd, window)
        self._status.setText(f"saved MSD fits loaded: {msd.height} tracks · {window.text()}")
        style_status_label(self._status, "ok")

    # -- the run --

    def _run(self) -> None:
        tracks = self.host.diffkit_tracks_for_fit()
        if tracks is None:
            return
        self._pending = self.window()
        self._status.setText("fitting…")
        style_status_label(self._status)
        worker = _msd_worker(tracks, self.host.dt_s, self.min_frames, self._pending, self.host.progress_callback)
        self.host.start_worker(worker, self._on_finished, self._on_error, [self._run_button], "MSD fits")

    def _on_finished(self, fits: pl.DataFrame) -> None:
        window = self._pending
        self._adopt(msd_track_table(fits), window)
        brownian = fits.filter(pl.col("model") == "brownian")
        n_ok = brownian.filter(pl.col("status") == "ok").height
        n_alpha = fits.filter((pl.col("model") == "power_law") & (pl.col("status") == "ok")).height
        self._status.setText(f"{n_ok} of {brownian.height} tracks fitted (α ok for {n_alpha}) · {window.text()}")
        style_status_label(self._status, "ok" if n_ok else "caution")

    def _adopt(self, msd: pl.DataFrame, window: MSDWindow) -> None:
        self._ran = window
        self._on_window_changed()
        self.host.set_msd_results(msd, window)

    def _on_error(self, exc: Exception) -> None:
        self._status.setText(f"error: {exc}")
        style_status_label(self._status, "error")

    # -- plots --

    def _show_track(self) -> None:
        track_id = self.host.selected_track_id
        tracks = self.host.diffkit_tracks_for_fit()
        if tracks is None:
            return
        if track_id is None:
            self._status.setText("select a track (table or viewer) first")
            style_status_label(self._status, "caution")
            return
        panel = msd_track_panel(tracks, self.host.dt_s, track_id, self.min_frames, self.window())
        if panel is None:
            self._status.setText(f"track {track_id} is too short for the MSD fits")
            style_status_label(self._status, "caution")
            return
        figure = plot_track_msd(panel)
        if self._track_window is None:
            self._track_window = PlotWindow("MSD: selected track", parent=self)
        if not self._track_window.isVisible():
            fit_window_to_figure(self._track_window, figure)
        self._track_window.show_figure(figure)

    def _show_ensemble(self) -> None:
        if self._ensemble_window is None:
            self._ensemble_window = EnsembleMSDWindow(self)
        self._ensemble_window.show()
        self._ensemble_window.refresh()


def _lag_box(value: int, tooltip: str) -> QSpinBox:
    box = QSpinBox()
    box.setRange(MSD_MIN_LAG, 1000)
    box.setValue(value)
    box.setToolTip(tooltip)
    return box


class EnsembleMSDWindow(PlotWindow):
    """The ensemble-averaged MSD over the tracks the filters pass, one row
    per region class (`diffusion.ensemble_msd_panels`): diffusionkit's
    textbook population curve. It needs no run -- only tracks -- and follows
    the filters like the other figures.

    Its window is explicit, since it changes the answer
    (`EnsembleWindowControls`): how far each track's MSD runs (`max_lag`,
    which recomputes the curves on a worker, bootstrap included), the shares
    of the averaged curve D and alpha are fitted over, and where the
    localization offset comes from; those only refit."""

    def __init__(self, tab: MSDTab) -> None:
        super().__init__("MSD: ensemble average", parent=tab)
        self._tab = tab
        self._ens = None
        self._key = None
        self._worker = None
        self._stale = False
        self._status = status_label("")
        controls = EnsembleWindowControls()
        controls.curve_changed.connect(self.refresh)
        controls.fit_changed.connect(self._redraw)
        self._controls = controls
        self.layout().insertWidget(0, controls)
        self.layout().insertWidget(1, self._status)

    def invalidate(self) -> None:
        """The tracks changed: the curves computed so far are not theirs."""
        self._ens = None
        self._key = None

    def refresh(self) -> None:
        """Recompute the curves when what they are over has changed (the
        filters, the tracks, `max_lag`, min points), else just redraw."""
        host = self._tab.host
        tracks = host.diffkit_tracks_for_fit()
        if tracks is None or tracks.height == 0:
            self._say("no tracks loaded", "caution")
            return
        ids = host.combined_filtered_track_ids()
        groups = host.group_track_ids(ids) or {"all": ids}
        min_frames, max_lag = self._tab.min_frames, self._controls.max_lag.value()
        key = (
            tuple((name, None if g is None else frozenset(g)) for name, g in groups.items()),
            min_frames,
            max_lag,
        )
        if key == self._key and self._ens is not None:
            self._redraw()
            return
        if self._worker is not None:
            self._stale = True
            return
        experiments = group_experiments(tracks, host.dt_s, groups)
        if not experiments:
            self._say("no tracks pass the current filters", "caution")
            return
        self._say("averaging every track's MSD…")
        worker = _ensemble_msd_worker(experiments, min_frames, max_lag)
        worker.returned.connect(lambda ens: self._ready(key, ens))
        worker.errored.connect(self._failed)
        self._worker = worker
        worker.start()

    def _ready(self, key, ens) -> None:
        self._worker = None
        if self._stale:
            self._stale = False
            self.refresh()
            return
        self._ens, self._key = ens, key
        self._redraw()

    def _failed(self, exc: Exception) -> None:
        self._worker = None
        self._stale = False
        self._say(f"ensemble MSD failed: {exc}", "error")

    def _redraw(self) -> None:
        if self._ens is None:
            return
        n_points, alpha_points = self._controls.windows()
        panels = ensemble_msd_panels(self._ens, n_points, self._controls.offset_key(), alpha_points=alpha_points)
        short = [p["name"] for p in panels if p["power_law"].get("status") == "insufficient_data"]
        if short:
            self._say(f"no fit for {', '.join(short)}: no track reaches lag {max(n_points, alpha_points)} -- "
                      "lower the curve's lag or the windows", "caution")
        else:
            self._say("")
        figure = plot_ensemble_msd(panels)
        fit_window_to_figure(self, figure)
        self.show_figure(figure)

    def _say(self, text: str, level: str = "neutral") -> None:
        self._status.setText(text)
        style_status_label(self._status, level)
        if text and not self.isVisible():
            self.show()
