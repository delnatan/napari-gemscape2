"""The diffusion widget's α · D tab: diffusionkit's fBm grid posteriors
(`napari_gemscape2.fbm`), per track, beside the Brownian D tab.

The D tab is the standard quantification of how fast tracks move. This one
answers a different question -- what each track's anomalous exponent alpha
is -- with a second posterior, whose amplitude it reports as D_ref at a
reference lag. D_ref is not a Brownian D (see `fbm`), and the tab says so
wherever D_ref is shown.

- **Run** -- every track's joint posterior over (alpha, D_ref): the alpha_* and
  D_ref_* columns in the tracks pane, so the Map tab can color by alpha and
  the filters can cut on `alpha_info_bits`.
- **α–D** -- every fitted track's alpha against D_ref with its 1-SD posterior
  ellipse, one panel per region class, over the tracks the filters pass
  (`joint_plot.plot_alpha_D`).
- **Track** -- the selected track's joint posterior with both margins, alpha's
  drawn beside the flat prior it started from (`joint_plot.plot_fbm_track`).

There is no ensemble here: a track's alpha curve is a posterior, so tracks
don't combine by adding them, and the population of alpha needs the 2D
(alpha, D) deconvolution, which runs outside the UI on pooled bundles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import polars as pl
from napari.qt.threading import thread_worker
from qtkit import CollapsibleSection, double_spinbox, flow_row, note_label, status_label, style_status_label
from qtkit.plot import PlotWindow
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from napari_gemscape2 import units
from napari_gemscape2.fbm import (
    MIN_BITS,
    TAU_REF_S,
    FbmAnalysis,
    analyze_fbm,
    fbm_options,
    scatter_panels,
    summarize_fbm,
    track_panel,
)
from napari_gemscape2.joint_plot import plot_alpha_D, plot_fbm_track
from napari_gemscape2.widgets.msd_widgets import fit_window_to_figure

if TYPE_CHECKING:
    from diffusionkit import Acquisition
    from diffusionkit.fbmpost import FbmPostOptions

    from napari_gemscape2.widgets.diffusion_panel import DiffusionAnalysisWidget


@thread_worker(start_thread=False)
def _fbm_worker(tracks: pl.DataFrame, acquisition: "Acquisition", options: "FbmPostOptions", progress) -> FbmAnalysis:
    """`fbm.analyze_fbm` off the GUI thread: ~25 ms per 40-frame track, ~1 s at 300 frames."""
    return analyze_fbm(tracks, acquisition, options, progress=progress)


_FBM_HELP = (
    "<b>What this is.</b> A second per-track posterior, beside the D tab's: the joint posterior of "
    "the anomalous exponent <b>α</b> and an amplitude under fractional Brownian motion, with the "
    "same exposure blur and localization SDs. At α = 1 its likelihood is the D tab's. Use the "
    "<b>D</b> tab for how fast tracks move; use this one when α is the question."
    "<br><br><b>Priors.</b> Every number here is a posterior, and the priors matter. α is flat on "
    "0.02&ndash;1.98; the amplitude is flat in ln SNR. A track that carries no information on α "
    "returns its prior: mean 1, interval 0.1&ndash;1.9, and <i>alpha_info_bits</i> near 0. The "
    f"α&ndash;D plot draws tracks with less than {MIN_BITS:g} bit grey, and the Track plot draws "
    "the prior beside the posterior, so how much of a track's α is data shows directly."
    "<br><br><b>D_ref is not a Brownian D.</b> It is the fBm amplitude expressed as "
    "D_ref = MSD<sub>1D</sub>(τ<sub>ref</sub>) / (2 τ<sub>ref</sub>), the true motion's apparent D at "
    "one lag, τ<sub>ref</sub>. It equals a Brownian D only at α = 1; otherwise it changes with "
    "τ<sub>ref</sub> (at α = 0.3, going from 0.1 s to 1 s divides it by 5), so compare D_ref only at "
    "the same τ<sub>ref</sub>. Its interval inherits the α prior: below its floor "
    "(<i>D_ref_floor_um2_s</i>) read it as an upper bound. Never compare it with <i>D_mean</i> or an "
    "MSD D, and never pair <i>D_mean</i> with this tab's α."
    "<br><br><b>Ellipses, not error bars.</b> Within a track, α and D_ref trade off (their "
    "correlation, <i>alpha_D_ref_corr</i>, reaches about &minus;0.75 near the floor), so each track's "
    "error bar is its 1-SD ellipse; its projections are &plusmn;1 SD on each axis."
    "<br><br><b>No population here.</b> A track's α curve is a posterior, not a likelihood, so "
    "tracks don't combine by adding their curves (in simulation that was biased), and the plot is "
    "not a population: a short track's α sits where the prior puts it. The population of α comes "
    "from the 2D deconvolution of the tracks' joint (α, D) likelihoods, run outside this widget "
    "on pooled bundles."
)


class FbmTab(QWidget):
    """Per-track fBm posteriors with their own Run, and the α–D and Track plots.
    Its exposure and min points are the shared `_AcquisitionRow`'s."""

    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host
        self._analysis: Optional[FbmAnalysis] = None
        self._scatter_window: Optional[PlotWindow] = None
        self._track_window: Optional[PlotWindow] = None

        intro = note_label(
            "Per-track α and D_ref under fractional Brownian motion: a second posterior, beside the "
            "D tab's Brownian D. D_ref is not a Brownian D."
        )
        self._tau_ref = double_spinbox(
            TAU_REF_S, 0.001, 1000.0, 0.01, decimals=3, suffix=" s",
            tooltip="The lag D_ref is reported at: D_ref = MSD_1D(τ_ref) / (2 τ_ref).\n"
            "It changes D_ref (unless α = 1), not α. Keep it the same, in\n"
            "seconds, across everything you compare.",
        )
        self._tau_ref.valueChanged.connect(lambda _v: self._on_tau_changed())
        self._tau_status = status_label("")
        prior = note_label("priors: α flat on 0.02–1.98 · amplitude flat in ln SNR")

        self._run_button = QPushButton("Run α · D posteriors")
        self._run_button.setToolTip(
            "Every track's joint (α, D_ref) posterior (or only the filtered ones,\n"
            "with the footer's switch): adds the alpha_* and D_ref_* columns.\n"
            "About 25 ms per 40-frame track; long tracks cost more (~1 s at 300)."
        )
        self._run_button.clicked.connect(self._run)
        self._status = status_label("")
        self._summary = status_label("")
        self._summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._scatter_button = QPushButton("α–D")
        self._scatter_button.setToolTip(
            "Every fitted track's α against its D_ref with its 1-SD posterior\n"
            "ellipse, over the tracks the filters pass, one panel per region\n"
            "class. Per-track posteriors side by side -- not a population."
        )
        self._scatter_button.clicked.connect(self._show_scatter)
        self._track_button = QPushButton("Track")
        self._track_button.setToolTip(
            "The selected track's joint posterior over (D_ref, α), with α's\n"
            "margin beside its flat prior. Stays open and follows the selection."
        )
        self._track_button.clicked.connect(self._show_track)
        plot_row = flow_row(QLabel("plot:"), self._scatter_button, self._track_button)

        help_text = note_label(_FBM_HELP)
        help_text.setTextFormat(Qt.TextFormat.RichText)
        self._help = CollapsibleSection("Reading α and D_ref", help_text, expanded=False)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(intro)
        layout.addWidget(flow_row(QLabel("τ_ref"), self._tau_ref, prior))
        layout.addWidget(self._tau_status)
        layout.addWidget(self._run_button)
        layout.addWidget(self._status)
        layout.addWidget(self._summary)
        layout.addWidget(plot_row)
        layout.addWidget(self._help)
        layout.addStretch()
        self.setLayout(layout)
        self._refresh_plot_buttons()
        self.refresh_inputs()

    def showEvent(self, event) -> None:  # noqa: N802
        # As the D tab: the image panel's exposure box writes into the
        # layer's metadata unannounced, so look again on coming into view.
        super().showEvent(event)
        self.host.sync_layer_exposure()

    # -- state the host drives --

    @property
    def analysis(self) -> Optional[FbmAnalysis]:
        return self._analysis

    def reset(self) -> None:
        """New tracks: the posteriors shown belong to the old ones."""
        self._analysis = None
        self._status.setText("")
        style_status_label(self._status)
        self._summary.setText("")
        self._refresh_plot_buttons()
        self._on_tau_changed()

    def refresh_inputs(self) -> None:
        self._run_button.setEnabled(self.host.has_tracks and self.host.run_exposure_s() is not None)

    def refresh(self) -> None:
        """The filters changed: restate the summary, redraw the open α–D plot."""
        if self._analysis is None:
            return
        self._summary.setText(self._format_summary())
        if self._scatter_window is not None and self._scatter_window.isVisible():
            self._show_scatter()

    def on_track_selected(self) -> None:
        if self._track_window is not None and self._track_window.isVisible():
            self._show_track()

    def restore(self, analysis: FbmAnalysis) -> None:
        """A saved run's columns, reopened with the settings they ran with."""
        self._set_tau(analysis.tau_ref_s)
        n_ok = self._adopt(analysis)
        self._status.setText(
            f"saved α · D posteriors loaded: {n_ok} of {analysis.fits.height} tracks fitted, "
            f"τ_ref {analysis.tau_ref_s:g} s — Run to re-fit"
        )
        style_status_label(self._status, "ok")

    def report_error(self, text: str) -> None:
        self._status.setText(text)
        style_status_label(self._status, "error")

    # -- tau_ref --

    def _set_tau(self, value: float) -> None:
        blocked = self._tau_ref.blockSignals(True)
        self._tau_ref.setValue(value)
        self._tau_ref.blockSignals(blocked)
        self._on_tau_changed()

    def _on_tau_changed(self) -> None:
        """Say when the columns shown were run at another tau_ref."""
        ran = self._analysis.tau_ref_s if self._analysis is not None else None
        if ran is not None and abs(ran - self._tau_ref.value()) > 1e-12:
            self._tau_status.setText(f"columns shown: τ_ref {ran:g} s — Run to use this one")
            style_status_label(self._tau_status, "caution")
        else:
            self._tau_status.setText("")
            style_status_label(self._tau_status)

    # -- the run --

    def _run(self) -> None:
        from diffusionkit import Acquisition

        self.host.sync_layer_exposure()
        tracks = self.host.diffkit_tracks_for_fit()
        exposure = self.host.run_exposure_s()
        if tracks is None or exposure is None:
            return
        try:
            options = fbm_options(self.host.min_frames, self._tau_ref.value())
        except ValueError as exc:
            self.report_error(f"settings: {exc}")
            return
        self._status.setText("running…")
        style_status_label(self._status)
        worker = _fbm_worker(tracks, Acquisition(self.host.dt_s, exposure), options, self.host.progress_callback)
        self.host.start_worker(worker, self._on_finished, self._on_error, [self._run_button], "α · D posteriors")

    def _on_finished(self, analysis: FbmAnalysis) -> None:
        n_ok = self._adopt(analysis)
        self._status.setText(f"{n_ok} of {analysis.fits.height} tracks fitted · τ_ref {analysis.tau_ref_s:g} s")
        style_status_label(self._status, "ok" if n_ok else "caution")
        if n_ok:
            self._show_scatter()

    def _adopt(self, analysis: FbmAnalysis) -> int:
        self._analysis = analysis
        self._refresh_plot_buttons()
        self._on_tau_changed()
        self.host.set_fbm_results(analysis)
        self._summary.setText(self._format_summary())
        return analysis.ok.height

    def _on_error(self, exc: Exception) -> None:
        self.report_error(f"error: {exc}")

    def _refresh_plot_buttons(self) -> None:
        has_run = self._analysis is not None and self._analysis.ok.height > 0
        self._scatter_button.setEnabled(has_run)
        self._track_button.setEnabled(has_run)

    def _format_summary(self) -> str:
        """Counts over the tracks the filters pass -- by status, how many say
        something about α, how many only bound D_ref -- and what the run was."""
        sm = summarize_fbm(self._analysis, self.host.combined_filtered_track_ids())
        others = [f"{n} {key[2:]}" for key, n in sm.items()
                  if key.startswith("n_") and key not in ("n_tracks", "n_ok", "n_alpha_informative",
                                                          "n_D_ref_upper_bound") and n]
        lines = [f"{sm['n_tracks']} tracks: {sm['n_ok']} fitted" + "".join(f" · {o}" for o in others),
                 f"{sm['n_alpha_informative']} with ≥ {MIN_BITS:g} bit on α (the rest are mostly the prior)"]
        if sm["n_D_ref_upper_bound"]:
            lines.append(f"{sm['n_D_ref_upper_bound']} only bound D_ref from above (D_ref_grid_edge = low)")
        if sm.get("D_ref_floor_median_um2_s") is not None:
            lines.append(f"D_ref floor {units.fmt(sm['D_ref_floor_median_um2_s'], 'D_ref_um2_s')} (median)")
        acq = self._analysis.acquisition
        lines.append(
            f"τ_ref {self._analysis.tau_ref_s:g} s · dt {units.fmt_unit(acq.dt_s, 's/frame')} · "
            f"exposure {units.fmt_unit(acq.exposure_s, units.SECONDS)}"
        )
        return "\n".join(lines)

    # -- plots --

    def _show_scatter(self) -> None:
        if self._analysis is None:
            return
        ids = self.host.combined_filtered_track_ids()
        groups = self.host.group_track_ids(ids) or {"all": ids}
        panels = scatter_panels(self._analysis, groups)
        if not panels:
            self._status.setText("no fitted tracks pass the current filters")
            style_status_label(self._status, "caution")
            return
        figure = plot_alpha_D(panels)
        if self._scatter_window is None:
            self._scatter_window = PlotWindow("α · D: every track", parent=self)
        if not self._scatter_window.isVisible():
            fit_window_to_figure(self._scatter_window, figure)
        self._scatter_window.show_figure(figure)

    def _show_track(self) -> None:
        if self._analysis is None:
            return
        track_id = self.host.selected_track_id
        if track_id is None:
            self._status.setText("select a track (table or viewer) first")
            style_status_label(self._status, "caution")
            return
        track = self.host.diffkit_track(track_id)
        panel = track_panel(self._analysis, track) if track is not None and track.height else None
        if panel is None:
            self._status.setText(f"track {track_id} was not fitted (see its fbm_status)")
            style_status_label(self._status, "caution")
            return
        figure = plot_fbm_track(panel)
        if self._track_window is None:
            self._track_window = PlotWindow("α · D: selected track", parent=self)
        if not self._track_window.isVisible():
            fit_window_to_figure(self._track_window, figure)
        self._track_window.show_figure(figure)
