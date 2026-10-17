"""The diffusion widget's α · D tab: diffusionkit's fBm profile-likelihood fits
(`napari_gemscape2.fbm`), per track, beside the Brownian D tab.

The D tab is the standard quantification of how fast tracks move. This one
answers a different question -- what each track's anomalous exponent alpha
is -- with a second fit, whose amplitude it reports as D_ref at a reference
lag. D_ref is not a Brownian D (see `fbm`), and the tab says so wherever
D_ref is shown.

- **Run** -- every track's alpha_hat with its likelihood-ratio interval, the
  test of alpha = 1 and D_ref: the alpha_* and D_ref_* columns in the tracks
  pane, so the Map tab can color by alpha and the filters can cut on them.
- **α–D** -- every fitted track's alpha_hat against D_ref with its intervals,
  one panel per region class, over the tracks the filters pass
  (`joint_plot.plot_alpha_D`).
- **Track** -- the selected track's likelihood over (alpha, D_ref) with its
  profiles (`joint_plot.plot_fbm_track`).
- **Population** -- the distribution of (alpha, D_ref) across the tracks the
  filters pass, per region class: diffusionkit's 2D deconvolution of their
  joint likelihoods (`fbm.alpha_population`, `joint_plot.plot_alpha_population`),
  on a worker (~10 s per thousand tracks), with the shared alpha beside it. Its
  numbers join the summary once computed, and Save computes them for the bundle.

The shared alpha (one alpha for every track, summed profiles) is in the summary
from the run on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import polars as pl
from napari.qt.threading import thread_worker
from qtkit import CollapsibleSection, double_spinbox, flow_row, note_label, status_label, style_status_label
from qtkit.plot import PlotWindow
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from napari_gemscape2 import units
from napari_gemscape2.fbm import (
    BROWNIAN_P,
    FBM_GRID_DEFAULTS,
    MIN_POPULATION_TRACKS,
    TAU_REF_S,
    FbmAnalysis,
    alpha_population,
    analyze_fbm,
    fbm_options,
    joint_grid,
    population_is_cached,
    population_panels,
    population_summary,
    scatter_panels,
    shared_alpha_summary,
    summarize_fbm,
    track_panel,
)
from napari_gemscape2.joint_plot import plot_alpha_D, plot_alpha_population, plot_fbm_track
from napari_gemscape2.widgets.msd_widgets import fit_window_to_figure
from napari_gemscape2.widgets.params_panel import _compact_form, _dspin, _ispin

if TYPE_CHECKING:
    from diffusionkit import Acquisition
    from diffusionkit.fbmpost import FbmPostOptions, JointGrid

    from napari_gemscape2.widgets.diffusion_panel import DiffusionAnalysisWidget


@thread_worker(start_thread=False)
def _fbm_worker(tracks: pl.DataFrame, acquisition: "Acquisition", options: "FbmPostOptions", joint: "JointGrid",
                progress) -> FbmAnalysis:
    """`fbm.analyze_fbm` off the GUI thread: ~40 ms per 40-frame track, ~2 s at 300 frames, per core."""
    return analyze_fbm(tracks, acquisition, options, joint=joint, progress=progress)


@thread_worker(start_thread=False)
def _population_worker(analysis: FbmAnalysis, groups: dict) -> list[dict]:
    """`fbm.population_panels` off the GUI thread: one 2D deconvolution per group not yet cached."""
    return population_panels(analysis, groups)


_FBM_HELP = (
    "<b>What this is.</b> A second per-track fit, beside the D tab's: the anomalous exponent "
    "<b>α</b> and an amplitude under fractional Brownian motion, with the same exposure blur and "
    "localization SDs. At α = 1 its likelihood is the D tab's. Use the <b>D</b> tab for how fast "
    "tracks move; use this one when α is the question."
    "<br><br><b>No prior.</b> For each α the amplitude is set to its best value (the profile "
    "likelihood), so <i>alpha_hat</i> is the maximum-likelihood α and <i>alpha_low</i>&ndash;"
    "<i>alpha_high</i> its 95% likelihood-ratio interval: every α whose profile is within 1.92 "
    "log-likelihood units of the best. <i>alpha_edge</i> says when the interval reaches an end of "
    "the α grid (0.02&ndash;1.98); \"both\" means the track says nothing about α. "
    "<i>alpha_p_brownian</i> is the likelihood-ratio test of α = 1. A track with no motion above its "
    "localization noise has no <i>alpha_hat</i>."
    "<br><br><b>α at the low end.</b> Noise beyond the reported SDs, and confinement whose cage "
    "relaxes within about a frame, both look like extra static noise, which is α &rarr; 0: such "
    "tracks put <i>alpha_hat</i> at 0.02."
    "<br><br><b>D_ref is not a Brownian D.</b> It is the fBm amplitude expressed as "
    "D_ref = MSD<sub>1D</sub>(τ<sub>ref</sub>) / (2 τ<sub>ref</sub>), the true motion's apparent D at "
    "one lag, τ<sub>ref</sub>. It equals a Brownian D only at α = 1; otherwise it changes with "
    "τ<sub>ref</sub>, so compare D_ref only at the same τ<sub>ref</sub>. Its interval has α profiled "
    "out; <i>D_ref_low_um2_s</i> = 0 means the track only bounds D_ref from above. Never compare it "
    "with <i>D_mean</i> or an MSD D, and never pair <i>D_mean</i> with this tab's α."
    "<br><br><b>The α–D plot is not a population.</b> It shows each track's own fit side by side."
    "<br><br><b>Shared α.</b> One α for every track the filters pass: each track's profile is a "
    "likelihood, so the profiles add, and the summed curve's maximum and 95% interval are the shared "
    "α, as the D tab's shared D. When tracks differ in α it lands near an information-weighted "
    "average, and its interval does not describe the spread."
    "<br><br><b>Population.</b> The distribution of α (and D_ref) across the tracks the filters "
    "pass: a smooth 2D density fitted to every track's joint (α, D_ref) likelihood, as the D tab's "
    "deconvolution is fitted to the tracks' D likelihoods. Uninformative tracks change nothing; the "
    "band is the population's uncertainty, not the spread of tracks. "
    f"It needs at least {MIN_POPULATION_TRACKS} fitted tracks. <i>Below the D_ref floor</i> the tracks "
    "barely tell D_ref values apart, and α there is mostly the smoothness prior continuing α from "
    "above, so the plot also shows α above the floor and how much of the population lies below it."
)


class FbmTab(QWidget):
    """Per-track fBm fits with their own Run, and the α–D, Track and Population plots.
    Its exposure and min points are the shared `_AcquisitionRow`'s."""

    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host
        self._analysis: Optional[FbmAnalysis] = None
        self._grid_valid = True
        self._scatter_window: Optional[PlotWindow] = None
        self._track_window: Optional[PlotWindow] = None

        intro = note_label(
            "Per-track α and D_ref under fractional Brownian motion, by profile likelihood: a second "
            "fit, beside the D tab's Brownian D. D_ref is not a Brownian D."
        )
        self._tau_ref = double_spinbox(
            TAU_REF_S, 0.001, 1000.0, 0.01, decimals=3, suffix=" s",
            tooltip="The lag D_ref is reported at: D_ref = MSD_1D(τ_ref) / (2 τ_ref).\n"
            "It changes D_ref (unless α = 1), not α. Keep it the same, in\n"
            "seconds, across everything you compare.",
        )
        self._tau_ref.valueChanged.connect(lambda _v: self._on_tau_changed())
        self._tau_status = status_label("")
        prior = note_label("profile likelihood: no prior · 95% likelihood-ratio intervals")

        # The grids the likelihood is evaluated on (`fbm.FBM_GRID_DEFAULTS`), read when Run is pressed.
        d = FBM_GRID_DEFAULTS
        self._grid_n_alpha = _ispin(
            d["n_alpha"], 3, 999,
            tooltip="Points of each track's α grid over 0.02–1.98 (default 99: steps of 0.02).\n"
            "α̂ is refined between them; the profile and its interval are read on them.\n"
            "Each point costs one eigendecomposition per track. Odd counts keep α = 1 on the grid.",
        )
        self._grid_alpha_group = _ispin(
            d["alpha_group"], 1, 20,
            tooltip="α grid points per population cell along α (default 3: cells 0.06 wide).\n"
            "The population's α resolution; odd counts keep α = 1 a cell centre.",
        )
        self._grid_d_log10_D = _dspin(
            d["d_log10_D"], 0.005, 0.5, 0.01, 3, suffix=" dec",
            tooltip="Width of a population cell along D, in decades of D at the frame interval\n"
            "(default 0.05). Narrower than the sharpest tracks' likelihood across D (~0.07\n"
            "decades for long tracks); wider cells blur a narrow population across alphas.\n"
            "The run's time with the population is in proportion to 1 / this width.",
        )
        self._grid_cell_points = _ispin(
            d["cell_points"], 1, 32,
            tooltip="Likelihood points averaged across each population cell along D (default 4).\n"
            "The run's time with the population is in proportion to this. On 0.05-decade\n"
            "cells 4 points stay within 0.08 nats of 8; use more for wider cells.",
        )
        grid_form = _compact_form(QFormLayout())
        grid_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        grid_form.addRow("α points (per track)", self._grid_n_alpha)
        grid_form.addRow("α points per cell", self._grid_alpha_group)
        grid_form.addRow("D cell width", self._grid_d_log10_D)
        grid_form.addRow("points per D cell", self._grid_cell_points)
        grid_reset = QPushButton("default grid")
        grid_reset.setToolTip("napari-gemscape2's defaults (diffusionkit's, with 4 points per D cell).")
        grid_reset.clicked.connect(lambda: self._set_grid(FBM_GRID_DEFAULTS))
        grid_body = QWidget()
        grid_layout = QVBoxLayout(grid_body)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.addLayout(grid_form)
        grid_layout.addWidget(grid_reset, alignment=Qt.AlignmentFlag.AlignLeft)
        self._grid_section = CollapsibleSection("Grid", grid_body, expanded=False)
        self._grid_status = status_label("")
        for box in self._grid_boxes():
            box.valueChanged.connect(lambda _v: self._on_grid_changed())

        self._run_button = QPushButton("Run α · D fits")
        self._run_button.setToolTip(
            "Every track's α and D_ref by profile likelihood (or only the filtered\n"
            "ones, with the footer's switch): adds the alpha_* and D_ref_* columns.\n"
            "About 40 ms per 40-frame track per core; long tracks cost more (~2 s at 300)."
        )
        self._run_button.clicked.connect(self._run)
        self._status = status_label("")
        self._summary = status_label("")
        self._summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._scatter_button = QPushButton("α–D")
        self._scatter_button.setToolTip(
            "Every fitted track's α̂ against its D_ref with its likelihood-ratio\n"
            "intervals, over the tracks the filters pass, one panel per region\n"
            "class. Per-track fits side by side -- not a population."
        )
        self._scatter_button.clicked.connect(self._show_scatter)
        self._track_button = QPushButton("Track")
        self._track_button.setToolTip(
            "The selected track's likelihood over (D_ref, α), with its profiles\n"
            "and 95% region. Stays open and follows the selection."
        )
        self._track_button.clicked.connect(self._show_track)
        self._population_button = QPushButton("Population")
        self._population_button.setToolTip(
            "The distribution of α and D_ref across the tracks the filters pass,\n"
            "one panel per region class: a 2D deconvolution of the tracks' joint\n"
            "likelihoods (~10 s per thousand tracks, on a worker), with the\n"
            "shared α beside it. Its numbers join the summary below."
        )
        self._population_button.clicked.connect(self._show_population)
        self._population_window: Optional[PlotWindow] = None
        plot_row = flow_row(QLabel("plot:"), self._scatter_button, self._track_button, self._population_button)

        help_text = note_label(_FBM_HELP)
        help_text.setTextFormat(Qt.TextFormat.RichText)
        self._help = CollapsibleSection("Reading α and D_ref", help_text, expanded=False)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(intro)
        layout.addWidget(flow_row(QLabel("τ_ref"), self._tau_ref, prior))
        layout.addWidget(self._tau_status)
        layout.addWidget(self._grid_section)
        layout.addWidget(self._grid_status)
        layout.addWidget(self._run_button)
        layout.addWidget(self._status)
        layout.addWidget(self._summary)
        layout.addWidget(plot_row)
        layout.addWidget(self._help)
        layout.addStretch()
        self.setLayout(layout)
        self._refresh_plot_buttons()
        self._on_grid_changed()

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
        """New tracks: the fits shown belong to the old ones."""
        self._analysis = None
        self._status.setText("")
        style_status_label(self._status)
        self._summary.setText("")
        self._refresh_plot_buttons()
        self._on_tau_changed()
        self._on_grid_changed()

    def refresh_inputs(self) -> None:
        self._run_button.setEnabled(self.host.has_tracks and self.host.run_exposure_s() is not None
                                    and self._grid_valid)

    def refresh(self) -> None:
        """The filters changed: restate the summary, redraw the open α–D plot."""
        if self._analysis is None:
            return
        self._summary.setText(self._format_summary())
        if self._scatter_window is not None and self._scatter_window.isVisible():
            self._show_scatter()
        if self._population_window is not None and self._population_window.isVisible():
            self._show_population()

    def on_track_selected(self) -> None:
        if self._track_window is not None and self._track_window.isVisible():
            self._show_track()

    def restore(self, analysis: FbmAnalysis) -> None:
        """A saved run's columns, reopened with the settings they ran with."""
        self._set_tau(analysis.tau_ref_s)
        self._set_grid(analysis.grid)
        n_ok = self._adopt(analysis)
        no_population = "" if analysis.likelihoods is not None else " (saved without likelihoods: no population)"
        self._status.setText(
            f"saved α · D fits loaded: {n_ok} of {analysis.fits.height} tracks fitted, "
            f"τ_ref {analysis.tau_ref_s:g} s{no_population} — Run to re-fit"
        )
        style_status_label(self._status, "ok" if not no_population else "caution")

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

    # -- the grid --

    def _grid_boxes(self) -> tuple:
        return (self._grid_n_alpha, self._grid_alpha_group, self._grid_d_log10_D, self._grid_cell_points)

    def grid(self) -> dict:
        """The next run's grid settings (`fbm.FBM_GRID_DEFAULTS`' keys), from the controls."""
        return {"n_alpha": self._grid_n_alpha.value(), "alpha_group": self._grid_alpha_group.value(),
                "d_log10_D": round(self._grid_d_log10_D.value(), 6), "cell_points": self._grid_cell_points.value()}

    def _set_grid(self, grid: dict) -> None:
        for box, key in zip(self._grid_boxes(), ("n_alpha", "alpha_group", "d_log10_D", "cell_points")):
            blocked = box.blockSignals(True)
            box.setValue(grid[key])
            box.blockSignals(blocked)
        self._on_grid_changed()

    def _on_grid_changed(self) -> None:
        """State the cells the grid makes, whether it is valid, and whether the
        columns shown ran on another one."""
        grid = self.grid()
        try:
            options, joint = fbm_options(grid=grid), joint_grid(grid)
        except ValueError as exc:
            self._grid_valid = False
            self._grid_status.setText(f"grid: {exc}")
            style_status_label(self._grid_status, "error")
            self.refresh_inputs()
            return
        self._grid_valid = True
        step = (options.alpha_max - options.alpha_min) / (options.n_alpha - 1)
        text = (f"α step {step:.3g} · population cells {joint.alpha_group * step:.3g} in α × "
                f"{joint.d_log10_D_ref:g} decades of D, {joint.cell_points} points each")
        ran = self._analysis.grid if self._analysis is not None else None
        if ran is not None and ran != grid:
            text += (f"\ncolumns shown ran on {ran['n_alpha']} α points, cells {ran['alpha_group']} α points × "
                     f"{ran['d_log10_D']:g} decades, {ran['cell_points']} points — Run to use this grid")
            style_status_label(self._grid_status, "caution")
        else:
            style_status_label(self._grid_status)
        self._grid_status.setText(text)
        self.refresh_inputs()

    # -- the run --

    def _run(self) -> None:
        from diffusionkit import Acquisition

        self.host.sync_layer_exposure()
        tracks = self.host.diffkit_tracks_for_fit()
        exposure = self.host.run_exposure_s()
        if tracks is None or exposure is None:
            return
        try:
            grid = self.grid()
            options = fbm_options(self.host.min_frames, self._tau_ref.value(), grid)
            joint = joint_grid(grid)
        except ValueError as exc:
            self.report_error(f"settings: {exc}")
            return
        self._status.setText("running…")
        style_status_label(self._status)
        worker = _fbm_worker(tracks, Acquisition(self.host.dt_s, exposure), options, joint,
                             self.host.progress_callback)
        self.host.start_worker(worker, self._on_finished, self._on_error, [self._run_button], "α · D fits")

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
        self._on_grid_changed()
        self.host.set_fbm_results(analysis)
        self._summary.setText(self._format_summary())
        return analysis.ok.height

    def _on_error(self, exc: Exception) -> None:
        self.report_error(f"error: {exc}")

    def _refresh_plot_buttons(self) -> None:
        has_run = self._analysis is not None and self._analysis.ok.height > 0
        self._scatter_button.setEnabled(has_run)
        self._track_button.setEnabled(has_run)
        self._population_button.setEnabled(has_run and self._analysis.likelihoods is not None)

    def _format_summary(self) -> str:
        """Counts over the tracks the filters pass -- by status, how many pin α,
        how many reject α = 1, how many only bound D_ref -- the shared α, and
        what the run was."""
        ids = self.host.combined_filtered_track_ids()
        sm = summarize_fbm(self._analysis, ids)
        counted = ("n_tracks", "n_ok", "n_alpha_identified", "n_alpha_bounded", "n_non_brownian", "n_D_ref_upper_bound")
        others = [f"{n} {key[2:]}" for key, n in sm.items() if key.startswith("n_") and key not in counted and n]
        lines = [f"{sm['n_tracks']} tracks: {sm['n_ok']} fitted" + "".join(f" · {o}" for o in others),
                 f"{sm['n_alpha_identified']} with α̂, {sm['n_alpha_bounded']} of them with the α interval inside "
                 f"0.02–1.98 · {sm['n_non_brownian']} reject α = 1 (p < {BROWNIAN_P:g})"]
        if sm["n_D_ref_upper_bound"]:
            lines.append(f"{sm['n_D_ref_upper_bound']} only bound D_ref from above")
        if sm.get("D_ref_floor_median_um2_s") is not None:
            lines.append(f"D_ref floor {units.fmt(sm['D_ref_floor_median_um2_s'], 'D_ref_um2_s')} (median)")
        shared = shared_alpha_summary(self._analysis, ids)
        if shared is not None:
            edge = " (reaches the grid end)" if shared["at_grid_edge"] else ""
            lines.append(f"shared α {shared['alpha']:.2f} [{shared['low']:.2f}, {shared['high']:.2f}]{edge} over "
                         f"{shared['n_tracks']} tracks · p(α = 1) {shared['p_brownian']:.2g}")
        lines.extend(self._population_lines())
        acq = self._analysis.acquisition
        lines.append(
            f"τ_ref {self._analysis.tau_ref_s:g} s · dt {units.fmt_unit(acq.dt_s, 's/frame')} · "
            f"exposure {units.fmt_unit(acq.exposure_s, units.SECONDS)}"
        )
        return "\n".join(lines)

    def _population_lines(self) -> list[str]:
        """The population's numbers over the tracks the filters pass, once computed (Population or Save)."""
        ids = self.host.combined_filtered_track_ids()
        if self._analysis.likelihoods is None or not population_is_cached(self._analysis, ids):
            return []
        pop = alpha_population(self._analysis, ids)
        if pop is None:
            return [f"population: needs ≥ {MIN_POPULATION_TRACKS} fitted tracks"]
        sm = population_summary(pop)

        def iv(v):
            return f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"

        return [
            (f"population ({sm['n_tracks']} tracks): P(α<1) {iv(sm['p_alpha_below_1'])} · "
             f"mean α {iv(sm['alpha_mean'])} · SD {iv(sm['alpha_sd'])}"),
            (f"  above the D_ref floor: P(α<1) {iv(sm['above_floor']['p_alpha_below_1'])} · "
             f"{sm['share_below_floor'][0]:.0%} of the population below it"),
        ]

    # -- plots --

    def _show_population(self) -> None:
        if self._analysis is None or self._analysis.likelihoods is None:
            return
        ids = self.host.combined_filtered_track_ids()
        groups = self.host.group_track_ids(ids) or {"all": ids}
        worker = _population_worker(self._analysis, groups)
        self.host.start_worker(worker, self._on_population, self._on_error, [self._population_button],
                               "α population")

    def _on_population(self, panels: list[dict]) -> None:
        self._summary.setText(self._format_summary())
        if not panels:
            self._status.setText(f"no group has ≥ {MIN_POPULATION_TRACKS} fitted tracks passing the filters")
            style_status_label(self._status, "caution")
            return
        figure = plot_alpha_population(panels)
        if self._population_window is None:
            self._population_window = PlotWindow("α · D: population", parent=self)
        if not self._population_window.isVisible():
            fit_window_to_figure(self._population_window, figure)
        self._population_window.show_figure(figure)

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
