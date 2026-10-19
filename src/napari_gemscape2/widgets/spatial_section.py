"""The Map tab's "Spatial classes" section: classes of D among the tracks, from the D run's likelihoods
(`napari_gemscape2.spatial`).

Run takes the D run on screen and the tracks the filters pass now, fits C = 1..4 classes with the Potts prior on
their neighbour graph, and tests for a spatial pattern against shuffled positions. Its per-track columns join the
tracks pane (and so the map's colour choices), and it is saved with the analysis.

Three figures, each class in its own colour in all of them:
- "Number of classes": the score against C, to choose C at the elbow; picking C in "classes" and running again
  uses it (the suggested C is only the default);
- "Class map": where the classes are, or only the tracks when there is no spatial pattern;
- "Class populations": each class's D population, deconvolved from its tracks, and alpha's when there is an
  α · D run with its likelihoods, as densities with their bands, all classes on one axis per parameter.

Every number of classes is a movie's own: classes of different movies are matched by the user, from their
populations, and pooled by `gemscape2 pool` (`spatial_class` on an input).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from diffusionkit.spatial import SpatialAnalysis
from napari.qt.threading import thread_worker
from qtkit import flow_row, note_label, status_label, style_status_label
from qtkit.plot import PlotWindow
from qtpy.QtWidgets import QComboBox, QFormLayout, QPushButton, QSpinBox, QVBoxLayout, QWidget

from napari_gemscape2.diffusion import LEVEL, ensemble
from napari_gemscape2.fbm import alpha_population
from napari_gemscape2.spatial import MAX_CLASSES, N_SHUFFLES, analyze_spatial, class_groups, describe, spatial_options
from napari_gemscape2.spatial_plot import plot_class_map, plot_class_populations, plot_class_selection
from napari_gemscape2.widgets.msd_widgets import fit_window_to_figure

if TYPE_CHECKING:
    from napari_gemscape2.widgets.diffusion_panel import DiffusionAnalysisWidget

_SUGGESTED = "suggested"


@thread_worker(start_thread=False)
def _spatial_worker(analysis, base, pixel_size_um, ids, options, progress) -> SpatialAnalysis:
    return analyze_spatial(analysis, base, pixel_size_um, ids, options, progress=progress)


@thread_worker(start_thread=False)
def _populations_worker(analysis, fbm, id_sets: list) -> None:
    """The deconvolutions the populations figure reads, off the GUI thread (cached on the analyses)."""
    for ids in id_sets:
        ensemble(analysis, ids)
        if fbm is not None:
            alpha_population(fbm, ids)


class SpatialSection(QWidget):
    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host
        self._analysis: Optional[SpatialAnalysis] = None
        self._populations_worker = None
        self._windows: dict[str, PlotWindow] = {}

        self._classes = QComboBox()
        self._classes.addItems([_SUGGESTED, *(str(c) for c in range(1, MAX_CLASSES + 1))])
        self._classes.setToolTip(
            "How many classes to use: the suggested number is the last whose gain clears\n"
            "0.02 nats per track. Look at \"Number of classes\" and pick the elbow."
        )
        self._shuffles = QSpinBox()
        self._shuffles.setRange(0, 999)
        self._shuffles.setValue(N_SHUFFLES)
        self._shuffles.setToolTip(
            "Shuffles of the track positions for the pattern test.\n"
            "99 resolve p down to 0.01, 19 is the fewest that can reach 0.05;\n"
            "0 skips the test (and the class map)."
        )
        self._run_button = QPushButton("Run")
        self._run_button.setToolTip("Classes of D among the tracks the filters pass, from the D run's likelihoods.")
        self._run_button.clicked.connect(self._run)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("classes", flow_row(self._classes, self._run_button))
        form.addRow("shuffles", self._shuffles)
        self._status = status_label("")
        self._summary = note_label("")
        self._summary.setVisible(False)

        self._selection_button = QPushButton("Number of classes")
        self._selection_button.clicked.connect(self._show_selection)
        self._map_button = QPushButton("Class map")
        self._map_button.clicked.connect(self._show_map)
        self._populations_button = QPushButton("Class populations")
        self._populations_button.clicked.connect(self._show_populations)
        self._figure_buttons = (self._selection_button, self._map_button, self._populations_button)
        # Shown once there is a run to draw: three greyed buttons beside
        # Run read as the whole section being off.
        self._figure_row = flow_row(*self._figure_buttons)
        self._figure_row.setVisible(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(note_label("Classes numbered by increasing D, from each track's D likelihood and its "
                                    "neighbours'."))
        layout.addLayout(form)
        layout.addWidget(self._status)
        layout.addWidget(self._summary)
        layout.addWidget(self._figure_row)
        self.setLayout(layout)
        self.reset()

    @property
    def analysis(self) -> Optional[SpatialAnalysis]:
        return self._analysis

    def reset(self) -> None:
        """Forget the run (new tracks, or a new D run it was read from).
        Run is on once there is a D run to read the likelihoods from."""
        self._analysis = None
        ready = self.host.posterior is not None
        self._run_button.setEnabled(ready)
        self._status.setText("" if ready else "needs a D run: run it on the D tab first")
        style_status_label(self._status, "neutral")
        self._summary.setText("")
        self._summary.setVisible(False)
        self._figure_row.setVisible(False)
        for window in self._windows.values():
            window.hide()

    def report_error(self, text: str) -> None:
        self._status.setText(text)
        style_status_label(self._status, "error")

    def adopt(self, s: SpatialAnalysis, how: str = "") -> None:
        """Show `s` (run here, or restored with the saved analysis)."""
        self._analysis = s
        self._status.setText(how or f"{s.n_classes} classes")
        style_status_label(self._status, "ok")
        self._summary.setText(describe(s))
        self._summary.setVisible(True)
        self._figure_row.setVisible(True)
        self._run_button.setEnabled(True)
        for button in self._figure_buttons:
            button.setEnabled(True)
        self._populations_button.setEnabled(s.n_classes > 1)
        for name, window in self._windows.items():
            if window.isVisible():
                {"selection": self._show_selection, "map": self._show_map,
                 "populations": self._show_populations}[name]()

    def _run(self) -> None:
        analysis = self.host.posterior
        if analysis is None:
            self._status.setText("run D first: the classes are read from its likelihoods")
            style_status_label(self._status, "caution")
            return
        text = self._classes.currentText()
        try:
            options = spatial_options(None if text == _SUGGESTED else int(text), self._shuffles.value())
        except ValueError as exc:
            self._status.setText(str(exc))
            style_status_label(self._status, "error")
            return
        self._status.setText("running…")
        style_status_label(self._status)
        worker = _spatial_worker(analysis, self.host.base_track_df, self.host.pixel_size_um,
                                 self.host.combined_filtered_track_ids(), options, self.host.progress_callback)
        self.host.start_worker(worker, self._on_finished, self._on_error, [self._run_button], "spatial classes",
                               unit="fits")

    def _on_finished(self, s: SpatialAnalysis) -> None:
        self.host.set_spatial_results(s)
        self.adopt(s)

    def _on_error(self, exc: Exception) -> None:
        self._status.setText(f"spatial classes failed: {exc}")
        style_status_label(self._status, "error")

    def _window(self, name: str, title: str, figure) -> None:
        window = self._windows.get(name)
        if window is None:
            window = self._windows[name] = PlotWindow(title, parent=self)
            fit_window_to_figure(window, figure)
        window.show_figure(figure)

    def _show_selection(self) -> None:
        if self._analysis is not None:
            self._window("selection", "Spatial classes: number of classes",
                         plot_class_selection(self._analysis, self.host.map_title()))

    def _show_map(self) -> None:
        if self._analysis is None:
            return
        found = self.host.map_regions()
        labels, classes = found if found is not None else (None, None)
        figure = plot_class_map(
            self._analysis,
            shape_px=None if labels is None else labels.shape,
            pixel_size_um=self.host.pixel_size_um,
            labels=labels,
            region_classes=classes,
            title=self.host.map_title(),
        )
        self._window("map", "Spatial classes: map", figure)

    def _population_sets(self) -> list:
        s = self._analysis
        return [set(s.track_ids.tolist()), *class_groups(s).values()]

    def _show_populations(self) -> None:
        s, analysis = self._analysis, self.host.posterior
        if s is None or analysis is None or self._populations_worker is not None:
            return
        fbm = self.host.fbm
        fbm = fbm if fbm is not None and fbm.likelihoods is not None else None
        self._populations_button.setEnabled(False)
        self._status.setText("deconvolving the classes' populations…")
        style_status_label(self._status)
        worker = _populations_worker(analysis, fbm, self._population_sets())
        worker.returned.connect(lambda _r: self._populations_ready(analysis, fbm))
        worker.errored.connect(self._populations_failed)
        self._populations_worker = worker
        worker.start()

    def _populations_failed(self, exc: Exception) -> None:
        self._populations_worker = None
        self._populations_button.setEnabled(True)
        self._status.setText(f"populations failed: {exc}")
        style_status_label(self._status, "error")

    def _populations_ready(self, analysis, fbm) -> None:
        self._populations_worker = None
        self._populations_button.setEnabled(True)
        s = self._analysis
        if s is None or analysis is not self.host.posterior:
            return
        self._status.setText(f"{s.n_classes} classes")
        style_status_label(self._status, "ok")
        sets = self._population_sets()
        names = [f"all {len(s.track_ids)} tracks of the run", *class_groups(s)]
        numbers = [None, *range(1, s.n_classes + 1)]
        d_panels, alpha_panels = [], []
        alpha_ids = None if fbm is None else fbm.likelihoods.track_ids
        for name, c, ids in zip(names, numbers, sets):
            ens = ensemble(analysis, ids)
            if ens is not None:
                d_panels.append({"name": name if c is None else f"class {c}", "class": c, "n_tracks": ens.n_tracks,
                                 "deconvolved": ens.deconvolved_D, "band": ens.deconvolved_D_band})
            pop = None if fbm is None else alpha_population(fbm, ids)
            if pop is not None:
                alpha_panels.append({"name": "all tracks of the run" if c is None else f"class {c}", "class": c,
                                     "n_tracks": int(np.isin(alpha_ids, list(ids)).sum()), "alphas": pop.alphas,
                                     "weights": pop.alpha_weights(), "band": pop.alpha_band(LEVEL)})
        figure = plot_class_populations(analysis.D_grid_um2_s, d_panels, alpha_panels or None, analysis.level,
                                        title=self.host.map_title())
        self._window("populations", "Spatial classes: populations", figure)
