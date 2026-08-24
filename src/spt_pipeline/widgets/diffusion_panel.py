"""Diffusion-analysis dock widget: pick a Tracks layer already in the
napari viewer and interactively explore its data quality and per-track
diffusion behavior, using diffusionkit's classical MSD fit and both
speeds of its Bayesian fit (fast batched MAP for a spatial overview,
full NUTS posterior for a track flagged interesting from that overview).

Four tabs share one widget (mirrors `params_panel.py`'s convention of
private per-stage tab classes in one file), because they share live
state -- unlike `PipelineParamsWidget`'s stage tabs, which only fire a
one-shot "run this stage" signal at their host, these tabs need to read
and write the *same* loaded track set, filters, and current-track
selection, so each tab holds a plain reference to this module's
`DiffusionAnalysisWidget` (`self.host`) rather than talking through Qt
signals:

- **Data Explorer** -- histograms of per-point quality fields
  (amplitude, sigma_x/y, bg, ...) read straight off the selected Tracks
  layer's own per-vertex properties (every points_df column rides along
  there already, aligned with track_id -- see `viewer.py`; the "points"
  layer itself has no track_id, it's the pre-linking detections table),
  with a `matplotlib.widgets.SpanSelector` on each histogram
  to set a numeric range filter; filters AND together (with the Track
  Explorer's min-track-length filter -- `combined_filtered_track_ids`)
  into a "tracks that pass every active filter" subset that immediately
  reshapes what the Track Explorer table shows, and that Classical/
  Bayesian fit runs can optionally restrict to (a per-tab checkbox).
- **Track Explorer** -- one row per track (`qt_helpers.DataFrameTableModel`
  in a `QTableView`), the single place all per-track numbers now live
  (classical MSD, bulk MAP, and any one-off per-track fit, each in its
  own column group) instead of a `QLabel` text dump, plus its own
  always-visible `min_track_length` filter (the direct complement to
  Data Explorer's range filters, which need a histogram opened first).
  Selecting a row is
  "the current track" for the Bayesian tab's per-track action, and is
  kept in sync with the viewer both ways: clicking a track in the Tracks
  layer selects its row here (napari's own `TrackManager.get_value` --
  see `DiffusionAnalysisWidget._make_click_callback` -- already resolves
  a click to a `track_id`, no proxy layer needed), and selecting a row
  here draws that track's vertices as a small highlight overlay
  (`self._highlight_layer`, a plain `Points` layer since Tracks layers
  have no selection-highlight of their own).
- **Classical (MSD)** -- `analysis.fit_population`, visually and
  column-wise separate from the Bayesian numbers per-request.
- **Bayesian** -- *Spatial MAP*: `bayes.fit_population(engine="map")`
  batched over every eligible track, fast enough to run on the whole
  field of view; besides filling in Track Explorer columns, it also
  places a `Points` layer in the viewer (`self._spatial_map_layer`, one
  point per track centroid, colored by the fitted parameter) -- the
  actual spatial map, and the reason this analysis benefits from staying
  inside napari next to the image at all. *Per-track*: `bayes.fit_track`
  (`method="map"` or `"nuts"`) against whichever track is selected in the
  Track Explorer -- the "promote this one track, flagged interesting
  from the map, to expensive inference" step; `method="nuts"` additionally
  renders a posterior corner plot.

All fits run through `napari.qt.threading.thread_worker`, sharing one
`self._worker` slot host-wide (`start_worker`) so only one fit -- of any
kind -- runs at a time.

Track/spatial-map positions used for viewer overlays are kept in
*pixels* (`self._tracks_df_px`, the same coordinate space as the image
and Tracks layer), separate from the *physical-unit* table
(`self._diffkit_tracks`) handed to diffusionkit -- conflating the two
would misplace every overlay relative to the image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import analysis as dk_analysis
import bayes as dk_bayes
import numpy as np
import polars as pl
from matplotlib.widgets import SpanSelector
from napari.layers import Points, Tracks
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from spt_pipeline.diffusion import tracks_to_diffusionkit_df
from spt_pipeline.experiment import load_diffusion_results, write_diffusion_results
from spt_pipeline.widgets.qt_helpers import DataFrameTableModel, PlotWindow, hline, style_status_label

# Points-layer look for the two viewer overlays this widget owns -- kept
# visually distinct from DETECTED_POINTS_STYLE's magenta "+" (viewer.py)
# so a highlighted/mapped track never gets mistaken for a raw detection.
_HIGHLIGHT_POINTS_STYLE = dict(symbol="ring", size=4, face_color="transparent", border_color="yellow", border_width=0.3)


@thread_worker(start_thread=False)
def _run_population_fit_worker(
    diffkit_tracks: pl.DataFrame, dt_s: float, min_track_length: int
) -> "dk_analysis.PopulationFit":
    return dk_analysis.fit_population(diffkit_tracks, dt_s, min_track_length=min_track_length)


@thread_worker(start_thread=False)
def _run_bulk_map_worker(diffkit_tracks: pl.DataFrame, dt_s: float, model: str) -> pl.DataFrame:
    return dk_bayes.fit_population(diffkit_tracks, dt_s, model=model, engine="map", show_progress=False)


@thread_worker(start_thread=False)
def _run_track_fit_worker(
    track_df: pl.DataFrame, dt_s: float, model: str, method: str
) -> "dk_bayes.TrackFit":
    return dk_bayes.fit_track(track_df, dt_s, model=model, method=method)


def _base_track_table(diffkit_tracks: pl.DataFrame, tracks_df_px: pl.DataFrame) -> pl.DataFrame:
    """One row per track: identity + context columns every tab's results
    get left-joined onto. Position columns are pixel-space (`tracks_df_px`,
    the viewer's own coordinate system), not the physical-unit table."""
    centroids = tracks_df_px.group_by("track_id").agg(
        pl.col("y").mean().alias("y_px"), pl.col("x").mean().alias("x_px")
    )
    lengths = diffkit_tracks.group_by("track_id").agg(pl.col("track_length").first())
    return lengths.join(centroids, on="track_id", how="left").sort("track_id")


def _normalize_map_table(table: pl.DataFrame, model: str) -> pl.DataFrame:
    """`bayes.fit_population(..., engine="map")`'s output, trimmed to one
    D-like column and one alpha-like column regardless of `model` --
    `normal` has no alpha, `anomalous` names its D column differently
    (`D_alpha_median_um2_s_alpha` vs `D_median_um2_s`)."""
    if model == "normal":
        return table.select(
            "track_id",
            pl.col("D_median_um2_s").alias("D_map_um2_s"),
            pl.lit(None, dtype=pl.Float64).alias("alpha_map"),
        )
    return table.select(
        "track_id",
        pl.col("D_alpha_median_um2_s_alpha").alias("D_map_um2_s"),
        pl.col("alpha").alias("alpha_map"),
    )


def _track_fit_row(fit: "dk_bayes.TrackFit") -> dict:
    """One ad-hoc single-track fit (`method` "map" or "nuts"), normalized
    the same way as `_normalize_map_table` so both land in the same
    `D_track_fit_um2_s`/`alpha_track_fit` Track Explorer columns --
    deliberately separate from the bulk MAP columns even when `method`
    happens to be "map" too, since a bulk fit and a one-off single-track
    fit are different actions the user can compare against each other."""
    D = fit.params["D"] if fit.model == "normal" else fit.params["D_alpha"]
    return {
        "track_id": fit.track_id,
        "model": fit.model,
        "method": fit.method,
        "D_track_fit_um2_s": D,
        "alpha_track_fit": fit.params.get("alpha"),
    }


def _format_summary(summary: dict) -> str:
    return "\n".join(f"{key} = {value}" for key, value in summary.items())


class _DataExplorerTab(QWidget):
    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host
        self._filters: dict[str, tuple[float, float]] = {}
        self._plot_window: Optional[PlotWindow] = None
        self._span_selector: Optional[SpanSelector] = None

        self._column_picker = QComboBox()
        self._show_button = QPushButton("Show histogram")
        self._show_button.clicked.connect(self._show_histogram)
        pick_row = QHBoxLayout()
        pick_row.addWidget(QLabel("column:"))
        pick_row.addWidget(self._column_picker, 1)
        pick_row.addWidget(self._show_button)

        self._filter_label = QLabel("no filters active")
        self._filter_label.setWordWrap(True)
        self._clear_button = QPushButton("Clear filters")
        self._clear_button.clicked.connect(self._clear_filters)
        filter_row = QHBoxLayout()
        filter_row.addWidget(self._filter_label, 1)
        filter_row.addWidget(self._clear_button)

        layout = QVBoxLayout()
        layout.addLayout(pick_row)
        layout.addWidget(hline())
        layout.addLayout(filter_row)
        layout.addStretch()
        self.setLayout(layout)

    def reset(self, points_df: Optional[pl.DataFrame]) -> None:
        self._filters = {}
        self._update_filter_label()
        self._column_picker.clear()
        if points_df is None:
            return
        skip = {"frame", "y", "x", "track_id"}
        numeric_cols = [
            c for c, dt in zip(points_df.columns, points_df.dtypes) if c not in skip and dt.is_numeric()
        ]
        self._column_picker.addItems(numeric_cols)

    def _show_histogram(self) -> None:
        points_df = self.host.points_df
        column = self._column_picker.currentText()
        if points_df is None or not column:
            return
        values = points_df[column].drop_nulls().to_numpy()
        if self._plot_window is None:
            self._plot_window = PlotWindow(f"Data explorer -- {column}", parent=self)
        else:
            self._plot_window.setWindowTitle(f"Data explorer -- {column}")
        import matplotlib.pyplot as plt

        figure = plt.Figure(figsize=(6, 4.5))
        ax = figure.add_subplot(111)
        ax.hist(values, bins=50, color="steelblue")
        ax.set_xlabel(column)
        ax.set_ylabel("count")
        figure.tight_layout()
        self._plot_window.show_figure(figure)

        def _on_select(vmin: float, vmax: float, _column=column) -> None:
            self._filters[_column] = (vmin, vmax)
            self._update_filter_label()
            self.host.on_filters_changed()

        self._span_selector = SpanSelector(
            ax, _on_select, "horizontal", useblit=True, props=dict(alpha=0.3, facecolor="crimson")
        )

    def _clear_filters(self) -> None:
        self._filters = {}
        self._update_filter_label()
        self.host.on_filters_changed()

    def _update_filter_label(self) -> None:
        if not self._filters:
            self._filter_label.setText("no filters active")
            return
        n_tracks = None
        ids = self.filtered_track_ids()
        if ids is not None:
            n_tracks = len(ids)
        parts = [f"{col}: [{lo:.4g}, {hi:.4g}]" for col, (lo, hi) in self._filters.items()]
        suffix = f" -- {n_tracks} tracks pass" if n_tracks is not None else ""
        self._filter_label.setText("; ".join(parts) + suffix)

    def filtered_track_ids(self) -> Optional[set]:
        points_df = self.host.points_df
        if not self._filters or points_df is None:
            return None
        expr = None
        for col, (lo, hi) in self._filters.items():
            if col not in points_df.columns:
                continue
            cond = (pl.col(col) >= lo) & (pl.col(col) <= hi)
            expr = cond if expr is None else expr & cond
        if expr is None:
            return None
        bad_ids = set(points_df.filter(~expr)["track_id"].unique().to_list())
        all_ids = set(points_df["track_id"].unique().to_list())
        return all_ids - bad_ids


class _TrackExplorerTab(QWidget):
    """The Track Explorer table is filtered live by two independent
    controls that AND together (see `DiffusionAnalysisWidget.
    combined_filtered_track_ids`): this tab's own `min_track_length`
    spinbox (a direct, always-visible track-level filter -- the
    complement to `_DataExplorerTab`'s per-point-quality range filters,
    which need a histogram opened first). Both immediately reshape what's
    shown here, not just what a fit run is optionally restricted to."""

    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host
        self._suppress_selection_signal = False

        self._min_track_length = QSpinBox()
        self._min_track_length.setRange(1, 10_000)
        self._min_track_length.setValue(1)
        self._min_track_length.setToolTip("Hide tracks shorter than this (1 = show everything).")
        self._min_track_length.valueChanged.connect(lambda _v: self.host.on_filters_changed())
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("min track length:"))
        filter_row.addWidget(self._min_track_length)
        filter_row.addStretch()

        self._model = DataFrameTableModel()
        self.table = QTableView()
        self.table.setModel(self._model)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        layout = QVBoxLayout()
        layout.addLayout(filter_row)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def min_track_length(self) -> int:
        return self._min_track_length.value()

    def reset(self) -> None:
        self._min_track_length.blockSignals(True)
        self._min_track_length.setValue(1)
        self._min_track_length.blockSignals(False)
        self._model.setDataFrame(pl.DataFrame())

    def set_dataframe(self, df: pl.DataFrame) -> None:
        current = self.host.selected_track_id
        self._model.setDataFrame(df)
        if current is not None:
            self.select_track_id(current)

    def _on_selection_changed(self, *_args) -> None:
        if self._suppress_selection_signal:
            return
        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            return
        track_id = self._model.row_dict(indexes[0].row()).get("track_id")
        if track_id is not None:
            self.host.on_table_row_selected(int(track_id))

    def select_track_id(self, track_id: int) -> None:
        df = self._model.dataframe()
        if df.height == 0 or "track_id" not in df.columns:
            return
        ids = df["track_id"].to_list()
        if track_id not in ids:
            return
        row = ids.index(track_id)
        self._suppress_selection_signal = True
        self.table.selectRow(row)
        self._suppress_selection_signal = False


class _ClassicalTab(QWidget):
    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host
        self._fit: Optional["dk_analysis.PopulationFit"] = None
        self._plot_window: Optional[PlotWindow] = None

        self._min_track_length = QSpinBox()
        self._min_track_length.setRange(2, 10_000)
        self._min_track_length.setValue(10)
        self._min_track_length.setToolTip(
            "Tracks shorter than this are excluded from the per-track table."
        )
        form = QFormLayout()
        form.addRow("min track length", self._min_track_length)

        self._restrict_checkbox = QCheckBox("restrict to filtered tracks (Data Explorer)")

        self._run_button = QPushButton("Run population fit")
        self._run_button.clicked.connect(self._run)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        style_status_label(self._status)
        run_row = QHBoxLayout()
        run_row.addWidget(self._run_button)
        run_row.addWidget(self._status, 1)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self._restrict_checkbox)
        layout.addLayout(run_row)
        layout.addWidget(self._summary)
        layout.addStretch()
        self.setLayout(layout)

    @property
    def fit(self) -> Optional["dk_analysis.PopulationFit"]:
        return self._fit

    def reset(self) -> None:
        self._fit = None
        self._status.setText("")
        style_status_label(self._status)
        self._summary.setText("")

    def report_saved(self, text: str) -> None:
        self._status.setText(text)
        style_status_label(self._status, "ok")

    def show_loaded_summary(self, text: str) -> None:
        self._summary.setText(text)

    def _run(self) -> None:
        tracks = self.host.diffkit_tracks_for_fit(self._restrict_checkbox.isChecked())
        if tracks is None:
            return
        self._status.setText("running...")
        style_status_label(self._status)
        worker = _run_population_fit_worker(tracks, self.host.dt_s, self._min_track_length.value())
        self.host.start_worker(worker, self._on_finished, self._on_error, [self._run_button])

    def _on_finished(self, fit: "dk_analysis.PopulationFit") -> None:
        self._fit = fit
        self._status.setText(f"fit {fit.per_track.height} tracks")
        style_status_label(self._status, "ok" if fit.per_track.height else "caution")

        normal = fit.ensemble_normal_fit
        anomalous = fit.ensemble_anomalous_fit
        self._summary.setText(
            f"D = {normal.D_um2_s:.4g} um^2/s (R^2={normal.r_squared:.3f})\n"
            f"alpha = {anomalous.alpha:.3f}, "
            f"D_alpha = {anomalous.D_alpha_um2_s_alpha:.4g} um^2/s^alpha "
            f"(R^2={anomalous.r_squared:.3f})\n"
            f"mean localization offset = {fit.mean_localization_offset_um2:.4g} um^2"
        )

        figure = dk_analysis.plot_ensemble_fit(fit.ensemble, normal, anomalous)
        if self._plot_window is None:
            self._plot_window = PlotWindow("Population MSD fit", parent=self)
        self._plot_window.show_figure(figure)

        if fit.per_track.height:
            classical_df = fit.per_track.select(
                "track_id",
                pl.col("D_um2_s").alias("D_classical_um2_s"),
                pl.col("r2_normal").alias("r2_classical"),
                pl.col("alpha").alias("alpha_classical"),
                pl.col("r2_anomalous").alias("r2_classical_anom"),
            )
        else:
            classical_df = None
        self.host.set_classical_results(fit.per_track, classical_df)

    def _on_error(self, exc: Exception) -> None:
        self._status.setText(f"error: {exc}")
        style_status_label(self._status, "error")


class _BayesianTab(QWidget):
    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host
        self._map_full: Optional[pl.DataFrame] = None
        self._map_model: Optional[str] = None
        self._track_plot_window: Optional[PlotWindow] = None

        self._model_picker = QComboBox()
        self._model_picker.addItems(["anomalous", "normal"])

        self._restrict_checkbox = QCheckBox("restrict to filtered tracks (Data Explorer)")

        self._map_button = QPushButton("Run MAP fit (all tracks)")
        self._map_button.clicked.connect(self._run_map)
        self._map_status = QLabel("")
        self._map_status.setWordWrap(True)
        style_status_label(self._map_status)
        map_row = QHBoxLayout()
        map_row.addWidget(self._map_button)
        map_row.addWidget(self._map_status, 1)

        self._method_picker = QComboBox()
        self._method_picker.addItems(["map", "nuts"])
        self._method_picker.setToolTip(
            "map: fast point estimate + interval.\n"
            "nuts: full MCMC posterior (slower -- tens of seconds), "
            "opens a posterior corner-plot pop-up when done."
        )
        self._track_fit_button = QPushButton("Fit selected track")
        self._track_fit_button.clicked.connect(self._run_track_fit)
        self._track_status = QLabel("")
        self._track_status.setWordWrap(True)
        style_status_label(self._track_status)
        track_row = QHBoxLayout()
        track_row.addWidget(self._track_fit_button)
        track_row.addWidget(self._track_status, 1)
        self._track_result_label = QLabel("")
        self._track_result_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("model:"))
        layout.addWidget(self._model_picker)
        layout.addWidget(hline())
        layout.addWidget(QLabel("<b>Spatial MAP (all tracks)</b>"))
        layout.addWidget(self._restrict_checkbox)
        layout.addLayout(map_row)
        layout.addWidget(hline())
        layout.addWidget(QLabel("<b>Per-track</b> (uses the Track Explorer selection)"))
        layout.addWidget(QLabel("method:"))
        layout.addWidget(self._method_picker)
        layout.addLayout(track_row)
        layout.addWidget(self._track_result_label)
        layout.addStretch()
        self.setLayout(layout)

    def reset(self) -> None:
        self._map_full = None
        self._map_model = None
        self._map_status.setText("")
        style_status_label(self._map_status)
        self._track_status.setText("")
        style_status_label(self._track_status)
        self._track_result_label.setText("")

    def _run_map(self) -> None:
        tracks = self.host.diffkit_tracks_for_fit(self._restrict_checkbox.isChecked())
        if tracks is None:
            return
        model = self._model_picker.currentText()
        self._map_status.setText("running...")
        style_status_label(self._map_status)
        worker = _run_bulk_map_worker(tracks, self.host.dt_s, model)
        self.host.start_worker(
            worker, lambda table, m=model: self._on_map_finished(table, m), self._on_map_error, [self._map_button]
        )

    def _on_map_finished(self, table: pl.DataFrame, model: str) -> None:
        self._map_full = table
        self._map_model = model
        self._map_status.setText(f"fit {table.height} tracks")
        style_status_label(self._map_status, "ok" if table.height else "caution")
        map_df = _normalize_map_table(table, model)
        color_by = "alpha_map" if model == "anomalous" else "D_map_um2_s"
        self.host.set_map_results(table, map_df, model, color_by)

    def _on_map_error(self, exc: Exception) -> None:
        self._map_status.setText(f"error: {exc}")
        style_status_label(self._map_status, "error")

    def _run_track_fit(self) -> None:
        track_id = self.host.selected_track_id
        if track_id is None:
            self._track_status.setText("select a track in Track Explorer first")
            style_status_label(self._track_status, "caution")
            return
        track_df = self.host.diffkit_track(track_id)
        if track_df is None or track_df.height == 0:
            return
        model = self._model_picker.currentText()
        method = self._method_picker.currentText()
        self._track_status.setText(
            "running... (NUTS sampling can take tens of seconds)"
            if method == "nuts"
            else "running... (first fit compiles, may take a few seconds)"
        )
        style_status_label(self._track_status)
        worker = _run_track_fit_worker(track_df, self.host.dt_s, model, method)
        self.host.start_worker(worker, self._on_track_fit_finished, self._on_track_fit_error, [self._track_fit_button])

    def _on_track_fit_finished(self, fit: "dk_bayes.TrackFit") -> None:
        self._track_status.setText(f"track {fit.track_id} fit (n={fit.track_length})")
        style_status_label(self._track_status, "ok")

        lines = [f"track {fit.track_id}, model={fit.model}, method={fit.method}"]
        for name, value in fit.params.items():
            lines.append(f"  {name} = {value:.4g}  [{fit.lo[name]:.4g}, {fit.hi[name]:.4g}]")
        self._track_result_label.setText("\n".join(lines))

        if fit.method == "nuts":
            samples_dict, _mcmc = fit.raw
            param_names = list(fit.params.keys())
            flat, _trace = dk_bayes.samples_dict_to_arrays(samples_dict, param_names)
            figure = dk_bayes.plot_posterior_corner(flat, param_names)
            if self._track_plot_window is None:
                self._track_plot_window = PlotWindow("Posterior (per-track)", parent=self)
            self._track_plot_window.show_figure(figure)

        self.host.set_track_fit_result(_track_fit_row(fit))

    def _on_track_fit_error(self, exc: Exception) -> None:
        self._track_status.setText(f"error: {exc}")
        style_status_label(self._track_status, "error")


class DiffusionAnalysisWidget(QWidget):
    def __init__(self, napari_viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer

        self._experiment_dir: Optional[Path] = None
        self.pixel_size_um = 1.0
        self.dt_s = 1.0
        self._diffkit_tracks: Optional[pl.DataFrame] = None
        self._tracks_df_px: Optional[pl.DataFrame] = None
        self._points_df: Optional[pl.DataFrame] = None
        self._base_track_df: Optional[pl.DataFrame] = None
        self._classical_full_df: Optional[pl.DataFrame] = None
        self._classical_df: Optional[pl.DataFrame] = None
        self._map_full_df: Optional[pl.DataFrame] = None
        self._map_model: Optional[str] = None
        self._map_df: Optional[pl.DataFrame] = None
        self._track_fit_rows: list[dict] = []
        self._current_track_id: Optional[int] = None
        self._worker = None

        self._tracks_layer: Optional[Tracks] = None
        self._highlight_layer: Optional[Points] = None
        self._spatial_map_layer: Optional[Points] = None
        self._mouse_callback = None

        self._layer_combo = QComboBox()
        self._layer_combo.currentIndexChanged.connect(self._on_layer_combo_changed)
        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel("Tracks layer:"))
        layer_row.addWidget(self._layer_combo, 1)

        self._source_label = QLabel("no Tracks layer in this viewer")
        self._source_label.setWordWrap(True)

        self._data_explorer = _DataExplorerTab(self)
        self._track_explorer = _TrackExplorerTab(self)
        self._classical = _ClassicalTab(self)
        self._bayesian = _BayesianTab(self)

        tabs = QTabWidget()
        tabs.addTab(self._data_explorer, "Data Explorer")
        tabs.addTab(self._track_explorer, "Track Explorer")
        tabs.addTab(self._classical, "Classical (MSD)")
        tabs.addTab(self._bayesian, "Bayesian")

        self._save_button = QPushButton("Save results")
        self._save_button.clicked.connect(self._save_results)
        self._save_button.setEnabled(False)

        layout = QVBoxLayout()
        layout.addLayout(layer_row)
        layout.addWidget(self._source_label)
        layout.addWidget(tabs, 1)
        layout.addWidget(self._save_button)
        self.setLayout(layout)

        self.viewer.layers.events.inserted.connect(self._refresh_layer_combo)
        self.viewer.layers.events.removed.connect(self._refresh_layer_combo)
        self.viewer.layers.events.reordered.connect(self._refresh_layer_combo)
        self._refresh_layer_combo()

    # -- shared read accessors used by the tabs --

    @property
    def points_df(self) -> Optional[pl.DataFrame]:
        return self._points_df

    @property
    def selected_track_id(self) -> Optional[int]:
        return self._current_track_id

    def diffkit_track(self, track_id: int) -> Optional[pl.DataFrame]:
        if self._diffkit_tracks is None:
            return None
        return self._diffkit_tracks.filter(pl.col("track_id") == track_id)

    def combined_filtered_track_ids(self) -> Optional[set]:
        """The Data Explorer's per-point-quality range filters ANDed with
        the Track Explorer's min-track-length filter -- `None` means "no
        restriction from either source". This is both what the Track
        Explorer table displays (`_rebuild_track_table`) and what a fit
        run optionally restricts to (`diffkit_tracks_for_fit`)."""
        ids = self._data_explorer.filtered_track_ids()
        min_len = self._track_explorer.min_track_length()
        if min_len > 1 and self._base_track_df is not None:
            length_ids = set(
                self._base_track_df.filter(pl.col("track_length") >= min_len)["track_id"].to_list()
            )
            ids = length_ids if ids is None else (ids & length_ids)
        return ids

    def on_filters_changed(self) -> None:
        self._rebuild_track_table()

    def diffkit_tracks_for_fit(self, restrict_to_filtered: bool) -> Optional[pl.DataFrame]:
        if self._diffkit_tracks is None:
            return None
        if not restrict_to_filtered:
            return self._diffkit_tracks
        ids = self.combined_filtered_track_ids()
        if ids is None:
            return self._diffkit_tracks
        return self._diffkit_tracks.filter(pl.col("track_id").is_in(list(ids)))

    def start_worker(self, worker, on_finished, on_error, busy_widgets: list) -> None:
        if self._worker is not None:
            return
        for widget in busy_widgets:
            widget.setEnabled(False)

        def _finished(result, _widgets=busy_widgets) -> None:
            self._worker = None
            for w in _widgets:
                w.setEnabled(True)
            on_finished(result)

        def _errored(exc, _widgets=busy_widgets) -> None:
            self._worker = None
            for w in _widgets:
                w.setEnabled(True)
            on_error(exc)

        worker.returned.connect(_finished)
        worker.errored.connect(_errored)
        self._worker = worker
        worker.start()

    # -- tracks-layer selection --

    def _tracks_layers(self) -> list[Tracks]:
        return [layer for layer in self.viewer.layers if isinstance(layer, Tracks)]

    def _refresh_layer_combo(self, event=None) -> None:
        current = self._layer_combo.currentData()
        layers = self._tracks_layers()
        self._layer_combo.blockSignals(True)
        self._layer_combo.clear()
        for layer in layers:
            self._layer_combo.addItem(layer.name, layer)
        if layers:
            idx = next((i for i, layer in enumerate(layers) if layer is current), 0)
            self._layer_combo.setCurrentIndex(idx)
        self._layer_combo.blockSignals(False)

        if not layers:
            self._clear_loaded_state()
        elif current not in layers:
            self._on_layer_combo_changed(self._layer_combo.currentIndex())

    def _on_layer_combo_changed(self, index: int) -> None:
        layer = self._layer_combo.itemData(index)
        if layer is None:
            self._clear_loaded_state()
        else:
            self._load_from_layer(layer)

    def _detach_mouse_callback(self) -> None:
        if self._mouse_callback is not None and self._tracks_layer is not None:
            try:
                self._tracks_layer.mouse_drag_callbacks.remove(self._mouse_callback)
            except ValueError:
                pass
        self._mouse_callback = None

    def _make_click_callback(self):
        def _on_click(layer, event):
            if event.type != "mouse_press":
                return
            track_id = layer.get_value(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
            if track_id is not None:
                self.on_viewer_track_clicked(int(track_id))

        return _on_click

    def _clear_loaded_state(self) -> None:
        self._detach_mouse_callback()
        self._experiment_dir = None
        self._tracks_layer = None
        self._diffkit_tracks = None
        self._tracks_df_px = None
        self._points_df = None
        self._base_track_df = None
        self._classical_full_df = None
        self._classical_df = None
        self._map_full_df = None
        self._map_model = None
        self._map_df = None
        self._track_fit_rows = []
        self._current_track_id = None
        self._source_label.setText("no Tracks layer in this viewer")
        self._data_explorer.reset(None)
        self._track_explorer.reset()
        self._classical.reset()
        self._bayesian.reset()
        self._save_button.setEnabled(False)
        self._clear_overlay_layers()

    def _load_from_layer(self, layer: Tracks) -> None:
        props = layer.properties
        if "sigma_x" not in props or "sigma_y" not in props:
            self._clear_loaded_state()
            self._source_label.setText(
                f"'{layer.name}' has no sigma_x/sigma_y properties -- only Tracks "
                "layers produced by this project's pipeline can be analyzed here."
            )
            return

        self._detach_mouse_callback()
        self._tracks_layer = layer

        data = layer.data
        base_cols = {
            "track_id": data[:, 0].astype(np.int64),
            "frame": data[:, 1].astype(np.int64),
            "y": data[:, 2],
            "x": data[:, 3],
        }
        # Every per-vertex property the layer carries (sigma_x/y plus
        # whatever QC/derived columns viewer.py put there -- amplitude,
        # bg, psf_sigma, track_length, ...), aligned with track_id: one
        # table serves both diffusionkit conversion (needs track_id/frame/
        # y/x/sigma_x/sigma_y) and the Data Explorer's QC histograms
        # (wants everything else too).
        extra_cols = {name: np.asarray(values) for name, values in props.items() if name != "track_id"}
        track_points_df = pl.DataFrame({**base_cols, **extra_cols})
        self._tracks_df_px = track_points_df
        self._points_df = track_points_df

        raw_experiment_dir = layer.metadata.get("experiment_dir")
        self._experiment_dir = Path(raw_experiment_dir) if raw_experiment_dir else None
        self.pixel_size_um = layer.metadata.get("pixel_size_um") or 1.0
        self.dt_s = layer.metadata.get("dt_s") or 1.0
        self._diffkit_tracks = tracks_to_diffusionkit_df(track_points_df, self.pixel_size_um, self.dt_s)
        self._base_track_df = _base_track_table(self._diffkit_tracks, track_points_df)

        self._classical_full_df = None
        self._classical_df = None
        self._map_full_df = None
        self._map_model = None
        self._map_df = None
        self._track_fit_rows = []
        self._current_track_id = None

        if self._experiment_dir is not None:
            self._source_label.setText(f"'{layer.name}' -> {self._experiment_dir}")
        else:
            self._source_label.setText(
                f"'{layer.name}' (no known experiment bundle -- results can't be saved)"
            )

        self._data_explorer.reset(self._points_df)
        self._track_explorer.reset()
        self._classical.reset()
        self._bayesian.reset()
        self._rebuild_track_table()
        self._clear_overlay_layers()

        self._mouse_callback = self._make_click_callback()
        layer.mouse_drag_callbacks.append(self._mouse_callback)

        saved_per_track, saved_summary = (
            load_diffusion_results(self._experiment_dir) if self._experiment_dir is not None else (None, None)
        )
        n_saved = saved_per_track.height if saved_per_track is not None else 0
        if saved_summary is not None:
            self._classical.show_loaded_summary("Loaded saved results:\n" + _format_summary(saved_summary))
        if n_saved:
            self._classical.report_saved(f"{n_saved} saved fit(s) found")
        self._update_save_enabled()

    def _update_save_enabled(self) -> None:
        has_results = (
            self._classical_full_df is not None or self._map_full_df is not None or bool(self._track_fit_rows)
        )
        self._save_button.setEnabled(has_results and self._experiment_dir is not None)

    # -- track table + selection --

    def _rebuild_track_table(self) -> None:
        if self._base_track_df is None:
            return
        df = self._base_track_df
        if self._classical_df is not None:
            df = df.join(self._classical_df, on="track_id", how="left")
        if self._map_df is not None:
            df = df.join(self._map_df, on="track_id", how="left")
        if self._track_fit_rows:
            fit_df = (
                pl.DataFrame(self._track_fit_rows)
                .group_by("track_id", maintain_order=True)
                .last()
                .select("track_id", "D_track_fit_um2_s", "alpha_track_fit", "model", "method")
                .rename({"model": "track_fit_model", "method": "track_fit_method"})
            )
            df = df.join(fit_df, on="track_id", how="left")

        ids = self.combined_filtered_track_ids()
        if ids is not None:
            df = df.filter(pl.col("track_id").is_in(list(ids)))
        self._track_explorer.set_dataframe(df)

    def set_classical_results(self, full_df: pl.DataFrame, display_df: Optional[pl.DataFrame]) -> None:
        self._classical_full_df = full_df
        self._classical_df = display_df
        self._rebuild_track_table()
        self._update_save_enabled()

    def set_map_results(self, full_df: pl.DataFrame, display_df: pl.DataFrame, model: str, color_by: str) -> None:
        self._map_full_df = full_df
        self._map_model = model
        self._map_df = display_df
        self._rebuild_track_table()
        self._update_spatial_map_layer(display_df, color_by)
        self._update_save_enabled()

    def set_track_fit_result(self, row: dict) -> None:
        self._track_fit_rows.append(row)
        self._rebuild_track_table()
        self._update_save_enabled()

    def on_table_row_selected(self, track_id: int) -> None:
        self._current_track_id = track_id
        self._update_highlight_layer(track_id)

    def on_viewer_track_clicked(self, track_id: int) -> None:
        self._current_track_id = track_id
        self._update_highlight_layer(track_id)
        self._track_explorer.select_track_id(track_id)

    # -- viewer overlays --

    def _clear_overlay_layers(self) -> None:
        if self._highlight_layer is not None:
            self._highlight_layer.data = np.empty((0, 2))
        if self._spatial_map_layer is not None:
            self._spatial_map_layer.data = np.empty((0, 2))

    def _update_highlight_layer(self, track_id: Optional[int]) -> None:
        if self._tracks_df_px is None or track_id is None:
            return
        track = self._tracks_df_px.filter(pl.col("track_id") == track_id)
        positions = track.select("y", "x").to_numpy()
        if self._highlight_layer is None:
            self._highlight_layer = self.viewer.add_points(
                positions, name="selected track", **_HIGHLIGHT_POINTS_STYLE
            )
        else:
            self._highlight_layer.data = positions

    def _update_spatial_map_layer(self, map_df: pl.DataFrame, color_by: str) -> None:
        if self._base_track_df is None:
            return
        merged = self._base_track_df.select("track_id", "y_px", "x_px").join(
            map_df, on="track_id", how="inner"
        )
        if merged.height == 0:
            return
        positions = merged.select("y_px", "x_px").to_numpy()
        values = merged[color_by].fill_null(merged[color_by].mean() or 0.0).to_numpy()
        if self._spatial_map_layer is None:
            self._spatial_map_layer = self.viewer.add_points(
                positions,
                name="diffusion map",
                features={color_by: values},
                face_color=color_by,
                face_colormap="viridis",
                symbol="disc",
                size=6,
                border_color="black",
                border_width=0.1,
            )
        else:
            self._spatial_map_layer.data = positions
            self._spatial_map_layer.features = {color_by: values}
            self._spatial_map_layer.face_color = color_by
            self._spatial_map_layer.face_colormap = "viridis"

    # -- persistence --

    def _save_results(self) -> None:
        if self._experiment_dir is None:
            return
        tables = []
        if self._classical_full_df is not None:
            tables.append(self._classical_full_df.with_columns(pl.lit("classic_msd").alias("method")))
        if self._map_full_df is not None:
            tables.append(
                self._map_full_df.with_columns(pl.lit(f"bayes_map_bulk_{self._map_model}").alias("method"))
            )
        if self._track_fit_rows:
            rows = []
            for row in self._track_fit_rows:
                tagged = dict(row)
                tagged["method"] = f"bayes_{row['method']}_{row['model']}"
                rows.append(tagged)
            tables.append(pl.DataFrame(rows))
        if not tables:
            return
        per_track_df = tables[0] if len(tables) == 1 else pl.concat(tables, how="diagonal_relaxed")

        summary = {}
        if self._classical_full_df is not None and self._classical.fit is not None:
            fit = self._classical.fit
            summary = {
                "ensemble_n_points_used": fit.ensemble_n_points_used,
                "mean_localization_offset_um2": fit.mean_localization_offset_um2,
                "normal_D_um2_s": fit.ensemble_normal_fit.D_um2_s,
                "normal_r_squared": fit.ensemble_normal_fit.r_squared,
                "anomalous_alpha": fit.ensemble_anomalous_fit.alpha,
                "anomalous_D_alpha_um2_s_alpha": fit.ensemble_anomalous_fit.D_alpha_um2_s_alpha,
                "anomalous_r_squared": fit.ensemble_anomalous_fit.r_squared,
            }

        write_diffusion_results(self._experiment_dir, per_track_df, summary)
        self._classical.report_saved(f"saved {per_track_df.height} fit(s) to {self._experiment_dir}")
