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

- **Data Explorer** -- a stack of histogram range filters
  (`widgets/feature_filters.FeatureFilterPanel`, the same component the
  params panel's Detect and Track tabs use) over per-point quality fields
  (flux, se_y/se_x, bg, fit_sigma, ...) read straight off the selected Tracks
  layer's own per-vertex properties (every points_df column rides along
  there already, aligned with track_id -- see `viewer.py`; the "points"
  layer itself has no track_id, it's the pre-linking detections table).
  A track is kept only if every one of its points passes every cut;
  those cuts AND together with the Track Explorer's two
  (`combined_filtered_track_ids`) into a "tracks that pass every active
  filter" subset that immediately reshapes what the Track Explorer table
  shows, and that Classical/Bayesian fit runs can optionally restrict to
  (a per-tab checkbox).
- **Track Explorer** -- one row per track (`qt_helpers.DataFrameTableModel`
  in a `QTableView`), the single place all per-track numbers now live
  (classical MSD, bulk MAP, and any one-off per-track fit, each in its
  own column group) instead of a `QLabel` text dump, plus its own
  `min_track_length` spinbox and a second `FeatureFilterPanel` over
  whatever columns that table currently holds -- which after a fit run
  includes `D_um2_s` and `alpha`, so a per-track fit result is filterable
  by the same drag as any other feature.
  Selecting a row is
  "the current track" for the Bayesian tab's per-track action, and is
  kept in sync with the viewer both ways: clicking a track in the Tracks
  layer selects its row here (napari's own `TrackManager.get_value` --
  see `DiffusionAnalysisWidget._make_click_callback` -- already resolves
  a click to a `track_id`, no proxy layer needed), and selecting a row
  here boxes that track in the viewer (`self._highlight_layer`, a
  `Shapes` layer holding one rectangle on the track's principal axes --
  see `oriented_track_box` -- since Tracks layers have no
  selection-highlight of their own).
- **Classical (MSD)** -- `analysis.fit_population`, visually and
  column-wise separate from the Bayesian numbers per-request.
- **Bayesian** -- *Spatial MAP*: `diffusionkit.bayes.fit_population`
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

import numpy as np
import polars as pl
from diffusionkit import bayes as dk_bayes
from diffusionkit import classic as dk_analysis
from diffusionkit.bayes import anisotropy as dk_anisotropy
from diffusionkit.bayes import viz as dk_bayes_viz
from diffusionkit.classic import viz as dk_analysis_viz
from napari.layers import Points, Shapes, Tracks
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
from spt_pipeline.joint_plot import numeric_columns, plot_property_joint
from spt_pipeline.pipeline import filter_mask
from spt_pipeline.widgets.feature_filters import FeatureFilterPanel
from spt_pipeline.widgets.qt_helpers import (
    DataFrameTableModel,
    HistogramRangeWidget,
    JointPlotControl,
    PlotWindow,
    hline,
    style_status_label,
    tabify_with_open_widget,
)

# Look for the two viewer overlays this widget owns -- kept visually
# distinct from DETECTED_POINTS_STYLE's magenta "+" (viewer.py) so a
# highlighted/mapped track never gets mistaken for a raw detection.
#
# The selected track is one rotated rectangle (`oriented_track_box`)
# rather than a marker per vertex: at a marker's scale a selected track is
# indistinguishable from the detections around it until you're already
# zoomed onto it, whereas a box reads at any zoom, and it leaves the
# Tracks layer's own colored path -- which is what actually shows where
# the molecule went -- unobscured. Drawn on the track's principal axes, so
# its aspect ratio is a free read on whether the motion is anisotropic,
# ahead of (and independent of) the Bayesian tab's formal test.
_TRACK_BOX_STYLE = dict(
    shape_type="rectangle",
    face_color="transparent",
    edge_color="yellow",
    # Data pixels, like every other size here. Thicker than the hairline
    # the detection markers use -- this one is meant to be findable in a
    # zoomed-out field, not to sit unobtrusively on top of a PSF.
    edge_width=1.0,
)
_TRACK_BOX_TEXT = dict(
    string="track {track_id}",
    size=8,
    color="yellow",
    anchor="upper_left",
    translation=[-4, 0],
)
# Padding around the track's own extent, and a floor on the half-extent so
# a confined (or single-point) track still gets a box you can see instead
# of a sub-pixel sliver -- the confined case being exactly the one worth
# looking at. Both pad the short side as much as the long one, so for a
# track only a few pixels across the box's aspect ratio understates the
# anisotropy; it's a read to follow up in the Bayesian tab, not a measure.
_TRACK_BOX_PAD_PX = 2.0
_TRACK_BOX_MIN_HALF_PX = 4.0


def oriented_track_box(
    positions: np.ndarray,
    pad_px: float = _TRACK_BOX_PAD_PX,
    min_half_px: float = _TRACK_BOX_MIN_HALF_PX,
) -> np.ndarray:
    """The four corners, in order, of the padded rectangle enclosing an
    (N, 2) array of y/x track positions on its own principal axes.

    The box is built in the PCA frame and mapped back, so it is centred on
    the track's centroid and its long side follows the track's dominant
    direction: for a directed or confined-but-elongated track it hugs the
    motion instead of the much larger axis-aligned box a diagonal track
    would get. Degenerate input (one point, or every point identical) has
    no principal direction, so it falls back to the image axes, where
    `min_half_px` keeps the result visible."""
    positions = np.asarray(positions, dtype=float)
    center = positions.mean(axis=0)
    centered = positions - center

    axes = np.eye(2)
    if len(positions) > 1:
        # eigh (not eig) since a covariance matrix is symmetric: real
        # eigenvalues, orthonormal eigenvectors, ascending order -- so the
        # principal axis is the LAST column, and [::-1] puts it first.
        _, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
        axes = eigenvectors.T[::-1]

    # Half-extent along each axis measured from the centroid, so the box is
    # symmetric about it -- max|projection| rather than the projections'
    # own range, which would only be centred for a symmetric track.
    half = np.abs(centered @ axes.T).max(axis=0) + pad_px
    half = np.maximum(half, min_half_px)

    signs = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    return center + (signs * half) @ axes


@thread_worker(start_thread=False)
def _run_population_fit_worker(
    diffkit_tracks: pl.DataFrame, dt_s: float, min_track_length: int
) -> "dk_analysis.PopulationFit":
    return dk_analysis.fit_population(diffkit_tracks, dt_s, min_track_length=min_track_length)


@thread_worker(start_thread=False)
def _run_bulk_map_worker(diffkit_tracks: pl.DataFrame, dt_s: float, model: str) -> pl.DataFrame:
    # No `engine=`: diffusionkit now always uses the batched exact-MAP
    # engine here. It dropped the SVI alternative because SVI reported
    # uncertainty 3-10x too narrow, so there is no longer a choice to pass.
    return dk_bayes.fit_population(diffkit_tracks, dt_s, model=model, show_progress=False)


@thread_worker(start_thread=False)
def _run_track_fit_worker(
    track_df: pl.DataFrame, dt_s: float, model: str, method: str
) -> "dk_bayes.TrackFit":
    return dk_bayes.fit_track(track_df, dt_s, model=model, method=method)


@thread_worker(start_thread=False)
def _run_anisotropy_worker(
    diffkit_tracks: pl.DataFrame,
    dt_s: float,
    min_track_length: int,
) -> pl.DataFrame:
    return dk_anisotropy.analyze(
        diffkit_tracks,
        dt_s,
        min_track_length=min_track_length,
        show_progress=False,
    )


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
    """`diffusionkit.bayes.fit_population`'s output, trimmed to one
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


_ANISOTROPY_DISPLAY_COLUMNS = [
    "track_id",
    "log_bf10",
    "log_bf10_stderr",
    "evidence",
    "eps_median",
    "eps_lo",
    "eps_hi",
    "psi_median_rad",
    "D_arith_mean_median_um2_s",
    "D_par_median_um2_s",
    "D_perp_median_um2_s",
]


def _normalize_anisotropy_table(per_track: pl.DataFrame) -> pl.DataFrame:
    """`anisotropy.analyze`'s table trimmed to the columns worth showing
    in the Track Explorer / offering as a spatial-map color choice.

    One nested-sampling run now yields the evidence AND the posterior it
    came from, so `eps_*`/`psi_*`/`D_*` are always present -- there is no
    longer a fast-vs-full split to degrade across. The intersection is
    still taken rather than assumed, so a diffusionkit that adds or drops
    a column doesn't break the table."""
    cols = [c for c in _ANISOTROPY_DISPLAY_COLUMNS if c in per_track.columns]
    return per_track.select(cols)


class _DataExplorerTab(QWidget):
    """Per-point QC filtering: a stack of histogram range filters
    (`widgets/feature_filters.FeatureFilterPanel`) over the detections
    behind the loaded tracks -- the same component the Detect tab uses
    while a run is being tuned, here post-hoc on a saved bundle.

    A track is kept only if EVERY one of its points passes every cut. A
    trajectory built partly from detections you have judged untrustworthy
    is not a shorter good trajectory; it is a trajectory whose links were
    scored against points you would have excluded."""

    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host

        self.filters = FeatureFilterPanel(
            noun="points",
            hint="Select a Tracks layer to filter its detections.",
        )
        self.filters.filtersChanged.connect(self.host.on_filters_changed)

        self._track_label = QLabel("")
        self._track_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(self.filters)
        layout.addWidget(hline())
        layout.addWidget(self._track_label)
        layout.addStretch()
        self.setLayout(layout)

    def reset(self, points_df: Optional[pl.DataFrame]) -> None:
        self.filters.set_filters(None)
        self.filters.set_source(points_df)
        self._update_track_label()

    def filtered_track_ids(self) -> Optional[set]:
        """Track ids where every point passes -- `None` for "no cut from
        this tab", which `combined_filtered_track_ids` reads as no
        restriction rather than as an empty set."""
        points_df = self.host.points_df
        filters = self.filters.filters()
        if not filters or points_df is None:
            return None
        passes = filter_mask(points_df, filters)
        bad_ids = set(points_df.filter(~passes)["track_id"].unique().to_list())
        all_ids = set(points_df["track_id"].unique().to_list())
        return all_ids - bad_ids

    def on_filters_changed(self) -> None:
        self._update_track_label()

    def _update_track_label(self) -> None:
        ids = self.filtered_track_ids()
        if ids is None:
            self._track_label.setText("no point filters active")
            return
        points_df = self.host.points_df
        total = points_df["track_id"].n_unique() if points_df is not None else 0
        self._track_label.setText(
            f"{len(ids)} of {total} tracks have every point inside these ranges"
        )


class _TrackExplorerTab(QWidget):
    """The Track Explorer table is filtered live by three independent
    controls that AND together (see `DiffusionAnalysisWidget.
    combined_filtered_track_ids`): this tab's own `min_track_length`
    spinbox, its histogram filters over the table's own per-track
    columns, and `_DataExplorerTab`'s per-point-quality cuts. All three
    immediately reshape what's shown here, not just what a fit run is
    optionally restricted to.

    The histogram filters are the same `FeatureFilterPanel` the params
    panel uses, pointed at this table -- which means they cover whatever
    columns it currently holds. Before any fit that is just
    `track_length`/`mean_step_um`/`duration_s`; after a Classical or
    Bayesian run it is also `D_um2_s`, `alpha` and the rest, so "keep the
    tracks whose fitted alpha is below 0.8 and look at where they are"
    is the same two drags as any other cut. That is the reason a per-track
    filter lives here rather than only on per-point quality.

    The "sync tracks display to filter" checkbox extends that same filter
    to the viewer's own `tracks` layer -- off by default, since it
    rewrites that layer's data/properties (restored from
    `DiffusionAnalysisWidget._tracks_df_px`, the full unfiltered table
    kept around specifically for this) rather than something this widget
    otherwise only reads from. Without it, filtering only ever narrowed
    this table and the spatial map, while the actual trajectories drawn
    in the viewer kept showing everything -- a real inconsistency between
    "what I filtered to" and "what's on screen"."""

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

        self._sync_display_checkbox = QCheckBox("sync tracks display to filter")
        self._sync_display_checkbox.setToolTip(
            "When checked, the viewer's own 'tracks' layer shows only the "
            "currently filtered tracks too, not just this table."
        )
        self._sync_display_checkbox.toggled.connect(lambda _checked: self.host.on_filters_changed())

        self.filters = FeatureFilterPanel(
            noun="tracks",
            hint="Filter on any per-track column below — including fit results once you run one.",
        )
        self.filters.filtersChanged.connect(self.host.on_filters_changed)

        self._model = DataFrameTableModel()
        self.table = QTableView()
        self.table.setModel(self._model)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        layout = QVBoxLayout()
        layout.addLayout(filter_row)
        layout.addWidget(self._sync_display_checkbox)
        layout.addWidget(self.filters)
        layout.addWidget(self.table, 1)
        self.setLayout(layout)

    def min_track_length(self) -> int:
        return self._min_track_length.value()

    def sync_display_enabled(self) -> bool:
        return self._sync_display_checkbox.isChecked()

    def filtered_track_ids(self) -> Optional[set]:
        """Track ids passing this tab's histogram cuts, or `None` for no
        cut. Read against the unfiltered joined table the host keeps
        (`DiffusionAnalysisWidget._joined_track_df`) rather than the
        displayed one, which has already had the other filters applied --
        filtering a filtered table would make each cut depend on the order
        the others were dragged in."""
        df = self.host.joined_track_df
        filters = self.filters.filters()
        if not filters or df is None or df.height == 0:
            return None
        return set(df.filter(filter_mask(df, filters))["track_id"].to_list())

    def set_filter_source(self, df: Optional[pl.DataFrame]) -> None:
        """Point the histograms at the current joined table. Silent, so
        re-running a fit (which adds columns) doesn't re-emit a filter
        change while `_rebuild_track_table` is midway through one."""
        self.filters.set_source(df)

    def reset(self) -> None:
        self._min_track_length.blockSignals(True)
        self._min_track_length.setValue(1)
        self._min_track_length.blockSignals(False)
        self._sync_display_checkbox.blockSignals(True)
        self._sync_display_checkbox.setChecked(False)
        self._sync_display_checkbox.blockSignals(False)
        self.filters.set_filters(None)
        self.filters.set_source(None)
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
        self._joint_plot_window: Optional[PlotWindow] = None

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

        self._joint_plot_control = JointPlotControl()
        self._joint_plot_control.plotRequested.connect(self._show_joint_plot)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self._restrict_checkbox)
        layout.addLayout(run_row)
        layout.addWidget(self._summary)
        layout.addWidget(hline())
        layout.addWidget(QLabel("<b>Joint plot</b> (any two per-track properties)"))
        layout.addWidget(self._joint_plot_control)
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
        self._joint_plot_control.clear()

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
        if fit.per_track.height:
            columns = numeric_columns(fit.per_track)
            self._joint_plot_control.set_columns(columns, prefer_x="D_um2_s", prefer_y="alpha")
        else:
            self._joint_plot_control.clear()

        normal = fit.ensemble_normal_fit
        anomalous = fit.ensemble_anomalous_fit
        self._summary.setText(
            f"D = {normal.D_um2_s:.4g} um^2/s (R^2={normal.r_squared:.3f})\n"
            f"alpha = {anomalous.alpha:.3f}, "
            f"D_alpha = {anomalous.D_alpha_um2_s_alpha:.4g} um^2/s^alpha "
            f"(R^2={anomalous.r_squared:.3f})\n"
            f"mean localization offset = {fit.mean_localization_offset_um2:.4g} um^2"
        )

        figure = dk_analysis_viz.plot_ensemble_fit(fit.ensemble, normal, anomalous)
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

    def _show_joint_plot(self) -> None:
        if self._fit is None or self._fit.per_track.height == 0:
            return
        x_col, y_col, log_x, log_y = self._joint_plot_control.selection()
        if not x_col or not y_col:
            return
        figure = plot_property_joint(
            self._fit.per_track, x_col, y_col, log_x=log_x, log_y=log_y, title="Classical MSD fit"
        )
        if self._joint_plot_window is None:
            self._joint_plot_window = PlotWindow("Classical: joint plot", parent=self)
        self._joint_plot_window.show_figure(figure)


class _BayesianTab(QWidget):
    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host
        self._map_full: Optional[pl.DataFrame] = None
        self._map_model: Optional[str] = None
        self._map_by_model: dict[str, pl.DataFrame] = {}
        self._track_plot_window: Optional[PlotWindow] = None
        self._joint_plot_window: Optional[PlotWindow] = None

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

        self._color_by_picker = QComboBox()
        self._color_by_picker.setEnabled(False)
        self._color_by_picker.currentTextChanged.connect(self._on_color_by_changed)
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("color map by:"))
        color_row.addWidget(self._color_by_picker, 1)
        self._map_histogram = HistogramRangeWidget()
        self._map_histogram.setEnabled(False)
        self._map_histogram.rangeChanged.connect(self._on_map_range_changed)

        self._joint_plot_control = JointPlotControl()
        self._joint_plot_control.setToolTip(
            "Columns from every MAP fit run so far (normal and/or anomalous) "
            "-- e.g. compare the generalized-diffusion K (D_alpha_median_um2_s_alpha) "
            "against the normal-model D (D_median_um2_s), or either against alpha."
        )
        self._joint_plot_control.plotRequested.connect(self._show_joint_plot)

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
        layout.addLayout(color_row)
        layout.addWidget(self._map_histogram)
        layout.addWidget(hline())
        layout.addWidget(QLabel("<b>Joint plot</b> (any two per-track properties)"))
        layout.addWidget(self._joint_plot_control)
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
        self._map_by_model = {}
        self._map_status.setText("")
        style_status_label(self._map_status)
        self._track_status.setText("")
        style_status_label(self._track_status)
        self._track_result_label.setText("")
        self._color_by_picker.blockSignals(True)
        self._color_by_picker.clear()
        self._color_by_picker.blockSignals(False)
        self._color_by_picker.setEnabled(False)
        self._map_histogram.setEnabled(False)
        self._joint_plot_control.clear()

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
        self._map_by_model[model] = table
        self._map_status.setText(f"fit {table.height} tracks")
        style_status_label(self._map_status, "ok" if table.height else "caution")

        combined = self._combined_map_table()
        columns = numeric_columns(combined) if combined is not None else []
        prefer_x = "D_alpha_median_um2_s_alpha" if "D_alpha_median_um2_s_alpha" in columns else "D_median_um2_s"
        prefer_y = "alpha" if "alpha" in columns else None
        self._joint_plot_control.set_columns(columns, prefer_x=prefer_x, prefer_y=prefer_y)

        map_df = _normalize_map_table(table, model)
        color_by = "alpha_map" if model == "anomalous" else "D_map_um2_s"
        self.host.set_map_results(table, map_df, model, color_by)

    def _on_map_error(self, exc: Exception) -> None:
        self._map_status.setText(f"error: {exc}")
        style_status_label(self._map_status, "error")

    def _combined_map_table(self) -> Optional[pl.DataFrame]:
        """Every MAP fit run so far (normal and/or anomalous), left-joined
        on track_id into one wide table -- the two models never share a
        result column name (besides track_id), so this is what lets the
        joint-plot picker offer e.g. the generalized-diffusion K from the
        anomalous fit *and* D from the normal fit at the same time, without
        requiring both to have been run in the same call."""
        tables = list(self._map_by_model.values())
        if not tables:
            return None
        combined = tables[0]
        for table in tables[1:]:
            combined = combined.join(table, on="track_id", how="full", coalesce=True)
        return combined

    def _show_joint_plot(self) -> None:
        combined = self._combined_map_table()
        if combined is None:
            return
        x_col, y_col, log_x, log_y = self._joint_plot_control.selection()
        if not x_col or not y_col:
            return
        figure = plot_property_joint(
            combined, x_col, y_col, log_x=log_x, log_y=log_y, title="Bayesian MAP fit"
        )
        if self._joint_plot_window is None:
            self._joint_plot_window = PlotWindow("Bayesian: joint plot", parent=self)
        self._joint_plot_window.show_figure(figure)

    def _on_color_by_changed(self, column: str) -> None:
        if not column:
            return
        self.host.set_map_color_by(column)
        self._refresh_map_histogram(reset_range=True)

    def _on_map_range_changed(self, vmin: float, vmax: float) -> None:
        self.host.set_map_contrast_limits(vmin, vmax)

    def on_spatial_source_registered(self, preferred_color_by: Optional[str] = None) -> None:
        """Called by the host whenever *any* tab (this one's own MAP fit,
        or the Anisotropy tab) adds a new spatial-map color choice --
        repopulates the "color map by" picker with the union of every
        registered source's columns."""
        self.refresh_color_by_choices(prefer=preferred_color_by)
        self._refresh_map_histogram(reset_range=True)

    def refresh_color_by_choices(self, prefer: Optional[str] = None) -> None:
        columns = self.host.spatial_color_by_columns()
        current = prefer or self._color_by_picker.currentText()
        self._color_by_picker.blockSignals(True)
        self._color_by_picker.clear()
        self._color_by_picker.addItems(columns)
        if current in columns:
            self._color_by_picker.setCurrentText(current)
        elif columns:
            self._color_by_picker.setCurrentText(columns[0])
        self._color_by_picker.blockSignals(False)
        enabled = bool(columns)
        self._color_by_picker.setEnabled(enabled)
        self._map_histogram.setEnabled(enabled)
        if enabled:
            self.host.set_map_color_by(self._color_by_picker.currentText())

    def refresh_map_histogram(self) -> None:
        """Called by the host after filters change -- the map's color
        range stays as the user set it, only the underlying data (and
        thus the histogram bars) needs to reflect the new filtered set."""
        self._refresh_map_histogram(reset_range=False)

    def _refresh_map_histogram(self, reset_range: bool) -> None:
        values = self.host.map_values_for_histogram()
        if values is None or values.size == 0:
            return
        self._map_histogram.blockSignals(True)
        self._map_histogram.set_data(values)
        if reset_range:
            lo, hi = np.percentile(values, [2, 98])
            if lo >= hi:
                lo, hi = self._map_histogram.data_range()
            self._map_histogram.set_range(float(lo), float(hi))
            self.host.set_map_contrast_limits(float(lo), float(hi))
        self._map_histogram.blockSignals(False)

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
            flat, _trace = dk_bayes_viz.samples_dict_to_arrays(samples_dict, param_names)
            figure = dk_bayes_viz.plot_posterior_corner(flat, param_names)
            if self._track_plot_window is None:
                self._track_plot_window = PlotWindow("Posterior (per-track)", parent=self)
            self._track_plot_window.show_figure(figure)

        self.host.set_track_fit_result(_track_fit_row(fit))

    def _on_track_fit_error(self, exc: Exception) -> None:
        self._track_status.setText(f"error: {exc}")
        style_status_label(self._track_status, "error")


class _AnisotropyTab(QWidget):
    """`diffusionkit.bayes.anisotropy.analyze` -- a model-*comparison*
    question (is this track diffusing anisotropically) rather than a point
    estimate, so it gets its own tab rather than a third `model=` choice
    on the Bayesian tab's existing MAP/per-track sections. One bulk run
    over every eligible track, like Bayesian's Spatial MAP; results
    register into the shared spatial-map color-by picker
    (`DiffusionAnalysisWidget.register_spatial_source`) and merge into the
    Track Explorer table, same as every other tab's.

    This tab used to offer two speeds -- a fast log-BF detector and a
    slower "full posterior" pass -- and an upper track-length cap.
    diffusionkit removed all three, and the tab follows rather than
    emulating them:

      - The evidence now comes from nested sampling, and ONE run yields
        both `log_bf10` and the eps/psi/D posterior. There is nothing left
        for a second, slower button to compute.
      - The old cap (`max_track_length=10`) existed because the previous
        Monte-Carlo estimator broke above it, which had the effect of
        restricting the analysis to exactly the tracks too short to answer
        the question. Nested sampling stays valid as tracks get long, so
        only a lower bound remains.
      - There is no ensemble score. Summing per-track log BF10 answers
        "does each track have its own independent anisotropy", not "do
        these tracks share an axis", and it manufactures anisotropy out of
        ordinary D-heterogeneity. The summary below therefore counts
        tracks by Jeffreys-scale `evidence` label instead of reporting one
        pooled number.

    Read `log_bf10` against `log_bf10_stderr`: on a short track the
    sampler's own uncertainty is larger than the evidence, and the
    `evidence` column says so in words rather than leaving a near-zero
    number to be over-read."""

    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host
        self._per_track: Optional[pl.DataFrame] = None
        self._plot_window: Optional[PlotWindow] = None

        self._min_track_length = QSpinBox()
        self._min_track_length.setRange(2, 10_000)
        self._min_track_length.setValue(5)
        self._min_track_length.setToolTip(
            "Skip tracks shorter than this. There is no upper bound: nested "
            "sampling stays valid as a track gets long, and long tracks are "
            "the ones that can actually answer the question."
        )
        form = QFormLayout()
        form.addRow("min track length", self._min_track_length)

        self._restrict_checkbox = QCheckBox("restrict to filtered tracks (Data Explorer)")

        self._run_button = QPushButton("Run anisotropy analysis")
        self._run_button.setToolTip(
            "Nested sampling per track -- scales with track count and can take\n"
            "tens of seconds or more. Yields the evidence and the eps/psi/D\n"
            "posterior in one pass."
        )
        self._run_button.clicked.connect(self._run)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        style_status_label(self._status)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)

        self._log_bf_button = QPushButton("Show log BF10 distribution")
        self._log_bf_button.setEnabled(False)
        self._log_bf_button.clicked.connect(self._show_log_bf_plot)
        self._eps_vs_bf_button = QPushButton("Show eps vs log BF10")
        self._eps_vs_bf_button.setEnabled(False)
        self._eps_vs_bf_button.clicked.connect(self._show_eps_vs_bf_plot)
        self._eps_forest_button = QPushButton("Show eps forest (top tracks)")
        self._eps_forest_button.setEnabled(False)
        self._eps_forest_button.clicked.connect(self._show_eps_forest_plot)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self._restrict_checkbox)
        layout.addWidget(self._run_button)
        layout.addWidget(self._status)
        layout.addWidget(self._summary)
        layout.addWidget(hline())
        layout.addWidget(self._log_bf_button)
        layout.addWidget(self._eps_vs_bf_button)
        layout.addWidget(self._eps_forest_button)
        layout.addStretch()
        self.setLayout(layout)

    def reset(self) -> None:
        self._per_track = None
        self._status.setText("")
        style_status_label(self._status)
        self._summary.setText("")
        self._log_bf_button.setEnabled(False)
        self._eps_vs_bf_button.setEnabled(False)
        self._eps_forest_button.setEnabled(False)

    def _run(self) -> None:
        tracks = self.host.diffkit_tracks_for_fit(self._restrict_checkbox.isChecked())
        if tracks is None:
            return
        self._status.setText("running... (nested sampling per track, can take a while)")
        style_status_label(self._status)
        worker = _run_anisotropy_worker(
            tracks, self.host.dt_s, self._min_track_length.value()
        )
        self.host.start_worker(worker, self._on_finished, self._on_error, [self._run_button])

    def _on_finished(self, per_track: pl.DataFrame) -> None:
        self._per_track = per_track
        n = per_track.height
        self._status.setText(f"analyzed {n} track(s)")
        style_status_label(self._status, "ok" if n else "caution")

        has_eps = "eps_median" in per_track.columns
        if n:
            self._summary.setText(self._summarize(per_track))
        self._log_bf_button.setEnabled(n > 0)
        self._eps_vs_bf_button.setEnabled(has_eps and n > 0)
        self._eps_forest_button.setEnabled(has_eps and n > 0)

        self.host.set_anisotropy_results(per_track, _normalize_anisotropy_table(per_track))

    @staticmethod
    def _summarize(per_track: pl.DataFrame) -> str:
        """Counts by Jeffreys-scale `evidence` label, plus the median eps.

        Deliberately not a pooled score: see this class's docstring for why
        summing per-track log BF10 answers a different question than the
        one it appears to. Counting labels keeps the per-track structure
        visible -- "3 of 200 tracks show strong evidence" is a statement a
        reader can act on, and a sum is not."""
        lines = []
        if "evidence" in per_track.columns:
            counts = (
                per_track.group_by("evidence")
                .len()
                .sort("len", descending=True)
            )
            lines.append(
                "  ".join(
                    f"{row['len']}x {row['evidence']}" for row in counts.iter_rows(named=True)
                )
            )
        if "eps_median" in per_track.columns:
            lines.append(f"median eps = {per_track['eps_median'].median():.3g}")
        return "\n".join(lines)

    def _on_error(self, exc: Exception) -> None:
        self._status.setText(f"error: {exc}")
        style_status_label(self._status, "error")

    def _show_plot(self, figure) -> None:
        if self._plot_window is None:
            self._plot_window = PlotWindow("Anisotropy", parent=self)
        self._plot_window.show_figure(figure)

    def _show_log_bf_plot(self) -> None:
        if self._per_track is not None:
            self._show_plot(dk_anisotropy.plot_log_bf_distribution(self._per_track, "log BF10 per track"))

    def _show_eps_vs_bf_plot(self) -> None:
        if self._per_track is not None:
            self._show_plot(dk_anisotropy.plot_eps_vs_log_bf(self._per_track, "eps vs. log BF10"))

    def _show_eps_forest_plot(self) -> None:
        if self._per_track is not None:
            self._show_plot(dk_anisotropy.plot_eps_forest(self._per_track, "eps posterior, top tracks"))


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
        self._joined_track_df: Optional[pl.DataFrame] = None
        self._classical_full_df: Optional[pl.DataFrame] = None
        self._classical_df: Optional[pl.DataFrame] = None
        self._map_full_df: Optional[pl.DataFrame] = None
        self._map_model: Optional[str] = None
        self._map_df: Optional[pl.DataFrame] = None
        self._map_color_by: Optional[str] = None
        self._anisotropy_full_df: Optional[pl.DataFrame] = None
        # name -> per-track df (track_id + one or more value columns) --
        # every tab that produces a per-track number registers here, and
        # the spatial map / its color-by picker draw from the union of
        # whatever's currently registered (see register_spatial_source).
        self._spatial_sources: dict[str, pl.DataFrame] = {}
        self._track_fit_rows: list[dict] = []
        self._current_track_id: Optional[int] = None
        self._worker = None

        self._tracks_layer: Optional[Tracks] = None
        self._highlight_layer: Optional[Shapes] = None
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
        self._anisotropy = _AnisotropyTab(self)

        tabs = QTabWidget()
        tabs.addTab(self._data_explorer, "Data Explorer")
        tabs.addTab(self._track_explorer, "Track Explorer")
        tabs.addTab(self._classical, "Classical (MSD)")
        tabs.addTab(self._bayesian, "Bayesian")
        tabs.addTab(self._anisotropy, "Anisotropy")

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
        tabify_with_open_widget(napari_viewer, self, "ExperimentListWidget")

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

    @property
    def joined_track_df(self) -> Optional[pl.DataFrame]:
        """The per-track table with every computed result joined in, BEFORE
        any filter -- what the Track Explorer's own histogram cuts are read
        against (see `_TrackExplorerTab.filtered_track_ids`)."""
        return self._joined_track_df

    def combined_filtered_track_ids(self) -> Optional[set]:
        """The Data Explorer's per-point-quality cuts ANDed with the Track
        Explorer's min-track-length spinbox and its per-track histogram
        cuts -- `None` means "no restriction from any source". This is both
        what the Track Explorer table displays (`_rebuild_track_table`) and
        what a fit run optionally restricts to
        (`diffkit_tracks_for_fit`)."""
        ids = self._data_explorer.filtered_track_ids()
        min_len = self._track_explorer.min_track_length()
        if min_len > 1 and self._base_track_df is not None:
            length_ids = set(
                self._base_track_df.filter(pl.col("track_length") >= min_len)["track_id"].to_list()
            )
            ids = length_ids if ids is None else (ids & length_ids)
        track_ids = self._track_explorer.filtered_track_ids()
        if track_ids is not None:
            ids = track_ids if ids is None else (ids & track_ids)
        return ids

    def on_filters_changed(self) -> None:
        self._data_explorer.on_filters_changed()
        self._rebuild_track_table()
        self._update_spatial_map_layer()
        self._bayesian.refresh_map_histogram()

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

    def _restore_previous_tracks_layer_if_synced(self) -> None:
        """If "sync tracks display to filter" left the *previous*
        tracks layer showing a filtered subset, put it back to its full
        set before this widget stops tracking it -- otherwise switching
        away (a different Tracks layer, or no layer at all) leaves that
        old layer silently stuck filtered in the viewer."""
        if (
            self._tracks_layer is not None
            and self._tracks_df_px is not None
            and self._track_explorer.sync_display_enabled()
        ):
            self._set_tracks_layer_data(self._tracks_df_px)

    def _clear_loaded_state(self) -> None:
        self._restore_previous_tracks_layer_if_synced()
        self._detach_mouse_callback()
        self._experiment_dir = None
        self._tracks_layer = None
        self._diffkit_tracks = None
        self._tracks_df_px = None
        self._points_df = None
        self._base_track_df = None
        self._joined_track_df = None
        self._classical_full_df = None
        self._classical_df = None
        self._map_full_df = None
        self._map_model = None
        self._map_df = None
        self._map_color_by = None
        self._anisotropy_full_df = None
        self._spatial_sources = {}
        self._track_fit_rows = []
        self._current_track_id = None
        self._source_label.setText("no Tracks layer in this viewer")
        self._data_explorer.reset(None)
        self._track_explorer.reset()
        self._classical.reset()
        self._bayesian.reset()
        self._anisotropy.reset()
        self._save_button.setEnabled(False)
        self._clear_overlay_layers()

    def _load_from_layer(self, layer: Tracks) -> None:
        props = layer.properties
        # `se_y`/`se_x` are spotsolve's per-detection CRLB (see
        # spt_pipeline.diffusion for the rename to diffusionkit's
        # `sigma_*`). Requiring them is how a Tracks layer from this
        # pipeline is told apart from any other Tracks layer in the
        # viewer: without a position error there is no noise term to fit,
        # so the check is load-bearing, not cosmetic.
        if "se_x" not in props or "se_y" not in props:
            self._clear_loaded_state()
            self._source_label.setText(
                f"'{layer.name}' has no se_x/se_y properties -- only Tracks "
                "layers produced by this project's pipeline can be analyzed here."
            )
            return

        self._restore_previous_tracks_layer_if_synced()
        self._detach_mouse_callback()
        self._tracks_layer = layer

        data = layer.data
        base_cols = {
            "track_id": data[:, 0].astype(np.int64),
            "frame": data[:, 1].astype(np.int64),
            "y": data[:, 2],
            "x": data[:, 3],
        }
        # Every per-vertex property the layer carries (se_y/se_x plus
        # whatever QC/derived columns viewer.py put there -- flux,
        # flux_snr, bg, fit_sigma, sigma_ratio, track_length, ...),
        # aligned with track_id: one table serves both the diffusionkit
        # conversion (needs track_id/frame/y/x/se_y/se_x) and the Data
        # Explorer's QC histograms (wants everything else too).
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
        self._map_color_by = None
        self._anisotropy_full_df = None
        self._spatial_sources = {}
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
        self._anisotropy.reset()
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
            self._classical_full_df is not None
            or self._map_full_df is not None
            or self._anisotropy_full_df is not None
            or bool(self._track_fit_rows)
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
        anisotropy_df = self._spatial_sources.get("anisotropy")
        if anisotropy_df is not None:
            df = df.join(anisotropy_df, on="track_id", how="left")
        if self._track_fit_rows:
            fit_df = (
                pl.DataFrame(self._track_fit_rows)
                .group_by("track_id", maintain_order=True)
                .last()
                .select("track_id", "D_track_fit_um2_s", "alpha_track_fit", "model", "method")
                .rename({"model": "track_fit_model", "method": "track_fit_method"})
            )
            df = df.join(fit_df, on="track_id", how="left")

        # Keep the unfiltered join around and hand it to the Track
        # Explorer's histograms BEFORE narrowing: a filter has to be drawn
        # against the whole population, or each cut would reshape the
        # distribution the next one is chosen on.
        self._joined_track_df = df
        self._track_explorer.set_filter_source(df)

        ids = self.combined_filtered_track_ids()
        if ids is not None:
            df = df.filter(pl.col("track_id").is_in(list(ids)))
        self._track_explorer.set_dataframe(df)

        if self._current_track_id is not None and self._current_track_id not in set(df["track_id"].to_list()):
            # The selected track just got filtered out -- clear it rather
            # than leave a stale highlight in the viewer and a stale
            # target for "Fit selected track" pointing at a track that
            # isn't even in the table anymore.
            self._current_track_id = None
            self._clear_highlight_layer()

        self._sync_tracks_layer_display(ids)

    def set_classical_results(self, full_df: pl.DataFrame, display_df: Optional[pl.DataFrame]) -> None:
        self._classical_full_df = full_df
        self._classical_df = display_df
        self._rebuild_track_table()
        self._update_save_enabled()

    def register_spatial_source(self, name: str, df: pl.DataFrame) -> None:
        """`df` must have `track_id` plus one or more value columns --
        registers it as a spatial-map color choice (`spatial_color_by_columns`),
        available to whichever tab produced it and every other one."""
        self._spatial_sources[name] = df

    def spatial_color_by_columns(self) -> list[str]:
        columns: list[str] = []
        for df in self._spatial_sources.values():
            columns.extend(c for c in df.columns if c != "track_id" and c not in columns)
        return columns

    def _table_for_column(self, column: str) -> Optional[pl.DataFrame]:
        for df in self._spatial_sources.values():
            if column in df.columns:
                return df.select("track_id", column)
        return None

    def set_map_results(self, full_df: pl.DataFrame, display_df: pl.DataFrame, model: str, color_by: str) -> None:
        self._map_full_df = full_df
        self._map_model = model
        self._map_df = display_df
        self.register_spatial_source("bulk_map", display_df)
        self._rebuild_track_table()
        self._bayesian.on_spatial_source_registered(color_by)
        self._update_save_enabled()

    def set_anisotropy_results(self, full_df: pl.DataFrame, display_df: pl.DataFrame) -> None:
        self._anisotropy_full_df = full_df
        self.register_spatial_source("anisotropy", display_df)
        self._rebuild_track_table()
        self._bayesian.on_spatial_source_registered("log_bf10")
        self._update_save_enabled()

    def set_map_color_by(self, color_by: str) -> None:
        self._map_color_by = color_by
        self._update_spatial_map_layer()

    def set_map_contrast_limits(self, vmin: float, vmax: float) -> None:
        if self._spatial_map_layer is not None and vmin < vmax:
            self._spatial_map_layer.face_contrast_limits = (vmin, vmax)

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
        self._clear_highlight_layer()
        if self._spatial_map_layer is not None:
            self._spatial_map_layer.data = np.empty((0, 2))

    def _clear_highlight_layer(self) -> None:
        if self._highlight_layer is not None:
            self._highlight_layer.data = []

    def _update_highlight_layer(self, track_id: Optional[int]) -> None:
        """Box the selected track (see `oriented_track_box`). The box is
        2D, so napari shows it on every frame rather than only the ones
        the track is alive for -- "where is the track I just selected" is
        a question asked while scrubbing through the stack."""
        if self._tracks_df_px is None or track_id is None:
            return
        track = self._tracks_df_px.filter(pl.col("track_id") == track_id)
        if track.height == 0:
            self._clear_highlight_layer()
            return
        corners = oriented_track_box(track.select("y", "x").to_numpy())
        features = {"track_id": np.array([track_id])}
        if self._highlight_layer is None:
            self._highlight_layer = self.viewer.add_shapes(
                [corners],
                name="selected track",
                features=features,
                text=dict(_TRACK_BOX_TEXT),
                **_TRACK_BOX_STYLE,
            )
        else:
            # Data first, then features: assigning `data` resizes the
            # feature table (filling new rows with defaults), so writing
            # the label before the box would leave the text showing the
            # default track_id of 0 whenever the box had been cleared.
            self._highlight_layer.data = [corners]
            self._highlight_layer.features = features

    def _sync_tracks_layer_display(self, filtered_ids: Optional[set]) -> None:
        """When Track Explorer's "sync tracks display to filter" is on,
        rewrite the viewer's own `tracks` layer to the filtered subset
        (restored from `self._tracks_df_px`, the full unfiltered table
        this widget keeps around specifically for this) -- otherwise
        restore it to the full set, in case it was left filtered from a
        moment ago when the checkbox was on."""
        if self._tracks_layer is None or self._tracks_df_px is None:
            return
        if self._track_explorer.sync_display_enabled() and filtered_ids is not None:
            df = self._tracks_df_px.filter(pl.col("track_id").is_in(list(filtered_ids)))
        else:
            df = self._tracks_df_px
        self._set_tracks_layer_data(df)

    def _set_tracks_layer_data(self, df: pl.DataFrame) -> None:
        layer = self._tracks_layer
        # Tracks.data's own setter clears .features (and with it,
        # color_by falls back to track_id) before this gets a chance to
        # hand the new properties back -- restore whatever it was
        # colored by afterward, or a resync silently strips the
        # track_length coloring viewer.py set up.
        previous_color_by = layer.color_by
        data = df.select("track_id", "frame", "y", "x").to_numpy()
        properties = {c: df[c].to_numpy() for c in df.columns if c not in ("track_id", "frame", "y", "x")}
        layer.data = data
        layer.properties = properties
        if previous_color_by in layer.properties_to_color_by:
            layer.color_by = previous_color_by

    def _filtered_map_df(self) -> Optional[pl.DataFrame]:
        """Whichever registered spatial source has the current color-by
        column, joined to pixel-space centroids and restricted to
        `combined_filtered_track_ids()` -- the *display* subset for both
        the spatial map layer and its remap histogram. The underlying fit
        results (`self._spatial_sources`) are untouched by this -- filters
        only change what's shown, not what was already computed."""
        if self._map_color_by is None or self._base_track_df is None:
            return None
        source = self._table_for_column(self._map_color_by)
        if source is None:
            return None
        merged = self._base_track_df.select("track_id", "y_px", "x_px").join(source, on="track_id", how="inner")
        ids = self.combined_filtered_track_ids()
        if ids is not None:
            merged = merged.filter(pl.col("track_id").is_in(list(ids)))
        return merged

    def map_values_for_histogram(self) -> Optional[np.ndarray]:
        merged = self._filtered_map_df()
        if merged is None or self._map_color_by is None or self._map_color_by not in merged.columns:
            return None
        return merged[self._map_color_by].drop_nulls().to_numpy()

    def _update_spatial_map_layer(self) -> None:
        merged = self._filtered_map_df()
        if merged is None or self._map_color_by is None:
            return
        color_by = self._map_color_by
        merged = merged.filter(pl.col(color_by).is_not_null())
        if merged.height == 0:
            if self._spatial_map_layer is not None:
                self._spatial_map_layer.data = np.empty((0, 2))
            return
        positions = merged.select("y_px", "x_px").to_numpy()
        values = merged[color_by].to_numpy()
        track_ids = merged["track_id"].to_numpy()
        if self._spatial_map_layer is None:
            self._spatial_map_layer = self.viewer.add_points(
                positions,
                name="diffusion map",
                features={color_by: values, "track_id": track_ids},
                face_color=color_by,
                face_colormap="viridis",
                symbol="disc",
                # Small and border-free on purpose: this sits directly on
                # top of the tracks layer, and a size-6+bordered marker
                # was obscuring the trajectory it's meant to annotate.
                size=2.5,
                border_width=0.0,
            )
            callback = self._make_spatial_map_click_callback()
            self._spatial_map_layer.mouse_drag_callbacks.append(callback)
        else:
            self._spatial_map_layer.data = positions
            self._spatial_map_layer.features = {color_by: values, "track_id": track_ids}
            self._spatial_map_layer.face_color = color_by
            self._spatial_map_layer.face_colormap = "viridis"

    def _make_spatial_map_click_callback(self):
        """Click a point on the spatial map -> select that track --
        `Points.get_value` returns an *index* into `.data` (unlike
        `Tracks.get_value`, which returns a `track_id` directly), so this
        looks the id up via the layer's own `track_id` feature."""

        def _on_click(layer, event):
            if event.type != "mouse_press":
                return
            index = layer.get_value(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
            if index is not None:
                track_id = int(layer.features["track_id"].iloc[index])
                self.on_viewer_track_clicked(track_id)

        return _on_click

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
        if self._anisotropy_full_df is not None:
            tables.append(self._anisotropy_full_df.with_columns(pl.lit("bayes_anisotropy").alias("method")))
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
