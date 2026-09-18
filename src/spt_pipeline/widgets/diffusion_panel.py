"""Diffusion-analysis dock widget: pick a Tracks layer already in the
napari viewer and interactively explore its per-track diffusion behavior,
using diffusionkit's classical per-track Brownian MLE, both speeds of its Bayesian fit
(fast batched MAP for a spatial overview, full NUTS posterior for a track
flagged interesting from that overview), and its nested-sampling
anisotropy test.

Layout: one permanent **tracks pane** on top, a stack of **analysis tabs**
below it, and a footer, split by a drag-resizable `QSplitter`.

The tracks pane is not a tab, because everything else in the widget reads
or writes it: every fit merges its results in as new columns, the Bayesian
tab's per-track action operates on whatever row is selected here, and the
filters here decide what a fit runs on. Behind a tab, running a fit and
seeing its result were two different screens, and "Fit selected track"
pointed at a selection you could not see.

- **Tracks pane** -- one row per track (`qtkit.ColumnTableModel` in
  a `QTableView`), the single place all per-track numbers live (classical
  MLE, bulk MAP, anisotropy, and any one-off per-track fit, each in its own
  column group), plus a `min_track_length` spinbox and a
  `FeatureFilterPanel` over whatever columns that table currently holds.
  Those columns include per-point detection quality aggregated to the
  track (`flux_min`, `se_x_max`, ... -- see `_qc_aggregate_table`), so a
  QC cut and a fit-result cut are the same gesture on the same table
  rather than two panels at two granularities.
  Selecting a row is "the current track", kept in sync with the viewer both
  ways: clicking a track in the Tracks layer selects its row here (napari's
  own `TrackManager.get_value` -- see
  `DiffusionAnalysisWidget._make_click_callback` -- already resolves a
  click to a `track_id`, no proxy layer needed), and selecting a row here
  boxes that track in the viewer (`self._highlight_layer`, a `Shapes` layer
  holding one rectangle on the track's principal axes -- see
  `oriented_track_box` -- since Tracks layers have no selection-highlight
  of their own).
- **Classical** -- `diffusionkit.classic.analyze_tracks`: per track, the
  Brownian displacement MLE's D (with blur from the camera exposure
  modelled) and its bootstrap-calibrated non-Brownian score z, read as a
  population (log D vs z plot, mean z ± SE); the old MSD D/alpha fits only
  behind an "MSD comparison" toggle. See `_ClassicalTab`.
- **Bayesian** -- *Spatial MAP*: `diffusionkit.bayes.fit_population`
  batched over every eligible track, fast enough to run on the whole
  field of view; besides filling in tracks-pane columns, it also
  places a `Points` layer in the viewer (`self._spatial_map_layer`, one
  point per track centroid, colored by the fitted parameter) -- the
  actual spatial map, and the reason this analysis benefits from staying
  inside napari next to the image at all. *Per-track*: `bayes.fit_track`
  (`method="map"` or `"nuts"`) against whichever track is selected in the
  tracks pane -- the "promote this one track, flagged interesting
  from the map, to expensive inference" step; `method="nuts"` additionally
  renders a posterior corner plot.
- **Anisotropy** -- `bayes.anisotropy.analyze`; see `_AnisotropyTab`.

Every tab holds a plain reference to this module's
`DiffusionAnalysisWidget` (`self.host`) rather than talking through Qt
signals (unlike `PipelineParamsWidget`'s stage tabs, which only fire a
one-shot "run this stage" signal at their host): these panes read and
write the *same* loaded track set, filters, and current-track selection,
so a signal round-trip would only obscure that they are one shared state.

"Restrict fits to filtered tracks" is one checkbox in the footer, not a
copy per tab. Whether a fit runs on the filtered subset or on everything
is a property of the session, not of which analysis you happen to be
looking at, and three independently-toggled copies of it could disagree
about what the last run actually covered.

Tight space is a hard constraint here, not a polish item: this dock shares
a napari window with the canvas and often with the experiment list, so
every tall region is either collapsible (`qtkit.CollapsibleSection`),
scrollable (`qtkit.scrolled`), or on a splitter, and every status
line comes from `qtkit.status_label` so a long error message can
never pin the dock's minimum width.

All fits run through `napari.qt.threading.thread_worker`, sharing one
`self._worker` slot host-wide (`start_worker`) so only one fit -- of any
kind -- runs at a time. That one slot is also why there is one progress
bar, in the footer, rather than one per tab: it shows tracks done / total
for the bulk fits diffusionkit can report on (classical MLE, MAP,
anisotropy), and a busy indicator for the ones it can't (a single-track
fit).

Track/spatial-map positions used for viewer overlays are kept in
*pixels* (`self._tracks_df_px`, the same coordinate space as the image
and Tracks layer), separate from the *physical-unit* table
(`self._diffkit_tracks`) handed to diffusionkit -- conflating the two
would misplace every overlay relative to the image.

Both unit systems are therefore on screen at once, which is why nothing
here shows a bare number:

  - the tracks pane's headers carry each column's unit
    (`_UnitHeaderModel` over `spt_pipeline.units`), since `se_x_max` (px)
    and `se_x_um_max` (µm) are adjacent columns of the same quantity;
  - each fit readout formats through `units.fmt`, including the per-track
    Bayesian fit, whose parameters are a D, a K, an alpha and a
    localization sigma with four different units (`_PARAM_COLUMNS`);
  - the spatial map gets a written color scale (`_map_scale_label`), a
    napari Points layer colored by a feature having no legend of its own;
  - and the two conversion factors everything above depends on --
    `pixel_size_um`, `dt_s`, read off the Tracks layer's metadata -- are
    shown next to the layer name, including the case where the layer
    carries none and they are placeholders (see `_update_source_label`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from diffusionkit import bayes as dk_bayes
from diffusionkit.bayes import anisotropy as dk_anisotropy
from diffusionkit.bayes import viz as dk_bayes_viz
from diffusionkit.classic import (
    Acquisition,
    ClassicAnalysis,
    MLEOptions,
    MSDOptions,
    analyze_tracks,
)

# Legacy, but still what the tracks pane's shape columns come from; taken
# from its own module rather than through `diffusionkit.classic`, whose
# fallback `__getattr__` warns on every legacy name it resolves.
from diffusionkit.classic.features import track_geometry
from napari.layers import Points, Shapes, Tracks
from napari.qt.threading import thread_worker
from qtpy.QtCore import QObject, Qt, QTimer, Signal
from qtkit import (
    CollapsibleSection,
    ColumnTableModel,
    HistogramRangeWidget,
    double_spinbox,
    flow_row,
    hline,
    note_label,
    scrolled,
    status_label,
    style_status_label,
    table_view,
)
from qtkit.napari import live_layer, tabify_with_open_widget
from qtkit.plot import AxisPicker, PlotWindow
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from spt_pipeline import units
from spt_pipeline.diffusion import (
    mle_rows,
    mle_track_table,
    msd_track_table,
    summarize_mle,
    summarize_mle_by_group,
    track_d_table,
    tracks_to_diffusionkit_df,
)
from spt_pipeline.results import load_diffusion_results, write_diffusion_results
from spt_pipeline.joint_plot import (
    numeric_columns,
    plot_d_histogram,
    plot_d_z_joint,
    plot_property_joint,
)
from spt_pipeline.pipeline import filter_mask
from spt_pipeline.viewer import set_tracks_layer_data
from spt_pipeline.widgets.feature_filters import FeatureFilterPanel

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


# The MSD comparison's lag window: diffusionkit's own default, and the
# window its validation compared the MLE against. Not exposed -- the MSD
# fits are a cross-check here, not an analysis to tune.
_MSD_MAX_LAG = 3


@thread_worker(start_thread=False)
def _run_classical_worker(
    diffkit_tracks: pl.DataFrame,
    dt_s: float,
    exposure_s: float,
    min_frames: int,
    n_boot: int,
    msd_comparison: bool,
    progress,
) -> tuple[ClassicAnalysis, Optional[ClassicAnalysis]]:
    """`(analysis, comparison)`: the Brownian MLE with the real exposure,
    and -- when asked for -- a second pass for the MSD fits alone.

    The second pass exists because diffusionkit only fits MSDs when
    `exposure_s == 0`: they have no blur model, so with the real exposure
    they come back `excluded`. The comparison therefore runs them with
    the exposure treated as 0, which is exactly the assumption that biases
    them, and is labelled as such everywhere it shows. Its MLE rows are
    discarded (and its bootstrap skipped) -- the MLE that counts is the
    first pass's."""
    options = MSDOptions(max_lag=_MSD_MAX_LAG, min_frames=min_frames, localization="provided")
    analysis = analyze_tracks(
        diffkit_tracks,
        Acquisition(dt_s=dt_s, exposure_s=exposure_s),
        options,
        MLEOptions(n_boot=n_boot),
        progress=progress,
    )
    comparison = None
    if msd_comparison:
        comparison = (
            analysis
            if exposure_s == 0
            else analyze_tracks(diffkit_tracks, Acquisition(dt_s=dt_s), options, MLEOptions(n_boot=0))
        )
    return analysis, comparison


@thread_worker(start_thread=False)
def _run_bulk_map_worker(diffkit_tracks: pl.DataFrame, dt_s: float, model: str, progress) -> pl.DataFrame:
    # No `engine=`: diffusionkit now always uses the batched exact-MAP
    # engine here. It dropped the SVI alternative because SVI reported
    # uncertainty 3-10x too narrow, so there is no longer a choice to pass.
    return dk_bayes.fit_population(
        diffkit_tracks, dt_s, model=model, show_progress=False, progress=progress
    )


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
    progress,
) -> pl.DataFrame:
    return dk_anisotropy.analyze(
        diffkit_tracks,
        dt_s,
        min_track_length=min_track_length,
        show_progress=False,
        progress=progress,
    )


class _UnitHeaderModel(ColumnTableModel):
    """`qtkit.ColumnTableModel` with the units in the header.

    The tracks pane's table is where this pipeline's two unit systems
    meet: `se_x_max` is in pixels, `se_x_um_max` and
    `radius_of_gyration_um` in µm, `flux_mean` in camera counts,
    `D_map_um2_s` in µm²/s -- 40-odd columns whose unit is a naming
    convention at best (`_um`) and absent at worst (`flux`, `se_x`,
    `fit_sigma`). So each header shows `spt_pipeline.units.header` (the
    name with its unit bracketed, the unit stated once) and each header's
    tooltip the exact column name, which is what the value is stored and
    filtered under.

    Headers are resolved for the table as a whole (`units.headers`), not
    column by column, so a pair like `se_x_max`/`se_x_um_max` doesn't come
    out as the same word twice.

    Only the header is relabelled: `column_names`/`row_dict`/`find_row`
    and every caller that reaches for `track_id` still see the real
    names."""

    def __init__(self, *args, **kwargs) -> None:
        self._headers: dict[str, str] = {}
        super().__init__(*args, **kwargs)

    def set_columns(self, columns) -> None:
        self._headers = units.headers(columns)
        super().set_columns(columns)

    def headerData(self, section: int, orientation, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if orientation == Qt.Orientation.Horizontal:
            names = self.column_names()
            if 0 <= section < len(names):
                name = names[section]
                if role == Qt.ItemDataRole.DisplayRole:
                    return self._headers.get(name, units.header(name))
                if role == Qt.ItemDataRole.ToolTipRole:
                    return units.tooltip(name)
        return super().headerData(section, orientation, role)


class _ProgressRelay(QObject):
    """Carries diffusionkit's `progress(done, total)` callback -- which it
    calls from the worker thread -- back to the GUI thread. The relay lives
    on the GUI thread, so emitting from the worker is a queued connection
    and the progress bar is only ever touched where Qt allows it."""

    progress = Signal(int, int)


# Per-vertex columns that are position, identity, or already per-track --
# nothing to summarize. `track_length` is taken from the diffusionkit table
# instead (guaranteed present there), and y/x become the centroid.
# `loc_id` is a detection's serial number: its min/mean/max are three
# columns of pure noise in a table that already runs past fifty.
_QC_SKIP_COLUMNS = frozenset({"track_id", "loc_id", "frame", "y", "x", "track_length", "roi_index"})

# Per-track MLE results this widget broadcasts onto the viewer's Tracks
# layer as properties, so the trajectories themselves can be colored by
# them (layer controls -> color by). `log10_D_mle` rather than D itself:
# a Tracks layer colormap spans min..max linearly, and D spans decades.
# Written by this widget, so they are dropped again whenever it reads
# the layer back (`_layer_track_table`) -- otherwise they would come back
# as "detection QC" columns and collide with the fit's own.
_TRACK_COLOR_COLUMNS = ("log10_D_mle", "z_nonbrownian")

# "No exposure seen yet" for `DiffusionAnalysisWidget._layer_exposure_s`,
# distinct from None ("the layer records none").
_UNSEEN = object()

# How each genuinely per-point column is collapsed to one number per track.
# min/max are what actually replaced the old per-point "Data Explorer": its
# rule was "keep a track only if EVERY one of its points passes this range",
# which is `col_min >= lo and col_max <= hi` -- the same cut, expressed as a
# track property, so it can sit in the same filter panel as `alpha` and be
# read against the same table. The mean is the plain descriptive statistic
# the min/max pair does not imply, and the one to filter on when the
# question is about the track's typical quality rather than its worst point.
_QC_AGGREGATIONS = (
    ("min", lambda col: pl.col(col).min()),
    ("mean", lambda col: pl.col(col).mean()),
    ("max", lambda col: pl.col(col).max()),
)


def _qc_aggregate_table(tracks_df_px: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Every per-vertex detection-quality column on the Tracks layer
    (`flux`, `se_y`/`se_x`, `bg`, `fit_sigma`, ...), collapsed to one row
    per track, plus the names of the columns that produced aggregates.

    Columns that are already constant within each track are passed through
    under their own name rather than tripled: `duration_s` and
    `mean_step_um` are per-track numbers that `viewer.py` broadcast onto
    every vertex, and `duration_s_min == duration_s_mean == duration_s_max`
    would be three identical columns and two useless filter choices."""
    numeric = [
        col
        for col, dtype in zip(tracks_df_px.columns, tracks_df_px.dtypes)
        if col not in _QC_SKIP_COLUMNS and dtype.is_numeric()
    ]
    if not numeric:
        return tracks_df_px.select("track_id").unique().sort("track_id"), []

    # One pass to find out which of those are per-point and which are
    # per-track-broadcast, rather than hardcoding a list that would go
    # stale the moment the detector gains a column.
    varies = tracks_df_px.group_by("track_id").agg(
        pl.col(col).n_unique().alias(col) for col in numeric
    )
    per_point = [col for col in numeric if varies[col].max() is not None and varies[col].max() > 1]
    per_track = [col for col in numeric if col not in per_point]

    exprs = [pl.col(col).first().alias(col) for col in per_track]
    for col in per_point:
        exprs.extend(agg(col).alias(f"{col}_{suffix}") for suffix, agg in _QC_AGGREGATIONS)
    return tracks_df_px.group_by("track_id").agg(exprs).sort("track_id"), per_point


def _base_track_table(
    diffkit_tracks: pl.DataFrame, tracks_df_px: pl.DataFrame
) -> tuple[pl.DataFrame, list[str]]:
    """One row per track: identity, position, shape
    (`diffusionkit.classic.track_geometry` -- radius of gyration,
    straightness, ...) and detection-quality context columns that every
    tab's results get left-joined onto, plus the names of
    the aggregate QC columns (so the tracks pane can hide that group from
    the table when it is not being used -- there are three per detector
    field, and they are the widest group in the table).

    Position columns are pixel-space (`tracks_df_px`, the viewer's own
    coordinate system), not the physical-unit table."""
    centroids = tracks_df_px.group_by("track_id").agg(
        pl.col("y").mean().alias("y_px"), pl.col("x").mean().alias("x_px")
    )
    lengths = diffkit_tracks.group_by("track_id").agg(pl.col("track_length").first())
    geometry = track_geometry(diffkit_tracks)
    qc, per_point = _qc_aggregate_table(tracks_df_px)
    qc_columns = [c for c in qc.columns if c != "track_id"]
    table = (
        lengths.join(centroids, on="track_id", how="left")
        .join(geometry, on="track_id", how="left")
        .join(qc, on="track_id", how="left")
        .sort("track_id")
    )
    if "roi" in tracks_df_px.columns:
        # Which region the track was linked in (`rois.label_points`) --
        # one per track, since tracking links each ROI on its own.
        rois = tracks_df_px.group_by("track_id").agg(pl.col("roi").first())
        table = table.join(rois, on="track_id", how="left")
    # Only the tripled columns are the hideable group; the passed-through
    # per-track ones (duration_s, mean_step_um) are core context.
    hideable = [c for c in qc_columns if any(c.startswith(f"{col}_") for col in per_point)]
    return table, hideable


def _shared_tracks_unchanged(old: Optional[pl.DataFrame], new: pl.DataFrame) -> bool:
    """Whether every track_id present in both vertex tables has identical
    vertices in each -- i.e. `new` differs from `old` only by whole tracks
    added or removed (a filter), not by tracks being re-linked. False when
    they share no track at all, since then nothing carries over anyway."""
    if old is None:
        return False
    columns = ["track_id", "frame", "y", "x"]
    shared = old.select("track_id").unique().join(new.select("track_id").unique(), on="track_id")
    if shared.height == 0:
        return False
    a = old.select(columns).join(shared, on="track_id").sort("track_id", "frame")
    b = new.select(columns).join(shared, on="track_id").sort("track_id", "frame")
    return a.equals(b)


def _normalize_map_table(table: pl.DataFrame, model: str) -> pl.DataFrame:
    """`diffusionkit.bayes.fit_population`'s output trimmed to its point
    estimates, under names that stay distinct across models so running
    both keeps both in the tracks pane: `D_map_um2_s` from `normal`,
    `K_map_um2_s_alpha` and `alpha_map` from `anomalous`. K, the
    generalized diffusion coefficient, has units of um^2/s^alpha -- it is
    not a D, and sharing a column with one would put two different
    quantities on the same axis.

    Physical units only. diffusionkit fits D in log space and also returns
    `log10_D`, but that is exactly `log10(D_median)`, so it would be a
    second copy of the same number for the joint plot's "log" checkbox to
    reproduce. The intervals, stderrs, sigma and `converged` stay in the
    full table that Save writes."""
    if model == "normal":
        return table.select("track_id", pl.col("D_median_um2_s").alias("D_map_um2_s"))
    return table.select(
        "track_id",
        pl.col("K_median_um2_s_alpha").alias("K_map_um2_s_alpha"),
        pl.col("alpha").alias("alpha_map"),
    )


# diffusionkit's per-track parameter names mapped to the column names
# `spt_pipeline.units` knows their units by. Only `sigma` actually needs
# the indirection, and it needs it badly: in a `TrackFit` it is the fitted
# LOCALIZATION error in µm (diffusionkit's own bulk table calls it
# `sigma_median_um`), while the same bare name in spotsolve's localization
# table is the PSF width in pixels. One name, two units, two orders of
# magnitude apart.
_PARAM_COLUMNS = {
    "D": "D_um2_s",
    "K": "K_um2_s_alpha",
    "alpha": "alpha",
    "sigma": "sigma_loc_um",
}


def _label_corner_axes(figure, param_names: list[str]) -> None:
    """Put units on `dk_bayes_viz.plot_posterior_corner`'s axes.

    diffusionkit labels them with the bare parameter names ("D", "K",
    "sigma"), which leaves a posterior over D reading the same as one over
    alpha. The corner is a d x d grid in row-major order, labelled along
    the bottom row (x) and the left column below the first row (y); each
    label is redone from `units.mpl_label` of the parameter's column, the
    same labels every other plot here uses."""
    d = len(param_names)
    axes = figure.axes
    if len(axes) != d * d:
        return
    labels = [units.mpl_label(_PARAM_COLUMNS.get(name, name)) for name in param_names]
    for j in range(d):
        axes[(d - 1) * d + j].set_xlabel(labels[j], fontsize=9)
    for i in range(1, d):
        axes[i * d].set_ylabel(labels[i], fontsize=9)


def _label_anisotropy_axes(figure) -> None:
    """Say on diffusionkit's anisotropy plots that eps and log BF10 are
    pure numbers -- their axes read "eps ..." and "log BF10" with no unit,
    which elsewhere in this widget would mean "unit unknown"."""
    for ax in figure.axes:
        for get, set_ in ((ax.get_xlabel, ax.set_xlabel), (ax.get_ylabel, ax.set_ylabel)):
            label = get()
            if label.startswith("eps "):
                set_(r"$\epsilon$" + label[3:] + " (dimensionless)")
            elif label == "log BF10":
                set_(r"$\log$ BF$_{10}$ (dimensionless)")


def _track_fit_row(fit: "dk_bayes.TrackFit") -> dict:
    """One ad-hoc single-track fit (`method` "map" or "nuts"), normalized
    the same way as `_normalize_map_table` so both land in the same
    `D_track_fit_um2_s` (normal) / `K_track_fit_um2_s_alpha` +
    `alpha_track_fit` (anomalous) tracks-pane columns --
    deliberately separate from the bulk MAP columns even when `method`
    happens to be "map" too, since a bulk fit and a one-off single-track
    fit are different actions the user can compare against each other."""
    return {
        "track_id": fit.track_id,
        "model": fit.model,
        "method": fit.method,
        "D_track_fit_um2_s": fit.params.get("D"),
        "K_track_fit_um2_s_alpha": fit.params.get("K"),
        "alpha_track_fit": fit.params.get("alpha"),
    }


def _format_summary(summary: dict) -> str:
    """A saved `diffusion_summary.json` as lines of "key = value unit".

    The keys are already unit-suffixed (`normal_D_um2_s`), which is how
    the file stays readable on its own; `units.fmt` restates the unit
    where it can, so a loaded summary reads like a freshly-computed one
    rather than like raw JSON."""
    lines = "\n".join(
        f"{key} = {units.fmt(value, key)}" for key, value in summary.items() if key != "by_roi"
    )
    return lines + _format_mle_by_roi(summary.get("by_roi"))


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
    in the tracks pane / offering as a spatial-map color choice.

    One nested-sampling run now yields the evidence AND the posterior it
    came from, so `eps_*`/`psi_*`/`D_*` are always present -- there is no
    longer a fast-vs-full split to degrade across. The intersection is
    still taken rather than assumed, so a diffusionkit that adds or drops
    a column doesn't break the table."""
    cols = [c for c in _ANISOTROPY_DISPLAY_COLUMNS if c in per_track.columns]
    return per_track.select(cols)


class _TracksPane(QWidget):
    """The per-track table and everything that decides which rows it shows.

    Not a tab: this is the shared context every analysis tab writes into
    and reads back, so it stays on screen while a fit runs. Three
    independent controls narrow it, ANDing together (see
    `DiffusionAnalysisWidget.combined_filtered_track_ids`): the
    `min_track_length` spinbox, the `FeatureFilterPanel`'s histogram cuts,
    and -- once a fit has added its columns -- cuts on the fit results
    themselves.

    The filter panel covers whatever columns the table currently holds.
    Before any fit that is `track_length`/`duration_s`/`mean_step_um` plus
    the per-point detection quality aggregated to the track (`flux_min`,
    `se_x_max`, ... -- see `_qc_aggregate_table`); after a Classical,
    Bayesian or Anisotropy run it is also `D_um2_s`, `alpha`, `log_bf10`
    and the rest. So "drop the tracks with a bad worst-point localization
    error, then keep the ones whose fitted alpha is below 0.8, and look at
    where they are" is three drags in one panel, against one table, at one
    granularity. That single granularity is the point: this pane replaced
    a separate per-point "Data Explorer" whose cuts were
    "every point must pass", which is a statement about a track's min and
    max and was only ever displayed as a per-point histogram because that
    was the table it happened to hold.

    The "Joint plot" section scatters any two of those same columns
    against each other, over the rows the table is currently showing -- so
    "D against flux_mean, for the tracks with alpha below 0.8" is a cut and
    a plot on one table. It used to be a separate picker on the Classical
    and Bayesian tabs, each over only that tab's own raw fit output; that
    could not put a fit result against a track property at all, and it
    offered diffusionkit's `log10_*` columns next to the physical ones they
    are the log of, duplicating the picker's own "log x"/"log y" toggles.

    Those QC aggregates come three-per-detector-field, which is the widest
    column group in the table and usually not what you are reading, so the
    "QC columns" checkbox hides them from the *view* only -- the filter
    panel keeps offering them either way, since it reads the host's
    unfiltered joined table rather than what is on screen.

    The "sync tracks display to filter" checkbox extends the filter to the
    viewer's own `tracks` layer -- off by default, since it rewrites that
    layer's data/properties (restored from
    `DiffusionAnalysisWidget._tracks_df_px`, the full unfiltered table kept
    around specifically for this) rather than something this widget
    otherwise only reads from. Without it, filtering only ever narrowed
    this table and the spatial map, while the actual trajectories drawn in
    the viewer kept showing everything -- a real inconsistency between
    "what I filtered to" and "what's on screen"."""

    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host
        self._suppress_selection_signal = False
        self._qc_columns: list[str] = []
        self._displayed_df: Optional[pl.DataFrame] = None
        self._joint_plot_window: Optional[PlotWindow] = None

        self._min_track_length = QSpinBox()
        self._min_track_length.setRange(1, 10_000)
        self._min_track_length.setValue(1)
        self._min_track_length.setToolTip(
            "Hide tracks with fewer localizations than this — counted in POINTS, "
            "not seconds (an n-point track spans (n-1) × dt). 1 = show everything."
        )
        self._min_track_length.valueChanged.connect(lambda _v: self.host.on_filters_changed())

        self._sync_display_checkbox = QCheckBox("sync viewer")
        self._sync_display_checkbox.setToolTip(
            "When checked, the viewer's own 'tracks' layer shows only the "
            "currently filtered tracks too, not just this table."
        )
        self._sync_display_checkbox.toggled.connect(lambda _checked: self.host.on_filters_changed())

        self._qc_columns_checkbox = QCheckBox("QC columns")
        self._qc_columns_checkbox.setToolTip(
            "Show the per-point detection-quality aggregates (flux_min, "
            "se_x_max, ...) in the table. They stay available to the filters "
            "below either way -- this only controls the table's width."
        )
        self._qc_columns_checkbox.toggled.connect(lambda _checked: self.host.on_filters_changed())

        # One reflowing row rather than a stacked spinbox row + two
        # checkbox rows: three lines of chrome above a table is most of a
        # short dock's usable height, but a fixed QHBoxLayout of these
        # four controls has a 350px minimum width -- on its own wider than
        # everything else in this widget put together. A FlowLayout wraps
        # to two lines only once the dock is actually too narrow for one.
        # Which region's tracks to show and fit -- only there when the
        # layer's tracks were linked per ROI (`rois.label_points`), so a
        # single-field run doesn't carry a control with one choice.
        self._roi_label = QLabel("ROI:")
        self._roi_picker = QComboBox()
        self._roi_picker.setToolTip(
            "Show (and, with \"restrict fits to filtered tracks\", fit) only the\n"
            "tracks linked inside one ROI. Each ROI was linked on its own, so\n"
            "no track spans two."
        )
        self._roi_picker.currentIndexChanged.connect(lambda _i: self.host.on_filters_changed())
        self._roi_label.setVisible(False)
        self._roi_picker.setVisible(False)

        length_label = QLabel("min length:")
        control_row = flow_row(
            length_label,
            self._min_track_length,
            self._roi_label,
            self._roi_picker,
            self._sync_display_checkbox,
            self._qc_columns_checkbox,
        )

        self.filters = FeatureFilterPanel(
            noun="tracks",
            hint="Filter on any per-track column — detection quality, or fit results once you run one.",
        )
        # On release, not per mouse-move: a change re-joins and redraws the
        # whole table, the spatial map and (if synced) the tracks layer.
        self.filters.filtersCommitted.connect(self.host.on_filters_changed)
        self._filter_section = CollapsibleSection("Filters", self.filters, expanded=False)

        self._joint_plot_control = AxisPicker()
        self._joint_plot_control.setToolTip(
            "Any two per-track columns from the table -- track shape, detection "
            "quality, and every fit run so far -- over the tracks currently shown."
        )
        self._joint_plot_control.plotRequested.connect(self._show_joint_plot)
        self._plot_section = CollapsibleSection("Joint plot", self._joint_plot_control, expanded=False)

        self._model = _UnitHeaderModel()
        # Can carry 40+ columns after three fits (hence fixed-width
        # columns), and is the one thing here that should soak up spare
        # height while giving all of it back on demand.
        self.table = table_view(self._model)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        self._count_label = status_label("")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(control_row)
        layout.addWidget(self._filter_section)
        layout.addWidget(self._plot_section)
        layout.addWidget(self.table, 1)
        layout.addWidget(self._count_label)
        self.setLayout(layout)

    def min_track_length(self) -> int:
        return self._min_track_length.value()

    def sync_display_enabled(self) -> bool:
        return self._sync_display_checkbox.isChecked()

    def show_qc_columns(self) -> bool:
        return self._qc_columns_checkbox.isChecked()

    def set_qc_columns(self, columns: list[str]) -> None:
        """Which columns the "QC columns" checkbox hides, and whether there
        are any to hide at all."""
        self._qc_columns = list(columns)
        self._qc_columns_checkbox.setEnabled(bool(columns))

    def hidden_columns(self) -> list[str]:
        return [] if self.show_qc_columns() else self._qc_columns

    _ALL_ROIS = "All"

    def set_roi_choices(self, names: list[str]) -> None:
        """Offer `names` (the loaded tracks' ROIs) in the ROI picker,
        keeping the current pick if it survived; hidden with fewer than
        two, where there is nothing to choose between."""
        current = self._roi_picker.currentText()
        blocked = self._roi_picker.blockSignals(True)
        self._roi_picker.clear()
        self._roi_picker.addItems([self._ALL_ROIS, *names])
        if current in names:
            self._roi_picker.setCurrentText(current)
        self._roi_picker.blockSignals(blocked)
        visible = len(names) > 1
        self._roi_label.setVisible(visible)
        self._roi_picker.setVisible(visible)

    def selected_roi(self) -> Optional[str]:
        """The ROI picked, or None for all of them."""
        if self._roi_picker.count() <= 2:  # "All" plus at most one ROI
            return None
        text = self._roi_picker.currentText()
        return None if not text or text == self._ALL_ROIS else text

    def filtered_track_ids(self) -> Optional[set]:
        """Track ids passing this pane's histogram cuts, or `None` for no
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
        self._qc_columns_checkbox.blockSignals(True)
        self._qc_columns_checkbox.setChecked(False)
        self._qc_columns_checkbox.blockSignals(False)
        self._qc_columns = []
        self.set_roi_choices([])
        self.filters.set_filters(None)
        self.filters.set_source(None)
        self._model.clear()
        self._displayed_df = None
        self._joint_plot_control.clear()
        self._count_label.setText("")

    def set_dataframe(self, df: pl.DataFrame, total: int) -> None:
        current = self.host.selected_track_id
        hidden = set(self.hidden_columns())
        self._displayed_df = df
        self._model.set_frame(df.select([c for c in df.columns if c not in hidden]))
        self._count_label.setText(f"{df.height} of {total} tracks")
        style_status_label(
            self._count_label, "ok" if df.height else "caution" if total else "neutral"
        )
        if current is not None:
            self.select_track_id(current)

    def set_plot_columns(
        self, df: pl.DataFrame, prefer_x: Optional[str] = None, prefer_y: Optional[str] = None
    ) -> None:
        """Offer every numeric column of the joined table (hidden QC columns
        included) as a joint-plot axis, keeping the current choice unless a
        fit that just finished names a better default."""
        columns = [c for c in numeric_columns(df) if c not in ("y_px", "x_px")]
        self._joint_plot_control.set_columns(
            columns,
            prefer_x=prefer_x if prefer_x in columns else None,
            prefer_y=prefer_y if prefer_y in columns else None,
        )

    def _show_joint_plot(self) -> None:
        df = self._displayed_df
        x_col, y_col, log_x, log_y = self._joint_plot_control.selection()
        if df is None or not x_col or not y_col:
            return
        usable = df.select(x_col, y_col).drop_nulls()
        if log_x:
            usable = usable.filter(pl.col(x_col) > 0)
        if log_y:
            usable = usable.filter(pl.col(y_col) > 0)
        if usable.height < 2:
            self._count_label.setText(
                f"not enough tracks with both {x_col} and {y_col}"
                + (" (> 0 for log)" if log_x or log_y else "")
                + " to plot"
            )
            style_status_label(self._count_label, "caution")
            return
        # No `title=`: `plot_property_joint` then titles the panel with
        # both axes' own labels (quantity + unit), which says more than
        # "Tracks" does when the axes are picked at runtime.
        figure = plot_property_joint(df, x_col, y_col, log_x=log_x, log_y=log_y)
        if self._joint_plot_window is None:
            self._joint_plot_window = PlotWindow("Tracks: joint plot", parent=self)
        self._joint_plot_window.show_figure(figure)

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
        row = self._model.find_row("track_id", track_id)
        if row is None:
            return
        self._suppress_selection_signal = True
        self.table.selectRow(row)
        self.table.scrollTo(self._model.index(row, 0))
        self._suppress_selection_signal = False


# Exposure longer than the frame interval by at most this fraction is read
# as a streaming acquisition (exposure == interval) whose measured median
# interval came out a hair short, and clamped to it -- with a note. Past
# it, the two numbers genuinely disagree and the run is refused.
_EXPOSURE_CLAMP_FRACTION = 0.01

_Z_HELP = (
    "<b>D</b> is per track: the maximum-likelihood diffusion coefficient of its "
    "displacements, modelling the provided localization errors and the motion blur "
    "of the exposure. <b>unresolved</b> means D̂ = 0 &mdash; localization noise "
    "explains all the motion &mdash; and gets an upper limit instead of a value. "
    "The histogram is of log D over resolved tracks; median and IQR are marked."
    "<br><br><b>z</b> (optional) is read across tracks, not per track. It is calibrated so "
    "that it is ~N(0,1) for Brownian tracks of any length, noise or D. At ~5 frames "
    "a single track can't be called non-Brownian (even α = 0.5 is flagged only "
    "4&ndash;8% of the time), but a mean-z shift of 0.4&ndash;0.7 is plain across "
    "~100 tracks. From ~20 frames single tracks become partly informative."
    "<br><br><b>Sign:</b> negative = sub-diffusive or confined; positive = "
    "super-diffusive or directed. z says the motion departs from Brownian, not why."
    "<br><br><b>Calibrate first:</b> run a bead or immobilized control with the same "
    "exposure and localization errors. Its mean z should be ~0. If it isn't, the "
    "exposure or the localization SDs are off, and a biological z shift isn't "
    "interpretable until they are fixed."
)


class _ClassicalTab(QWidget):
    """diffusionkit's rebuilt classical analysis
    (`classic.analyze_tracks`): the per-track Brownian displacement MLE.

    Routine work is D: each track's maximum-likelihood D with its upper
    limit, shown as a log-D histogram with the median and IQR. That costs
    about 2 s for ~500 tracks. The calibrated non-Brownian score z is
    opt-in ("non-Brownian score z"), because it is what the run's time
    goes into -- a parametric bootstrap per track, ~15x the cost -- and
    it changes nothing about D. The old MSD fits stay available only as a
    labelled comparison.

    The one input here that is not already on the layer is the camera
    **exposure** -- separate from the frame interval, and the input the
    MLE's result depends on most: treating a 20 ms exposure as
    instantaneous biases D by about -25% and mean z by +0.3 to +0.7. So it
    is pre-filled from the layer's metadata (the file's record, or the
    override in the experiment list's image panel) and otherwise has to be
    typed -- the box starts at "not set", never at 0, and Run stays off
    until it has a value."""

    # The exposure box's "not set" value -- one step below 0, which is a
    # legitimate (stroboscopic) exposure and must not double as "unknown".
    _EXPOSURE_UNSET = -0.0001

    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host
        self._analysis: Optional[ClassicAnalysis] = None
        self._comparison: Optional[ClassicAnalysis] = None
        self._summary_values: Optional[dict] = None
        # `summarize_mle` per ROI, when the analysed tracks span more than
        # one -- shown under the pooled summary and saved beside it.
        self._summary_by_roi: Optional[dict] = None
        self._plot_window: Optional[PlotWindow] = None
        self._hist_window: Optional[PlotWindow] = None
        self._msd_plot_window: Optional[PlotWindow] = None
        # What the layer said, so the exposure note can say where the box's
        # value came from (and "reset" has something to go back to).
        self._layer_exposure_s: Optional[float] = None

        self._exposure = double_spinbox(
            self._EXPOSURE_UNSET, self._EXPOSURE_UNSET, 3600.0, 0.001, decimals=4, suffix=" s",
            tooltip="Camera exposure per frame -- how long the sensor integrates, not\n"
            "the frame interval. The MLE models the blur of a continuous\n"
            "exposure; entering 0 when the camera really exposed for 20 ms\n"
            "biases D by about -25% and shifts mean z by +0.3 to +0.7.\n\n"
            "Pre-filled from the layer when the file (or the image panel's\n"
            "override) recorded it.",
        )
        self._exposure.setSpecialValueText("not set")
        self._exposure.valueChanged.connect(lambda _v: self.refresh_inputs())
        self._exposure_note = status_label("")

        # A fit requirement, not a display filter: shorter tracks come back
        # `excluded`, and are counted. Distinct from the tracks pane's
        # `min length`, which only decides what the table shows.
        self._min_frames = QSpinBox()
        self._min_frames.setRange(2, 10_000)
        self._min_frames.setValue(MSDOptions().min_frames)
        self._min_frames.setToolTip(
            "Tracks with fewer localizations than this are excluded (and counted\n"
            "as such). diffusionkit's default is 5; the MLE and z are calibrated\n"
            "for short tracks, so there is no need to raise it for accuracy."
        )

        # Off by default: D, its upper limit and p_motion come from the
        # likelihood alone, and the bootstrap that calibrates z is nearly
        # all of a run's cost.
        self._compute_z = QCheckBox("non-Brownian score z")
        self._compute_z.setToolTip(
            "Also compute z, a per-track score calibrated to N(0,1) under\n"
            "Brownian motion, read across tracks (mean z ± SE). It needs a\n"
            "parametric bootstrap per track -- ~50 ms per track at 500 reps,\n"
            "about 15x the cost of D alone -- and leaves D unchanged."
        )
        self._n_boot = QSpinBox()
        self._n_boot.setRange(100, 100_000)
        self._n_boot.setSingleStep(100)
        self._n_boot.setValue(MLEOptions().n_boot)
        self._n_boot.setSuffix(" reps")
        self._n_boot.setToolTip("Parametric-bootstrap replicates per track that calibrate z to N(0,1).")
        self._n_boot.setEnabled(False)
        self._compute_z.toggled.connect(self._n_boot.setEnabled)

        self._msd_comparison = QCheckBox("MSD comparison (D, α)")
        self._msd_comparison.setToolTip(
            "Also fit the old 3-lag MSD models (Brownian D; power-law K and α),\n"
            "for comparison only. They have no confidence intervals and no blur\n"
            "model, so they are run with the exposure treated as 0 -- which biases\n"
            "them on real data -- and their α has a null spread that fills [0, 2]\n"
            "for short tracks. Adds D_msd / α_msd columns and a D vs α plot."
        )

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.addRow("exposure", self._exposure)
        form.addRow("", self._exposure_note)
        form.addRow("min points", self._min_frames)
        form.addRow(flow_row(self._compute_z, self._n_boot))

        self._run_button = QPushButton("Run D (Brownian MLE)")
        self._run_button.clicked.connect(self._run)
        self._status = status_label("")
        self._summary = status_label("")
        self._summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._hist_button = QPushButton("D histogram")
        self._hist_button.setToolTip("Reopen the log D histogram of the last run.")
        self._hist_button.clicked.connect(self._show_histogram)
        self._plot_button = QPushButton("D vs z plot")
        self._plot_button.setToolTip("Reopen the log D vs z population plot (runs with z only).")
        self._plot_button.clicked.connect(self._show_plot)
        self._msd_plot_button = QPushButton("MSD comparison plot")
        self._msd_plot_button.setToolTip(
            "D vs α from the MSD fits (exposure treated as 0; no confidence intervals)."
        )
        self._msd_plot_button.clicked.connect(self._show_msd_plot)
        plot_row = flow_row(self._hist_button, self._plot_button, self._msd_plot_button)

        help_text = note_label(_Z_HELP)
        help_text.setTextFormat(Qt.TextFormat.RichText)
        self._help = CollapsibleSection("Reading D (and z)", help_text, expanded=False)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addLayout(form)
        layout.addWidget(self._msd_comparison)
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
        # The image panel's exposure box writes into the layer's metadata,
        # which napari doesn't announce -- so look again whenever this tab
        # comes into view.
        super().showEvent(event)
        self.host.sync_layer_exposure()

    @property
    def analysis(self) -> Optional[ClassicAnalysis]:
        return self._analysis

    @property
    def summary_values(self) -> Optional[dict]:
        return self._summary_values

    @property
    def summary_by_roi(self) -> Optional[dict]:
        return self._summary_by_roi

    def reset(self) -> None:
        self._analysis = None
        self._comparison = None
        self._summary_values = None
        self._summary_by_roi = None
        self._status.setText("")
        style_status_label(self._status)
        self._summary.setText("")
        self._refresh_plot_buttons()

    def set_layer_exposure(self, exposure_s: Optional[float]) -> None:
        """What the newly loaded layer records -- pre-fills the box, or
        parks it on "not set" so the last layer's value can't carry over
        to an acquisition it doesn't describe."""
        self._layer_exposure_s = exposure_s
        blocked = self._exposure.blockSignals(True)
        self._exposure.setValue(exposure_s if exposure_s is not None else self._EXPOSURE_UNSET)
        self._exposure.blockSignals(blocked)
        self.refresh_inputs()

    def exposure_s(self) -> Optional[float]:
        value = self._exposure.value()
        return value if value >= 0 else None

    def _exposure_for_run(self) -> tuple[Optional[float], str, str]:
        """`(exposure to run with, note, level)`. None when the run can't
        go ahead, with the note saying why."""
        exposure = self.exposure_s()
        dt = self.host.dt_s
        if exposure is None:
            return None, "not recorded on this layer — enter the camera exposure to run", "error"
        if exposure > dt * (1 + _EXPOSURE_CLAMP_FRACTION):
            return (
                None,
                f"longer than the frame interval ({units.fmt_unit(dt, 's/frame')}) — check both",
                "error",
            )
        if exposure > dt:
            return (
                dt,
                f"{units.fmt_unit(exposure, units.SECONDS)} is within 1% over the frame "
                f"interval — run with exposure = interval ({units.fmt_unit(dt, units.SECONDS)})",
                "caution",
            )
        if self._layer_exposure_s is not None and exposure == self._layer_exposure_s:
            return exposure, "from the layer's metadata", "neutral"
        if exposure == 0:
            return exposure, "0 = instantaneous (stroboscopic) — no blur modelled", "caution"
        return exposure, "entered here — not recorded on the layer", "neutral"

    def refresh_inputs(self) -> None:
        exposure, note, level = self._exposure_for_run()
        self._exposure_note.setText(note)
        style_status_label(self._exposure_note, level)
        self._run_button.setEnabled(exposure is not None and self.host.has_tracks)

    def _refresh_plot_buttons(self) -> None:
        self._hist_button.setEnabled(self._analysis is not None)
        self._plot_button.setEnabled(bool((self._summary_values or {}).get("n_z")))
        self._msd_plot_button.setEnabled(self._comparison is not None)

    def report_saved(self, text: str) -> None:
        self._status.setText(text)
        style_status_label(self._status, "ok")

    def show_loaded_summary(self, text: str) -> None:
        self._summary.setText(text)

    def _run(self) -> None:
        tracks = self.host.diffkit_tracks_for_fit()
        exposure, _note, _level = self._exposure_for_run()
        if tracks is None or exposure is None:
            return
        self._status.setText("running...")
        style_status_label(self._status)
        worker = _run_classical_worker(
            tracks,
            self.host.dt_s,
            exposure,
            self._min_frames.value(),
            self._n_boot.value() if self._compute_z.isChecked() else 0,
            self._msd_comparison.isChecked(),
            self.host.progress_callback,
        )
        self.host.start_worker(
            worker,
            self._on_finished,
            self._on_error,
            [self._run_button],
            "Brownian MLE",
            reports_progress=True,
        )

    def _on_finished(self, result: tuple[ClassicAnalysis, Optional[ClassicAnalysis]]) -> None:
        analysis, comparison = result
        self._analysis = analysis
        self._comparison = comparison
        summary = summarize_mle(analysis.fits)
        self._summary_values = summary
        groups = self.host.track_rois()
        if groups is not None:
            groups = groups.filter(pl.col("track_id").is_in(mle_rows(analysis.fits)["track_id"]))
        self._summary_by_roi = (
            summarize_mle_by_group(analysis.fits, groups)
            if groups is not None and groups["group"].n_unique() > 1
            else None
        )
        self._refresh_plot_buttons()
        self.refresh_inputs()

        n_ok = summary.get("n_ok", 0)
        self._status.setText(f"{n_ok} of {summary['n_tracks']} tracks resolved motion")
        style_status_label(self._status, "ok" if summary["median_D_um2_s"] is not None else "caution")
        self._summary.setText(
            _format_mle_summary(summary, analysis) + _format_mle_by_roi(self._summary_by_roi)
        )

        display = mle_track_table(analysis.fits)
        if comparison is not None:
            display = display.join(msd_track_table(comparison.fits), on="track_id", how="left")
        self.host.set_classical_results(analysis, comparison, display)
        if summary["median_D_um2_s"] is not None:
            self._show_histogram()
        if summary["n_z"]:
            self._show_plot()
        if comparison is not None:
            self._show_msd_plot()

    def _show_histogram(self) -> None:
        if self._analysis is None or self._summary_values is None:
            return
        figure = plot_d_histogram(
            mle_rows(self._analysis.fits),
            self._summary_values,
            groups=self.host.track_rois() if self._summary_by_roi else None,
        )
        if self._hist_window is None:
            self._hist_window = PlotWindow("Brownian MLE: D histogram", parent=self)
        self._hist_window.show_figure(figure)

    def _show_plot(self) -> None:
        if self._analysis is None or not (self._summary_values or {}).get("n_z"):
            return
        figure = plot_d_z_joint(
            mle_rows(self._analysis.fits),
            self._summary_values,
            groups=self.host.track_rois() if self._summary_by_roi else None,
        )
        if self._plot_window is None:
            self._plot_window = PlotWindow("Brownian MLE: D vs z", parent=self)
        self._plot_window.show_figure(figure)

    def _show_msd_plot(self) -> None:
        if self._comparison is None:
            return
        df = msd_track_table(self._comparison.fits)
        usable = df.filter(
            pl.col("D_msd_um2_s").is_not_null()
            & (pl.col("D_msd_um2_s") > 0)
            & pl.col("alpha_msd").is_not_null()
        )
        if usable.height < 2:
            self._status.setText("MSD comparison: too few tracks with a positive D and an α to plot")
            style_status_label(self._status, "caution")
            return
        figure = plot_property_joint(
            usable,
            "D_msd_um2_s",
            "alpha_msd",
            log_x=True,
            title="MSD comparison — exposure treated as 0, no CIs",
        )
        if self._msd_plot_window is None:
            self._msd_plot_window = PlotWindow("MSD comparison: D vs α", parent=self)
        self._msd_plot_window.show_figure(figure)

    def _on_error(self, exc: Exception) -> None:
        self._status.setText(f"error: {exc}")
        style_status_label(self._status, "error")


def _format_mle_by_roi(by_roi: Optional[dict]) -> str:
    """One line per ROI under the pooled summary: each region was linked
    on its own, and whether its motion differs from the others' is the
    reason it was drawn."""
    if not by_roi:
        return ""
    lines = ["", "by ROI:"]
    for name, summary in by_roi.items():
        line = (
            f"  {name}: {summary['n_tracks']} tracks · median D = "
            f"{units.fmt(summary.get('median_D_um2_s'), 'D_um2_s')}"
        )
        if summary.get("mean_z") is not None:
            se = summary.get("se_mean_z")
            line += f" · z {summary['mean_z']:+.2f}" + (f" ± {se:.2f}" if se is not None else "")
            line += f" · |z| > 1.96: {100 * summary['frac_abs_z_gt_1_96']:.0f}%"
        lines.append(line)
    return "\n".join(lines)


def _format_mle_summary(summary: dict, analysis: ClassicAnalysis) -> str:
    """The population summary as a few lines of text beside the plot --
    counts by diffusionkit status first, since `unresolved`/`excluded`/
    `invalid_input` tracks are part of the result, not noise to drop."""
    n_ok = summary.get("n_ok", 0)
    others = {
        key[2:]: value
        for key, value in summary.items()
        if key.startswith("n_") and key not in ("n_tracks", "n_ok", "n_z") and not key.startswith("n_frames")
    }
    counts = f"{n_ok} ok"
    if others:
        counts += " · " + " · ".join(f"{count} {status}" for status, count in sorted(others.items()))
    lines = [f"{summary['n_tracks']} tracks: {counts}"]
    if summary.get("n_frames_median") is not None:
        lines.append(
            "length (min / median / max): "
            f"{summary['n_frames_min']:.0f} / {summary['n_frames_median']:.0f} / "
            f"{summary['n_frames_max']:.0f} {units.POINTS}"
        )
    median = f"median D = {units.fmt(summary.get('median_D_um2_s'), 'D_um2_s')}"
    if summary.get("q25_D_um2_s") is not None and summary.get("q75_D_um2_s") is not None:
        median += f" (IQR {summary['q25_D_um2_s']:.3g}–{summary['q75_D_um2_s']:.3g})"
    lines.append(median + ", resolved tracks")
    if summary.get("n_unresolved"):
        lines.append(
            f"{summary['n_unresolved']} unresolved (D̂ = 0): median upper limit "
            f"D < {units.fmt(summary.get('unresolved_D_upper_median_um2_s'), 'D_um2_s')}"
        )
    if summary.get("mean_z") is not None:
        se = summary.get("se_mean_z")
        sd = summary.get("sd_z")
        lines.append(
            f"z: mean {summary['mean_z']:+.3f}"
            + (f" ± {se:.3f} (SE)" if se is not None else "")
            + (f" · SD {sd:.2f}" if sd is not None else "")
        )
        lines.append(
            f"|z| > 1.96: {100 * summary['frac_abs_z_gt_1_96']:.1f}% of tracks (≈5% if Brownian)"
        )
    acquisition = analysis.acquisition
    lines.append(
        f"dt {units.fmt_unit(acquisition.dt_s, 's/frame')} · exposure "
        f"{units.fmt_unit(acquisition.exposure_s, units.SECONDS)} · "
        + (f"z from {analysis.mle_options.n_boot} bootstrap reps" if analysis.mle_options.n_boot else "no z")
    )
    return "\n".join(lines)


class _BayesianTab(QWidget):
    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host
        self._track_plot_window: Optional[PlotWindow] = None

        self._model_picker = QComboBox()
        self._model_picker.addItems(["anomalous", "normal"])
        self._model_picker.setToolTip(
            "Shared by both actions below -- the bulk map and the per-track "
            "fit answer the same question at two costs, so fitting one model "
            "in bulk and a different one per track would make the two "
            "uncomparable."
        )
        model_form = QFormLayout()
        model_form.setContentsMargins(0, 0, 0, 0)
        model_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        model_form.addRow("model", self._model_picker)

        self._map_button = QPushButton("Run MAP fit (all tracks)")
        self._map_button.clicked.connect(self._run_map)
        self._map_status = status_label("")

        self._color_by_picker = QComboBox()
        self._color_by_picker.setEnabled(False)
        self._color_by_picker.currentTextChanged.connect(self._on_color_by_changed)
        # D spans orders of magnitude across a field of tracks, so a linear
        # histogram is one tall spike at the low end -- log bins are what
        # make the rest of the distribution visible enough to drag a handle
        # into.
        self._map_log_scale_check = QCheckBox("log scale")
        self._map_log_scale_check.setEnabled(False)
        self._map_log_scale_check.toggled.connect(self._on_map_log_scale_toggled)
        color_form = QFormLayout()
        color_form.setContentsMargins(0, 0, 0, 0)
        color_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        color_form.addRow("color map by", self._color_by_picker)
        color_form.addRow("", self._map_log_scale_check)
        # What the map's colors mean, in words and in a unit: the picker
        # above it names a column and the histogram below sets the color
        # range, but neither said what the numbers on that range are --
        # so a viridis field of dots carried no scale at all. Updated by
        # `_on_color_by_changed`/`refresh_color_by_choices`.
        self._map_scale_label = status_label("")
        self._map_histogram = HistogramRangeWidget()
        self._map_histogram.setEnabled(False)
        self._map_histogram.rangeChanged.connect(self._on_map_range_changed)

        self._method_picker = QComboBox()
        self._method_picker.addItems(["map", "nuts"])
        self._method_picker.setToolTip(
            "map: fast point estimate + interval.\n"
            "nuts: full MCMC posterior (slower -- tens of seconds), "
            "opens a posterior corner-plot pop-up when done."
        )
        method_form = QFormLayout()
        method_form.setContentsMargins(0, 0, 0, 0)
        method_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        method_form.addRow("method", self._method_picker)
        self._track_fit_button = QPushButton("Fit selected track")
        self._track_fit_button.setToolTip("Fits whichever track is selected in the table above.")
        self._track_fit_button.clicked.connect(self._run_track_fit)
        self._track_status = status_label("")
        self._track_result_label = status_label("")

        map_body = QWidget()
        map_layout = QVBoxLayout(map_body)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(4)
        map_layout.addWidget(self._map_button)
        map_layout.addWidget(self._map_status)
        map_layout.addLayout(color_form)
        map_layout.addWidget(self._map_scale_label)
        map_layout.addWidget(self._map_histogram)

        track_body = QWidget()
        track_layout = QVBoxLayout(track_body)
        track_layout.setContentsMargins(0, 0, 0, 0)
        track_layout.setSpacing(4)
        track_layout.addLayout(method_form)
        track_layout.addWidget(self._track_fit_button)
        track_layout.addWidget(self._track_status)
        track_layout.addWidget(self._track_result_label)

        # Three collapsible sections rather than one long scroll of rules
        # and bold labels: this is the tallest tab, and the two actions are
        # used at different moments (map the whole field first, then
        # promote one track to expensive inference), so only one of them is
        # usually wanted on screen. Spatial MAP starts open because it is
        # the entry point -- the per-track fit needs a selection that the
        # map is how you find.
        self._map_section = CollapsibleSection(
            "Spatial MAP (all tracks)", map_body, expanded=True
        )
        self._track_section = CollapsibleSection("Per-track fit", track_body)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addLayout(model_form)
        layout.addWidget(hline())
        layout.addWidget(self._map_section)
        layout.addWidget(self._track_section)
        layout.addStretch()
        self.setLayout(layout)

    def reset(self) -> None:
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
        self._map_log_scale_check.setEnabled(False)
        self._map_scale_label.setText("")

    def _run_map(self) -> None:
        tracks = self.host.diffkit_tracks_for_fit()
        if tracks is None:
            return
        model = self._model_picker.currentText()
        self._map_status.setText("running...")
        style_status_label(self._map_status)
        worker = _run_bulk_map_worker(tracks, self.host.dt_s, model, self.host.progress_callback)
        self.host.start_worker(
            worker,
            lambda table, m=model: self._on_map_finished(table, m),
            self._on_map_error,
            [self._map_button],
            f"MAP ({model})",
            reports_progress=True,
        )

    def _on_map_finished(self, table: pl.DataFrame, model: str) -> None:
        self._map_status.setText(f"fit {table.height} tracks ({model})")
        style_status_label(self._map_status, "ok" if table.height else "caution")
        map_df = _normalize_map_table(table, model)
        color_by = "alpha_map" if model == "anomalous" else "D_map_um2_s"
        self.host.set_map_results(table, map_df, model, color_by)

    def _on_map_error(self, exc: Exception) -> None:
        self._map_status.setText(f"error: {exc}")
        style_status_label(self._map_status, "error")

    def _on_color_by_changed(self, column: str) -> None:
        if not column:
            return
        self.host.set_map_color_by(column)
        self._refresh_map_histogram(reset_range=True)

    def _on_map_range_changed(self, vmin: float, vmax: float) -> None:
        self.host.set_map_contrast_limits(vmin, vmax)
        self._update_map_scale_label(vmin, vmax)

    def _on_map_log_scale_toggled(self, enabled: bool) -> None:
        """Re-bin the map histogram on a log10 axis. `set_log_scale` is
        silent and may clamp the range off zero/negative values, so the
        host's contrast limits are re-synced from it explicitly -- unlike
        a drag, nothing else would tell them the range moved."""
        self._map_histogram.set_log_scale(enabled)
        vmin, vmax = self._map_histogram.range()
        self.host.set_map_contrast_limits(vmin, vmax)
        self._update_map_scale_label(vmin, vmax)

    def _update_map_scale_label(
        self, vmin: Optional[float] = None, vmax: Optional[float] = None
    ) -> None:
        """State what the map's color ramp is showing and over what range,
        in that quantity's own unit -- the legend a napari Points layer
        colored by a feature doesn't come with."""
        column = self._color_by_picker.currentText()
        if not column:
            self._map_scale_label.setText("")
            return
        if vmin is None or vmax is None:
            vmin, vmax = self._map_histogram.range()
        unit = units.unit_of(column)
        self._map_scale_label.setText(
            f"map color: {units.header(column)}   "
            f"{vmin:.4g} → {vmax:.4g}{' ' + unit if unit else ''}"
        )
        self._map_scale_label.setToolTip(units.tooltip(column))

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
        self._map_log_scale_check.setEnabled(enabled)
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
        self._update_map_scale_label()

    def _run_track_fit(self) -> None:
        track_id = self.host.selected_track_id
        if track_id is None:
            self._track_section.set_expanded(True)
            self._track_status.setText("select a track in the table above first")
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
        self.host.start_worker(
            worker,
            self._on_track_fit_finished,
            self._on_track_fit_error,
            [self._track_fit_button],
            f"{method.upper()} fit, track {track_id}",
        )

    def _on_track_fit_finished(self, fit: "dk_bayes.TrackFit") -> None:
        self._track_status.setText(f"track {fit.track_id} fit (n={fit.track_length})")
        style_status_label(self._track_status, "ok")

        # Parameter names come from diffusionkit ("D", "K", "alpha",
        # "sigma"), and each carries a different unit -- which this
        # readout used to leave off entirely, so a D and an alpha were
        # printed identically. `_PARAM_COLUMNS` maps each to the column
        # name `spt_pipeline.units` knows it by.
        lines = [f"track {fit.track_id}, model={fit.model}, method={fit.method}"]
        for name, value in fit.params.items():
            column = _PARAM_COLUMNS.get(name, name)
            interval = f"[{fit.lo[name]:.4g}, {fit.hi[name]:.4g}]"
            lines.append(f"  {name} = {units.fmt(value, column)}  {interval}")
        self._track_result_label.setText("\n".join(lines))

        if fit.method == "nuts":
            samples_dict, _mcmc = fit.raw
            param_names = list(fit.params.keys())
            flat, _trace = dk_bayes_viz.samples_dict_to_arrays(samples_dict, param_names)
            figure = dk_bayes_viz.plot_posterior_corner(flat, param_names)
            _label_corner_axes(figure, param_names)
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
    tracks-pane table, same as every other tab's.

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
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.addRow("min points for fit", self._min_track_length)

        self._run_button = QPushButton("Run anisotropy analysis")
        self._run_button.setToolTip(
            "Nested sampling per track -- scales with track count and can take\n"
            "tens of seconds or more. Yields the evidence and the eps/psi/D\n"
            "posterior in one pass."
        )
        self._run_button.clicked.connect(self._run)
        self._status = status_label("")
        self._summary = status_label("")

        self._log_bf_button = QPushButton("Show log BF10 distribution")
        self._log_bf_button.setEnabled(False)
        self._log_bf_button.clicked.connect(self._show_log_bf_plot)
        self._eps_vs_bf_button = QPushButton("Show eps vs log BF10")
        self._eps_vs_bf_button.setEnabled(False)
        self._eps_vs_bf_button.clicked.connect(self._show_eps_vs_bf_plot)
        self._eps_forest_button = QPushButton("Show eps forest (top tracks)")
        self._eps_forest_button.setEnabled(False)
        self._eps_forest_button.clicked.connect(self._show_eps_forest_plot)

        plots_body = QWidget()
        plots_layout = QVBoxLayout(plots_body)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(4)
        plots_layout.addWidget(self._log_bf_button)
        plots_layout.addWidget(self._eps_vs_bf_button)
        plots_layout.addWidget(self._eps_forest_button)
        # Open once there is something to plot (see `_on_finished`) and
        # folded away before then, so three permanently-disabled buttons
        # aren't the tallest thing on the tab.
        self._plots_section = CollapsibleSection("Plots", plots_body)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addLayout(form)
        layout.addWidget(self._run_button)
        layout.addWidget(self._status)
        layout.addWidget(self._summary)
        layout.addWidget(hline())
        layout.addWidget(self._plots_section)
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
        self._plots_section.set_expanded(False)

    def _run(self) -> None:
        tracks = self.host.diffkit_tracks_for_fit()
        if tracks is None:
            return
        self._status.setText("running... (nested sampling per track, can take a while)")
        style_status_label(self._status)
        worker = _run_anisotropy_worker(
            tracks, self.host.dt_s, self._min_track_length.value(), self.host.progress_callback
        )
        self.host.start_worker(
            worker,
            self._on_finished,
            self._on_error,
            [self._run_button],
            "anisotropy",
            reports_progress=True,
        )

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
        self._plots_section.set_expanded(n > 0)

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
            # eps is tanh(|h|) in diffusionkit's log-Euclidean
            # coordinates, i.e. (D∥ − D⊥)/(D∥ + D⊥): a normalized
            # difference in [0, 1), 0 being isotropic. Dimensionless, and
            # said so, since every other number this widget reports does
            # carry a unit and a bare figure would read as an oversight.
            lines.append(
                f"median ε = {per_track['eps_median'].median():.3g} "
                "  (dimensionless: (D∥−D⊥)/(D∥+D⊥), 0 = isotropic)"
            )
        return "\n".join(lines)

    def _on_error(self, exc: Exception) -> None:
        self._status.setText(f"error: {exc}")
        style_status_label(self._status, "error")

    def _show_plot(self, figure) -> None:
        _label_anisotropy_axes(figure)
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

        self._result_dir: Optional[Path] = None
        self.pixel_size_um = 1.0
        self.dt_s = 1.0
        # Whether the two above came from a real calibration or are the
        # placeholders that let the conversion run at all -- set from the
        # layer's metadata in `_adopt_track_table`.
        self.units_known = False
        self._diffkit_tracks: Optional[pl.DataFrame] = None
        self._tracks_df_px: Optional[pl.DataFrame] = None
        self._base_track_df: Optional[pl.DataFrame] = None
        # Names of the aggregate detection-quality columns in
        # `_base_track_df` -- the group the tracks pane can hide from its
        # table (see `_qc_aggregate_table`).
        self._qc_columns: list[str] = []
        self._joined_track_df: Optional[pl.DataFrame] = None
        # The last Classical run (`ClassicAnalysis`, all models' rows),
        # its optional MSD comparison pass, and the per-track columns the
        # tracks pane shows from them.
        self._classical_analysis: Optional[ClassicAnalysis] = None
        self._classical_comparison: Optional[ClassicAnalysis] = None
        self._classical_df: Optional[pl.DataFrame] = None
        # model name -> that model's latest bulk MAP table: the full
        # diffusionkit output (what Save writes) and its trimmed point
        # estimates (what the tracks pane shows). Keyed by model so running
        # "normal" after "anomalous" adds columns instead of replacing them.
        self._map_full_by_model: dict[str, pl.DataFrame] = {}
        self._map_df_by_model: dict[str, pl.DataFrame] = {}
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
        # The Tracks layer's recorded exposure as last seen, to tell a
        # changed record from a re-read of the same one (`_adopt_track_table`).
        self._layer_exposure_s = _UNSEEN
        self._progress_label = ""
        self._progress_relay = _ProgressRelay(self)
        self._progress_relay.progress.connect(self._on_worker_progress)

        self._tracks_layer: Optional[Tracks] = None
        self._highlight_layer: Optional[Shapes] = None
        self._spatial_map_layer: Optional[Points] = None
        self._mouse_callback = None
        # True while this widget is itself rewriting the tracks layer
        # ("sync viewer"), so that write isn't mistaken for someone else
        # changing the tracks under us.
        self._writing_tracks_layer = False
        # The tracks layer's data and properties are set in two steps (and
        # its data setter empties the properties in between), so a change
        # from outside is picked up once, on the next event-loop pass,
        # after both have landed.
        self._external_change_timer = QTimer(self)
        self._external_change_timer.setSingleShot(True)
        self._external_change_timer.setInterval(0)
        self._external_change_timer.timeout.connect(self._on_tracks_layer_changed_externally)

        self._layer_combo = QComboBox()
        self._layer_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._layer_combo.currentIndexChanged.connect(self._on_layer_combo_changed)
        layer_row = QHBoxLayout()
        layer_row.setContentsMargins(0, 0, 0, 0)
        layer_row.addWidget(QLabel("Tracks layer:"))
        layer_row.addWidget(self._layer_combo, 1)

        self._source_label = status_label("no Tracks layer in this viewer")

        self._tracks_pane = _TracksPane(self)
        self._classical = _ClassicalTab(self)
        self._bayesian = _BayesianTab(self)
        self._anisotropy = _AnisotropyTab(self)

        # Each page scrolls independently rather than the whole tab widget
        # scrolling: wrapping the QTabWidget itself would carry the tab bar
        # off the top as soon as you scrolled down inside a page, which is
        # exactly when you want to be able to switch away from it.
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(scrolled(self._classical), "Classical")
        tabs.addTab(scrolled(self._bayesian), "Bayesian")
        tabs.addTab(scrolled(self._anisotropy), "Anisotropy")
        tabs.setTabToolTip(
            0, "Per-track Brownian displacement MLE: D and non-Brownian z (diffusionkit.classic.analyze_tracks)"
        )
        tabs.setTabToolTip(1, "Bayesian MAP / NUTS fits (diffusionkit.bayes)")
        tabs.setTabToolTip(2, "Anisotropy model comparison (diffusionkit.bayes.anisotropy)")

        # One session-wide switch, not a copy on each tab -- see this
        # module's docstring.
        self._restrict_checkbox = QCheckBox("restrict to filtered")
        self._restrict_checkbox.setToolTip(
            "When checked, every fit runs only on the tracks currently "
            "passing the filters above, instead of on all of them."
        )

        self._save_button = QPushButton("Save results")
        self._save_button.clicked.connect(self._save_results)
        self._save_button.setEnabled(False)

        footer = flow_row(self._restrict_checkbox, self._save_button)

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        self._progress_bar.hide()

        # The tracks table and the analysis tabs both want more height than
        # a docked panel has, and which one deserves it changes by the
        # minute (scanning rows vs. reading a fit's output), so it is a
        # drag, not a fixed ratio. Neither pane is collapsible: dragging
        # either to zero would hide the selection the other one acts on.
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._tracks_pane)
        splitter.addWidget(tabs)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([320, 260])

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addLayout(layer_row)
        layout.addWidget(self._source_label)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._progress_bar)
        layout.addWidget(footer)
        self.setLayout(layout)

        self.viewer.layers.events.inserted.connect(self._refresh_layer_combo)
        self.viewer.layers.events.removed.connect(self._refresh_layer_combo)
        self.viewer.layers.events.reordered.connect(self._refresh_layer_combo)
        self._refresh_layer_combo()
        tabify_with_open_widget(napari_viewer, self, "ExperimentListWidget")

    # -- shared read accessors used by the tabs --

    @property
    def selected_track_id(self) -> Optional[int]:
        return self._current_track_id

    @property
    def has_tracks(self) -> bool:
        return self._diffkit_tracks is not None and self._diffkit_tracks.height > 0

    def diffkit_track(self, track_id: int) -> Optional[pl.DataFrame]:
        if self._diffkit_tracks is None:
            return None
        return self._diffkit_tracks.filter(pl.col("track_id") == track_id)

    @property
    def joined_track_df(self) -> Optional[pl.DataFrame]:
        """The per-track table with every computed result joined in, BEFORE
        any filter -- what the tracks pane's own histogram cuts are read
        against (see `_TracksPane.filtered_track_ids`)."""
        return self._joined_track_df

    def combined_filtered_track_ids(self) -> Optional[set]:
        """The tracks pane's min-track-length spinbox ANDed with its
        per-track histogram cuts (which now cover the aggregated per-point
        quality columns too) -- `None` means "no restriction from any
        source". This is both what the table displays
        (`_rebuild_track_table`) and what a fit run optionally restricts to
        (`diffkit_tracks_for_fit`)."""
        ids = None
        min_len = self._tracks_pane.min_track_length()
        if min_len > 1 and self._base_track_df is not None:
            ids = set(
                self._base_track_df.filter(pl.col("track_length") >= min_len)["track_id"].to_list()
            )
        roi = self._tracks_pane.selected_roi()
        if roi is not None and self._base_track_df is not None and "roi" in self._base_track_df.columns:
            roi_ids = set(self._base_track_df.filter(pl.col("roi") == roi)["track_id"].to_list())
            ids = roi_ids if ids is None else (ids & roi_ids)
        track_ids = self._tracks_pane.filtered_track_ids()
        if track_ids is not None:
            ids = track_ids if ids is None else (ids & track_ids)
        return ids

    def track_rois(self) -> Optional[pl.DataFrame]:
        """`(track_id, group)` naming each loaded track's ROI, or None when
        the tracks weren't linked across more than one -- what per-ROI
        summaries and plot colors are keyed on."""
        base = self._base_track_df
        if base is None or "roi" not in base.columns or base["roi"].drop_nulls().n_unique() < 2:
            return None
        return base.select("track_id", pl.col("roi").alias("group"))

    def on_filters_changed(self) -> None:
        self._rebuild_track_table()
        self._update_spatial_map_layer()
        self._bayesian.refresh_map_histogram()

    def diffkit_tracks_for_fit(self) -> Optional[pl.DataFrame]:
        """The tracks a fit should run on, honoring the footer's one
        "restrict fits to filtered tracks" switch."""
        if self._diffkit_tracks is None:
            return None
        if not self._restrict_checkbox.isChecked():
            return self._diffkit_tracks
        ids = self.combined_filtered_track_ids()
        if ids is None:
            return self._diffkit_tracks
        return self._diffkit_tracks.filter(pl.col("track_id").is_in(list(ids)))

    @property
    def progress_callback(self):
        """What to hand diffusionkit as `progress=`: safe to call from the
        worker thread (see `_ProgressRelay`)."""
        return self._progress_relay.progress.emit

    def start_worker(
        self,
        worker,
        on_finished,
        on_error,
        busy_widgets: list,
        label: str,
        reports_progress: bool = False,
    ) -> None:
        """Run `worker` in the one host-wide slot. `reports_progress` means
        its function was given `progress_callback`; until the first report
        arrives (and throughout, for a worker that never reports) the bar
        is a busy indicator, since a bulk fit's first batch includes JAX
        compilation and can sit at 0 for a while."""
        if self._worker is not None:
            return
        for widget in busy_widgets:
            widget.setEnabled(False)
        self._progress_label = label
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat(f"{label}...")
        self._progress_bar.show()

        def _done(_widgets=busy_widgets) -> None:
            self._worker = None
            self._progress_bar.hide()
            for w in _widgets:
                w.setEnabled(True)

        def _finished(result) -> None:
            _done()
            on_finished(result)

        def _errored(exc) -> None:
            _done()
            on_error(exc)

        worker.returned.connect(_finished)
        worker.errored.connect(_errored)
        self._worker = worker
        worker.start()

    def _on_worker_progress(self, done: int, total: int) -> None:
        if self._worker is None or total <= 0 or done <= 0:
            # Stay a busy indicator through the first (compiling) batch; a
            # determinate bar parked at 0% reads as hung.
            return
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(done)
        self._progress_bar.setFormat(f"{self._progress_label}: {done}/{total} tracks")

    # -- tracks-layer selection --

    def _tracks_layers(self) -> list[Tracks]:
        return [layer for layer in self.viewer.layers if isinstance(layer, Tracks)]

    def _refresh_layer_combo(self, event=None) -> None:
        current = self._layer_combo.currentData()
        layers = self._tracks_layers()
        for layer in layers:
            # A rename is an event on the layer, not the list. napari's
            # emitters ignore a duplicate connect, so this can run freely.
            layer.events.name.connect(self._on_tracks_layer_renamed)
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

    def _on_tracks_layer_renamed(self, event=None) -> None:
        for index in range(self._layer_combo.count()):
            layer = self._layer_combo.itemData(index)
            if layer is not None:
                self._layer_combo.setItemText(index, layer.name)
        if self._tracks_layer is not None:
            self._update_source_label()

    def _detach_mouse_callback(self) -> None:
        if self._tracks_layer is not None:
            if self._mouse_callback is not None:
                try:
                    self._tracks_layer.mouse_drag_callbacks.remove(self._mouse_callback)
                except ValueError:
                    pass
            self._tracks_layer.events.data.disconnect(self._on_tracks_layer_event)
            self._tracks_layer.events.properties.disconnect(self._on_tracks_layer_event)
        self._external_change_timer.stop()
        self._mouse_callback = None

    def _on_tracks_layer_event(self, event=None) -> None:
        if not self._writing_tracks_layer:
            self._external_change_timer.start()

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
            and self._tracks_pane.sync_display_enabled()
        ):
            self._set_tracks_layer_data(self._tracks_df_px)

    def _clear_loaded_state(self) -> None:
        self._restore_previous_tracks_layer_if_synced()
        self._detach_mouse_callback()
        self._result_dir = None
        self._tracks_layer = None
        self._layer_exposure_s = _UNSEEN
        self._diffkit_tracks = None
        self._tracks_df_px = None
        self._base_track_df = None
        self._qc_columns = []
        self._joined_track_df = None
        self._classical_analysis = None
        self._classical_comparison = None
        self._classical_df = None
        self._map_full_by_model = {}
        self._map_df_by_model = {}
        self._map_color_by = None
        self._anisotropy_full_df = None
        self._spatial_sources = {}
        self._track_fit_rows = []
        self._current_track_id = None
        self.pixel_size_um = 1.0
        self.dt_s = 1.0
        self.units_known = False
        self._source_label.setText("no Tracks layer in this viewer")
        style_status_label(self._source_label)
        self._tracks_pane.reset()
        self._classical.reset()
        self._classical.set_layer_exposure(None)
        self._bayesian.reset()
        self._anisotropy.reset()
        self._save_button.setEnabled(False)
        self._clear_overlay_layers()

    @staticmethod
    def _layer_track_table(layer: Tracks) -> Optional[pl.DataFrame]:
        """The layer's vertices plus every per-vertex property (se_y/se_x
        and whatever QC/derived columns viewer.py put there -- flux, bg,
        fit_sigma, track_length, ...), aligned with track_id: one table
        serves both the diffusionkit conversion (needs
        track_id/frame/y/x/se_y/se_x) and the per-track QC aggregates.

        None when the layer has no `se_y`/`se_x` (spotsolve's per-detection
        CRLB, renamed for diffusionkit in spt_pipeline.diffusion). Requiring
        them is how a Tracks layer from this pipeline is told apart from
        any other: without a position error there is no noise term to fit,
        so the check is load-bearing, not cosmetic."""
        props = layer.properties
        if "se_x" not in props or "se_y" not in props:
            return None
        data = layer.data
        base_cols = {
            "track_id": data[:, 0].astype(np.int64),
            "frame": data[:, 1].astype(np.int64),
            "y": data[:, 2],
            "x": data[:, 3],
        }
        extra_cols = {
            name: np.asarray(values)
            for name, values in props.items()
            if name != "track_id" and name not in _TRACK_COLOR_COLUMNS
        }
        table = pl.DataFrame({**base_cols, **extra_cols})
        # The layer carries the ROI as a number (Tracks properties are
        # numeric); its name comes from the layer's `roi_names` metadata.
        roi_names = list(layer.metadata.get("roi_names") or [])
        if "roi_index" in table.columns and roi_names:
            lookup = pl.DataFrame(
                {"roi_index": list(range(len(roi_names))), "roi": roi_names},
                schema={"roi_index": table.schema["roi_index"], "roi": pl.Utf8},
            )
            table = table.join(lookup, on="roi_index", how="left")
        return table

    def _adopt_track_table(self, layer: Tracks, track_points_df: pl.DataFrame) -> None:
        """Take `track_points_df` (read off `layer`) as the loaded track set:
        the pixel-space table, its diffusionkit conversion, the per-track
        base table and the layer's units/bundle metadata. Leaves every fit
        result alone -- callers decide whether those still apply."""
        self._tracks_df_px = track_points_df
        raw_result_dir = layer.metadata.get("result_dir")
        self._result_dir = Path(raw_result_dir) if raw_result_dir else None
        self.pixel_size_um = layer.metadata.get("pixel_size_um") or 1.0
        self.dt_s = layer.metadata.get("dt_s") or 1.0
        # Layers this app builds say whether those two are real
        # (`viewer.layer_units_metadata`); a Tracks layer from anywhere
        # else doesn't, and the 1.0s above are then placeholders rather
        # than a calibration -- see `_update_source_label`.
        self.units_known = bool(
            layer.metadata.get(
                "units_known",
                layer.metadata.get("pixel_size_um") is not None
                and layer.metadata.get("dt_s") is not None,
            )
        )
        self._diffkit_tracks = tracks_to_diffusionkit_df(track_points_df, self.pixel_size_um, self.dt_s)
        self._base_track_df, self._qc_columns = _base_track_table(
            self._diffkit_tracks, track_points_df
        )
        self._update_source_label()
        # The exposure box follows the layer's record only when that record
        # changed (a new layer, or the image panel's override edited): a
        # filter redraw re-adopts the same layer, and must not wipe an
        # exposure typed into the tab.
        self.sync_layer_exposure()

    def sync_layer_exposure(self) -> None:
        """Hand the loaded layer's recorded exposure to the Classical tab
        if it changed since last looked at -- see `_adopt_track_table`."""
        layer = self._tracks_layer
        layer_exposure = layer.metadata.get("exposure_s") if layer is not None else None
        if self._layer_exposure_s is _UNSEEN or layer_exposure != self._layer_exposure_s:
            self._layer_exposure_s = layer_exposure
            self._classical.set_layer_exposure(layer_exposure)
        else:
            self._classical.refresh_inputs()

    def _update_source_label(self) -> None:
        """Name the loaded layer, its bundle, and -- the part that is not
        cosmetic -- the two conversion factors every physical column in
        this widget is computed with.

        `tracks_to_diffusionkit_df` multiplies pixel positions by
        `pixel_size_um` and frame indices by `dt_s`, so every `_um`/`_s`
        column here, and every D and K fitted from them, is those two
        numbers. A layer that carries neither still gets 1.0 for both
        (see `viewer.layer_units_metadata`) because the conversion has to
        run on something -- and then `D_map_um2_s` is really px²/frame
        under a µm²/s name. That case gets said out loud rather than
        rendered identically to a calibrated one."""
        layer = self._tracks_layer
        if layer is None:
            return
        where = (
            f"→ {self._result_dir}"
            if self._result_dir is not None
            else "(no known results bundle — results can't be saved)"
        )
        scale = (
            f"{units.fmt_unit(self.pixel_size_um, units.UM + '/px')} · "
            f"{units.fmt_unit(self.dt_s, 's/frame')}"
        )
        if self.units_known:
            self._source_label.setText(f"'{layer.name}' {where}\n{scale}")
            style_status_label(self._source_label, "neutral")
        else:
            self._source_label.setText(
                f"'{layer.name}' {where}\nno pixel size / frame interval on this layer — "
                "every µm and s column below is really px and frames"
            )
            style_status_label(self._source_label, "caution")

    def _load_from_layer(self, layer: Tracks) -> None:
        track_points_df = self._layer_track_table(layer)
        if track_points_df is None:
            self._clear_loaded_state()
            self._source_label.setText(
                f"'{layer.name}' has no se_x/se_y properties -- only Tracks "
                "layers produced by this project's pipeline can be analyzed here."
            )
            return

        if layer is not self._tracks_layer:
            # Not when re-reading the same layer after an outside change:
            # "restoring" it would write the old tracks back over the new.
            self._restore_previous_tracks_layer_if_synced()
        self._detach_mouse_callback()
        self._tracks_layer = layer
        # A full (re)load is a new data set: the exposure box follows this
        # layer's record, whatever was typed for the last one.
        self._layer_exposure_s = _UNSEEN
        self._adopt_track_table(layer, track_points_df)

        self._classical_analysis = None
        self._classical_comparison = None
        self._classical_df = None
        self._map_full_by_model = {}
        self._map_df_by_model = {}
        self._map_color_by = None
        self._anisotropy_full_df = None
        self._spatial_sources = {}
        self._track_fit_rows = []
        self._current_track_id = None

        self._tracks_pane.reset()
        self._tracks_pane.set_qc_columns(self._qc_columns)
        self._tracks_pane.set_roi_choices(self._roi_names_loaded())
        self._classical.reset()
        self._bayesian.reset()
        self._anisotropy.reset()
        self._rebuild_track_table()
        self._tracks_pane.set_plot_columns(
            self._joined_track_df, prefer_x="radius_of_gyration_um", prefer_y="flux_mean"
        )
        self._clear_overlay_layers()

        self._mouse_callback = self._make_click_callback()
        layer.mouse_drag_callbacks.append(self._mouse_callback)
        layer.events.data.connect(self._on_tracks_layer_event)
        layer.events.properties.connect(self._on_tracks_layer_event)

        saved_per_track, saved_summary = (
            load_diffusion_results(self._result_dir) if self._result_dir is not None else (None, None)
        )
        n_saved = saved_per_track.height if saved_per_track is not None else 0
        if saved_summary is not None:
            self._classical.show_loaded_summary("Loaded saved results:\n" + _format_summary(saved_summary))
        if n_saved:
            self._classical.report_saved(f"{n_saved} saved fit(s) found")
        self._update_save_enabled()

    def _on_tracks_layer_changed_externally(self) -> None:
        """The loaded Tracks layer was redrawn in place by someone else --
        the experiment list narrowing it to its Track-tab filters, or
        replacing it with a re-linked result.

        Fit results are keyed by track_id, so they stay valid for as long
        as the tracks they were fitted on are the same tracks. A filter
        change only adds or removes whole tracks and leaves every shared
        one vertex-for-vertex identical, so the fits are kept and the table
        is re-joined over the new track set. A re-link renumbers and
        re-shapes tracks, so a shared track_id no longer means the same
        track: that is a new data set, and everything resets as if the
        layer had just been picked."""
        layer = self._tracks_layer
        if layer is None or layer not in self.viewer.layers:
            return
        track_points_df = self._layer_track_table(layer)
        if track_points_df is None or not _shared_tracks_unchanged(self._tracks_df_px, track_points_df):
            self._load_from_layer(layer)
            return
        self._adopt_track_table(layer, track_points_df)
        self._tracks_pane.set_qc_columns(self._qc_columns)
        self._tracks_pane.set_roi_choices(self._roi_names_loaded())
        self._rebuild_track_table()
        self._update_spatial_map_layer()
        self._bayesian.refresh_map_histogram()
        self._update_save_enabled()

    def _roi_names_loaded(self) -> list[str]:
        """The ROIs the loaded tracks were linked in, in the layer's own
        order (its `roi_names` metadata) where that is known."""
        base = self._base_track_df
        if base is None or "roi" not in base.columns:
            return []
        present = set(base["roi"].drop_nulls().to_list())
        layer = self._tracks_layer
        ordered = list(layer.metadata.get("roi_names") or []) if layer is not None else []
        return [n for n in ordered if n in present] + sorted(present - set(ordered))

    def _update_save_enabled(self) -> None:
        has_results = (
            self._classical_analysis is not None
            or bool(self._map_full_by_model)
            or self._anisotropy_full_df is not None
            or bool(self._track_fit_rows)
        )
        self._save_button.setEnabled(has_results and self._result_dir is not None)

    # -- track table + selection --

    def _rebuild_track_table(self) -> None:
        if self._base_track_df is None:
            return
        df = self._base_track_df
        if self._classical_df is not None:
            df = df.join(self._classical_df, on="track_id", how="left")
        for map_df in self._map_df_by_model.values():
            df = df.join(map_df, on="track_id", how="left")
        anisotropy_df = self._spatial_sources.get("anisotropy")
        if anisotropy_df is not None:
            df = df.join(anisotropy_df, on="track_id", how="left")
        if self._track_fit_rows:
            fit_df = (
                pl.DataFrame(self._track_fit_rows)
                .group_by("track_id", maintain_order=True)
                .last()
                .select(
                    "track_id", "D_track_fit_um2_s", "K_track_fit_um2_s_alpha", "alpha_track_fit", "model", "method"
                )
                .rename({"model": "track_fit_model", "method": "track_fit_method"})
            )
            df = df.join(fit_df, on="track_id", how="left")

        # Keep the unfiltered join around and hand it to the Track
        # Explorer's histograms BEFORE narrowing: a filter has to be drawn
        # against the whole population, or each cut would reshape the
        # distribution the next one is chosen on.
        self._joined_track_df = df
        self._tracks_pane.set_filter_source(df)
        self._tracks_pane.set_plot_columns(df)

        total = df.height
        ids = self.combined_filtered_track_ids()
        if ids is not None:
            df = df.filter(pl.col("track_id").is_in(list(ids)))
        self._tracks_pane.set_dataframe(df, total)

        if self._current_track_id is not None and self._current_track_id not in set(df["track_id"].to_list()):
            # The selected track just got filtered out -- clear it rather
            # than leave a stale highlight in the viewer and a stale
            # target for "Fit selected track" pointing at a track that
            # isn't even in the table anymore.
            self._current_track_id = None
            self._clear_highlight_layer()

        self._sync_tracks_layer_display(ids)

    def set_classical_results(
        self,
        analysis: ClassicAnalysis,
        comparison: Optional[ClassicAnalysis],
        display_df: pl.DataFrame,
    ) -> None:
        """A Classical run finished: keep it for Save, join its per-track
        columns into the tracks pane, offer D and z as spatial-map colors,
        and put them on the Tracks layer itself (`_TRACK_COLOR_COLUMNS`)."""
        self._classical_analysis = analysis
        self._classical_comparison = comparison
        self._classical_df = display_df
        numeric = [c for c, dtype in zip(display_df.columns, display_df.dtypes) if dtype.is_numeric()]
        self.register_spatial_source("classical_mle", display_df.select(numeric))
        self._rebuild_track_table()
        with_z = "z_nonbrownian" in display_df.columns
        self._tracks_pane.set_plot_columns(
            self._joined_track_df,
            prefer_x="D_mle_um2_s",
            prefer_y="z_nonbrownian" if with_z else "track_length",
        )
        self._bayesian.on_spatial_source_registered("D_mle_um2_s")
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
        self._map_full_by_model[model] = full_df
        self._map_df_by_model[model] = display_df
        self.register_spatial_source(f"bulk_map_{model}", display_df)
        self._rebuild_track_table()
        if model == "anomalous":
            prefer_x, prefer_y = "K_map_um2_s_alpha", "alpha_map"
        else:
            prefer_x, prefer_y = "D_map_um2_s", "alpha_map" if "anomalous" in self._map_df_by_model else None
        self._tracks_pane.set_plot_columns(self._joined_track_df, prefer_x=prefer_x, prefer_y=prefer_y)
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
        if self._live(self._spatial_map_layer) is not None and vmin < vmax:
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
        self._tracks_pane.select_track_id(track_id)

    # -- viewer overlays --

    def _live(self, layer):
        """`layer` if it is still in the viewer (`qtkit.napari.live_layer`).
        The overlay layers this widget adds can be deleted by anyone -- the
        user, or the experiment list clearing the viewer for the next
        image."""
        return live_layer(self.viewer, layer)

    def _clear_overlay_layers(self) -> None:
        self._clear_highlight_layer()
        if self._live(self._spatial_map_layer) is not None:
            self._spatial_map_layer.data = np.empty((0, 2))

    def _clear_highlight_layer(self) -> None:
        if self._live(self._highlight_layer) is not None:
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
        if self._live(self._highlight_layer) is None:
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
        """When the tracks pane's "sync viewer" checkbox is on,
        rewrite the viewer's own `tracks` layer to the filtered subset
        (restored from `self._tracks_df_px`, the full unfiltered table
        this widget keeps around specifically for this) -- otherwise
        restore it to the full set, in case it was left filtered from a
        moment ago when the checkbox was on."""
        if self._tracks_layer is None or self._tracks_df_px is None:
            return
        if self._tracks_pane.sync_display_enabled() and filtered_ids is not None:
            df = self._tracks_df_px.filter(pl.col("track_id").is_in(list(filtered_ids)))
        else:
            df = self._tracks_df_px
        self._set_tracks_layer_data(self._with_track_colors(df))

    def _with_track_colors(self, df: pl.DataFrame) -> pl.DataFrame:
        """`df` with the Classical run's per-track `_TRACK_COLOR_COLUMNS`
        broadcast onto its vertices, when there is a run. Tracks without
        a value (unresolved, excluded) get NaN, which napari leaves
        uncolored rather than pinning to one end of the colormap."""
        if self._classical_df is None:
            return df
        present = [c for c in _TRACK_COLOR_COLUMNS[1:] if c in self._classical_df.columns]
        colors = self._classical_df.select(
            "track_id",
            pl.when(pl.col("D_mle_um2_s") > 0)
            .then(pl.col("D_mle_um2_s").log10())
            .otherwise(None)
            .alias("log10_D_mle"),
            *present,
        )
        return df.join(colors, on="track_id", how="left").with_columns(
            pl.col(c).cast(pl.Float64).fill_null(float("nan")) for c in ["log10_D_mle", *present]
        )

    def _set_tracks_layer_data(self, df: pl.DataFrame) -> None:
        layer = self._tracks_layer
        if layer is None or layer not in self.viewer.layers:
            return
        self._writing_tracks_layer = True
        try:
            set_tracks_layer_data(layer, df)
        finally:
            self._writing_tracks_layer = False

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
            if self._live(self._spatial_map_layer) is not None:
                self._spatial_map_layer.data = np.empty((0, 2))
            return
        positions = merged.select("y_px", "x_px").to_numpy()
        values = merged[color_by].to_numpy()
        track_ids = merged["track_id"].to_numpy()
        if self._live(self._spatial_map_layer) is None:
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
        if self._result_dir is None:
            return
        tables = []
        if self._classical_analysis is not None:
            # diffusionkit's own `method` (displacement_mle / msd_ols /
            # msd_nls) is kept, prefixed like the other analyses' tags.
            # MSD rows come from the comparison pass when there was one --
            # the main pass's are all `excluded` whenever exposure > 0 --
            # tagged `_exposure0` since that pass ignored the blur.
            fits = self._classical_analysis.fits
            comparison = self._classical_comparison
            if comparison is not None and comparison is not self._classical_analysis:
                fits = pl.concat(
                    [
                        fits.filter(pl.col("model") == "brownian_mle").with_columns(
                            ("classic_" + pl.col("method")).alias("method")
                        ),
                        comparison.fits.filter(pl.col("model") != "brownian_mle").with_columns(
                            ("classic_" + pl.col("method") + "_exposure0").alias("method")
                        ),
                    ]
                )
            else:
                fits = fits.with_columns(("classic_" + pl.col("method")).alias("method"))
            tables.append(fits)
        for model, map_full_df in self._map_full_by_model.items():
            tables.append(map_full_df.with_columns(pl.lit(f"bayes_map_bulk_{model}").alias("method")))
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
        groups = self.track_rois()
        if groups is not None and "track_id" in per_track_df.columns:
            # Which ROI each fitted track was linked in, so the saved
            # table can be split the same way without the tracks file.
            per_track_df = per_track_df.join(
                groups.rename({"group": "roi"}).with_columns(
                    pl.col("track_id").cast(per_track_df.schema["track_id"])
                ),
                on="track_id",
                how="left",
            )

        summary = {}
        analysis = self._classical_analysis
        if analysis is not None and self._classical.summary_values is not None:
            # The settings travel with the tables -- diffusionkit's own
            # guidance, since the fits alone don't say what exposure or
            # bootstrap produced them. Flat keys so `_format_summary` can
            # label each with its unit when the bundle is reopened.
            summary = {
                "classical_analysis": "diffusionkit.classic.analyze_tracks (brownian_mle)",
                "dt_s": analysis.acquisition.dt_s,
                "exposure_s": analysis.acquisition.exposure_s,
                "min_frames": analysis.options.min_frames,
                "localization": analysis.options.localization,
                "n_boot": analysis.mle_options.n_boot,
                "bootstrap_seed": analysis.mle_options.seed,
                "D_upper_level": analysis.mle_options.upper_level,
                "msd_comparison": self._classical_comparison is not None,
                "msd_comparison_max_lag": analysis.options.max_lag,
                "msd_comparison_exposure_s": 0.0 if self._classical_comparison is not None else None,
                **self._classical.summary_values,
            }
            if self._classical.summary_by_roi:
                summary["by_roi"] = self._classical.summary_by_roi

        track_d = (
            track_d_table(analysis.fits, self.track_rois()) if analysis is not None else None
        )
        write_diffusion_results(self._result_dir, per_track_df, summary, track_d)
        self._classical.report_saved(f"saved {per_track_df.height} fit(s) to {self._result_dir}")
