"""Diffusion-analysis dock widget: pick a Tracks layer already in the
napari viewer and explore its per-track diffusion, using diffusionkit's
grid posteriors over D (and alpha), the ensemble read across tracks, the
classic MSD fits as a comparison, and -- optionally -- a full NUTS
posterior for one selected track.

Layout: one permanent **tracks pane** on top, a stack of **analysis tabs**
below it, and a footer, split by a drag-resizable `QSplitter`.

The tracks pane is not a tab, because everything else in the widget reads
or writes it: every fit merges its results in as new columns, the NUTS
tab's per-track action operates on whatever row is selected here, and the
filters here decide what a fit runs on. Behind a tab, running a fit and
seeing its result were two different screens, and "Fit selected track"
pointed at a selection you could not see.

- **Tracks pane** -- one row per track (`qtkit.ColumnTableModel` in
  a `QTableView`, folded away by default behind a "Table" header), the single place all per-track numbers live (posterior,
  MSD comparison and NUTS results, each in its own column group), plus a `min_track_length` spinbox and a
  `FeatureFilterPanel` over whatever columns that table currently holds.
  Those columns include per-point detection quality aggregated to the
  track (`flux_min`, `se_x_max`, ... -- see `diffusion.qc_aggregate_table`), so a
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
  of their own) and moves the time slider to the track's last frame, so
  its tail is drawn in full inside the box.
- **Posterior** -- `diffusion.analyze_posteriors` (diffusionkit.gridpost):
  per track, the posterior median of D and its 5%/95% quantiles (with
  the camera exposure's blur modelled), alpha when exposure is 0, and the
  ensemble -- the summed (shared-D) posterior and the deconvolved
  distribution of D across tracks -- over whatever the tracks pane
  passes, per region class. The MSD fits are a labelled opt-in
  comparison. See `_PosteriorTab`.
- **Map** -- a `Points` layer in the viewer (`self._spatial_map_layer`,
  one point per track centroid) colored by any per-track result: the
  spatial map, and the reason this analysis stays inside napari next to
  the image. See `_MapTab`.
- **NUTS** -- `diffusionkit.bayes.fit_track` on the selected track, with
  its corner plot; needs the optional `[bayes]` extra. See `_NutsTab`.

Saving writes the per-track summary (`tracks_summary.csv`), every
track's posterior (`posterior_D.parquet`, `posterior_alpha.parquet`),
the ensemble distributions (`distributions_*.csv`) and the settings and
population numbers (`diffusion_summary.json`) into the layer's bundle --
see `results.py`. Picking a layer whose bundle has a saved analysis
restores it (`diffusion.restore_analysis`) when the tracks it was fitted
on are still the layer's, which the summary's `tracks_sha256` decides:
the plots, per-track columns, filters and settings come back as saved,
with no fit re-run. A mismatch shows the saved numbers as text only.

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
for the posterior run, and a busy indicator for a single-track NUTS fit.

Track/spatial-map positions used for viewer overlays are kept in
*pixels* (`self._tracks_df_px`, the same coordinate space as the image
and Tracks layer), separate from the *physical-unit* table
(`self._diffkit_tracks`) handed to diffusionkit -- conflating the two
would misplace every overlay relative to the image.

Both unit systems are therefore on screen at once, which is why nothing
here shows a bare number:

  - the tracks pane's headers carry each column's unit
    (`_UnitHeaderModel` over `napari_gemscape2.units`), since `se_x_max` (px)
    and `se_x_um_max` (µm) are adjacent columns of the same quantity;
  - each fit readout formats through `units.fmt`, including the NUTS fit,
    whose parameters are a D, a K, an alpha and a localization sigma with
    four different units (`_PARAM_COLUMNS`);
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
from diffusionkit import Acquisition
from napari.layers import Points, Shapes, Tracks
from napari.qt.threading import thread_worker
from qtpy.QtCore import QObject, Qt, QTimer, Signal
from qtkit import (
    CollapsibleSection,
    ColumnTableModel,
    HistogramRangeWidget,
    double_spinbox,
    flow_row,
    note_label,
    scrolled,
    status_label,
    style_status_label,
    table_view,
)
from qtkit.napari import live_layer, tabify_with_open_widget
from qtkit.plot import AxisPicker, PlotWindow
from qtpy.QtGui import QValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QComboBox,
    QFileDialog,
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

from napari_gemscape2 import units
from diffusionkit.gridpost import GridPostOptions

from napari_gemscape2.diffusion import (
    EXPOSURE_CLAMP_FRACTION,
    MIN_FRAMES,
    Deconvolution,
    PosteriorAnalysis,
    SavedAnalysis,
    StaleAnalysisError,
    analysis_summary,
    analysis_tables,
    analyze_posteriors,
    base_track_table,
    ensemble_panels,
    filter_record,
    grid_record,
    msd_fits_blur_free,
    msd_track_table,
    passing_track_ids,
    posterior_options,
    posterior_results_table,
    region_class_groups,
    restore_analysis,
    summarize,
    track_posterior,
    tracks_summary_table,
    tracks_to_diffusionkit_df,
)
from napari_gemscape2.results import (
    TRACKS_SUMMARY_FILENAME,
    load_diffusion_results,
    load_diffusion_summary,
    repo_shas,
    write_diffusion_results,
)
from napari_gemscape2.joint_plot import (
    numeric_columns,
    plot_d_ensemble,
    plot_d_posteriors,
    plot_property_joint,
    plot_track_posterior,
)
from napari_gemscape2.pipeline import filter_mask
from napari_gemscape2.viewer import set_tracks_layer_data
from napari_gemscape2.widgets.feature_filters import FeatureFilterPanel
from napari_gemscape2.widgets.params_panel import _compact_form, _dspin, _ispin

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
# its aspect ratio is a free read on whether the motion is anisotropic.
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
# anisotropy; it's a read, not a measure.
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
def _run_posterior_worker(
    diffkit_tracks: pl.DataFrame,
    dt_s: float,
    exposure_s: float,
    options: GridPostOptions,
    msd_comparison: bool,
    progress,
) -> tuple[PosteriorAnalysis, Optional[pl.DataFrame]]:
    """`(analysis, msd_fits)`: the grid posteriors with the real exposure,
    on `options`' grids, and -- when asked for -- diffusionkit.classic's
    MSD fits, run with the exposure treated as 0 (see
    `diffusion.msd_fits_blur_free`)."""
    acquisition = Acquisition(dt_s=dt_s, exposure_s=exposure_s)
    analysis = analyze_posteriors(diffkit_tracks, acquisition, options, progress=progress)
    msd_fits = None
    if msd_comparison:
        msd_fits = msd_fits_blur_free(diffkit_tracks, dt_s, options.min_frames)
    return analysis, msd_fits


@thread_worker(start_thread=False)
def _run_nuts_worker(track_df: pl.DataFrame, dt_s: float, model: str):
    from diffusionkit import bayes as dk_bayes

    return dk_bayes.fit_track(track_df, dt_s, model=model)


class _UnitHeaderModel(ColumnTableModel):
    """`qtkit.ColumnTableModel` with the units in the header.

    The tracks pane's table is where this pipeline's two unit systems
    meet: `se_x_max` is in pixels, `se_x_um_max` and
    `radius_of_gyration_um` in µm, `flux_mean` in camera counts,
    `D_median_um2_s` in µm²/s -- 40-odd columns whose unit is a naming
    convention at best (`_um`) and absent at worst (`flux`, `se_x`,
    `fit_sigma`). So each header shows `napari_gemscape2.units.header` (the
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


# Per-track posterior results this widget broadcasts onto the viewer's
# Tracks layer as properties, so the trajectories themselves can be
# colored by them (layer controls -> color by). `log10_D_median` rather
# than D itself: a Tracks layer colormap spans min..max linearly, and D
# spans decades. Written by this widget, so they are dropped again
# whenever it reads the layer back (`_layer_track_table`) -- otherwise
# they would come back as "detection QC" columns.
_TRACK_COLOR_COLUMNS = ("log10_D_median", "alpha_median")

# "No exposure seen yet" for `DiffusionAnalysisWidget._layer_exposure_s`,
# distinct from None ("the layer records none").
_UNSEEN = object()


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


# diffusionkit's per-track parameter names mapped to the column names
# `napari_gemscape2.units` knows their units by. Only `sigma` actually needs
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


def _analysis_repo_shas() -> dict:
    """Provenance for saved diffusion results: the diffusionkit and
    napari_gemscape2 checkouts that computed them (the bundle's manifest
    already records what produced the tracks)."""
    import diffusionkit
    import napari_gemscape2

    return repo_shas(diffusionkit, napari_gemscape2)


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


def _nuts_row(fit) -> dict:
    """One NUTS fit as tracks-pane columns: the posterior median and 90%
    HPDI of each parameter, under `_nuts` names that stay distinct from the
    grid posterior's own columns (`D_nuts_um2_s`, `D_nuts_low_um2_s`, ...)."""
    names = {"D": ("D", "_um2_s"), "K": ("K", "_um2_s_alpha"), "alpha": ("alpha", "")}
    row = {"track_id": fit.track_id, "nuts_model": fit.model}
    for name, (base, unit) in names.items():
        if name in fit.params:
            row[f"{base}_nuts{unit}"] = fit.params[name]
            row[f"{base}_nuts_low{unit}"] = fit.lo[name]
            row[f"{base}_nuts_high{unit}"] = fit.hi[name]
    return row


def _format_summary(summary: dict) -> str:
    """A saved `diffusion_summary.json` as lines of "key = value unit".

    The keys are already unit-suffixed (`normal_D_um2_s`), which is how
    the file stays readable on its own; `units.fmt` restates the unit
    where it can, so a loaded summary reads like a freshly-computed one
    rather than like raw JSON."""
    lines = "\n".join(
        f"{key} = {units.fmt(value, key)}"
        for key, value in summary.items()
        # Nested records (by_region_class, filters, provenance) aren't one number.
        if not isinstance(value, dict)
    )
    return lines + _format_by_group(summary.get("by_region_class"))


# Qt's "no maximum" for widget sizes; not exported by every qtpy binding.
_QWIDGETSIZE_MAX = (1 << 24) - 1


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
    `se_x_max`, ... -- see `diffusion.qc_aggregate_table`); after a posterior run
    it is also `D_median_um2_s`, `D_low_um2_s`, `alpha_median` and the
    rest. So "drop the tracks with a bad worst-point localization
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
    a plot on one table. It used to be a separate picker on each analysis
    tab, each over only that tab's own raw fit output; that
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
        self._total = 0
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
        # layer's tracks span more than one region class
        # (`regions.label_points`), so a single-field run doesn't carry a
        # control with one choice.
        self._region_label = QLabel("Region:")
        self._region_picker = QComboBox()
        self._region_picker.setToolTip(
            "Show (and, with \"restrict fits to filtered tracks\", fit) only the\n"
            "tracks of one region class (nucleus, cytoplasm, ...). Each region\n"
            "was linked on its own, so no track spans two."
        )
        self._region_picker.currentIndexChanged.connect(lambda _i: self.host.on_filters_changed())
        # A row of its own, hidden whole: qtkit's FlowLayout still spaces a
        # hidden item, which left a blank line in the control row.
        self._region_row = flow_row(self._region_label, self._region_picker)
        self._region_row.setVisible(False)

        length_label = QLabel("min length:")
        control_row = flow_row(
            length_label,
            self._min_track_length,
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
        # Folded by default: the rows are for troubleshooting a run, not
        # for reading every time, and unfolded they take most of the dock.
        # Selection still works folded -- a track clicked in the viewer
        # selects its row, and the count line below names it.
        self._table_section = CollapsibleSection("Table", self.table, expanded=False)

        self._count_label = status_label("")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(control_row)
        layout.addWidget(self._region_row)
        layout.addWidget(self._filter_section)
        layout.addWidget(self._plot_section)
        layout.addWidget(self._table_section, 1)
        layout.addWidget(self._count_label)
        layout.addStretch(0)
        self.setLayout(layout)

        for section in (self._table_section, self._filter_section, self._plot_section):
            section.toggled.connect(lambda _expanded: QTimer.singleShot(0, self._fit_height))
        self._fit_height()

    def _fit_height(self) -> None:
        """With the table folded, cap this pane at its natural height so the
        splitter hands everything below it to the analysis tabs, rather
        than leaving a table-sized gap; unfolded, the cap comes off and the
        splitter is draggable again. Re-run whenever a section folds, since
        the natural height changes with it."""
        # Folded, the section must not claim the stretch either, or it
        # parks a gap under its header.
        expanded = self._table_section.is_expanded()
        self.layout().setStretchFactor(self._table_section, 1 if expanded else 0)
        if expanded:
            self.setMaximumHeight(_QWIDGETSIZE_MAX)
        else:
            layout = self.layout()
            layout.activate()
            # At the width it actually has: the flow rows wrap, and a plain
            # sizeHint doesn't know by how much.
            height = (
                layout.totalHeightForWidth(self.width())
                if layout.hasHeightForWidth() and self.width() > 0
                else self.sizeHint().height()
            )
            self.setMaximumHeight(height)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width() and not self._table_section.is_expanded():
            QTimer.singleShot(0, self._fit_height)

    def min_track_length(self) -> int:
        return self._min_track_length.value()

    def set_saved_filters(self, min_track_length: int, ranges: dict) -> None:
        """A saved analysis's cuts, put back silently -- the caller
        refreshes (`host.on_filters_changed`) once."""
        blocked = self._min_track_length.blockSignals(True)
        self._min_track_length.setValue(min_track_length)
        self._min_track_length.blockSignals(blocked)
        self.filters.set_filters(ranges or None)

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

    _ALL_REGIONS = "All"

    def set_region_choices(self, names: list[str]) -> None:
        """Offer `names` (the loaded tracks' region classes) in the picker,
        keeping the current pick if it survived; hidden with fewer than
        two, where there is nothing to choose between."""
        current = self._region_picker.currentText()
        blocked = self._region_picker.blockSignals(True)
        self._region_picker.clear()
        self._region_picker.addItems([self._ALL_REGIONS, *names])
        if current in names:
            self._region_picker.setCurrentText(current)
        self._region_picker.blockSignals(blocked)
        self._region_row.setVisible(len(names) > 1)

    def selected_region_class(self) -> Optional[str]:
        """The region class picked, or None for all of them."""
        if self._region_picker.count() <= 2:  # "All" plus at most one class
            return None
        text = self._region_picker.currentText()
        return None if not text or text == self._ALL_REGIONS else text

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
        self.set_region_choices([])
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
        self._total = total
        style_status_label(
            self._count_label, "ok" if df.height else "caution" if total else "neutral"
        )
        if current is not None:
            self.select_track_id(current)
        self._update_count_label()
        QTimer.singleShot(0, self._fit_height)

    def _update_count_label(self) -> None:
        """"n of N tracks", plus the selected track when there is one --
        with the table folded, this line is the only place the selection
        the NUTS tab and the track plot act on is written down."""
        if self._displayed_df is None:
            return
        text = f"{self._displayed_df.height} of {self._total} tracks"
        rows = self.table.selectionModel().selectedRows()
        if rows:
            track_id = self._model.row_dict(rows[0].row()).get("track_id")
            if track_id is not None:
                text += f" · track {int(track_id)} selected"
        self._count_label.setText(text)

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
        self._update_count_label()
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


_POSTERIOR_HELP = (
    "<b>Per track</b>, D is the median of its posterior (flat prior in ln D), and "
    "<b>low/high</b> are the 5% and 95% quantiles: a 90% credible interval. The "
    "likelihood is exact: each frame's localization error and the motion blur of "
    "the exposure are both modelled. A short track has a wide interval, and that "
    "width is the honest answer, not a failure."
    "<br><br><b>Grid</b>: each posterior is evaluated on a grid in ln D, and the flat "
    "prior is zero outside its range &mdash; so <i>D min</i>/<i>D max</i> are part of "
    "the analysis, not a numerical detail. A track whose posterior is cut by an edge is "
    "marked <i>D_at_grid_edge</i>: its median and interval move if the edge does. "
    "Near-immobile tracks reach <i>D min</i> this way, since localization error only "
    "lets the data bound D from above. These are diffusionkit's <tt>GridPostOptions</tt> "
    "fields, saved with the analysis, so a script can repeat the run exactly."
    "<br><br><b>Ensemble</b>: <i>summed</i> adds every track's log posterior &mdash; "
    "the posterior of one D shared by all of them. It is sharp, but only meaningful "
    "if they really do share a D. <i>Deconvolved</i> is how D is distributed across "
    "tracks, with each track's own uncertainty taken out (a smoothed nonparametric "
    "maximum likelihood). Its peak locations and the mass under each peak are "
    "robust; its peak widths are resolution-limited, not measured. <i>Pooled</i> "
    "averages the tracks' posteriors: where they put D, blurred by each one's own "
    "uncertainty &mdash; the deconvolution's starting point."
    "<br><br><b>Deconvolution</b>: more <i>iterations</i> remove more of that blur; "
    "<i>smoothing</i> (grid cells) keeps peaks from collapsing into "
    "spikes &mdash; less sharpens them, more widens them. D is spread over the D grid, "
    "the same range every track's prior has."
    "<br><br><b>α</b> (fBm exponent, K integrated out) has no motion-blur model, so "
    "it is only available for exposure 0. It costs ~30x D."
)


class _LogSpinBox(QDoubleSpinBox):
    """A positive value spanning decades (a D grid edge): shown as `%g`
    (1e-04, 10), typed in any float notation, and stepped by a factor of
    10 -- a linear spinbox's fixed decimals and additive steps suit
    neither end of 1e-4 to 10."""

    def __init__(self, value: float, minimum: float, maximum: float, tooltip: str = "") -> None:
        super().__init__()
        self.setDecimals(12)  # the stored precision; the text is `%g`
        self.setRange(minimum, maximum)
        self.setValue(value)
        self.setToolTip(tooltip)
        # Its size hint comes from the range ends' short text (1e-09),
        # which clips a value like 0.0001.
        self.setFixedWidth(_LOG_SPIN_WIDTH)

    def textFromValue(self, value: float) -> str:  # noqa: N802
        return f"{value:g}"

    def valueFromText(self, text: str) -> float:  # noqa: N802
        return float(text)

    def validate(self, text: str, pos: int):
        try:
            value = float(text)
        except ValueError:
            return QValidator.State.Intermediate, text, pos
        if self.minimum() <= value <= self.maximum():
            return QValidator.State.Acceptable, text, pos
        return QValidator.State.Intermediate, text, pos

    def stepBy(self, steps: int) -> None:  # noqa: N802
        self.setValue(self.value() * 10.0**steps)


_LOG_SPIN_WIDTH = 104


class _PosteriorTab(QWidget):
    """diffusionkit's grid posteriors (`diffusion.analyze_posteriors`):
    per-track D with its 90% interval, and the ensemble read across tracks.

    The one input not already on the layer is the camera **exposure** --
    separate from the frame interval, and one the result depends on: the D
    likelihood models the blur of a continuous exposure, so treating 20 ms
    as instantaneous biases D low. It is pre-filled from the layer's
    metadata and otherwise has to be typed -- the box starts at "not set",
    never at 0, and Run stays off until it has a value. Exposure 0 is also
    what makes the alpha posterior available (it has no blur model).

    The summary and the Ensemble and Posteriors figures are read over the
    tracks the tracks pane currently passes (and grouped by region class
    when there are several), so a filter change -- or a deconvolution
    setting -- updates them without a re-run: the per-track posteriors
    don't depend on which other tracks are in view."""

    # The exposure box's "not set" value -- one step below 0, which is a
    # legitimate (stroboscopic) exposure and must not double as "unknown".
    _EXPOSURE_UNSET = -0.0001

    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host
        self._analysis: Optional[PosteriorAnalysis] = None
        self._msd_df: Optional[pl.DataFrame] = None
        self._summary_values: Optional[dict] = None
        self._summary_by_group: Optional[dict] = None
        self._ensemble_window: Optional[PlotWindow] = None
        self._posteriors_window: Optional[PlotWindow] = None
        self._track_window: Optional[PlotWindow] = None
        self._msd_plot_window: Optional[PlotWindow] = None
        # What the layer said, so the exposure note can say where the box's
        # value came from.
        self._layer_exposure_s: Optional[float] = None

        self._exposure = double_spinbox(
            self._EXPOSURE_UNSET, self._EXPOSURE_UNSET, 3600.0, 0.001, decimals=4, suffix=" s",
            tooltip="Camera exposure per frame -- how long the sensor integrates, not\n"
            "the frame interval. The D posterior models the blur of a continuous\n"
            "exposure; entering 0 when the camera exposed for 20 ms biases D low.\n\n"
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
        self._min_frames.setRange(MIN_FRAMES, 10_000)
        self._min_frames.setValue(MIN_FRAMES)
        self._min_frames.setToolTip(
            "Tracks with fewer localizations than this are excluded (and counted\n"
            f"as such). {MIN_FRAMES} is diffusionkit's own minimum; a short track\n"
            "just gets a wide posterior, so there is no need to raise it for accuracy."
        )

        self._alpha = QCheckBox("α (slow)")
        self._alpha.setToolTip(
            "Also compute the posterior over the fBm exponent α, with K\n"
            "integrated out. No motion-blur model exists for it, so it needs\n"
            "exposure = 0. About 30x the cost of D (~50 ms per track)."
        )
        self._msd_comparison = QCheckBox("MSD")
        self._msd_comparison.setToolTip(
            "Also fit the classic 3-lag MSD models (Brownian D; power-law K and α)\n"
            "for comparison. They have no uncertainties and no blur model, so with\n"
            "exposure > 0 they are run with the exposure treated as 0 -- which\n"
            "biases them. Adds D_msd / α_msd columns and a D vs α plot."
        )

        # Flow rows rather than a form: a form's field column is too narrow
        # in a dock for three controls, and clipped them.
        exposure_row = flow_row(QLabel("exposure"), self._exposure)
        options_row = flow_row(QLabel("min points"), self._min_frames, self._alpha, self._msd_comparison)

        # The posterior grids -- diffusionkit's `GridPostOptions` fields,
        # read when Run is pressed. D's range is the flat prior's support,
        # so it stays in view; the alpha/K grids fold away.
        grid_default = GridPostOptions()
        d_range_tip = (
            "The range of D (µm²/s) the posterior is evaluated over -- the flat\n"
            "prior's support, so it is part of the analysis: a track whose\n"
            "posterior reaches an edge is cut there (and flagged D_at_grid_edge).\n"
            f"diffusionkit's default: {grid_default.D_min_um2_s:g} to {grid_default.D_max_um2_s:g}."
        )
        self._grid_D_min = _LogSpinBox(grid_default.D_min_um2_s, 1e-9, 1e4, tooltip=d_range_tip)
        self._grid_D_max = _LogSpinBox(grid_default.D_max_um2_s, 1e-9, 1e4, tooltip=d_range_tip)
        self._grid_n_D = _ispin(
            grid_default.n_D, 2, 100_000,
            tooltip="Grid points in ln D, spaced evenly between D min and D max.\n"
            f"diffusionkit's default {grid_default.n_D} gives ~2.3% steps over its default range.",
        )
        self._grid_alpha_min = _dspin(grid_default.alpha_min, 0.01, 1.99, 0.05, decimals=2,
                                      tooltip="Lowest α on the α grid (above 0, where fGn degenerates).")
        self._grid_alpha_max = _dspin(grid_default.alpha_max, 0.01, 1.99, 0.05, decimals=2,
                                      tooltip="Highest α on the α grid (below 2, where fGn degenerates).")
        self._grid_n_alpha = _ispin(grid_default.n_alpha, 2, 10_000, tooltip="Grid points in α.")
        self._grid_n_K = _ispin(
            grid_default.n_K, 2, 100_000,
            tooltip="Grid points in ln K, the nuisance parameter α's posterior integrates\n"
            "out. K's grid spans the same numeric range as D's.",
        )
        grid_form = _compact_form(QFormLayout())
        grid_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        grid_form.addRow("D min (µm²/s)", self._grid_D_min)
        grid_form.addRow("D max (µm²/s)", self._grid_D_max)
        grid_form.addRow("D points", self._grid_n_D)
        alpha_grid_box = QWidget()
        alpha_grid_form = _compact_form(QFormLayout(alpha_grid_box))
        alpha_grid_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        alpha_grid_form.addRow("α min", self._grid_alpha_min)
        alpha_grid_form.addRow("α max", self._grid_alpha_max)
        alpha_grid_form.addRow("α points", self._grid_n_alpha)
        alpha_grid_form.addRow("K points", self._grid_n_K)
        self._alpha_grid_section = CollapsibleSection("α / K grid", alpha_grid_box, expanded=False)
        self._grid_reset = QPushButton("default grid")
        self._grid_reset.setToolTip("diffusionkit's default grids (GridPostOptions()).")
        self._grid_reset.clicked.connect(lambda: self.set_grid(GridPostOptions()))
        self._grid_status = status_label("")
        for box in self._grid_boxes():
            box.valueChanged.connect(lambda _v: self._on_grid_changed())

        self._run_button = QPushButton("Run posteriors")
        self._run_button.clicked.connect(self._run)
        self._status = status_label("")
        self._summary = status_label("")
        self._summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._ensemble_button = QPushButton("Ensemble")
        self._ensemble_button.setToolTip(
            "Per-track medians, the deconvolved distribution and the summed\n"
            "(shared-D) posterior, over the tracks the filters pass -- one\n"
            "panel per region class when there are several."
        )
        self._ensemble_button.clicked.connect(self._show_ensemble)
        self._track_button = QPushButton("Track")
        self._track_button.setToolTip(
            "The selected track's posterior. Stays open and follows the selection."
        )
        self._track_button.clicked.connect(self._show_track)
        self._msd_plot_button = QPushButton("MSD")
        self._msd_plot_button.setToolTip("D vs α from the MSD fits (no uncertainties).")
        self._msd_plot_button.clicked.connect(self._show_msd_plot)
        self._posteriors_button = QPushButton("Posteriors")
        self._posteriors_button.setToolTip(
            "Every track's posterior as one row of a heat map, sorted by its\n"
            "median, beside the pooled posterior, the histogram of medians and\n"
            "the deconvolved distribution -- over the tracks the filters pass."
        )
        self._posteriors_button.clicked.connect(self._show_posteriors)
        plot_row = flow_row(
            QLabel("plot:"),
            self._ensemble_button,
            self._posteriors_button,
            self._track_button,
            self._msd_plot_button,
        )

        # The ensemble's deconvolution: read on commit (no keyboard
        # tracking), since each change re-deconvolves every group.
        default = Deconvolution()
        self._deconv_iters = _ispin(
            default.iters, 1, 20_000,
            tooltip="EM iterations. One is the pooled posterior; more remove more of the\n"
            "blur each track's own uncertainty adds. Cost is linear in it.",
        )
        self._deconv_iters.setSingleStep(100)
        # Units live in the row labels, not as suffixes: a suffix widens
        # every field to fit it, which is what made this section too wide.
        self._deconv_smooth = _dspin(
            default.smooth, 0.0, 20.0, 0.25, decimals=2,
            tooltip="Gaussian smoothing per iteration, in D grid cells.\n"
            "0 lets peaks sharpen toward spikes; more widens them. Peak positions\n"
            "and the mass under each are robust to it, widths are not.",
        )
        for box in (self._deconv_iters, self._deconv_smooth):
            box.setKeyboardTracking(False)
            box.valueChanged.connect(lambda _v: self._on_deconvolution_changed())
        self._deconv_reset = QPushButton("defaults")
        self._deconv_reset.clicked.connect(lambda: self.set_deconvolution(Deconvolution(), notify=True))
        self._deconv_status = status_label("")
        deconv_box = QWidget()
        deconv_layout = QVBoxLayout(deconv_box)
        deconv_layout.setContentsMargins(0, 0, 0, 0)
        deconv_layout.setSpacing(2)
        deconv_form = _compact_form(QFormLayout())
        deconv_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        deconv_form.addRow("iterations", self._deconv_iters)
        deconv_form.addRow("smoothing (cells)", self._deconv_smooth)
        deconv_form.addRow("", self._deconv_reset)
        deconv_layout.addLayout(deconv_form)
        deconv_layout.addWidget(self._deconv_status)
        self._deconv_section = CollapsibleSection("Deconvolution", deconv_box, expanded=False)

        help_text = note_label(_POSTERIOR_HELP)
        help_text.setTextFormat(Qt.TextFormat.RichText)
        self._help = CollapsibleSection("Reading the posteriors", help_text, expanded=False)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(exposure_row)
        layout.addWidget(self._exposure_note)
        layout.addWidget(options_row)
        layout.addLayout(grid_form)
        layout.addWidget(self._alpha_grid_section)
        layout.addWidget(self._grid_reset, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._grid_status)
        layout.addWidget(self._run_button)
        layout.addWidget(self._status)
        layout.addWidget(self._summary)
        layout.addWidget(plot_row)
        layout.addWidget(self._deconv_section)
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
    def analysis(self) -> Optional[PosteriorAnalysis]:
        return self._analysis

    @property
    def msd_df(self) -> Optional[pl.DataFrame]:
        return self._msd_df

    @property
    def summary_values(self) -> Optional[dict]:
        return self._summary_values

    @property
    def summary_by_group(self) -> Optional[dict]:
        return self._summary_by_group

    def deconvolution(self) -> Deconvolution:
        return Deconvolution(iters=self._deconv_iters.value(), smooth=self._deconv_smooth.value())

    def set_deconvolution(self, deconvolution: Deconvolution, *, notify: bool = False) -> None:
        for box, value in ((self._deconv_iters, deconvolution.iters), (self._deconv_smooth, deconvolution.smooth)):
            blocked = box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(blocked)
        if notify:
            self._on_deconvolution_changed()

    def _on_deconvolution_changed(self) -> None:
        self.refresh_summary()

    def _grid_boxes(self) -> tuple:
        return (
            self._grid_D_min, self._grid_D_max, self._grid_n_D,
            self._grid_alpha_min, self._grid_alpha_max, self._grid_n_alpha, self._grid_n_K,
        )

    def options(self) -> GridPostOptions:
        """The next run's `GridPostOptions`, from the controls. Raises
        ValueError for an invalid grid (see `_on_grid_changed`)."""
        grid = dict(
            D_min_um2_s=self._grid_D_min.value(),
            D_max_um2_s=self._grid_D_max.value(),
            n_D=self._grid_n_D.value(),
            alpha_min=self._grid_alpha_min.value(),
            alpha_max=self._grid_alpha_max.value(),
            n_alpha=self._grid_n_alpha.value(),
            n_K=self._grid_n_K.value(),
        )
        return posterior_options(self._min_frames.value(), self._alpha.isChecked(), grid)

    def set_grid(self, options: GridPostOptions) -> None:
        values = (
            options.D_min_um2_s, options.D_max_um2_s, options.n_D,
            options.alpha_min, options.alpha_max, options.n_alpha, options.n_K,
        )
        for box, value in zip(self._grid_boxes(), values):
            blocked = box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(blocked)
        self._on_grid_changed()

    def _on_grid_changed(self) -> None:
        """Say whether the grid is valid, and whether the analysis shown
        was run on a different one (its numbers are that grid's)."""
        try:
            options = self.options()
        except ValueError as exc:
            self._grid_status.setText(str(exc))
            style_status_label(self._grid_status, "error")
            self.refresh_inputs()
            return
        ran = self._analysis.options if self._analysis is not None else None
        if ran is not None and grid_record(ran) != grid_record(options):
            self._grid_status.setText(
                f"shown: run on D {ran.D_min_um2_s:g}–{ran.D_max_um2_s:g} µm²/s, {ran.n_D} points — Run to use this grid"
            )
            style_status_label(self._grid_status, "caution")
        else:
            self._grid_status.setText("")
            style_status_label(self._grid_status)
        self.refresh_inputs()

    def reset(self) -> None:
        self._analysis = None
        self._msd_df = None
        self._summary_values = None
        self._summary_by_group = None
        self._status.setText("")
        style_status_label(self._status)
        self._summary.setText("")
        self._refresh_plot_buttons()
        self._on_grid_changed()

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
        if exposure > dt * (1 + EXPOSURE_CLAMP_FRACTION):
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
            note = "from the layer's metadata"
        elif exposure == 0:
            return exposure, "0 = instantaneous (stroboscopic) — no blur modelled", "caution"
        else:
            note = "entered here — not recorded on the layer"
        return exposure, note + (" · α needs 0" if exposure > 0 else ""), "neutral"

    def refresh_inputs(self) -> None:
        exposure, note, level = self._exposure_for_run()
        self._exposure_note.setText(note)
        style_status_label(self._exposure_note, level)
        try:
            self.options()
            grid_ok = True
        except ValueError:
            grid_ok = False
        self._run_button.setEnabled(exposure is not None and grid_ok and self.host.has_tracks)
        alpha_possible = exposure == 0
        self._alpha.setEnabled(alpha_possible)
        if not alpha_possible:
            self._alpha.setChecked(False)

    def _refresh_plot_buttons(self) -> None:
        has_run = self._analysis is not None and len(self._analysis.fitted_ids) > 0
        self._ensemble_button.setEnabled(has_run)
        self._posteriors_button.setEnabled(has_run)
        self._track_button.setEnabled(has_run)
        self._msd_plot_button.setEnabled(self._msd_df is not None)

    def report_saved(self, text: str) -> None:
        self._status.setText(text)
        style_status_label(self._status, "ok")

    def report_error(self, text: str) -> None:
        self._status.setText(text)
        style_status_label(self._status, "error")

    def report_caution(self, text: str) -> None:
        self._status.setText(text)
        style_status_label(self._status, "caution")

    def show_loaded_summary(self, text: str) -> None:
        self._summary.setText(text)

    def _run(self) -> None:
        tracks = self.host.diffkit_tracks_for_fit()
        exposure, _note, _level = self._exposure_for_run()
        if tracks is None or exposure is None:
            return
        try:
            options = self.options()
        except ValueError as exc:
            self.report_error(f"grid: {exc}")
            return
        self._status.setText("running…")
        style_status_label(self._status)
        worker = _run_posterior_worker(
            tracks,
            self.host.dt_s,
            exposure,
            options,
            self._msd_comparison.isChecked(),
            self.host.progress_callback,
        )
        self.host.start_worker(worker, self._on_finished, self._on_error, [self._run_button], "posteriors")

    def _on_finished(self, result: tuple[PosteriorAnalysis, Optional[pl.DataFrame]]) -> None:
        analysis, msd_fits = result
        n_ok = self._adopt(analysis, msd_track_table(msd_fits) if msd_fits is not None else None)
        self._status.setText(f"{n_ok} of {analysis.fits.height} tracks fitted")
        style_status_label(self._status, "ok" if n_ok else "caution")
        self.refresh_summary()
        if n_ok:
            self._show_ensemble()

    def _adopt(self, analysis: PosteriorAnalysis, msd_df: Optional[pl.DataFrame]) -> int:
        """Make `analysis` this tab's, and hand its per-track columns to
        the host. Returns how many tracks were fitted."""
        self._analysis = analysis
        self._msd_df = msd_df
        self._refresh_plot_buttons()
        self._on_grid_changed()
        self.host.set_posterior_results(analysis, posterior_results_table(analysis, msd_df))
        return len(analysis.fitted_ids)

    def restore(self, saved: SavedAnalysis) -> None:
        """A bundle's saved analysis, reopened (see the module docstring):
        as if its run had just finished here, with the controls set to what
        it was run and summarized with, so Run re-fits the same way. The
        caller refreshes the summary once the filters are back too."""
        analysis = saved.analysis
        self._min_frames.setValue(analysis.min_frames)
        self._alpha.setChecked(analysis.alpha_ids is not None and self._alpha.isEnabled())
        self._msd_comparison.setChecked(saved.msd is not None)
        self.set_deconvolution(saved.deconvolution)
        n_ok = self._adopt(analysis, saved.msd)
        self.set_grid(analysis.options)
        self._status.setText(
            f"saved analysis loaded: {n_ok} of {analysis.fits.height} tracks fitted, "
            f"exposure {units.fmt_unit(analysis.acquisition.exposure_s, units.SECONDS)} — Run to re-fit"
        )
        style_status_label(self._status, "ok")

    def refresh_summary(self) -> None:
        """Recompute the population summary over the tracks the pane passes
        -- called after a run and whenever the filters change."""
        if self._analysis is None:
            return
        ids = self.host.combined_filtered_track_ids()
        deconvolution = self.deconvolution()
        self._summary_values = summarize(self._analysis, ids, deconvolution)
        groups = self.host.group_track_ids(ids)
        self._summary_by_group = (
            {name: summarize(self._analysis, group_ids, deconvolution) for name, group_ids in groups.items()}
            if groups
            else None
        )
        self._summary.setText(
            _format_posterior_summary(self._summary_values, self._analysis)
            + _format_by_group(self._summary_by_group)
        )
        if self._ensemble_window is not None and self._ensemble_window.isVisible():
            self._show_ensemble()
        if self._posteriors_window is not None and self._posteriors_window.isVisible():
            self._show_posteriors()

    def _panels(self, *, track_posteriors: bool = False) -> list[dict]:
        """`ensemble_panels` over the tracks the filters pass, one per
        region class when there are several -- or [] (and says so)."""
        if self._analysis is None:
            return []
        ids = self.host.combined_filtered_track_ids()
        groups = self.host.group_track_ids(ids) or {"all": ids}
        panels = ensemble_panels(
            self._analysis, groups, self.deconvolution(), track_posteriors=track_posteriors
        )
        if not panels:
            self._status.setText("no fitted tracks pass the current filters")
            style_status_label(self._status, "caution")
        return panels

    def _show_posteriors(self) -> None:
        panels = self._panels(track_posteriors=True)
        if not panels:
            return
        if self._posteriors_window is None:
            self._posteriors_window = PlotWindow("Posterior: every track", parent=self)
        self._posteriors_window.show_figure(plot_d_posteriors(self._analysis.D_grid_um2_s, panels))

    def _show_ensemble(self) -> None:
        panels = self._panels()
        if not panels:
            return
        figure = plot_d_ensemble(
            self._analysis.D_grid_um2_s, panels, self._analysis.alpha_grid if self._analysis.has_alpha else None
        )
        if self._ensemble_window is None:
            self._ensemble_window = PlotWindow("Posterior: ensemble", parent=self)
        self._ensemble_window.show_figure(figure)

    def _show_track(self) -> None:
        track_id = self.host.selected_track_id
        if self._analysis is None:
            return
        if track_id is None:
            self._status.setText("select a track (table or viewer) first")
            style_status_label(self._status, "caution")
            return
        data = track_posterior(self._analysis, track_id)
        if data is None:
            self._status.setText(f"track {track_id} was not fitted (see its posterior_status)")
            style_status_label(self._status, "caution")
            return
        if self._track_window is None:
            self._track_window = PlotWindow("Posterior: selected track", parent=self)
        self._track_window.show_figure(plot_track_posterior(**data))

    def on_track_selected(self) -> None:
        """Follow the selection while the track window is open."""
        if self._track_window is not None and self._track_window.isVisible():
            self._show_track()

    def _show_msd_plot(self) -> None:
        if self._msd_df is None:
            return
        usable = self._msd_df.filter(
            pl.col("D_msd_um2_s").is_not_null()
            & (pl.col("D_msd_um2_s") > 0)
            & pl.col("alpha_msd").is_not_null()
        )
        if usable.height < 2:
            self._status.setText("MSD comparison: too few tracks with a positive D and an α to plot")
            style_status_label(self._status, "caution")
            return
        exposure0 = self._analysis is not None and self._analysis.acquisition.exposure_s > 0
        figure = plot_property_joint(
            usable,
            "D_msd_um2_s",
            "alpha_msd",
            log_x=True,
            title="MSD comparison — no uncertainties" + (", exposure treated as 0" if exposure0 else ""),
        )
        if self._msd_plot_window is None:
            self._msd_plot_window = PlotWindow("MSD comparison: D vs α", parent=self)
        self._msd_plot_window.show_figure(figure)

    def _on_error(self, exc: Exception) -> None:
        self._status.setText(f"error: {exc}")
        style_status_label(self._status, "error")


def _format_by_group(by_group: Optional[dict]) -> str:
    """One line per region class under the pooled summary: each region was
    linked on its own, and whether its motion differs is why it was drawn."""
    if not by_group:
        return ""
    lines = ["", "by region class:"]
    for name, summary in by_group.items():
        line = (
            f"  {name}: {summary['n_tracks']} · median D "
            f"{units.fmt(summary.get('median_D_um2_s'), 'D_um2_s')}"
        )
        if summary.get("summed_D_median_um2_s") is not None:
            line += f" · shared {summary['summed_D_median_um2_s']:.3g}"
        if summary.get("median_alpha") is not None:
            line += f" · α {summary['median_alpha']:.2f}"
        lines.append(line)
    return "\n".join(lines)


def _format_posterior_summary(summary: dict, analysis: Optional[PosteriorAnalysis] = None) -> str:
    """The population summary as a few short lines -- counts by
    diffusionkit status first, since excluded/invalid tracks are part of
    the result, not noise to drop."""
    others = {
        key[2:]: value
        for key, value in summary.items()
        if key.startswith("n_")
        and key not in ("n_tracks", "n_ok", "n_alpha", "n_D_at_grid_edge")
        and not key.startswith("n_frames")
    }
    counts = f"{summary.get('n_ok', 0)} fitted"
    if others:
        counts += " · " + " · ".join(f"{count} {status}" for status, count in sorted(others.items()))
    lines = [f"{summary['n_tracks']} tracks: {counts}"]
    if summary.get("n_frames_median") is not None:
        lines.append(
            f"length {summary['n_frames_min']:.0f} / {summary['n_frames_median']:.0f} / "
            f"{summary['n_frames_max']:.0f} {units.POINTS} (min / median / max)"
        )
    if summary.get("n_D_at_grid_edge"):
        lines.append(
            f"{summary['n_D_at_grid_edge']} cut by the D grid's edge (D_at_grid_edge) — "
            "their numbers depend on the grid range"
        )
    if summary.get("median_D_um2_s") is not None:
        lines.append(
            f"median D {units.fmt(summary['median_D_um2_s'], 'D_um2_s')} "
            f"(IQR {summary['q25_D_um2_s']:.3g}–{summary['q75_D_um2_s']:.3g})"
        )
    if summary.get("summed_D_median_um2_s") is not None:
        lines.append(
            f"shared D {summary['summed_D_median_um2_s']:.3g} "
            f"[{summary['summed_D_low_um2_s']:.3g}, {summary['summed_D_high_um2_s']:.3g}] · "
            f"deconvolved mode {summary['deconvolved_D_mode_um2_s']:.3g}"
        )
    if summary.get("median_alpha") is not None:
        lines.append(
            f"median α {summary['median_alpha']:.2f} (IQR {summary['q25_alpha']:.2f}–"
            f"{summary['q75_alpha']:.2f}) · shared α {summary['summed_alpha_median']:.2f}"
        )
    if analysis is not None:
        acquisition = analysis.acquisition
        lines.append(
            f"dt {units.fmt_unit(acquisition.dt_s, 's/frame')} · exposure "
            f"{units.fmt_unit(acquisition.exposure_s, units.SECONDS)}"
        )
    return "\n".join(lines)


class _MapTab(QWidget):
    """The spatial map: one point per track at its centroid, colored by any
    per-track number a run produced (`DiffusionAnalysisWidget.
    register_spatial_source`) -- the reason this analysis stays inside
    napari next to the image. Nothing to run here; the color range is set
    by dragging the histogram, and a written scale says what the colors
    mean, since a napari Points layer colored by a feature has no legend."""

    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host
        self._hint = note_label("Run the posteriors first — the map colors each track's centroid by a result.")
        self._color_by_picker = QComboBox()
        self._color_by_picker.setEnabled(False)
        self._color_by_picker.currentTextChanged.connect(self._on_color_by_changed)
        # D spans orders of magnitude across a field of tracks, so a linear
        # histogram is one tall spike at the low end.
        self._map_log_scale_check = QCheckBox("log")
        self._map_log_scale_check.setEnabled(False)
        self._map_log_scale_check.toggled.connect(self._on_map_log_scale_toggled)
        color_form = QFormLayout()
        color_form.setContentsMargins(0, 0, 0, 0)
        color_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        color_form.addRow("color by", flow_row(self._color_by_picker, self._map_log_scale_check))
        self._map_scale_label = status_label("")
        self._map_histogram = HistogramRangeWidget()
        self._map_histogram.setEnabled(False)
        self._map_histogram.rangeChanged.connect(self._on_map_range_changed)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self._hint)
        layout.addLayout(color_form)
        layout.addWidget(self._map_scale_label)
        layout.addWidget(self._map_histogram)
        layout.addStretch()
        self.setLayout(layout)

    def reset(self) -> None:
        self._color_by_picker.blockSignals(True)
        self._color_by_picker.clear()
        self._color_by_picker.blockSignals(False)
        self._color_by_picker.setEnabled(False)
        self._map_histogram.setEnabled(False)
        self._map_log_scale_check.setEnabled(False)
        self._map_scale_label.setText("")
        self._hint.setVisible(True)

    def _on_color_by_changed(self, column: str) -> None:
        if not column:
            return
        self.host.set_map_color_by(column)
        self._refresh_map_histogram(reset_range=True)

    def _on_map_range_changed(self, vmin: float, vmax: float) -> None:
        self.host.set_map_contrast_limits(vmin, vmax)
        self._update_map_scale_label(vmin, vmax)

    def _on_map_log_scale_toggled(self, enabled: bool) -> None:
        """Re-bin the histogram on a log10 axis. `set_log_scale` is silent
        and may clamp the range off zero/negative values, so the host's
        contrast limits are re-synced from it explicitly."""
        self._map_histogram.set_log_scale(enabled)
        vmin, vmax = self._map_histogram.range()
        self.host.set_map_contrast_limits(vmin, vmax)
        self._update_map_scale_label(vmin, vmax)

    def _update_map_scale_label(self, vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
        column = self._color_by_picker.currentText()
        if not column:
            self._map_scale_label.setText("")
            return
        if vmin is None or vmax is None:
            vmin, vmax = self._map_histogram.range()
        unit = units.unit_of(column)
        self._map_scale_label.setText(
            f"{units.header(column)}: {vmin:.4g} → {vmax:.4g}{' ' + unit if unit else ''}"
        )
        self._map_scale_label.setToolTip(units.tooltip(column))

    def on_spatial_source_registered(self, preferred_color_by: Optional[str] = None) -> None:
        """A run added color choices: repopulate the picker with the union
        of every registered source's columns."""
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
        self._hint.setVisible(not enabled)
        self._color_by_picker.setEnabled(enabled)
        self._map_histogram.setEnabled(enabled)
        self._map_log_scale_check.setEnabled(enabled)
        if enabled:
            self.host.set_map_color_by(self._color_by_picker.currentText())

    def refresh_map_histogram(self) -> None:
        """After a filter change: the color range stays as set, only the
        histogram's data follows the new filtered set."""
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


def _bayes_available() -> bool:
    """Whether diffusionkit's NUTS stack (JAX, NumPyro) is installed -- the
    optional `[bayes]` extra. Checked without importing it: importing jax
    takes seconds and switches it to float64 process-wide."""
    import importlib.util

    return all(importlib.util.find_spec(name) is not None for name in ("jax", "numpyro"))


class _NutsTab(QWidget):
    """`diffusionkit.bayes.fit_track`: a full NUTS posterior for the one
    track selected in the tracks pane -- the per-track diagnostic for when
    a posterior's shape matters (the K/alpha/sigma correlations the 1D
    grid posteriors don't show). Tens of seconds per track, so there is no
    bulk version. Needs the optional `[bayes]` extra (JAX, NumPyro)."""

    def __init__(self, host: "DiffusionAnalysisWidget") -> None:
        super().__init__()
        self.host = host
        self._plot_window: Optional[PlotWindow] = None
        self._available = _bayes_available()

        self._model_picker = QComboBox()
        self._model_picker.addItems(["anomalous", "normal"])
        self._model_picker.setToolTip(
            "anomalous: K, α and the localization σ. normal: D and σ."
        )
        self._fit_button = QPushButton("Fit selected track")
        self._fit_button.setToolTip("Full NUTS posterior (4 chains) for the track selected above.")
        self._fit_button.clicked.connect(self._run)
        self._status = status_label("")
        self._result_label = status_label("")
        self._result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        row = flow_row(QLabel("model"), self._model_picker, self._fit_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(row)
        layout.addWidget(self._status)
        layout.addWidget(self._result_label)
        layout.addStretch()
        self.setLayout(layout)
        if not self._available:
            self._fit_button.setEnabled(False)
            self._model_picker.setEnabled(False)
            self._status.setText("needs the [bayes] extra: uv sync --extra bayes")
            style_status_label(self._status, "caution")

    def reset(self) -> None:
        if self._available:
            self._status.setText("")
            style_status_label(self._status)
        self._result_label.setText("")

    def _run(self) -> None:
        track_id = self.host.selected_track_id
        if track_id is None:
            self._status.setText("select a track (table or viewer) first")
            style_status_label(self._status, "caution")
            return
        track_df = self.host.diffkit_track(track_id)
        if track_df is None or track_df.height < 3:
            self._status.setText(f"track {track_id} is too short to fit")
            style_status_label(self._status, "caution")
            return
        model = self._model_picker.currentText()
        self._status.setText("running… (compiles on first use; tens of seconds)")
        style_status_label(self._status)
        worker = _run_nuts_worker(track_df, self.host.dt_s, model)
        self.host.start_worker(
            worker, self._on_finished, self._on_error, [self._fit_button], f"NUTS, track {track_id}"
        )

    def _on_finished(self, fit) -> None:
        from diffusionkit.bayes import viz as dk_bayes_viz

        self._status.setText(f"track {fit.track_id} fit (n={fit.track_length})")
        style_status_label(self._status, "ok")
        # Parameter names come from diffusionkit ("D", "K", "alpha",
        # "sigma"), each with its own unit; `_PARAM_COLUMNS` maps each to
        # the column name `napari_gemscape2.units` knows it by.
        lines = [f"track {fit.track_id}, {fit.model} (median, 90% HPDI)"]
        for name, value in fit.params.items():
            column = _PARAM_COLUMNS.get(name, name)
            lines.append(f"  {name} = {units.fmt(value, column)}  [{fit.lo[name]:.4g}, {fit.hi[name]:.4g}]")
        self._result_label.setText("\n".join(lines))

        samples_dict, _mcmc = fit.raw
        param_names = list(fit.params.keys())
        flat, _trace = dk_bayes_viz.samples_dict_to_arrays(samples_dict, param_names)
        figure = dk_bayes_viz.plot_posterior_corner(flat, param_names)
        _label_corner_axes(figure, param_names)
        if self._plot_window is None:
            self._plot_window = PlotWindow("NUTS posterior (selected track)", parent=self)
        self._plot_window.show_figure(figure)
        self.host.set_nuts_result(_nuts_row(fit))

    def _on_error(self, exc: Exception) -> None:
        self._status.setText(f"error: {exc}")
        style_status_label(self._status, "error")


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
        # table (see `diffusion.qc_aggregate_table`).
        self._qc_columns: list[str] = []
        self._joined_track_df: Optional[pl.DataFrame] = None
        # The last posterior run, and the per-track columns the tracks
        # pane shows from it (plus the MSD comparison's, when run).
        self._posterior: Optional[PosteriorAnalysis] = None
        self._posterior_df: Optional[pl.DataFrame] = None
        self._map_color_by: Optional[str] = None
        # name -> per-track df (track_id + one or more value columns) --
        # every tab that produces a per-track number registers here, and
        # the spatial map / its color-by picker draw from the union of
        # whatever's currently registered (see register_spatial_source).
        self._spatial_sources: dict[str, pl.DataFrame] = {}
        self._nuts_rows: list[dict] = []
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
        self._posterior_tab = _PosteriorTab(self)
        self._map_tab = _MapTab(self)
        self._nuts_tab = _NutsTab(self)

        # Each page scrolls independently rather than the whole tab widget
        # scrolling: wrapping the QTabWidget itself would carry the tab bar
        # off the top as soon as you scrolled down inside a page, which is
        # exactly when you want to be able to switch away from it.
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(scrolled(self._posterior_tab), "Posterior")
        tabs.addTab(scrolled(self._map_tab), "Map")
        tabs.addTab(scrolled(self._nuts_tab), "NUTS")
        tabs.setTabToolTip(0, "Per-track grid posteriors over D (and α), and the ensemble (diffusionkit.gridpost)")
        tabs.setTabToolTip(1, "Color each track's centroid in the viewer by a result")
        tabs.setTabToolTip(2, "Full NUTS posterior for the selected track (diffusionkit.bayes)")

        # One session-wide switch, not a copy on each tab -- see this
        # module's docstring.
        self._restrict_checkbox = QCheckBox("restrict to filtered")
        self._restrict_checkbox.setToolTip(
            "When checked, every fit runs only on the tracks currently "
            "passing the filters above, instead of on all of them."
        )

        # Not "Save results": that is the experiment list's button, which
        # writes the tracks themselves -- this one writes the analysis of
        # them into the same bundle.
        self._save_button = QPushButton("Save analysis")
        self._save_button.setToolTip(
            f"Write the per-track summary ({TRACKS_SUMMARY_FILENAME}), every\n"
            "track's posterior (posterior_D/_alpha.parquet), the ensemble\n"
            "distributions (distributions_D/_alpha.csv, over the tracks the\n"
            "filters pass) and the settings and population summary\n"
            "(diffusion_summary.json) into this layer's bundle."
        )
        self._save_button.clicked.connect(self._save_results)
        self._save_button.setEnabled(False)

        self._export_csv_button = QPushButton("Export CSV…")
        self._export_csv_button.setToolTip(
            "Write the per-track summary (the same table Save writes as\n"
            f"{TRACKS_SUMMARY_FILENAME}) as CSV, for a spreadsheet or Prism."
        )
        self._export_csv_button.clicked.connect(self._export_tracks_csv)
        self._export_csv_button.setEnabled(False)

        footer = flow_row(self._restrict_checkbox, self._save_button, self._export_csv_button)

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        self._progress_bar.hide()

        # The tracks table and the analysis tabs both want more height than
        # a docked panel has, and which one deserves it changes by the
        # minute (scanning rows vs. reading a fit's output), so it is a
        # drag, not a fixed ratio. Neither pane is collapsible: dragging
        # either to zero would hide the selection the other one acts on.
        # The table itself folds instead (`_TracksPane._fit_height`), and
        # while it is folded the tracks pane is capped at its own height.
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
        ids = self._passing_track_ids()
        region = self._tracks_pane.selected_region_class()
        if region is not None and self._base_track_df is not None and "region_class" in self._base_track_df.columns:
            region_ids = set(self._base_track_df.filter(pl.col("region_class") == region)["track_id"].to_list())
            ids = region_ids if ids is None else (ids & region_ids)
        return ids

    def group_track_ids(self, ids: Optional[set]) -> Optional[dict[str, set]]:
        """`{region class: its track ids}` restricted to `ids` (None = all),
        or None when the loaded tracks don't span more than one class."""
        if self._base_track_df is None:
            return None
        return region_class_groups(self._base_track_df, ids)

    def track_groups(self) -> Optional[pl.DataFrame]:
        """`(track_id, group)` naming each loaded track's region class, or
        None when the tracks don't span more than one -- what per-class
        summaries and plot colors are keyed on."""
        base = self._base_track_df
        if (
            base is None
            or "region_class" not in base.columns
            or base["region_class"].drop_nulls().n_unique() < 2
        ):
            return None
        return base.select("track_id", pl.col("region_class").alias("group"))

    def on_filters_changed(self) -> None:
        self._rebuild_track_table()
        self._update_spatial_map_layer()
        self._map_tab.refresh_map_histogram()
        self._posterior_tab.refresh_summary()

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
    ) -> None:
        """Run `worker` in the one host-wide slot. Until the worker's first
        report through `progress_callback` arrives (and throughout, for one
        that never reports) the bar is a busy indicator."""
        if self._worker is not None:
            return
        for widget in busy_widgets:
            widget.setEnabled(False)
        self._progress_label = label
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat(f"{label}…")
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
            # Stay a busy indicator until there is progress to show; a
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

    def _reset_analysis_state(self) -> None:
        """Forget every fit and the current track -- they belonged to the
        track set that was loaded, not to the one about to be."""
        self._posterior = None
        self._posterior_df = None
        self._map_color_by = None
        self._spatial_sources = {}
        self._nuts_rows = []
        self._current_track_id = None

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
        self._reset_analysis_state()
        self.pixel_size_um = 1.0
        self.dt_s = 1.0
        self.units_known = False
        self._source_label.setText("no Tracks layer in this viewer")
        style_status_label(self._source_label)
        self._tracks_pane.reset()
        self._posterior_tab.reset()
        self._posterior_tab.set_layer_exposure(None)
        self._map_tab.reset()
        self._nuts_tab.reset()
        self._save_button.setEnabled(False)
        self._export_csv_button.setEnabled(False)
        self._clear_overlay_layers()

    @staticmethod
    def _layer_track_table(layer: Tracks) -> Optional[pl.DataFrame]:
        """The layer's vertices plus every per-vertex property (se_y/se_x
        and whatever QC/derived columns viewer.py put there -- flux, bg,
        fit_sigma, track_length, ...), aligned with track_id: one table
        serves both the diffusionkit conversion (needs
        track_id/frame/y/x/se_y/se_x) and the per-track QC aggregates.

        None when the layer has no `se_y`/`se_x` (spotsolve's per-detection
        CRLB, renamed for diffusionkit in napari_gemscape2.diffusion). Requiring
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
        # A missing value (a track's first `link_margin`, say) crossed into
        # the layer as NaN; back to null, or every per-track mean over the
        # column is NaN too (`diffusion.qc_aggregate_table`).
        table = table.with_columns(
            pl.col(c).fill_nan(None) for c, dtype in table.schema.items() if dtype.is_float()
        )
        # The layer carries the region as its label (Tracks properties are
        # numeric); its class comes from the layer's `region_classes`
        # metadata ({label: class}).
        classes = dict(layer.metadata.get("region_classes") or {})
        if "region" in table.columns and classes:
            lookup = pl.DataFrame(
                {"region": [int(k) for k in classes], "region_class": list(classes.values())},
                schema={"region": table.schema["region"], "region_class": pl.Utf8},
            )
            table = table.join(lookup, on="region", how="left", maintain_order="left")
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
        self._base_track_df, self._qc_columns = base_track_table(
            self._diffkit_tracks, track_points_df
        )
        self._update_source_label()
        # The exposure box follows the layer's record only when that record
        # changed (a new layer, or the image panel's override edited): a
        # filter redraw re-adopts the same layer, and must not wipe an
        # exposure typed into the tab.
        self.sync_layer_exposure()

    def sync_layer_exposure(self) -> None:
        """Hand the loaded layer's recorded exposure to the Posterior tab
        if it changed since last looked at -- see `_adopt_track_table`."""
        layer = self._tracks_layer
        layer_exposure = layer.metadata.get("exposure_s") if layer is not None else None
        if self._layer_exposure_s is _UNSEEN or layer_exposure != self._layer_exposure_s:
            self._layer_exposure_s = layer_exposure
            self._posterior_tab.set_layer_exposure(layer_exposure)
        else:
            self._posterior_tab.refresh_inputs()

    def _update_source_label(self) -> None:
        """Name the loaded layer, its bundle, and -- the part that is not
        cosmetic -- the two conversion factors every physical column in
        this widget is computed with.

        `tracks_to_diffusionkit_df` multiplies pixel positions by
        `pixel_size_um` and frame indices by `dt_s`, so every `_um`/`_s`
        column here, and every D and K fitted from them, is those two
        numbers. A layer that carries neither still gets 1.0 for both
        (see `viewer.layer_units_metadata`) because the conversion has to
        run on something -- and then `D_median_um2_s` is really px²/frame
        under a µm²/s name. That case gets said out loud rather than
        rendered identically to a calibrated one."""
        layer = self._tracks_layer
        if layer is None:
            return
        # The bundle by name: its full path runs to several lines in a
        # dock, so it lives in the tooltip.
        where = (
            f"→ {self._result_dir.name}"
            if self._result_dir is not None
            else "(no results bundle — can't save)"
        )
        self._source_label.setToolTip(str(self._result_dir) if self._result_dir is not None else "")
        scale = (
            f"{units.fmt_unit(self.pixel_size_um, units.UM + '/px')} · "
            f"{units.fmt_unit(self.dt_s, 's/frame')}"
        )
        if self.units_known:
            self._source_label.setText(f"'{layer.name}' {where} · {scale}")
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

        self._reset_analysis_state()

        self._tracks_pane.reset()
        self._tracks_pane.set_qc_columns(self._qc_columns)
        self._tracks_pane.set_region_choices(self._region_classes_loaded())
        self._posterior_tab.reset()
        self._map_tab.reset()
        self._nuts_tab.reset()
        self._rebuild_track_table()
        self._tracks_pane.set_plot_columns(
            self._joined_track_df, prefer_x="radius_of_gyration_um", prefer_y="flux_mean"
        )
        self._clear_overlay_layers()

        self._mouse_callback = self._make_click_callback()
        layer.mouse_drag_callbacks.append(self._mouse_callback)
        layer.events.data.connect(self._on_tracks_layer_event)
        layer.events.properties.connect(self._on_tracks_layer_event)

        self._restore_saved_analysis()
        self._update_save_enabled()

    def _restore_saved_analysis(self) -> None:
        """Reopen the bundle's saved analysis when it was fitted on the
        tracks now loaded -- see the module docstring. When it wasn't (or
        can't be read), its numbers are shown as text only, and Run is the
        way to plots."""
        if self._result_dir is None or self._diffkit_tracks is None:
            return
        try:
            tables = load_diffusion_results(self._result_dir)
            if tables is None:
                self._show_saved_summary_only(None)
                return
            saved = restore_analysis(self._diffkit_tracks, **tables)
        except (StaleAnalysisError, OSError, ValueError, KeyError, pl.exceptions.PolarsError) as exc:
            self._show_saved_summary_only(exc)
            return
        self._nuts_rows = list(saved.nuts_rows)
        self._posterior_tab.restore(saved)
        # The tracks pane's cuts it was summarized under -- after the
        # results are joined in, since a cut may be on a result column.
        record = saved.summary.get("tracks_summary_filters") or {}
        self._tracks_pane.set_saved_filters(
            record.get("min_track_length", 1),
            {col: tuple(bounds) for col, bounds in (record.get("ranges") or {}).items()},
        )
        self.on_filters_changed()

    def _show_saved_summary_only(self, reason: Optional[Exception]) -> None:
        summary = load_diffusion_summary(self._result_dir)
        if summary is None:
            return
        self._posterior_tab.show_loaded_summary("Saved analysis:\n" + _format_summary(summary))
        if reason is None:
            self._posterior_tab.report_saved("this bundle has a saved summary — Run to redo it")
        else:
            self._posterior_tab.report_caution(f"saved analysis not loaded: {reason} — Run to re-fit")

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
        self._tracks_pane.set_region_choices(self._region_classes_loaded())
        self._rebuild_track_table()
        self._update_spatial_map_layer()
        self._map_tab.refresh_map_histogram()
        self._posterior_tab.refresh_summary()
        self._update_save_enabled()

    def _region_classes_loaded(self) -> list[str]:
        """The region classes the loaded tracks were linked in."""
        base = self._base_track_df
        if base is None or "region_class" not in base.columns:
            return []
        return sorted(set(base["region_class"].drop_nulls().to_list()))

    def _update_save_enabled(self) -> None:
        has_results = self._posterior is not None or bool(self._nuts_rows)
        self._save_button.setEnabled(has_results and self._result_dir is not None)
        self._export_csv_button.setEnabled(self._base_track_df is not None)

    # -- track table + selection --

    def _rebuild_track_table(self) -> None:
        if self._base_track_df is None:
            return
        df = self._base_track_df
        results = self._per_track_results()
        if results is not None:
            df = df.join(results, on="track_id", how="left")

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
            # target for the NUTS fit pointing at a track that
            # isn't even in the table anymore.
            self._current_track_id = None
            self._clear_highlight_layer()

        self._sync_tracks_layer_display(ids)

    def _per_track_results(self) -> Optional[pl.DataFrame]:
        """Every analysis column, one row per track: the posterior run's
        (and MSD comparison's), and the latest NUTS fit of each track."""
        df = self._posterior_df
        if self._nuts_rows:
            nuts = pl.DataFrame(self._nuts_rows).group_by("track_id", maintain_order=True).last()
            df = nuts if df is None else df.join(nuts, on="track_id", how="full", coalesce=True)
        return df

    def set_posterior_results(self, analysis: PosteriorAnalysis, display_df: pl.DataFrame) -> None:
        """A posterior run finished: keep it for Save, join its per-track
        columns into the tracks pane, offer them as spatial-map colors, and
        put `_TRACK_COLOR_COLUMNS` on the Tracks layer itself."""
        self._posterior = analysis
        self._posterior_df = display_df
        numeric = [c for c, dtype in zip(display_df.columns, display_df.dtypes) if dtype.is_numeric()]
        self.register_spatial_source("posterior", display_df.select(numeric))
        self._rebuild_track_table()
        self._tracks_pane.set_plot_columns(
            self._joined_track_df,
            prefer_x="D_median_um2_s",
            prefer_y="alpha_median" if analysis.has_alpha else "track_length",
        )
        self._map_tab.on_spatial_source_registered("D_median_um2_s")
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

    def set_map_color_by(self, color_by: str) -> None:
        self._map_color_by = color_by
        self._update_spatial_map_layer()

    def set_map_contrast_limits(self, vmin: float, vmax: float) -> None:
        if self._live(self._spatial_map_layer) is not None and vmin < vmax:
            self._spatial_map_layer.face_contrast_limits = (vmin, vmax)

    def set_nuts_result(self, row: dict) -> None:
        self._nuts_rows.append(row)
        self._rebuild_track_table()
        self._update_save_enabled()

    def on_table_row_selected(self, track_id: int) -> None:
        self._current_track_id = track_id
        self._update_highlight_layer(track_id)
        self._jump_to_track_end(track_id)
        self._posterior_tab.on_track_selected()

    def on_viewer_track_clicked(self, track_id: int) -> None:
        self._current_track_id = track_id
        self._update_highlight_layer(track_id)
        self._tracks_pane.select_track_id(track_id)
        self._posterior_tab.on_track_selected()

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

    def _jump_to_track_end(self, track_id: int) -> None:
        """Move the time slider to the track's last frame, where the Tracks
        layer draws its tail in full, ending inside the box. Only for
        table selections -- a track clicked in the viewer is already on
        screen, and yanking the slider away from it would be jarring."""
        layer = self._live(self._tracks_layer)
        if layer is None or self._tracks_df_px is None:
            return
        last = self._tracks_df_px.filter(pl.col("track_id") == track_id)["frame"].max()
        if last is None:
            return
        # Tracks data is [id, t, y, x]: time is the layer's first axis.
        world_axis = self.viewer.dims.ndim - layer.ndim
        self.viewer.dims.set_point(world_axis, last * layer.scale[0] + layer.translate[0])

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
        """`df` with the posterior run's per-track `_TRACK_COLOR_COLUMNS`
        broadcast onto its vertices, when there is a run. Tracks without a
        value (excluded) get NaN, which napari leaves uncolored rather than
        pinning to one end of the colormap."""
        if self._posterior_df is None:
            return df
        present = [c for c in _TRACK_COLOR_COLUMNS[1:] if c in self._posterior_df.columns]
        colors = self._posterior_df.select(
            "track_id",
            pl.col("D_median_um2_s").log10().alias("log10_D_median"),
            *present,
        )
        return df.join(colors, on="track_id", how="left").with_columns(
            pl.col(c).cast(pl.Float64).fill_null(float("nan")) for c in ["log10_D_median", *present]
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

    def _passing_track_ids(self) -> Optional[set]:
        """Tracks passing the pane's length and histogram cuts -- the
        quality filters, as opposed to its region picker, which is a view
        (every row carries its `region_class`). None when no cut is set.
        Read against the unfiltered joined table (see
        `_TracksPane.filtered_track_ids`)."""
        if self._joined_track_df is None:
            return None
        return passing_track_ids(
            self._joined_track_df,
            self._tracks_pane.min_track_length(),
            self._tracks_pane.filters.filters(),
        )

    def _summary_filter_record(self) -> dict:
        """What `passes_filters` in the saved summary means, for the JSON
        beside it."""
        return filter_record(self._tracks_pane.min_track_length(), self._tracks_pane.filters.filters())

    def tracks_summary(self) -> Optional[pl.DataFrame]:
        """The per-track summary (`diffusion.tracks_summary_table`) for the
        loaded layer and the current results, if any."""
        if self._base_track_df is None:
            return None
        result_id = self._result_dir.name if self._result_dir is not None else None
        if result_id is None and self._tracks_layer is not None:
            result_id = self._tracks_layer.name
        return tracks_summary_table(
            self._base_track_df,
            self._per_track_results(),
            result_id=result_id,
            pixel_size_um=self.pixel_size_um,
            passing_ids=self._passing_track_ids(),
        )

    def _export_tracks_csv(self) -> None:
        table = self.tracks_summary()
        if table is None:
            return
        stem = Path(TRACKS_SUMMARY_FILENAME).stem
        default_dir = self._result_dir if self._result_dir is not None else Path.home()
        default_name = f"{table['result_id'][0]}_{stem}.csv" if table.height else f"{stem}.csv"
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export per-track summary", str(default_dir / default_name), "CSV (*.csv)"
        )
        if not path:
            return
        try:
            table.write_csv(path)
        except OSError as exc:
            self._posterior_tab.report_error(f"could not write {path}: {exc}")
            return
        self._posterior_tab.report_saved(f"exported {table.height} tracks to {path}")

    def _save_results(self) -> None:
        if self._result_dir is None:
            return
        table = self.tracks_summary()
        if table is None:
            return
        analysis = self._posterior
        summary: dict = {}
        tables: dict = {}
        if analysis is not None:
            # The ensemble is over the tracks the filters pass -- the same
            # set `passes_filters` marks in the summary table -- split by
            # region class when there are several.
            ids = self.combined_filtered_track_ids()
            by_class = self.group_track_ids(ids)
            deconvolution = self._posterior_tab.deconvolution()
            tables = analysis_tables(analysis, ids, by_class, deconvolution)
            summary = analysis_summary(
                analysis,
                ids,
                by_class,
                msd_comparison=self._posterior_tab.msd_df is not None,
                deconvolution=deconvolution,
            )
        summary["tracks_summary_filters"] = self._summary_filter_record()
        summary["repo_shas"] = _analysis_repo_shas()
        write_diffusion_results(self._result_dir, tracks_summary=table, summary=summary, **tables)
        self._posterior_tab.report_saved(f"saved {table.height} tracks' analysis to {self._result_dir.name}/")
