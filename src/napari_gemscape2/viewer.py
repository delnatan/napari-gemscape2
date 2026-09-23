"""Load a results bundle's image + points + tracks as napari layers.

Also the single place that decides how each of those layers *looks*.
Both this module's `add_result_layers` and the interactive stepwise
path (`widgets/experiment_list.py`) build the same three kinds of layer,
and the stepwise path swaps its "(preview)" layers for final ones when a
bundle is saved -- so any styling that lives at only one of those call
sites shows up as the display changing under the user at save time. The
`add_image_layer`/`add_points_layer`/`add_tracks_layer` helpers here are
what both sides call, so a preview layer and its final counterpart are
pixel-for-pixel the same but for the name.

Layer *coordinates* are pixels throughout -- no `scale=` is set on any
layer, so points and tracks land on the image they were detected in.
The physical units live in the layers' `metadata` instead
(`layer_units_metadata`), which is what a downstream widget converts with
and, just as importantly, what tells it whether there was a real
calibration to convert by.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from napari_gemscape2.io_formats import StackMetadata
from napari_gemscape2.results import load_result
from napari_gemscape2.pipeline import load_stack, track_features_df
from napari_gemscape2.regions import LABELS_DTYPE, Regions

# color_by defaults to track_length so a broken/short track (a linking
# failure) stands out from a long one at a glance, instead of napari's
# default track_id coloring, which carries no quality signal at all.
TRACKS_COLOR_BY = "track_length"

# Not napari's default gray: a perceptually-uniform ramp separates the
# faint-spot/background range these images live in far better than
# luminance alone, and it leaves magenta (DETECTED_POINTS_STYLE below)
# free as a marker color. Swap this one constant to change every image
# layer this app adds.
IMAGE_COLORMAP = "viridis"

# Percentile stretch for the initial contrast limits. napari's default is
# the full min..max of the data, which on a 16-bit camera stack with a few
# hot pixels renders as a near-black frame the user has to hand-adjust
# every single time. The low cut is deliberately gentle (0.1, not 1) --
# for sparse single molecules the vast majority of pixels ARE background,
# so cutting a whole percent off the bottom would start clipping it.
IMAGE_PERCENTILES = (0.1, 99.0)

# Frames sampled when estimating those limits: a full percentile over a
# (2000, 2048, 2048) stack is seconds of work for a number that a handful
# of frames pins down just as well, and this runs on every selection
# change.
IMAGE_PERCENTILE_FRAMES = 8

# Shared look for every "detected spot" Points layer (the final "points"
# layer here, and experiment_list.py's stepwise "points (preview)") --
# a "+" so spots that are close together stay distinguishable at any
# zoom level, instead of overlapping discs merging into a blob. Size is
# in data pixels, so the marker keeps its scale relative to the image as
# you zoom; napari's "cross" symbol is a filled plus whose arms are a
# third of that wide, which at size 4 is thin enough to read the PSF
# through. Transparent border so only the "+" face is visible; magenta
# since it's a hue absent from both viridis and gray (this app's two
# expected image colormaps), so markers stay visible regardless of which
# one the image layer is using.
DETECTED_POINTS_STYLE = dict(
    symbol="cross",
    size=4,
    face_color="magenta",
    border_color="transparent",
)

# Display properties carried across a layer swap (see
# `image_display_carryover`) -- everything the user can reach from
# napari's layer controls for an Image layer without changing what the
# data is.
IMAGE_DISPLAY_PROPERTIES = ("colormap", "contrast_limits", "gamma")


def layer_units_metadata(
    pixel_size_um: Optional[float],
    dt_s: Optional[float],
    result_dir=None,
    exposure_s: Optional[float] = None,
) -> dict:
    """The `metadata=` dict every points/tracks layer this app adds
    carries: the two conversion factors, whether they are real, and which
    bundle the layer belongs to.

    `pixel_size_um`/`dt_s` ride along so a widget reading either layer
    (see `widgets/diffusion_panel.py`, which reads the "tracks" layer) can
    convert its pixel-space data straight to physical units and write
    results back to the right bundle -- no separate "load a result" step
    of its own.

    `units_known` is the part that matters for honesty: a layer whose
    source never recorded a pixel size still needs *some* factor for the
    conversion to run at all, and 1.0 is the only neutral choice -- but
    the resulting columns are then pixels and frames wearing `_um` and
    `_s` names. The flag is how a reader can say so out loud instead of
    presenting px²/frame as µm²/s (see
    `DiffusionAnalysisWidget._update_source_label`).

    `exposure_s` has no placeholder: None stays None ("not known"), since
    0 would be a real claim -- an instantaneous exposure -- that the
    diffusion analysis would act on (see `io_formats`).
    """
    known = _positive_or_none(pixel_size_um) is not None and _positive_or_none(dt_s) is not None
    return {
        "pixel_size_um": _positive_or_none(pixel_size_um) or 1.0,
        "dt_s": _positive_or_none(dt_s) or 1.0,
        "units_known": known,
        "exposure_s": _nonnegative_or_none(exposure_s),
        "result_dir": str(Path(result_dir).resolve()) if result_dir is not None else None,
    }


def _nonnegative_or_none(value) -> Optional[float]:
    """An exposure counts if it is finite and >= 0 -- zero is a real
    (stroboscopic) exposure, unlike a zero pixel size."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number >= 0 else None


def _positive_or_none(value) -> Optional[float]:
    """A conversion factor only counts if it is a finite positive number:
    a manifest/layer carrying 0.0 or None for one is carrying no value,
    and multiplying by it would be worse than admitting that."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def percentile_contrast_limits(image, percentiles=IMAGE_PERCENTILES) -> tuple[float, float]:
    """`(low, high)` intensity cuts at `percentiles` of `image`, sampled
    over at most `IMAGE_PERCENTILE_FRAMES` evenly-spaced frames of a
    (T, Y, X) stack. Falls back to a unit-wide window on a flat image, so
    the returned pair is always a valid (strictly increasing) contrast
    range for napari."""
    arr = np.asarray(image)
    if arr.ndim > 2 and arr.shape[0] > IMAGE_PERCENTILE_FRAMES:
        arr = arr[:: max(1, arr.shape[0] // IMAGE_PERCENTILE_FRAMES)]
    finite = arr[np.isfinite(arr)] if not np.all(np.isfinite(arr)) else arr
    if finite.size == 0:
        return 0.0, 1.0
    low, high = (float(v) for v in np.percentile(finite, percentiles))
    if not high > low:
        high = low + 1.0
    return low, high


def image_display_carryover(viewer, name: str) -> dict:
    """The display settings of the Image layer called `name`, if the
    viewer currently has one, as kwargs for `add_image_layer`.

    Reloading a bundle re-adds the same image under the same name (a save
    on the row that's already displayed), and
    re-deriving the contrast there would throw away a stretch the user
    hand-tuned -- for no reason, since it's the same pixels. Empty dict
    when there's no such layer, i.e. a genuinely new image, which then
    gets the percentile default."""
    from napari.layers import Image

    layer = viewer.layers[name] if name in viewer.layers else None
    if not isinstance(layer, Image):
        return {}
    return {prop: getattr(layer, prop) for prop in IMAGE_DISPLAY_PROPERTIES}


def add_image_layer(viewer, image, name: str, **kwargs):
    """`viewer.add_image` with this app's colormap and a percentile
    contrast stretch, so a freshly-loaded stack is readable without a trip
    to the layer controls. Any explicit kwarg (e.g. one carried over by
    `image_display_carryover`) wins over the defaults."""
    display = dict(colormap=IMAGE_COLORMAP)
    display.update(kwargs)
    if "contrast_limits" not in display:
        display["contrast_limits"] = percentile_contrast_limits(image)
    return viewer.add_image(image, name=name, **display)


def add_points_layer(viewer, points_df, name: str, metadata: dict | None = None):
    """Add a detections Points layer named `name` from a localization
    table, replacing any existing layer of that name. Every column rides
    along in `features`, so napari's status bar shows a hovered spot's
    full fit record (`flux`, `fit_sigma`, `se_*`, ...), not just its
    position. Returns None (and leaves no layer behind) for an empty
    table."""
    if name in viewer.layers:
        del viewer.layers[name]
    if points_df is None or points_df.height == 0:
        return None
    return viewer.add_points(
        points_df.select("frame", "y", "x").to_numpy(),
        name=name,
        features={col: points_df[col].to_numpy() for col in points_df.columns},
        metadata=dict(metadata or {}),
        **DETECTED_POINTS_STYLE,
    )


def add_tracks_layer(
    viewer, tracks_df, pixel_size_um: float, dt_s: float, name: str, metadata: dict | None = None
):
    """Add a Tracks layer named `name` from a linked table, replacing any
    existing layer of that name.

    Every `track_features_df` column rides along as a per-vertex property,
    not just the derived track_length/mean_step_um/duration_s -- this is
    what lets widgets/diffusion_panel.py read per-point QC fields (flux,
    se_y/se_x, bg, fit_sigma, ...) straight off this one layer, already
    aligned with track_id, instead of a separate "points" layer lookup
    (which has no track_id -- it's the pre-linking detections table, see
    results.py)."""
    if name in viewer.layers:
        del viewer.layers[name]
    if tracks_df is None or tracks_df.height == 0 or "track_id" not in tracks_df.columns:
        return None
    data, properties = _tracks_layer_arrays(track_features_df(tracks_df, pixel_size_um, dt_s))
    return viewer.add_tracks(
        data,
        name=name,
        properties=properties,
        color_by=TRACKS_COLOR_BY,
        metadata=dict(metadata or {}),
    )


# The Tracks layer's own `data` columns; every other numeric column on a
# track table rides along as a per-vertex property. Non-numeric ones (the
# `region_class` name, see `napari_gemscape2.regions.label_points`) stay off:
# a Tracks layer's properties feed its colormaps, and the region is carried
# there as its `region` label instead, named through the layer's
# `region_classes` metadata ({label: class}).
_TRACK_DATA_COLUMNS = ("track_id", "frame", "y", "x")


def _tracks_layer_arrays(feat_df: pl.DataFrame) -> tuple[np.ndarray, dict]:
    data = feat_df.select(*_TRACK_DATA_COLUMNS).to_numpy()
    properties = {
        col: feat_df[col].to_numpy()
        for col, dtype in zip(feat_df.columns, feat_df.dtypes)
        if col not in _TRACK_DATA_COLUMNS and (dtype.is_numeric() or dtype == pl.Boolean)
    }
    return data, properties


def set_tracks_layer_data(layer, feat_df: pl.DataFrame) -> None:
    """Replace an existing Tracks layer's vertices and properties in place.

    In place rather than delete-and-re-add, so that redrawing it (a filter
    handle being dragged, a re-run link step) keeps the layer's identity,
    its position in the layer list and whatever the user set in its layer
    controls -- and so a widget holding on to the layer (the diffusion
    panel) sees the same layer change, rather than one layer vanish and a
    stranger appear.

    `feat_df` carries `track_id`/`frame`/`y`/`x` plus any property columns
    (e.g. `track_features_df`'s output). `Tracks.data`'s setter wipes the
    features table, which drops `color_by` back to `track_id`, so the
    coloring in force beforehand is put back once the properties are."""
    previous_color_by = layer.color_by
    data, properties = _tracks_layer_arrays(feat_df)
    with warnings.catch_warnings():
        # The setter's own "color_by not present, falling back" warning is
        # exactly the reset this function undoes two lines later.
        warnings.filterwarnings("ignore", message="Previous color_by key", category=UserWarning)
        layer.data = data
    layer.properties = properties
    if previous_color_by in layer.properties_to_color_by:
        layer.color_by = previous_color_by


def set_points_layer_data(layer, points_df: pl.DataFrame) -> None:
    """Replace an existing detections Points layer's positions and
    features in place -- the `add_points_layer` counterpart of
    `set_tracks_layer_data`, for the same reasons. Every point is shown
    again afterwards; callers that filter set `layer.shown` next."""
    layer.data = points_df.select("frame", "y", "x").to_numpy()
    layer.features = {col: points_df[col].to_numpy() for col in points_df.columns}
    layer.shown = True


@dataclass
class ImageDisplay:
    """One image stack, loaded and ready to become a layer: everything
    `show_image` needs that is slow to compute, gathered off the GUI thread
    (`load_image_display`)."""

    path: Path
    image: np.ndarray
    metadata: StackMetadata
    contrast_limits: tuple[float, float]

    @property
    def pixel_size_um(self) -> Optional[float]:
        return self.metadata.pixel_size_um

    @property
    def dt_s(self) -> Optional[float]:
        return self.metadata.dt_s


@dataclass
class ResultDisplay:
    """One saved bundle plus its source image, loaded and ready to show."""

    bundle_dir: Path
    image: ImageDisplay
    points_df: pl.DataFrame
    tracks_df: pl.DataFrame
    manifest: dict
    labels: np.ndarray | None
    regions: Regions | None


def load_image_display(image_path: str | Path, channel: int = 0, z_index: int = 0) -> ImageDisplay:
    """Read a stack and its initial contrast. Touches no viewer, so it is
    safe to run in a worker thread -- the read and the percentile are the
    slow part of showing an image, and doing them on the GUI thread froze
    the file list on every row change."""
    image_path = Path(image_path)
    image, metadata = load_stack(image_path, channel=channel, z_index=z_index)
    return ImageDisplay(
        path=image_path,
        image=image,
        metadata=metadata,
        contrast_limits=percentile_contrast_limits(image),
    )


def load_result_display(bundle_dir: str | Path) -> ResultDisplay:
    """Read a bundle's tables and its source image. Thread-safe, like
    `load_image_display`."""
    points_df, tracks_df, manifest, labels, regions = load_result(bundle_dir)
    params = manifest.get("params", {})
    image = load_image_display(
        manifest["source_image_path"],
        channel=params.get("channel", 0),
        z_index=params.get("z_index", 0),
    )
    return ResultDisplay(Path(bundle_dir), image, points_df, tracks_df, manifest, labels, regions)


def show_image(viewer, loaded: ImageDisplay):
    """Clear `viewer` and show one image. If this same image is already on
    screen, its (possibly hand-tuned) display settings carry over instead
    of resetting -- read before the clear, since the clear is what removes
    them."""
    name = loaded.path.stem
    display = dict(contrast_limits=loaded.contrast_limits)
    display.update(image_display_carryover(viewer, name))
    viewer.layers.clear()
    return add_image_layer(viewer, loaded.image, name, **display)


def add_result_layers(viewer, result_dir: str | Path) -> None:
    """Load a bundle and show it (`load_result_display` + `show_result`),
    blocking -- for the standalone viewer, where there is no UI to keep
    responsive."""
    show_result(viewer, load_result_display(result_dir))


def region_classes(regions: Regions | None) -> dict[int, str]:
    """`{label: class}` -- what a layer's `region_classes` metadata holds."""
    return {label: r.class_ for label, r in regions.table.items()} if regions else {}


REGIONS_LAYER_NAME = "regions"


def add_regions_layer(viewer, labels: np.ndarray, regions: Regions | None = None, name: str = REGIONS_LAYER_NAME):
    """A 2D Labels layer holding a regions image (see
    `napari_gemscape2.regions`), its `Regions` table kept on the layer's
    metadata where `widgets.regions_panel` reads it. 2D, fewer dims than
    the image stack, so it shows on every frame."""
    return viewer.add_labels(
        np.asarray(labels, dtype=LABELS_DTYPE),
        name=name,
        opacity=0.3,
        metadata={"regions": regions if regions is not None else Regions()},
    )


def show_result(viewer, loaded: ResultDisplay) -> None:
    """Clear `viewer` and add the image/points/tracks/regions layers for
    one loaded bundle -- saved regions (see `napari_gemscape2.regions`) come
    back as a Labels layer, so the regions used for detection are visible
    again (and reusable, or editable), not just the results."""
    show_image(viewer, loaded.image)
    result_dir = loaded.bundle_dir
    points_df, tracks_df = loaded.points_df, loaded.tracks_df
    params = loaded.manifest.get("params", {})

    layer_metadata = layer_units_metadata(
        params.get("pixel_size_um"), params.get("dt_s"), result_dir, params.get("exposure_s")
    )
    pixel_size_um = layer_metadata["pixel_size_um"]
    dt_s = layer_metadata["dt_s"]
    # What each `region` label on the rows is (`regions.label_points`).
    layer_metadata["region_classes"] = region_classes(loaded.regions)

    if loaded.labels is not None:
        add_regions_layer(viewer, loaded.labels, loaded.regions)
    add_points_layer(viewer, points_df, "points", layer_metadata)
    add_tracks_layer(viewer, tracks_df, pixel_size_um, dt_s, "tracks", layer_metadata)


def launch_viewer(result_dir: str | Path):
    """Standalone entry point (`gemscape2 view <dir>`) -- opens a fresh napari
    window with one bundle's layers loaded, blocking until it's closed."""
    import napari

    viewer = napari.Viewer()
    add_result_layers(viewer, result_dir)
    napari.run()
    return viewer
