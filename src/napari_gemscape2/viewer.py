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
from napari_gemscape2.results import load_mask, load_result
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

# Initial contrast limits, set from the background's own statistics:
# median - IMAGE_NOISE_BELOW*sigma .. median + IMAGE_NOISE_ABOVE*sigma, with
# sigma the robust (MAD) noise. napari's default is the full min..max of
# the data, which on a 16-bit camera stack with a few hot pixels renders
# as a near-black frame. A plain percentile stretch is no better for
# sparse single molecules: spots cover well under 1% of the pixels, so
# even the 99th percentile lands inside the background noise -- the
# background renders speckled and every spot saturates. The noise-based
# window does not depend on how many spots are in the field, and hot
# pixels cannot drag it. One sigma below the median keeps the background
# dark gray rather than clipped black, so its texture stays readable.
IMAGE_NOISE_BELOW = 1.0
IMAGE_NOISE_ABOVE = 12.0

# Frames sampled when estimating those limits (and every
# IMAGE_STATS_PIXEL_STRIDE-th pixel along y and x within each): the
# median of a (2000, 2048, 2048) stack is seconds of work for a number
# that a subsample pins down just as well, and this runs on every
# selection change.
IMAGE_STATS_FRAMES = 8
IMAGE_STATS_PIXEL_STRIDE = 2

# Fallback stretch for an image without measurable noise (MAD of 0:
# clipped, binary, or synthetic data).
IMAGE_PERCENTILES = (0.1, 99.0)

# Shared look for every "detected spot" Points layer (the final "points"
# layer here, and experiment_list.py's stepwise "points (preview)") -- a
# thin hollow circle centered on the fit. Nothing covers the center, so
# the PSF peak shows through, and the eye finds a circle's center to
# well under a pixel. (napari's "cross" is a *filled* plus with arms
# size/3 wide, which covers most of the very pixel you want to see.)
#
# Size is in data pixels and the border is a fraction of it, so both
# scale with zoom. When zoomed out, vispy clamps each marker up to
# canvas_size_limits[0] screen pixels and widens its border to at least
# half that -- i.e. the ring fills in to a solid dot, which keeps every
# spot visible in a full-frame view. (It is also why a transparent
# border made markers vanish when zoomed out: the clamped border ate the
# whole marker.) Magenta since it's a hue absent from both viridis and
# gray (this app's two expected image colormaps), so markers stay
# visible regardless of which one the image layer is using.
DETECTED_POINTS_STYLE = dict(
    symbol="disc",
    size=2,
    face_color="transparent",
    border_color="magenta",
    border_width=0.08,
    border_width_is_relative=True,
    canvas_size_limits=(5, 10000),
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


def _stats_sample(image) -> np.ndarray:
    """The finite pixels of at most `IMAGE_STATS_FRAMES` evenly-spaced
    frames of a (T, Y, X) stack, strided by `IMAGE_STATS_PIXEL_STRIDE`
    along y and x -- what the contrast estimate is computed on."""
    arr = np.asarray(image)
    if arr.ndim > 2 and arr.shape[0] > IMAGE_STATS_FRAMES:
        arr = arr[:: max(1, arr.shape[0] // IMAGE_STATS_FRAMES)]
    if arr.ndim >= 2 and min(arr.shape[-2:]) >= 64:
        arr = arr[..., ::IMAGE_STATS_PIXEL_STRIDE, ::IMAGE_STATS_PIXEL_STRIDE]
    arr = arr.astype(np.float64, copy=False).ravel()
    return arr if np.all(np.isfinite(arr)) else arr[np.isfinite(arr)]


def initial_contrast_limits(image) -> tuple[float, float]:
    """`(low, high)` = median -/+ (`IMAGE_NOISE_BELOW`,
    `IMAGE_NOISE_ABOVE`) robust noise sigmas of `image`, from a subsample
    (`_stats_sample`). Falls back to the `IMAGE_PERCENTILES` stretch when
    the noise is unmeasurable (MAD of 0), and to a unit-wide window on a
    flat image, so the returned pair is always a valid (strictly
    increasing) contrast range for napari."""
    sample = _stats_sample(image)
    if sample.size == 0:
        return 0.0, 1.0
    median = float(np.median(sample))
    sigma = 1.4826 * float(np.median(np.abs(sample - median)))
    if sigma > 0:
        low = median - IMAGE_NOISE_BELOW * sigma
        high = median + IMAGE_NOISE_ABOVE * sigma
    else:
        low, high = (float(v) for v in np.percentile(sample, IMAGE_PERCENTILES))
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
    gets the noise-based default."""
    from napari.layers import Image

    layer = viewer.layers[name] if name in viewer.layers else None
    if not isinstance(layer, Image):
        return {}
    return {prop: getattr(layer, prop) for prop in IMAGE_DISPLAY_PROPERTIES}


def add_image_layer(viewer, image, name: str, **kwargs):
    """`viewer.add_image` with this app's colormap and a noise-based
    contrast stretch, so a freshly-loaded stack is readable without a trip
    to the layer controls. Any explicit kwarg (e.g. one carried over by
    `image_display_carryover`) wins over the defaults."""
    display = dict(colormap=IMAGE_COLORMAP)
    display.update(kwargs)
    if "contrast_limits" not in display:
        display["contrast_limits"] = initial_contrast_limits(image)
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
    # The mask the results were made with (`results.load_regions`) -- what
    # the rows' `region` labels refer to.
    labels: np.ndarray | None
    regions: Regions | None
    # The movie's mask as painted now (`results.load_mask`): the one shown,
    # edited, and used by the next run. Differs from the above once it has
    # been edited since the results were saved.
    mask_labels: np.ndarray | None = None
    mask_regions: Regions | None = None


def load_image_display(image_path: str | Path, channel: int = 0, z_index: int = 0) -> ImageDisplay:
    """Read a stack and its initial contrast. Touches no viewer, so it is
    safe to run in a worker thread -- the read and the contrast estimate are
    the slow part of showing an image, and doing them on the GUI thread froze
    the file list on every row change."""
    image_path = Path(image_path)
    image, metadata = load_stack(image_path, channel=channel, z_index=z_index)
    return ImageDisplay(
        path=image_path,
        image=image,
        metadata=metadata,
        contrast_limits=initial_contrast_limits(image),
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
    mask_labels, mask_regions = load_mask(bundle_dir)
    return ResultDisplay(
        Path(bundle_dir), image, points_df, tracks_df, manifest, labels, regions, mask_labels, mask_regions
    )


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
    one loaded bundle -- the movie's mask (see `napari_gemscape2.regions`)
    comes back as a Labels layer, so its regions are visible again (and
    reusable, or editable), not just the results."""
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

    if loaded.mask_labels is not None:
        add_regions_layer(viewer, loaded.mask_labels, loaded.mask_regions)
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
