"""Load an experiment bundle's image + points + tracks as napari layers.

Also the single place that decides how each of those layers *looks*.
Both this module's `add_experiment_layers` and the interactive stepwise
path (`widgets/experiment_list.py`) build the same three kinds of layer,
and the stepwise path swaps its "(preview)" layers for final ones when a
bundle is saved -- so any styling that lives at only one of those call
sites shows up as the display changing under the user at save time. The
`add_image_layer`/`add_points_layer`/`add_tracks_layer` helpers here are
what both sides call, so a preview layer and its final counterpart are
pixel-for-pixel the same but for the name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spt_pipeline.experiment import load_experiment
from spt_pipeline.pipeline import load_stack, track_features_df
from spt_pipeline.rois import roi_to_shapes_kwargs

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
    or a batch run finishing on the row that's already displayed), and
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
    display = dict(colormap=IMAGE_COLORMAP, contrast_limits=percentile_contrast_limits(image))
    display.update(kwargs)
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
    experiment.py)."""
    if name in viewer.layers:
        del viewer.layers[name]
    if tracks_df is None or tracks_df.height == 0 or "track_id" not in tracks_df.columns:
        return None
    feat_df = track_features_df(tracks_df, pixel_size_um, dt_s)
    properties = {
        col: feat_df[col].to_numpy()
        for col in feat_df.columns
        if col not in ("track_id", "frame", "y", "x")
    }
    return viewer.add_tracks(
        feat_df.select("track_id", "frame", "y", "x").to_numpy(),
        name=name,
        properties=properties,
        color_by=TRACKS_COLOR_BY,
        metadata=dict(metadata or {}),
    )


def add_experiment_layers(viewer, experiment_dir: str | Path) -> None:
    """Clear `viewer` and add the image/points/tracks/ROI layers for one
    bundle -- any saved ROI (see `spt_pipeline.rois`) is added back as a
    `"polygon"`-type Shapes layer under its original napari layer name, so
    the region used for detection is visible again, not just the results."""
    points_df, tracks_df, manifest, rois = load_experiment(experiment_dir)
    image_path = Path(manifest["source_image_path"])
    params = manifest.get("params", {})
    image, _, _ = load_stack(
        image_path, channel=params.get("channel", 0), z_index=params.get("z_index", 0)
    )

    # Read before the clear: if this same image is already on screen, its
    # (possibly hand-tuned) contrast comes with it instead of resetting.
    carryover = image_display_carryover(viewer, image_path.stem)
    viewer.layers.clear()
    add_image_layer(viewer, image, image_path.stem, **carryover)

    pixel_size_um = params.get("pixel_size_um") or 1.0
    dt_s = params.get("dt_s") or 1.0
    # pixel_size_um/dt_s/experiment_dir ride along as layer metadata on both
    # the "points" and "tracks" layers so a widget reading either one (see
    # widgets/diffusion_panel.py, which reads the "tracks" layer) can
    # convert its data straight to physical units and write results back
    # to the right bundle -- no separate "load an experiment" step of its
    # own.
    layer_metadata = {
        "pixel_size_um": pixel_size_um,
        "dt_s": dt_s,
        "experiment_dir": str(Path(experiment_dir).resolve()),
    }

    add_points_layer(viewer, points_df, "points", layer_metadata)
    add_tracks_layer(viewer, tracks_df, pixel_size_um, dt_s, "tracks", layer_metadata)

    for roi in rois:
        viewer.add_shapes(**roi_to_shapes_kwargs(roi), edge_color="yellow")


def launch_viewer(experiment_dir: str | Path):
    """Standalone entry point (`spt view <dir>`) -- opens a fresh napari
    window with one bundle's layers loaded, blocking until it's closed."""
    import napari

    viewer = napari.Viewer()
    add_experiment_layers(viewer, experiment_dir)
    napari.run()
    return viewer
