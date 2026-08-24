"""Load an experiment bundle's image + points + tracks as napari layers."""

from __future__ import annotations

from pathlib import Path

from spt_pipeline.experiment import load_experiment
from spt_pipeline.pipeline import load_stack, track_features_df
from spt_pipeline.rois import roi_to_shapes_kwargs

# color_by defaults to track_length so a broken/short track (a linking
# failure) stands out from a long one at a glance, instead of napari's
# default track_id coloring, which carries no quality signal at all.
TRACKS_COLOR_BY = "track_length"

# Shared look for every "detected spot" Points layer (the final "points"
# layer here, and experiment_list.py's stepwise "points (preview)") --
# a small "+" so spots that are close together stay distinguishable at any
# zoom level, instead of overlapping discs merging into a blob. Transparent
# border so only the "+" face is visible; magenta since it's a hue absent
# from both viridis and gray (this app's two expected image colormaps), so
# markers stay visible regardless of which one the image layer is using.
DETECTED_POINTS_STYLE = dict(
    symbol="cross",
    size=1.5,
    face_color="magenta",
    border_color="transparent",
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

    viewer.layers.clear()
    viewer.add_image(image, name=image_path.stem)

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

    if points_df.height > 0:
        points = points_df.select("frame", "y", "x").to_numpy()
        features = {col: points_df[col].to_numpy() for col in points_df.columns}
        viewer.add_points(
            points,
            name="points",
            features=features,
            metadata=dict(layer_metadata),
            **DETECTED_POINTS_STYLE,
        )

    if tracks_df.height > 0 and "track_id" in tracks_df.columns:
        feat_df = track_features_df(tracks_df, pixel_size_um, dt_s)
        tracks = feat_df.select("track_id", "frame", "y", "x").to_numpy()
        # Every feat_df column rides along as a per-vertex property, not
        # just the derived track_length/mean_step_um/duration_s -- this is
        # what lets widgets/diffusion_panel.py read per-point QC fields
        # (amplitude, sigma_x/y, bg, ...) straight off this one layer,
        # already aligned with track_id, instead of a separate "points"
        # layer lookup (which has no track_id -- it's the pre-linking
        # detections table, see experiment.py).
        properties = {
            col: feat_df[col].to_numpy() for col in feat_df.columns if col not in ("track_id", "frame", "y", "x")
        }
        viewer.add_tracks(
            tracks,
            name="tracks",
            properties=properties,
            color_by=TRACKS_COLOR_BY,
            metadata=dict(layer_metadata),
        )

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
