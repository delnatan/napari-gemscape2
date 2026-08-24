"""Load an experiment bundle's image + points + tracks as napari layers."""

from __future__ import annotations

from pathlib import Path

from spt_pipeline.experiment import load_experiment
from spt_pipeline.pipeline import load_stack, track_features_df
from spt_pipeline.rois import roi_to_shapes_kwargs

# Per-vertex properties on the "tracks" layer (see track_features_df) --
# color_by defaults to track_length so a broken/short track (a linking
# failure) stands out from a long one at a glance, instead of napari's
# default track_id coloring, which carries no quality signal at all.
TRACKS_PROPERTY_COLUMNS = ("track_length", "mean_step_um", "duration_s")
TRACKS_COLOR_BY = "track_length"

# Shared look for every "detected spot" Points layer (the final "points"
# layer here, and experiment_list.py's stepwise "points (preview)") --
# transparent face so overlapping markers don't occlude each other or the
# underlying image, magenta border since it's a hue absent from both
# viridis and gray (this app's two expected image colormaps), so markers
# stay visible regardless of which one the image layer is using.
DETECTED_POINTS_STYLE = dict(
    symbol="disc",
    size=7,
    face_color="transparent",
    border_color="magenta",
    border_width=0.15,
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

    if points_df.height > 0:
        points = points_df.select("frame", "y", "x").to_numpy()
        features = {col: points_df[col].to_numpy() for col in points_df.columns}
        viewer.add_points(
            points,
            name="points",
            features=features,
            **DETECTED_POINTS_STYLE,
        )

    if tracks_df.height > 0 and "track_id" in tracks_df.columns:
        pixel_size_um = params.get("pixel_size_um") or 1.0
        dt_s = params.get("dt_s") or 1.0
        feat_df = track_features_df(tracks_df, pixel_size_um, dt_s)
        tracks = feat_df.select("track_id", "frame", "y", "x").to_numpy()
        properties = {col: feat_df[col].to_numpy() for col in TRACKS_PROPERTY_COLUMNS}
        viewer.add_tracks(tracks, name="tracks", properties=properties, color_by=TRACKS_COLOR_BY)

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
