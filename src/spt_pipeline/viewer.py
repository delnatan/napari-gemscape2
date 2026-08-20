"""Load an experiment bundle's image + points + tracks as napari layers."""

from __future__ import annotations

from pathlib import Path

from spt_pipeline.experiment import load_experiment
from spt_pipeline.pipeline import load_stack
from spt_pipeline.rois import roi_to_shapes_kwargs


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
            size=4,
            symbol="ring",
            features=features,
        )

    if tracks_df.height > 0 and "track_id" in tracks_df.columns:
        tracks = tracks_df.select("track_id", "frame", "y", "x").to_numpy()
        viewer.add_tracks(tracks, name="tracks")

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
