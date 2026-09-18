"""Napari Shapes-layer ROIs, persisted as polygons.

Every shape (rectangle, ellipse, polygon, line, path, ...) is captured as
a flat list of `(y, x)` polygon vertices -- once a shape is reduced to
its own boundary vertices, its `shape_type` no longer matters for
reconstructing a mask or a Shapes layer, and a polygon is both cheaper to
store (a plain vertex list, no per-shape-type fields) and, mask-for-mask,
equivalent to re-adding the original typed shape. Ellipses are the one
shape whose native `Shapes.data` is a bounding box, not its own boundary,
so they're tessellated into an n-gon approximation before storing; every
other shape type's own vertices already trace its boundary.

One ROI dict per napari Shapes layer: `{"name": <layer name>, "polygons":
[[[y, x], ...], ...]}` -- multiple layers (e.g. differently-named regions
drawn for reference vs. as the detect-step mask) round-trip as separate
named Shapes layers, not merged into one.

Several ROIs can also *label* a run rather than just bound it:
`label_image` rasterizes a list of ROI records into one integer image
(-1 outside all of them) and `label_points` stamps each detection with the
ROI it fell in (`roi` by name, `roi_index` by position in the list, the
same order `rois.json` is written in). Where ROIs overlap -- a nucleus
drawn inside a cell, say -- the one listed first wins, so callers put the
region that should win first (the widget orders them as napari's layer
list shows them, top first).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from skimage.draw import polygon as _fill_polygon

ELLIPSE_N_VERTICES = 64


def _ellipse_polygon(bbox: np.ndarray, n: int = ELLIPSE_N_VERTICES) -> np.ndarray:
    """napari's 4-corner ellipse bounding box (`Shapes.data` for an
    'ellipse' shape) -> an n-vertex polygon tracing the ellipse inscribed
    in it."""
    center = bbox.mean(axis=0)
    radii = (bbox.max(axis=0) - bbox.min(axis=0)) / 2.0
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dy = radii[0] * np.sin(theta)
    dx = radii[1] * np.cos(theta)
    return np.stack([center[0] + dy, center[1] + dx], axis=1)


def shapes_layer_to_roi(layer) -> dict:
    """A napari `Shapes` layer -> `{"name": ..., "polygons": [...]}`, one
    polygon per shape on the layer, in the layer's own display order.
    Only the trailing 2 `(y, x)` coordinates of each vertex are kept --
    ROIs in this pipeline are always a single full-frame region, not
    drawn per-frame, so any leading (non-displayed) dims a shape's `data`
    might carry aren't meaningful here."""
    polygons = []
    for data, shape_type in zip(layer.data, layer.shape_type):
        verts = np.asarray(data)[:, -2:]
        if shape_type == "ellipse":
            verts = _ellipse_polygon(verts)
        polygons.append(verts.tolist())
    return {"name": layer.name, "polygons": polygons}


def roi_to_shapes_kwargs(roi: dict) -> dict:
    """A saved ROI dict -> kwargs for `viewer.add_shapes`, reconstructing
    every polygon as a napari 'polygon' shape (see this module's docstring
    for why the original shape_type doesn't need preserving)."""
    return dict(
        data=[np.asarray(p) for p in roi["polygons"]],
        shape_type="polygon",
        name=roi["name"],
        face_color="transparent",
    )


def label_image(rois: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """`(H, W)` int16 image: the index into `rois` of the ROI covering each
    pixel, -1 where none does. Every polygon of an ROI counts as that ROI
    (a record is one Shapes layer, possibly several shapes). Overlaps go to
    the ROI earliest in `rois`, which is why it is painted last."""
    labels = np.full(shape, -1, dtype=np.int16)
    for index in range(len(rois) - 1, -1, -1):
        for poly in rois[index]["polygons"]:
            verts = np.asarray(poly, dtype=float)
            if verts.ndim != 2 or len(verts) < 3:
                continue
            rr, cc = _fill_polygon(verts[:, 0], verts[:, 1], shape=shape)
            labels[rr, cc] = index
    return labels


def overlap_pixels(rois: list[dict], shape: tuple[int, int]) -> int:
    """How many pixels more than one ROI covers -- reported next to a
    labeled run, since those pixels silently went to the first-listed
    ROI (see `label_image`)."""
    counts = np.zeros(shape, dtype=np.int16)
    for roi in rois:
        covered = np.zeros(shape, dtype=bool)
        for poly in roi["polygons"]:
            verts = np.asarray(poly, dtype=float)
            if verts.ndim != 2 or len(verts) < 3:
                continue
            rr, cc = _fill_polygon(verts[:, 0], verts[:, 1], shape=shape)
            covered[rr, cc] = True
        counts += covered
    return int((counts > 1).sum())


def label_points(points_df: pl.DataFrame, labels: np.ndarray, names: list[str]) -> pl.DataFrame:
    """`points_df` plus `roi_index` (Int16, -1 outside every ROI) and `roi`
    (the ROI's name, null outside), read off `labels` at each detection's
    rounded `(y, x)`. Any existing `roi`/`roi_index` columns are replaced,
    so re-labeling a loaded table is safe."""
    points_df = points_df.drop([c for c in ("roi", "roi_index") if c in points_df.columns])
    if points_df.height == 0:
        return points_df.with_columns(
            pl.lit(None, dtype=pl.Int16).alias("roi_index"), pl.lit(None, dtype=pl.Utf8).alias("roi")
        )
    h, w = labels.shape
    yi = np.clip(np.rint(points_df["y"].to_numpy()).astype(np.int64), 0, h - 1)
    xi = np.clip(np.rint(points_df["x"].to_numpy()).astype(np.int64), 0, w - 1)
    index = labels[yi, xi].astype(np.int16)
    name_lookup = np.array(list(names) + [None], dtype=object)
    roi = name_lookup[np.where(index >= 0, index, len(names))]
    return points_df.with_columns(
        pl.Series("roi_index", index, dtype=pl.Int16),
        pl.Series("roi", roi.tolist(), dtype=pl.Utf8),
    )

