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
"""

from __future__ import annotations

import numpy as np

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
