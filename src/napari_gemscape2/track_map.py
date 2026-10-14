"""The spatial map as a matplotlib figure: every track drawn as a line in
its per-track value's color, over the regions it was linked in.

The Map tab's napari layer marks each localization with a dot, since a
Tracks layer can't be colored by an arbitrary value with a chosen range.
Dots sized for a zoomed-in view vanish or clump when zoomed out, and the
canvas has no colorbar to save with it. Here a track is one polyline (a
`LineCollection`, so a PDF/SVG keeps it vector), colored with the tab's
colormap and range, with a labelled colorbar; the figure window's toolbar
saves it.

The regions are the mask the tracks were made with (`results.load_regions`),
not whatever is painted now: they are what each track's `region_class`
refers to. Each class is a translucent fill with an outline, in a hue
outside viridis's range so a region never reads as a value.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from napari_gemscape2 import units

MAP_COLORMAP = "viridis"
# Region classes take these in label order, never cycled: orange, gray,
# magenta, red -- none of them on viridis's purple-blue-green-yellow path,
# and the first two apart even as faint fills. A fifth class onward is a
# lighter gray.
_REGION_COLORS = ("#eb6834", "#7a7a7a", "#e87ba4", "#e34948")
_REGION_OTHER = "#b0b0b0"
_REGION_FILL_ALPHA = 0.18
_TRACK_LINEWIDTH = 0.8


def track_segments(tracks_px: pl.DataFrame, track_ids: np.ndarray) -> list[np.ndarray]:
    """Each of `track_ids`' vertices in frame order, as an `(n, 2)` array of
    `(x, y)` pixels -- the order `track_ids` gives."""
    vertices = (
        tracks_px.select("track_id", "frame", "x", "y")
        .filter(pl.col("track_id").is_in(track_ids.tolist()))
        .sort("track_id", "frame")
    )
    ids = vertices["track_id"].to_numpy()
    xy = vertices.select("x", "y").to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    by_id = dict(zip(ids[starts].tolist(), np.split(xy, starts[1:])))
    return [by_id[i] for i in track_ids.tolist() if i in by_id]


def plot_track_map(
    tracks_px: pl.DataFrame,
    values: pl.DataFrame,
    column: str,
    vmin: float,
    vmax: float,
    log: bool = False,
    pixel_size_um: Optional[float] = None,
    labels: Optional[np.ndarray] = None,
    region_classes: Optional[dict[int, str]] = None,
    shape: Optional[tuple[int, int]] = None,
    title: Optional[str] = None,
) -> Figure:
    """`tracks_px` holds every vertex (`track_id`, `frame`, `y`, `x` in
    pixels); `values` one row per track to draw (`track_id`, `column`).
    Values outside `vmin`..`vmax` take the colormap's ends, and the colorbar
    says so with an arrow. `log` maps colors on a log scale (needs
    `vmin` > 0). Axes are in µm with `pixel_size_um`, else pixels.

    `labels` (a regions image) and `region_classes` (`{label: class}`)
    draw the regions; `shape` (H, W) frames the field of view, and falls
    back to `labels`' shape, then to the tracks' extent."""
    values = values.select("track_id", column).drop_nulls()
    track_ids = values["track_id"].to_numpy()
    segments = track_segments(tracks_px, track_ids)
    present = set(tracks_px["track_id"].unique().to_list())
    track_values = values.filter(pl.col("track_id").is_in(list(present)))[column].to_numpy().astype(float)

    scale = pixel_size_um or 1.0
    axis_unit = units.UM if pixel_size_um else units.PX
    segments = [segment * scale for segment in segments]
    if shape is None and labels is not None:
        shape = labels.shape

    norm = LogNorm(vmin, vmax) if log and vmin > 0 else Normalize(vmin, vmax)
    if shape is not None:
        height, width = shape
        aspect = height / width
    else:
        aspect = 1.0
    fig_width = 7.0
    figure = Figure(figsize=(fig_width, min(max(fig_width * 0.85 * aspect, 3.0), 10.0)), layout="constrained")
    ax = figure.add_subplot()

    if labels is not None and region_classes:
        handles = _draw_regions(ax, labels, region_classes, scale)
        if handles:
            ax.legend(
                handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9, title="regions",
                title_fontsize=9, alignment="left",
            )

    lines = LineCollection(
        segments, array=track_values, cmap=MAP_COLORMAP, norm=norm,
        linewidths=_TRACK_LINEWIDTH, capstyle="round", joinstyle="round",
    )
    ax.add_collection(lines)

    if shape is not None:
        ax.set_xlim(-0.5 * scale, (width - 0.5) * scale)
        ax.set_ylim((height - 0.5) * scale, -0.5 * scale)
    elif segments:
        xy = np.concatenate(segments)
        pad = 0.03 * float(np.ptp(xy, axis=0).max() or 1.0)
        ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
        ax.set_ylim(xy[:, 1].max() + pad, xy[:, 1].min() - pad)
    ax.set_aspect("equal")
    ax.set_xlabel(f"x ({axis_unit})")
    ax.set_ylabel(f"y ({axis_unit})")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)

    below = track_values.size and track_values.min() < vmin
    above = track_values.size and track_values.max() > vmax
    extend = {(True, True): "both", (True, False): "min", (False, True): "max"}.get((bool(below), bool(above)), "neither")
    colorbar = figure.colorbar(lines, ax=ax, extend=extend, shrink=0.8, pad=0.02, location="bottom", aspect=40)
    colorbar.set_label(units.mpl_label(column), fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    count = f"{len(segments)} track{'s' if len(segments) != 1 else ''}"
    ax.set_title(f"{title} · {count}" if title else count, fontsize=10, loc="left")
    return figure


def _draw_regions(ax, labels: np.ndarray, region_classes: dict[int, str], scale: float) -> list[Patch]:
    """Fill each region class translucently and outline it; return its
    legend entries. One RGBA image for the fills, so overlapping
    transparency never darkens a shared edge."""
    height, width = labels.shape
    present = set(np.unique(labels).tolist()) - {0}
    by_class: dict[str, list[int]] = {}
    for label, name in sorted(region_classes.items()):
        if label in present:
            by_class.setdefault(name, []).append(label)
    if not by_class:
        return []
    fill = np.zeros((height, width, 4))
    xs = np.arange(width) * scale
    ys = np.arange(height) * scale
    handles = []
    for index, (name, members) in enumerate(by_class.items()):
        color = _REGION_COLORS[index] if index < len(_REGION_COLORS) else _REGION_OTHER
        mask = np.isin(labels, members)
        fill[mask] = to_rgba(color, _REGION_FILL_ALPHA)
        ax.contour(xs, ys, mask.astype(float), levels=[0.5], colors=[color], linewidths=0.8)
        handles.append(Patch(facecolor=to_rgba(color, _REGION_FILL_ALPHA), edgecolor=color, linewidth=0.8, label=name))
    extent = (-0.5 * scale, (width - 0.5) * scale, (height - 0.5) * scale, -0.5 * scale)
    ax.imshow(fill, extent=extent, origin="upper", interpolation="nearest", zorder=0)
    return handles
