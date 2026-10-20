"""The spatial classes' figures (`spatial`), each class in its own colour (`spatial.class_color`) in all three.

- `plot_class_selection`: the score against C, for choosing C at the elbow, with the suggested C's rule drawn.
- `plot_class_map`: where each class is: the class probabilities from the nearest tracks, blank where no track
  is near, and the tracks' centroids in their class colours. Without a spatial pattern there is no field, only
  the tracks: a map of a plain mixture would show noise as places.
- `plot_class_populations`: each class's populations as densities on one axis per parameter: D deconvolved
  from the class's tracks with its band (`diffusion.ensemble`), and alpha's population across them when an
  α · D run is there (`fbm.alpha_population`). No number summarizes a class here.
- `plot_D_map`: the smooth map of D (`spatial.analyze_map`): each track at its mapped D, and how sure the map is.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from diffusionkit.spatial import SpatialAnalysis
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from napari_gemscape2 import units
from napari_gemscape2.spatial import FIELD_REACH, class_color

_INK = "#0b0b0b"
_MUTED_INK = "#52514e"
_ALL_COLOR = "#9a9993"
# Cells across the field's longer side.
_FIELD_CELLS = 320
# How far a sure cell of the field goes from white toward its class colour: pale, so the tracks' dots in the
# full colour read on top of it.
_FIELD_STRENGTH = 0.45


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED_INK)
    ax.tick_params(colors=_MUTED_INK, labelsize=8)
    ax.grid(axis="y", color="0.92", lw=0.6)
    ax.set_axisbelow(True)


def plot_class_selection(s: SpatialAnalysis, title: Optional[str] = None) -> Figure:
    """How much each extra class adds, per track: above, the score over C = 1; below, the gain over C - 1, with
    the suggested C's threshold. The C in use is marked in both."""
    sel = s.selection
    cs = sel["n_classes"].to_numpy()
    n = len(s.track_ids)
    score = (sel["score"].to_numpy() - sel["score"][0]) / n
    gain = sel["gain_per_track"].to_numpy()
    threshold = s.options.gain_per_track
    suggested = int(sel.filter(sel["suggested"])["n_classes"].item())

    fig = Figure(figsize=(5.2, 4.6), layout="constrained")
    top, bottom = fig.subplots(2, 1, sharex=True)
    for ax in (top, bottom):
        _style(ax)
        ax.axvspan(s.n_classes - 0.4, s.n_classes + 0.4, color=class_color(1), alpha=0.08, lw=0)
    top.plot(cs, score, "o-", color=_INK, lw=1.4, ms=5)
    top.set_ylabel("score over 1 class\n(nats per track)", fontsize=8, color=_MUTED_INK)
    top.set_title("neighbour-predictive score: each track's data under its neighbours' classes",
                  fontsize=8, loc="left", color=_MUTED_INK)

    g = np.nan_to_num(gain[1:].astype(float), nan=0.0)
    colors = [_INK if v >= threshold else _ALL_COLOR for v in g]
    bottom.bar(cs[1:], g, width=0.5, color=colors)
    bottom.axhline(threshold, color=_MUTED_INK, lw=0.9, ls="--")
    bottom.text(cs[-1] + 0.45, threshold, f" {threshold:g}", va="center", ha="left", fontsize=7, color=_MUTED_INK)
    bottom.axhline(0, color=_MUTED_INK, lw=0.6)
    bottom.set_ylabel("gain over C − 1\n(nats per track)", fontsize=8, color=_MUTED_INK)
    bottom.set_xlabel("classes C", fontsize=9)
    bottom.set_xticks(cs)
    bottom.set_xlim(cs[0] - 0.6, cs[-1] + 0.9)
    bottom.set_title(f"suggested: the last C whose gain clears {threshold:g} (dashed), counting up: C = {suggested}",
                     fontsize=8, loc="left", color=_MUTED_INK)
    head = f"{title} · " if title else ""
    fig.suptitle(f"{head}choosing the number of classes · {n} tracks · in use: C = {s.n_classes}",
                 fontsize=10, x=0.01, ha="left", color=_INK)
    return fig


def class_field(s: SpatialAnalysis, extent_um: tuple[float, float, float, float], cells: int = _FIELD_CELLS
                ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """The class map as an RGBA image over `extent_um` (x0, x1, y0, y1), rows along y: each cell a pale tint of
    its most probable class, faded to white as that class's probability falls to 1/C, and transparent where its
    k-th nearest track is beyond `FIELD_REACH` typical spacings. Returns the image and its imshow extent."""
    x0, x1, y0, y1 = extent_um
    step = max(x1 - x0, y1 - y0) / cells
    xs = np.arange(x0 + step / 2, x1, step)
    ys = np.arange(y0 + step / 2, y1, step)
    gx, gy = np.meshgrid(xs, ys)
    p, dist = s.field(np.column_stack([gx.ravel(), gy.ravel()]))
    c = s.n_classes
    best = p.argmax(axis=1)
    sure = np.clip((p.max(axis=1) - 1 / c) / (1 - 1 / c), 0, 1) if c > 1 else np.ones(len(p))
    palette = np.array([to_rgb(class_color(k + 1)) for k in range(c)])
    rgb = 1 - _FIELD_STRENGTH * sure[:, None] * (1 - palette[best])
    alpha = (dist <= FIELD_REACH * s.typical_spacing_um()).astype(float)
    image = np.concatenate([rgb, alpha[:, None]], axis=1).reshape(len(ys), len(xs), 4)
    return image, (xs[0] - step / 2, xs[-1] + step / 2, ys[-1] + step / 2, ys[0] - step / 2)


def plot_class_map(
    s: SpatialAnalysis,
    shape_px: Optional[tuple[int, int]] = None,
    pixel_size_um: Optional[float] = None,
    labels: Optional[np.ndarray] = None,
    region_classes: Optional[dict[int, str]] = None,
    title: Optional[str] = None,
) -> Figure:
    """Where the classes are. `shape_px` (H, W) frames the field of view (else the tracks' extent);
    `labels`/`region_classes` outline the regions. Axes in µm (the run's positions are)."""
    xy = s.xy_um
    if shape_px is not None and pixel_size_um:
        h, w = shape_px
        extent = (-0.5 * pixel_size_um, (w - 0.5) * pixel_size_um, -0.5 * pixel_size_um, (h - 0.5) * pixel_size_um)
    else:
        pad = 0.03 * float(np.ptp(xy, axis=0).max() or 1.0)
        extent = (xy[:, 0].min() - pad, xy[:, 0].max() + pad, xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    aspect = (extent[3] - extent[2]) / max(extent[1] - extent[0], 1e-9)
    fig = Figure(figsize=(7.0, min(max(7.0 * 0.85 * aspect, 3.0), 10.0)), layout="constrained")
    ax = fig.add_subplot()

    if s.patterned:
        image, im_extent = class_field(s, extent)
        ax.imshow(image, extent=im_extent, origin="upper", interpolation="nearest", zorder=0)
    if labels is not None and region_classes and pixel_size_um:
        present = set(np.unique(labels).tolist()) - {0}
        hh, ww = labels.shape
        xs, ys = np.arange(ww) * pixel_size_um, np.arange(hh) * pixel_size_um
        for label in sorted(present & set(region_classes)):
            ax.contour(xs, ys, (labels == label).astype(float), levels=[0.5], colors=[_MUTED_INK], linewidths=0.7)

    cls = s.columns()["spatial_class"].to_numpy()
    colors = [class_color(int(k)) for k in cls]
    # Smaller dots for more tracks, so the shade between them stays readable.
    size = float(np.clip(5000 / len(xy), 1.0, 8.0))
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=colors, edgecolors=_INK, linewidths=0.1 if size < 3 else 0.25, zorder=2)

    report = s.report()
    handles = [Patch(facecolor=class_color(int(r["spatial_class"])), edgecolor=_INK, linewidth=0.3,
                     label=f"class {r['spatial_class']} · {r['n_tracks']} tracks")
               for r in report.iter_rows(named=True)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8,
              title="classes (lowest D first),\ntracks by their neighbours' class", title_fontsize=8, alignment="left")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[3], extent[2])
    ax.set_aspect("equal")
    ax.set_xlabel(f"x ({units.UM})")
    ax.set_ylabel(f"y ({units.UM})")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)

    if s.null is None:
        verdict = "one class: no map" if s.n_classes == 1 else "no pattern test run: no map"
    elif s.patterned:
        verdict = (f"spatial pattern: neighbour gain {s.null.gain:.1f} nats, p = {s.null.p:.2g}\n"
                   f"shade: the most probable class there, paler where less sure,\n"
                   f"blank beyond {FIELD_REACH:g} track spacings · dots: each track's own class")
    else:
        verdict = (f"no spatial pattern (p = {s.null.p:.2g}): the classes are a mixture with no places,\n"
                   "so no map; dots: each track's own class")
    head = f"{title} · " if title else ""
    ax.set_title(f"{head}{len(xy)} tracks, {s.n_classes} classes\n{verdict}", fontsize=8, loc="left", color=_INK)
    return fig


def _draw_density(ax, x, weights, band, color, label, lw=1.6) -> float:
    dx = np.gradient(x)
    if band is not None:
        ax.fill_between(x, band[0] / dx, band[1] / dx, color=color, alpha=0.18, lw=0)
    dens = weights / dx
    ax.plot(x, dens, color=color, lw=lw, label=label)
    return float(dens.max())


def _mass_span(weights: np.ndarray, x: np.ndarray, tail: float = 0.005) -> tuple[float, float]:
    cdf = np.cumsum(weights) / weights.sum()
    return float(np.interp(tail, cdf, x)), float(np.interp(1 - tail, cdf, x))


def plot_class_populations(
    d_grid: np.ndarray,
    d_panels: list[dict],
    alpha_panels: Optional[list[dict]] = None,
    level: float = 0.9,
    title: Optional[str] = None,
) -> Figure:
    """Each class's populations as densities, all classes on one axis per parameter.

    `d_panels`: one dict per group with `name`, `class` (its number, None for the reference group of every
    track), `n_tracks`, `deconvolved` (weights on `d_grid`) and `band`. `alpha_panels`: the same with
    `alphas`, `weights` (alpha's marginal across tracks) and `band`, or None when there is no α · D run."""
    has_alpha = bool(alpha_panels)
    fig = Figure(figsize=(6.6, 2.9 * (2 if has_alpha else 1) + 0.4), layout="constrained")
    axes = fig.subplots(2 if has_alpha else 1, 1, squeeze=False)[:, 0]

    ax = axes[0]
    _style(ax)
    x = np.log10(d_grid)
    spans, top = [], 0.0
    for p in d_panels:
        color = _ALL_COLOR if p["class"] is None else class_color(p["class"])
        lw = 1.0 if p["class"] is None else 1.8
        top = max(top, _draw_density(ax, x, p["deconvolved"], p.get("band") if p["class"] is not None else None,
                                     color, f"{p['name']} · {p['n_tracks']} tracks", lw))
        spans.append(_mass_span(p["deconvolved"], x))
    if spans:
        ax.set_xlim(min(a for a, _ in spans) - 0.25, max(b for _, b in spans) + 0.25)
    ax.set_ylim(0, top * 1.12 if top > 0 else 1)
    ax.set_xlabel(units.mpl_log_label("D_um2_s"), fontsize=9)
    ax.set_ylabel(r"density per $\log_{10} D$", fontsize=8, color=_MUTED_INK)
    ax.set_title(f"D: each class's population, deconvolved from its tracks' likelihoods, {level:.0%} band",
                 fontsize=8, loc="left", color=_MUTED_INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    if has_alpha:
        ax = axes[1]
        _style(ax)
        top = 0.0
        for p in alpha_panels:
            color = _ALL_COLOR if p["class"] is None else class_color(p["class"])
            lw = 1.0 if p["class"] is None else 1.8
            top = max(top, _draw_density(ax, p["alphas"], p["weights"], p.get("band") if p["class"] is not None else None,
                                         color, f"{p['name']} · {p['n_tracks']} tracks", lw))
        ax.axvline(1, color=_MUTED_INK, lw=0.6, ls=(0, (2, 3)))
        ax.set_xlim(0, 2)
        ax.set_ylim(0, top * 1.12 if top > 0 else 1)
        ax.set_xlabel(r"$\alpha$", fontsize=9)
        ax.set_ylabel("density across tracks", fontsize=8, color=_MUTED_INK)
        ax.set_title(f"α: each class's population, from the 2D deconvolution of its tracks' joint likelihoods (fBm), "
                     f"{level:.0%} band", fontsize=8, loc="left", color=_MUTED_INK)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    head = f"{title} · " if title else ""
    fig.suptitle(f"{head}class populations · tracks grouped by their neighbours' class",
                 fontsize=10, x=0.01, ha="left", color=_INK)
    return fig


def plot_D_map(
    m,
    shape_px: Optional[tuple[int, int]] = None,
    pixel_size_um: Optional[float] = None,
    labels: Optional[np.ndarray] = None,
    region_classes: Optional[dict[int, str]] = None,
    title: Optional[str] = None,
) -> Figure:
    """The smooth map of D (`diffusionkit.spatial.SmoothMap`): left, each track at its mapped D on a log colour
    scale; right, the map's posterior SD of ln D there. `shape_px` frames the field of view (else the tracks'
    extent); `labels`/`region_classes` outline the regions. Axes in µm."""
    xy = m.xy_um
    if shape_px is not None and pixel_size_um:
        h, w = shape_px
        extent = (-0.5 * pixel_size_um, (w - 0.5) * pixel_size_um, -0.5 * pixel_size_um, (h - 0.5) * pixel_size_um)
    else:
        pad = 0.03 * float(np.ptp(xy, axis=0).max() or 1.0)
        extent = (xy[:, 0].min() - pad, xy[:, 0].max() + pad, xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    aspect = (extent[3] - extent[2]) / max(extent[1] - extent[0], 1e-9)
    fig = Figure(figsize=(11.0, min(max(5.0 * aspect, 3.0), 8.0) + 0.9), layout="constrained")
    left, right = fig.subplots(1, 2)
    d = np.exp(m.ln_D)
    lo, hi = np.percentile(d, [1, 99])
    size = float(np.clip(5000 / len(xy), 1.5, 10.0))
    panels = [
        (left, d, dict(cmap="viridis", norm=LogNorm(lo, hi if hi > lo else lo * 1.01)),
         f"mapped D ({units.UM2_S})", "D: the map's value at each track"),
        (right, m.ln_D_sd, dict(cmap="magma_r", vmin=0), "posterior SD of ln D", "how sure the map is there"),
    ]
    for ax, values, cmap, label, sub in panels:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=size, linewidths=0, zorder=2, **cmap)
        fig.colorbar(sc, ax=ax, shrink=0.8, label=label)
        if labels is not None and region_classes and pixel_size_um:
            present = set(np.unique(labels).tolist()) - {0}
            hh, ww = labels.shape
            xs, ys = np.arange(ww) * pixel_size_um, np.arange(hh) * pixel_size_um
            for label_id in sorted(present & set(region_classes)):
                ax.contour(xs, ys, (labels == label_id).astype(float), levels=[0.5], colors=[_MUTED_INK],
                           linewidths=0.7)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[3], extent[2])
        ax.set_aspect("equal")
        ax.set_xlabel(f"x ({units.UM})")
        ax.set_ylabel(f"y ({units.UM})")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_title(sub, fontsize=8, loc="left", color=_MUTED_INK)
    if m.null is None:
        verdict = f"evidence gain over no space {m.gain:.1f} nats (no pattern test)"
    elif m.patterned:
        verdict = f"spatial pattern: evidence gain over no space {m.gain:.1f} nats, p = {m.null.p:.2g}"
    else:
        verdict = (f"no spatial pattern (gain {m.gain:.1f} nats, p = {m.null.p:.2g}): "
                   "differences between places here are noise")
    head = f"{title} · " if title else ""
    fig.suptitle(f"{head}smooth map of D · {len(xy)} tracks\n{verdict}", fontsize=10, x=0.01, ha="left",
                 color=_INK)
    return fig
