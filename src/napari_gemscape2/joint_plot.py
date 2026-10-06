"""Generic 2D joint-distribution plot for any pair of per-track
properties. diffusionkit ships this exact diagnostic for D vs alpha
(`bayes.plot_K_joint`), but hardcodes a log10 transform on the D
column and always draws an alpha=1 Brownian reference line -- both specific
to that one comparison. This generalizes it to arbitrary columns/labels/log
scaling so any two trajectory properties (classical vs Bayesian D, alpha,
r2, track length, ...) can be compared the same way.

Few tracks are drawn as points over a KDE; many (`DENSITY_MIN_POINTS` or
more, with `style="auto"`) as a 2D histogram, because stacked points
saturate: at opacity 0.6, three overlapping dots are already 94% opaque,
so the dense core -- the part worth reading -- goes uniformly dark. Bins
holding fewer than `SPARSE_MAX_COUNT` tracks are left empty and their tracks
drawn as points instead, so outliers stay visible one track at a time.

The posterior figures (`plot_d_ensemble`, `plot_d_posteriors`,
`plot_d_by_length`, `plot_track_posterior`) follow one color assignment by
role, not by series: per-track medians are a neutral histogram, the
deconvolved distribution is blue, the shared-value posterior is orange, the
mean of the tracks' posteriors is aqua, and the per-track heat map is one
blue ramp. Track-length groups are ordered, so they are steps of one ramp,
light for short tracks and dark for long ones. Every D axis carries the localization floor -- the D at which a track's
motion per frame equals its localization noise -- as a dotted line at the
tracks' median floor over a band of their 10-90% range: a scale to read D
against, not a cut. Region classes are separate panels stacked on one
shared x axis rather than more hues, so every panel reads the same way.

Axis labels default to `napari_gemscape2.units.mpl_label(column)` rather than
to the raw column name: since the axes here are picked at runtime from
whatever the tracks pane holds, a plot could otherwise put
`mean_step_um` (µm) against `se_x_max` (px) with nothing on
either axis saying which is which. The labels come out in the same
mathtext style as diffusionkit's own figures -- "$D$ ($\\mu$m$^2$/s)" --
so a joint plot and an MSD plot from the same session look like they came
from one program.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.figure import Figure

from napari_gemscape2 import units


# `style="auto"` switches from points to the binned density at this many
# tracks; bins with fewer than `SPARSE_MAX_COUNT` tracks are drawn as points.
DENSITY_MIN_POINTS = 1000
SPARSE_MAX_COUNT = 3
JOINT_STYLES = ("auto", "points", "density")
# The density's bin count per axis: Freedman-Diaconis, but at most
# sqrt(n)/2 (a 2D grid spreads n tracks over the square of it), held to
# this range. Its colors go on a log scale when the fullest bin holds this
# many times the median occupied bin.
_BINS_RANGE = (15, 60)
_LOG_COLOR_RATIO = 10


def plot_property_joint(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    display_quantiles: tuple[float, float] = (0.01, 0.99),
    style: str = "auto",
    references: dict[str, tuple[float, float, float]] | None = None,
) -> Figure:
    """`style` is "points" (scatter over a KDE), "density" (2D histogram,
    sparse bins as points), or "auto": density from `DENSITY_MIN_POINTS`
    tracks up. `references` maps an axis column to a (low, mid, high)
    band drawn across the plot at that axis's value -- the localization
    floor on a D axis (`diffusion.floor_references`)."""
    if style not in JOINT_STYLES:
        raise ValueError(f"style must be one of {JOINT_STYLES}, not {style!r}")
    sub = df.select(x_col, y_col).drop_nulls()
    if log_x:
        sub = sub.filter(pl.col(x_col) > 0)
    if log_y:
        sub = sub.filter(pl.col(y_col) > 0)
    x = sub[x_col].to_numpy().astype(float)
    y = sub[y_col].to_numpy().astype(float)
    if log_x:
        x = np.log10(x)
    if log_y:
        y = np.log10(y)

    x_lo, x_hi = np.quantile(x, display_quantiles)
    y_lo, y_hi = np.quantile(y, display_quantiles)
    in_view = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
    n_dropped = int((~in_view).sum())
    pdf = pd.DataFrame({"x": x[in_view], "y": y[in_view]})

    r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else float("nan")
    density = style == "density" or (style == "auto" and len(pdf) >= DENSITY_MIN_POINTS)

    g = sns.JointGrid(data=pdf, x="x", y="y", height=6, ratio=4)
    if density:
        x_edges = _bin_edges(pdf["x"].to_numpy(), integer=not log_x and _is_integer(x))
        y_edges = _bin_edges(pdf["y"].to_numpy(), integer=not log_y and _is_integer(y))
        _draw_density(g, pdf["x"].to_numpy(), pdf["y"].to_numpy(), x_edges, y_edges)
        x_bins, y_bins = x_edges, y_edges
    else:
        sns.kdeplot(
            data=pdf, x="x", y="y", ax=g.ax_joint,
            fill=True, cmap="Blues", alpha=0.6, thresh=0.05, levels=12, zorder=0,
        )
        sns.scatterplot(
            data=pdf, x="x", y="y", ax=g.ax_joint,
            s=18, alpha=0.6, color="0.15", edgecolor="none", zorder=1,
        )
        x_bins = y_bins = 30
    g.ax_marg_x.hist(pdf["x"], bins=x_bins, color="steelblue", edgecolor="white")
    g.ax_marg_y.hist(pdf["y"], bins=y_bins, color="steelblue", edgecolor="white", orientation="horizontal")
    for col, log, vertical in ((x_col, log_x, True), (y_col, log_y, False)):
        band = (references or {}).get(col)
        if band is not None:
            band = tuple(np.log10(band)) if log else band
            for ax in (g.ax_joint, g.ax_marg_x if vertical else g.ax_marg_y):
                _draw_floor(ax, band, vertical=vertical)
            if vertical:
                g.ax_joint.text(band[1], 0.99, " localization floor", transform=g.ax_joint.get_xaxis_transform(),
                                ha="left", va="top", fontsize=7, color=_MUTED_INK, rotation=90)
            else:
                g.ax_joint.text(0.99, band[1], "localization floor", transform=g.ax_joint.get_yaxis_transform(),
                                ha="right", va="bottom", fontsize=7, color=_MUTED_INK)

    # A log axis keeps its quantity's unit -- it is the unit the log was
    # taken of, and dropping it (the old "log10(D_map_um2_s)") leaves the
    # reader to guess whether a -1 is a small D or a large one in other
    # units. `units.mpl_log_label` writes it the way diffusionkit's own
    # log-D axes do.
    xlabel = x_label or (units.mpl_log_label(x_col) if log_x else units.mpl_label(x_col))
    ylabel = y_label or (units.mpl_log_label(y_col) if log_y else units.mpl_label(y_col))
    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel(ylabel)

    subtitle = f"{title or f'{ylabel} vs {xlabel}'} (r={r:.2f}, n={len(x)}"
    if n_dropped:
        pct = int(100 * (display_quantiles[1] - display_quantiles[0]))
        subtitle += f", {n_dropped} outside {pct}% display range"
    subtitle += ")"
    g.ax_marg_x.set_title(subtitle, fontsize=10, loc="left", wrap=True)
    # Laid out again for the title: JointGrid's own layout ran before it
    # existed, and a long one otherwise runs off the top of the figure.
    g.figure.tight_layout()
    return g.figure


def _is_integer(v: np.ndarray) -> bool:
    return bool(np.all(v == np.round(v)))


def _bin_edges(v: np.ndarray, *, integer: bool) -> np.ndarray:
    """Freedman-Diaconis edges, at most sqrt(n)/2 and held to
    `_BINS_RANGE` bins. An integer
    column (track length) gets edges between its integers, whole numbers
    per bin, so no bin catches one more integer than its neighbours and
    stripes the histogram."""
    lo, hi = float(v.min()), float(v.max())
    if integer:
        step = max(1, int(np.ceil((hi - lo + 1) / _BINS_RANGE[1])))
        return np.arange(lo - 0.5, hi + 0.5 + step, step)
    if hi == lo:
        return np.array([lo - 0.5, lo + 0.5])
    n = min(len(np.histogram_bin_edges(v, bins="fd")) - 1, np.sqrt(len(v)) / 2)
    return np.linspace(lo, hi, int(np.clip(n, *_BINS_RANGE)) + 1)


def _draw_density(g, x: np.ndarray, y: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> None:
    """The 2D histogram on `g.ax_joint`, bins below `SPARSE_MAX_COUNT` left
    empty and their tracks drawn as points, and its color bar in the
    grid's empty top-right corner."""
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    ix = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, len(x_edges) - 2)
    iy = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, len(y_edges) - 2)
    sparse = counts[ix, iy] < SPARSE_MAX_COUNT
    shown = np.ma.masked_less(counts, SPARSE_MAX_COUNT)
    ax = g.ax_joint
    if shown.count():
        occupied = shown.compressed()
        norm = (
            LogNorm(vmin=SPARSE_MAX_COUNT, vmax=occupied.max())
            if occupied.max() >= _LOG_COLOR_RATIO * np.median(occupied)
            else Normalize(vmin=SPARSE_MAX_COUNT, vmax=occupied.max())
        )
        mesh = ax.pcolormesh(x_edges, y_edges, shown.T, cmap=_DENSITY_CMAP, norm=norm, rasterized=True, zorder=0)
        corner = g.figure.add_subplot(ax.get_subplotspec().get_gridspec()[0, -1])
        corner.set_axis_off()
        cax = corner.inset_axes([0.1, 0.4, 0.8, 0.14])
        bar = g.figure.colorbar(mesh, cax=cax, orientation="horizontal")
        bar.ax.tick_params(labelsize=7, length=2)
        bar.outline.set_visible(False)
        cax.set_title("tracks per bin", fontsize=8, color=_MUTED_INK)
    ax.scatter(x[sparse], y[sparse], s=8, color=_INK, edgecolor="none", zorder=1)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])


def numeric_columns(df: pl.DataFrame, exclude: tuple[str, ...] = ("track_id",)) -> list[str]:
    return [c for c, dtype in zip(df.columns, df.dtypes) if c not in exclude and dtype.is_numeric()]


# Roles, from the reference categorical palette's first two slots plus a
# neutral; text stays in ink colors, never a series color.
_HIST_COLOR = "#c9c8c2"
_DECONVOLVED_COLOR = "#2a78d6"
_SHARED_COLOR = "#eb6834"
_MEAN_POSTERIOR_COLOR = "#1baf7a"
# One hue, light to dark, its zero end receding into the figure's surface:
# the per-track heat map is magnitude, not identity.
_POSTERIOR_CMAP = LinearSegmentedColormap.from_list(
    "posterior", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
)
# Track-length groups, short to long: the same blue ramp.
_LENGTH_CMAP = LinearSegmentedColormap.from_list(
    "length", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
)
# The joint plot's density: the same ramp without its surface-colored end,
# so the emptiest drawn bin still reads against the empty ones.
_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "density", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
)
_INK = "#0b0b0b"
_MUTED_INK = "#52514e"


def _draw_floor(ax, band: tuple[float, float, float], *, vertical: bool = True, label: str | None = None) -> None:
    """The localization floor on `ax`, in the axis's own coordinates
    (already log10 on a log10-D axis): a band over (low, high) and a
    dotted line at mid."""
    low, mid, high = band
    span, line = (ax.axvspan, ax.axvline) if vertical else (ax.axhspan, ax.axhline)
    span(low, high, color=_MUTED_INK, alpha=0.08, lw=0, zorder=0)
    line(mid, color=_MUTED_INK, lw=1, ls=":", zorder=1, label=label)


def _floor_label(band: tuple[float, float, float]) -> str:
    return f"localization floor {band[1]:.2g} µm²/s (10–90% of tracks)"


def _style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED_INK)
    ax.tick_params(colors=_MUTED_INK, labelsize=8)
    ax.grid(axis="y", color="0.92", lw=0.6)
    ax.set_axisbelow(True)


def _density(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Grid weights (sum to 1) as a density over `x` (the plotted axis),
    so a curve and a density histogram share one y scale."""
    return weights / np.gradient(x)


def _draw_deconvolved(ax, x, weights, band, level, lw: float) -> float:
    """The deconvolved distribution as a density over `x`, with its
    pointwise band shaded (absent for None); returns the curve's maximum.

    The band does not set the y scale: a peak narrower than the tracks
    resolve has an unidentified height, and its band would dwarf the curve,
    so it runs off the top of the axis instead."""
    dens = _density(weights, x)
    if band is not None:
        lo, hi = (_density(b, x) for b in band)
        ax.fill_between(x, lo, hi, color=_DECONVOLVED_COLOR, alpha=0.2, lw=0,
                        label=f"deconvolved, {level:.0%} band")
    ax.plot(x, dens, color=_DECONVOLVED_COLOR, lw=lw, label="deconvolved")
    return dens.max()


def _mass_range(weights: np.ndarray, x: np.ndarray, tail: float = 1e-3) -> tuple[float, float]:
    """The span of `x` holding all but `tail` of `weights` at each end."""
    cdf = np.cumsum(weights)
    return float(np.interp(tail, cdf, x)), float(np.interp(1 - tail, cdf, x))


# Histogram bin width on a log10-D axis: about 12% in D, a few grid cells.
_LOG_D_BIN = 0.05
# Padding around the plotted D range, in decades.
_LOG_D_PAD = 0.25


def _interval_marker(ax, low: float, median: float, high: float, y: float, color: str) -> None:
    ax.plot([low, high], [y, y], color=color, lw=2, solid_capstyle="round", zorder=4)
    ax.plot([median], [y], "o", color=color, ms=6, mec="white", mew=1.2, zorder=5)


def _log_d_range(panels: list[dict], x: np.ndarray) -> tuple[float, float]:
    """One log10-D range for every panel: wherever any panel has medians,
    deconvolved mass or its localization floor, rather than the whole grid, which spans five
    decades and would squeeze the data into a sliver."""
    spans = []
    for panel in panels:
        # Not the mean posterior: short tracks give it a tail decades long.
        spans.append(_mass_range(panel["deconvolved"], x, tail=0.01))
        medians = np.asarray(panel["medians"], dtype=float)
        medians = medians[medians > 0]
        if len(medians):
            spans.append((np.log10(medians.min()), np.log10(medians.max())))
        if panel.get("floor") is not None:  # always in view: it is what D is read against
            spans.append((np.log10(panel["floor"][0]), np.log10(panel["floor"][2])))
    x_lo = max(min(lo for lo, _ in spans) - _LOG_D_PAD, x[0])
    x_hi = min(max(hi for _, hi in spans) + _LOG_D_PAD, x[-1])
    return x_lo, x_hi


def plot_d_ensemble(d_grid: np.ndarray, panels: list[dict], title: str | None = None) -> Figure:
    """The population read of a posterior run: one row per panel (a region
    class, or "all"), over log10 D.

    Each panel dict holds `name`, `n_tracks`, `medians` (per-track D
    posterior medians), `deconvolved` (weights on `d_grid`) with its
    pointwise `deconvolved_band` (low, high weights) at credible `level`,
    `shared_interval` (low, median, high) and `floor` (the localization
    floor's (10%, median, 90%) or None) (`diffusion.ensemble_panels`).

    On one density axis: the histogram of per-track medians (what the
    typical track says), and the deconvolved distribution (how D is spread
    across tracks, with each track's own uncertainty removed -- widths
    are resolution-limited). The shared posterior -- one D shared by every
    track -- is far narrower than either, so rather than a curve that
    would flatten the other two it is a marker with its 90% interval
    above them. The localization floor sits behind all three.
    """
    n = len(panels)
    fig = Figure(figsize=(6.0, 1.2 + 2.3 * n), layout="constrained")
    axes = fig.subplots(n, 1, squeeze=False, sharex="col")
    x = np.log10(d_grid)
    x_lo, x_hi = _log_d_range(panels, x)
    bins = np.arange(x_lo, x_hi + _LOG_D_BIN, _LOG_D_BIN)
    for row, panel in enumerate(panels):
        ax = axes[row][0]
        _style_axis(ax)
        medians = np.asarray(panel["medians"], dtype=float)
        medians = np.log10(medians[medians > 0])
        hist_top = 0.0
        if len(medians):
            counts, _, _ = ax.hist(medians, bins=bins, density=True, color=_HIST_COLOR, edgecolor="white",
                                   lw=0.5, label="per-track medians")
            hist_top = counts.max()
        dens_top = _draw_deconvolved(ax, x, panel["deconvolved"], panel.get("deconvolved_band"),
                                     panel.get("level", 0.9), lw=1.8)
        if panel.get("floor") is not None:
            _draw_floor(ax, tuple(np.log10(panel["floor"])), label=_floor_label(panel["floor"]))
        top = max(dens_top, hist_top)
        low, median, high = (np.log10(v) for v in panel["shared_interval"])
        _interval_marker(ax, low, median, high, top * 1.08, _SHARED_COLOR)
        ax.plot([], [], "o-", color=_SHARED_COLOR, lw=2, ms=5, label="shared D, 90%")
        ax.set_ylim(0, top * 1.18)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylabel("density", fontsize=8, color=_MUTED_INK)
        ax.set_title(
            f"{panel['name']} · {panel['n_tracks']} tracks · shared D = "
            f"{panel['shared_interval'][1]:.3g} µm²/s",
            fontsize=9, loc="left", color=_INK,
        )
        if row == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[-1][0].set_xlabel(units.mpl_log_label("D_um2_s"), fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    return fig


def plot_d_posteriors(d_grid: np.ndarray, panels: list[dict], title: str | None = None) -> Figure:
    """Every track's posterior next to what the population makes of them:
    one row per panel (`ensemble_panels(..., track_posteriors=True)`).

    Left, a heat map with one row per track, sorted by its posterior
    median: a well-determined track is a short bright streak, a short or
    noisy one a long faint smear. Right, on one density axis: the mean of
    the tracks' posteriors (where they put D, blurred by each one's
    uncertainty), the histogram of per-track medians, and the deconvolved
    distribution (that blur removed). The shared posterior -- one D
    shared by every track -- is far narrower than any of them, so it is
    the marker with its 90% interval above the curves, as in
    `plot_d_ensemble`. The localization floor is on both panels."""
    n = len(panels)
    fig = Figure(figsize=(10.5, 1.0 + 3.4 * n), layout="constrained")
    axes = fig.subplots(n, 2, squeeze=False, sharex=True, gridspec_kw={"width_ratios": [1, 1.1]})
    x = np.log10(d_grid)
    dx = float(np.mean(np.diff(x)))
    x_lo, x_hi = _log_d_range(panels, x)
    bins = np.arange(x_lo, x_hi + _LOG_D_BIN, _LOG_D_BIN)
    # One color scale for every heat map, so rows of different panels
    # compare; the top 0.5% saturate rather than wash the rest out.
    all_dens = np.concatenate([p["track_posteriors"].ravel() for p in panels]) / dx
    vmax = float(np.percentile(all_dens, 99.5)) or 1.0
    for row, panel in enumerate(panels):
        ax_map, ax = axes[row]
        _style_axis(ax_map)
        ax_map.grid(False)
        posts = panel["track_posteriors"]
        image = ax_map.imshow(
            posts / dx, aspect="auto", origin="lower", cmap=_POSTERIOR_CMAP, interpolation="nearest",
            extent=[x[0] - dx / 2, x[-1] + dx / 2, 0, len(posts)], vmin=0, vmax=vmax,
        )
        ax_map.set_ylabel("tracks, by posterior median", fontsize=8, color=_MUTED_INK)
        ax_map.set_title(
            f"{panel['name']} · {panel['n_tracks']} tracks' posteriors (one row each)",
            fontsize=9, loc="left", color=_INK,
        )
        if panel.get("floor") is not None:
            ax_map.axvline(np.log10(panel["floor"][1]), color=_INK, lw=1, ls=":")
        colorbar = fig.colorbar(image, ax=ax_map, pad=0.01, shrink=0.85)
        colorbar.set_label(r"density per $\log_{10} D$", fontsize=8, color=_MUTED_INK)
        colorbar.ax.tick_params(colors=_MUTED_INK, labelsize=7)
        colorbar.outline.set_visible(False)

        _style_axis(ax)
        medians = np.asarray(panel["medians"], dtype=float)
        medians = np.log10(medians[medians > 0])
        hist_top = 0.0
        if len(medians):
            counts, _, _ = ax.hist(medians, bins=bins, density=True, color=_HIST_COLOR, edgecolor="white",
                                   lw=0.5, label="per-track medians")
            hist_top = counts.max()
        mean_posterior = _density(panel["mean_posterior"], x)
        ax.plot(x, mean_posterior, color=_MEAN_POSTERIOR_COLOR, lw=2, label="mean of track posteriors")
        deconvolved_top = _draw_deconvolved(ax, x, panel["deconvolved"], panel.get("deconvolved_band"),
                                            panel.get("level", 0.9), lw=2)
        if panel.get("floor") is not None:
            _draw_floor(ax, tuple(np.log10(panel["floor"])), label=_floor_label(panel["floor"]))
        top = max(mean_posterior.max(), deconvolved_top, hist_top)
        low, median, high = (np.log10(v) for v in panel["shared_interval"])
        _interval_marker(ax, low, median, high, top * 1.08, _SHARED_COLOR)
        ax.plot([], [], "o-", color=_SHARED_COLOR, lw=2, ms=5, label="shared D, 90%")
        ax.set_ylim(0, top * 1.18)
        ax.set_ylabel(r"density per $\log_{10} D$", fontsize=8, color=_MUTED_INK)
        ax.set_title(
            f"shared D = {panel['shared_interval'][1]:.3g} µm²/s", fontsize=9, loc="left", color=_INK
        )
        if row == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[-1][0].set_xlim(x_lo, x_hi)
    for ax in axes[-1]:
        ax.set_xlabel(units.mpl_log_label("D_um2_s"), fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    return fig


def plot_d_by_length(panels: list[dict], title: str | None = None) -> Figure:
    """Which tracks each part of the distribution of D comes from: one row
    per panel (`diffusion.length_panels`: `name`, `composition` -- a
    diffusionkit `LengthComposition` -- and `floor`).

    Left, the mean of the tracks' posteriors and, middle, the deconvolved
    distribution (each track's posterior under it, mean over its draws),
    both stacked by track-length group: each group's band is its share of
    all the tracks (or detections, as the composition was weighted), so
    the stack is the whole distribution. Right, each group's own
    deconvolved distribution, scaled to its peak, with its tracks and
    detections at the right: short tracks come mostly from fast particles,
    which leave the focal depth within a few frames."""
    n = len(panels)
    fig = Figure(figsize=(12.0, 1.0 + 3.0 * n), layout="constrained")
    axes = fig.subplots(n, 3, squeeze=False, sharex=True, gridspec_kw={"width_ratios": [1, 1, 1.15]})
    x_lo = x_hi = None
    for row, panel in enumerate(panels):
        comp = panel["composition"]
        x = comp.u / np.log(10)
        dx = float(np.mean(np.diff(x)))
        mean = comp.deconvolved.mean(axis=0)
        lo, hi = _mass_range(mean.sum(axis=0), x, tail=0.005)
        if panel.get("floor") is not None:
            lo, hi = min(lo, np.log10(panel["floor"][0])), max(hi, np.log10(panel["floor"][2]))
        x_lo = lo if x_lo is None else min(x_lo, lo)
        x_hi = hi if x_hi is None else max(x_hi, hi)
        labels = comp.labels()
        colors = _LENGTH_CMAP(np.linspace(0.15, 1.0, len(labels)))
        top = 0.0
        for col, (contrib, what) in enumerate(((comp.pooled, "mean of track posteriors"), (mean, "deconvolved"))):
            ax = axes[row][col]
            _style_axis(ax)
            bottom = np.zeros(len(x))
            for j, label in enumerate(labels):
                upper = bottom + contrib[j] / dx
                ax.fill_between(x, bottom, upper, color=colors[j], lw=0, label=label)
                bottom = upper
            top = max(top, bottom.max())
            if panel.get("floor") is not None:
                _draw_floor(ax, tuple(np.log10(panel["floor"])))
            ax.set_title(f"{panel['name']} · {what}, by track length", fontsize=9, loc="left", color=_INK)
            ax.set_ylabel(f"{comp.weight} per $\\log_{{10}} D$", fontsize=8, color=_MUTED_INK)
        for col in (0, 1):
            axes[row][col].set_ylim(0, top * 1.05)
        axes[row][1].legend(title="frames", fontsize=7, title_fontsize=7, frameon=False, loc="upper left",
                            reverse=True)

        ax = axes[row][2]
        _style_axis(ax)
        ax.grid(False)
        own = mean / np.maximum(mean.max(axis=1, keepdims=True), 1e-300)
        ax.imshow(own, aspect="auto", origin="lower", cmap=_POSTERIOR_CMAP, interpolation="nearest",
                  extent=[x[0] - dx / 2, x[-1] + dx / 2, -0.5, len(labels) - 0.5], vmin=0, vmax=1)
        if panel.get("floor") is not None:
            ax.axvline(np.log10(panel["floor"][1]), color=_INK, lw=1, ls=":")
        ax.set_yticks(range(len(labels)), labels)
        for j in range(len(labels)):
            ax.text(1.01, j, f"{comp.n_tracks[j]} / {comp.n_detections[j]}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7, color=_MUTED_INK)
        ax.set_ylabel("track length (frames)", fontsize=8, color=_MUTED_INK)
        ax.set_title("each length's own distribution (peak = 1) · tracks / detections", fontsize=9, loc="left",
                     color=_INK)
    axes[-1][0].set_xlim(x_lo - _LOG_D_PAD, x_hi + _LOG_D_PAD)
    for ax in axes[-1]:
        ax.set_xlabel(units.mpl_log_label("D_um2_s"), fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    return fig


def plot_track_posterior(
    track_id: int,
    d_grid: np.ndarray,
    d_weights: np.ndarray,
    d_interval: tuple[float, float, float],
    n_frames: int | None = None,
    d_floor: float | None = None,
) -> Figure:
    """One track's posterior over log10 D, with its median and the shaded
    90% interval and its localization floor. A short track's posterior is
    wide, and that width is the answer, not a defect."""
    fig = Figure(figsize=(4.2, 2.8), layout="constrained")
    ax = fig.subplots()
    _style_axis(ax)
    x = np.log10(d_grid)
    low, median, high = (np.log10(v) for v in d_interval)
    dens = _density(d_weights, x)
    inside = (x >= low) & (x <= high)
    ax.fill_between(x, dens, where=inside, color=_DECONVOLVED_COLOR, alpha=0.18, lw=0)
    ax.plot(x, dens, color=_DECONVOLVED_COLOR, lw=1.8)
    ax.axvline(median, color=_INK, lw=1)
    ax.set_xlabel(units.mpl_log_label("D_um2_s"), fontsize=9)
    ax.set_ylim(bottom=0)
    lo, hi = _mass_range(d_weights, x)
    if d_floor is not None and d_floor > 0:
        floor = np.log10(d_floor)
        ax.axvline(floor, color=_MUTED_INK, lw=1, ls=":")
        ax.text(floor, 0.98, " floor", transform=ax.get_xaxis_transform(), ha="left", va="top",
                fontsize=7, color=_MUTED_INK)
        lo, hi = min(lo, floor), max(hi, floor)
    ax.set_xlim(lo - _LOG_D_PAD, hi + _LOG_D_PAD)
    ax.set_ylabel("posterior density", fontsize=8, color=_MUTED_INK)
    low, median, high = d_interval
    head = f"track {track_id}" + (f" · {n_frames} frames" if n_frames else "")
    ax.set_title(f"{head}\nD = {median:.3g} µm²/s [{low:.3g}, {high:.3g}]", fontsize=9, loc="left", color=_INK)
    return fig
