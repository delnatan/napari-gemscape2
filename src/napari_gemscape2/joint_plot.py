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
role, not by series: the log-normal population is aqua, the deconvolved
distribution is blue, the shared D is orange, and the per-track heat map is
one blue ramp. The population is always the tracks' log-likelihoods combined
under a model of it -- never a histogram of their medians or an average of
their posteriors (see `diffusion`). Track-length groups are ordered, so they are steps of one ramp,
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

import textwrap

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

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
_LOGNORMAL_COLOR = "#1baf7a"
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


def _draw_band_curve(ax, x, weights, band, color, label, lw: float, band_alpha: float = 0.2) -> float:
    """Grid weights as a density over `x`, with their pointwise band shaded
    (absent for None); returns the curve's maximum.

    The band does not set the y scale: a peak narrower than the tracks
    resolve has an unidentified height, and its band would dwarf the curve,
    so it runs off the top of the axis instead."""
    dens = _density(weights, x)
    if band is not None:
        lo, hi = (_density(b, x) for b in band)
        ax.fill_between(x, lo, hi, color=color, alpha=band_alpha, lw=0)
    ax.plot(x, dens, color=color, lw=lw, label=label)
    return dens.max()


def _draw_population(ax, x, panel: dict) -> float:
    """One panel's population reads on a density axis over log10 D: the
    log-normal (the headline, with its band), the deconvolved distribution
    (thinner: any shape, the check on the log-normal's), and the shared D
    as a marker with its interval above both -- far narrower than either,
    it would flatten them as a curve. Returns the y top it set."""
    level = panel.get("level", 0.9)
    top = _draw_band_curve(ax, x, panel["lognormal"], panel.get("lognormal_band"), _LOGNORMAL_COLOR,
                           f"log-normal population, {level:.0%} band", lw=2.2)
    top = max(top, _draw_band_curve(ax, x, panel["deconvolved"], panel.get("deconvolved_band"), _DECONVOLVED_COLOR,
                                    "deconvolved (any shape)", lw=1.2, band_alpha=0.1))
    if panel.get("floor") is not None:
        _draw_floor(ax, tuple(np.log10(panel["floor"])), label=_floor_label(panel["floor"]))
    low, median, high = (np.log10(v) for v in panel["shared_interval"])
    _interval_marker(ax, low, median, high, top * 1.08, _SHARED_COLOR)
    ax.plot([], [], "o-", color=_SHARED_COLOR, lw=2, ms=5, label=f"shared D, {level:.0%}")
    ax.set_ylim(0, top * 1.18)
    return top


def _population_title(panel: dict) -> str:
    """The log-normal's numbers, and on a second line the shared D's, as a panel title."""
    sm = panel["summary"]
    return (
        f"median D {_with_interval(sm['lognormal_D_median_um2_s'], sm['lognormal_D_median_low_um2_s'], sm['lognormal_D_median_high_um2_s'])}"
        f" µm²/s · σ(ln D) {_with_interval(sm['lognormal_sigma_ln_D'], sm['lognormal_sigma_ln_D_low'], sm['lognormal_sigma_ln_D_high'], '.2f')}"
        f"\nshared D {_with_interval(*np.array(panel['shared_interval'])[[1, 0, 2]])} µm²/s (one D for every track)"
    )


def _mass_range(weights: np.ndarray, x: np.ndarray, tail: float = 1e-3) -> tuple[float, float]:
    """The span of `x` holding all but `tail` of `weights` at each end."""
    cdf = np.cumsum(weights)
    return float(np.interp(tail, cdf, x)), float(np.interp(1 - tail, cdf, x))


# Padding around the plotted D range, in decades.
_LOG_D_PAD = 0.25


def _interval_marker(ax, low: float, median: float, high: float, y: float, color: str) -> None:
    ax.plot([low, high], [y, y], color=color, lw=2, solid_capstyle="round", zorder=4)
    ax.plot([median], [y], "o", color=color, ms=6, mec="white", mew=1.2, zorder=5)


def _log_d_range(panels: list[dict], x: np.ndarray) -> tuple[float, float]:
    """One log10-D range for every panel: wherever any panel's populations
    have mass, or its localization floor is, rather than the whole grid,
    which spans five decades and would squeeze the data into a sliver."""
    spans = []
    for panel in panels:
        for key in ("lognormal", "deconvolved"):
            spans.append(_mass_range(panel[key], x, tail=0.005))
        spans.append(tuple(np.log10([panel["shared_interval"][0], panel["shared_interval"][2]])))
        if panel.get("floor") is not None:  # always in view: it is what D is read against
            spans.append((np.log10(panel["floor"][0]), np.log10(panel["floor"][2])))
    x_lo = max(min(lo for lo, _ in spans) - _LOG_D_PAD, x[0])
    x_hi = min(max(hi for _, hi in spans) + _LOG_D_PAD, x[-1])
    return x_lo, x_hi


def plot_d_ensemble(d_grid: np.ndarray, panels: list[dict], title: str | None = None) -> Figure:
    """The population read of a posterior run: one row per panel (a region
    class, or "all"), over log10 D (`diffusion.ensemble_panels`).

    On one density axis, the tracks' log-likelihoods combined three ways:
    the log-normal population (its median D and spread in the title), the
    deconvolved distribution (any shape: where it shows two modes, the
    log-normal's numbers describe the wrong shape), and the shared D (one D
    for every track) as a marker with its interval. The localization floor
    sits behind all three."""
    n = len(panels)
    fig = Figure(figsize=(6.4, 1.2 + 2.5 * n), layout="constrained")
    axes = fig.subplots(n, 1, squeeze=False, sharex="col")
    x = np.log10(d_grid)
    x_lo, x_hi = _log_d_range(panels, x)
    for row, panel in enumerate(panels):
        ax = axes[row][0]
        _style_axis(ax)
        _draw_population(ax, x, panel)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylabel(r"density per $\log_{10} D$", fontsize=8, color=_MUTED_INK)
        ax.set_title(f"{panel['name']} · {panel['n_tracks']} tracks\n{_population_title(panel)}",
                     fontsize=9, loc="left", color=_INK)
        if row == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[-1][0].set_xlabel(units.mpl_log_label("D_um2_s"), fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    return fig


def plot_d_posteriors(d_grid: np.ndarray, panels: list[dict], title: str | None = None) -> Figure:
    """Every track's posterior next to what the population makes of them:
    one row per panel (`ensemble_panels(..., track_posteriors=True)`).

    Left, a heat map with one row per track (its flat-prior posterior),
    sorted by where its likelihood peaks: a well-determined track is a
    short bright streak, a short or noisy one a long faint smear. Right,
    the population reads as in `plot_d_ensemble`. The localization floor
    is on both panels."""
    n = len(panels)
    fig = Figure(figsize=(10.5, 1.0 + 3.4 * n), layout="constrained")
    axes = fig.subplots(n, 2, squeeze=False, sharex=True, gridspec_kw={"width_ratios": [1, 1.1]})
    x = np.log10(d_grid)
    dx = float(np.mean(np.diff(x)))
    x_lo, x_hi = _log_d_range(panels, x)
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
        ax_map.set_ylabel("tracks, by likelihood peak", fontsize=8, color=_MUTED_INK)
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
        _draw_population(ax, x, panel)
        ax.set_ylabel(r"density per $\log_{10} D$", fontsize=8, color=_MUTED_INK)
        ax.set_title(_population_title(panel), fontsize=9, loc="left", color=_INK)
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

    Left, the tracks' own flat-prior posteriors (unpooled) and, middle,
    each track's posterior under the deconvolved distribution (partially
    pooled, mean over its draws), both stacked by track-length group: each group's band is its share of
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
        mean = comp.partially_pooled.mean(axis=0)
        lo, hi = _mass_range(mean.sum(axis=0), x, tail=0.005)
        if panel.get("floor") is not None:
            lo, hi = min(lo, np.log10(panel["floor"][0])), max(hi, np.log10(panel["floor"][2]))
        x_lo = lo if x_lo is None else min(x_lo, lo)
        x_hi = hi if x_hi is None else max(x_hi, hi)
        labels = comp.labels()
        colors = _LENGTH_CMAP(np.linspace(0.15, 1.0, len(labels)))
        top = 0.0
        for col, (contrib, what) in enumerate(((comp.unpooled, "unpooled"), (mean, "partially pooled"))):
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
    if title is None:
        title = (
            "counted per track: each track once -- the stack is the deconvolved distribution of D across tracks"
            if panels[0]["composition"].weight == "tracks"
            else "counted per detection: each track once per frame, so long (often slow) tracks weigh more -- "
            "the make-up of the spots in focus"
        )
    fig.suptitle(title, fontsize=8, x=0.01, ha="left", color=_MUTED_INK)
    return fig


def _with_interval(value, low, high, fmt: str = ".3g") -> str:
    """`value [low–high]`, the interval left off when it was not estimated."""
    if value is None:
        return "—"
    if low is None or high is None:
        return f"{value:{fmt}}"
    return f"{value:{fmt}} [{low:{fmt}}–{high:{fmt}}]"


def _msd_points(ax, tau, y, yerr, inside, curve: str = "ensemble MSD") -> None:
    """MSD points: filled inside the fit window, hollow past it."""
    yerr = np.asarray(yerr, float)
    for mask, face, label in ((inside, _INK, "fit window"), (~inside, "white", "past the window")):
        if mask.any():
            err = yerr[..., mask]
            ax.errorbar(tau[mask], y[mask], yerr=None if np.isnan(err).all() else err, fmt="o", ms=4, color=_INK,
                        mfc=face, mec=_INK, ecolor=_MUTED_INK, elinewidth=0.8, capsize=0, zorder=3,
                        label=f"{curve}, {label}")


def _fit_line(ax, t, y, n_solid: int, color: str, style: str, label: str) -> None:
    """A fit drawn `style` over its window (the first `n_solid` points) and dotted past it."""
    ax.plot(t[:n_solid], y[:n_solid], style, color=color, lw=1.5, label=label)
    if n_solid < len(t):
        ax.plot(t[n_solid - 1:], y[n_solid - 1:], ":", color=color, lw=1.1)


def _plain_log_ticks(ax) -> None:
    """Log axes labelled at 1, 2 and 5 per decade as plain numbers: an MSD
    curve spans a decade or two, where matplotlib's default labels every
    minor tick in scientific notation and they run into each other."""
    plain = FuncFormatter(lambda v, _pos: f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(NullFormatter())


def _fit_problem(fit: dict) -> list[str]:
    """A fit's status and message, wrapped to sit inside half a figure's
    width -- or nothing when it is ok."""
    if fit.get("status") in (None, "ok"):
        return []
    return textwrap.wrap(f"{fit['status']}: {fit.get('message')}", 46)


def _nm(value_um):
    return None if value_um is None else 1000 * value_um


def plot_ensemble_msd(panels: list[dict], title: str | None = None) -> Figure:
    """The ensemble-averaged MSD of each group (`diffusion.ensemble_msd_panels`),
    one row per group: on linear axes with the linear fit that gives D, and
    on log-log axes net of the localization offset, with the power law that
    gives α. Each fit has its own window -- D the first `n_points` lags, α
    the first `alpha_points`, usually more -- filled with its fit solid; the
    lags past it are hollow and the fit continues dotted, so where the curve
    leaves the model shows. The α panel says how many decades of τ its
    window spans and how many tracks still reach its last lag: the late
    lags rest on the long tracks alone, which are not a random sample of
    the particles (fast ones leave the focus sooner). Error bars are ±1 SEM
    of the averaged curve over tracks, the independent unit: the bootstrap
    SD over resampled tracks, which is that SEM (pair-weighted) without a
    formula -- not the spread of the tracks (SD, √n times wider, a
    population's heterogeneity rather than the mean's uncertainty), and not
    an SEM over displacement pairs, which overlap and are correlated, so
    that one comes out several times too small. The intervals in the text
    are the fits' own, from refitting every resample."""
    if title is None:
        offset_rule = "offset fitted (intercept)" if panels[0]["offset"] == "fit" else "offset from the SDs"
        title = (
            f"Ensemble-averaged MSD, pair-weighted · {offset_rule} · exposure treated as 0 (no blur model)\n"
            f"bars ±1 SEM over tracks · fit intervals {panels[0]['level']:.0%}, bootstrap over tracks"
        )
    return _plot_msd_rows(panels, title)


def plot_track_msd(panel: dict, title: str | None = None) -> Figure:
    """One track's time-averaged MSD (`diffusion.msd_track_panel`), drawn
    as `plot_ensemble_msd` draws a group: D's fit on linear axes, alpha's
    log-log fit on log axes, each over its own window -- the fits behind
    the track's D_msd / alpha_msd. The linear axes add the fit with a free
    intercept (dotted grey), whose offset is read off the curve instead of
    the SDs; the localization SDs the two imply are compared in the text.
    No error bars: the lags of one track share their displacements."""
    if title is None:
        title = (
            "Time-averaged MSD of one track · offset from the SDs · exposure treated as 0 (no blur model)\n"
            "no error bars: a track's lags share their displacements, so they are not independent points"
        )
    return _plot_msd_rows([panel], title)


def _plot_msd_rows(panels: list[dict], title: str) -> Figure:
    """The MSD rows `plot_ensemble_msd` and `plot_track_msd` share: per
    panel, linear axes with D's fit and log-log axes with alpha's."""
    fig = Figure(figsize=(8.6, 0.7 + 2.7 * len(panels)), layout="constrained")
    axes = fig.subplots(len(panels), 2, sharex="col", squeeze=False)
    for row, p in enumerate(panels):
        tau, msd, se, off = p["tau_s"], p["msd_um2"], p["se_um2"], p["offset_um2"]
        k = min(p["n_points"], len(tau))
        m = min(p.get("alpha_points", k), len(tau))
        inside = np.arange(len(tau)) < k
        inside_log = np.arange(len(tau)) < m
        lin, pw = p["linear"], p["power_law"]
        per_track = p.get("unit") == "pairs"
        curve = "track MSD" if per_track else "ensemble MSD"
        ax_lin, ax_log = axes[row]
        for ax in (ax_lin, ax_log):
            _style_axis(ax)

        # Linear axes: D. The SDs' offset is known per lag only, so that line
        # runs through the lags; a fitted intercept is drawn from τ = 0.
        _msd_points(ax_lin, tau, msd, se, inside, curve)
        D = lin.get("D_um2_s")
        if D is not None and p["offset"] == "fit" and lin.get("offset_um2") is not None:
            t = np.concatenate([[0.0], tau])
            _fit_line(ax_lin, t, 4 * D * t + lin["offset_um2"], k + 1, _SHARED_COLOR, "-", "linear (D), fitted offset")
        elif D is not None:
            _fit_line(ax_lin, tau, 4 * D * tau + off, k, _SHARED_COLOR, "-", "linear (D), SDs' offset")
        text = [f"D = {_with_interval(D, lin.get('D_um2_s_lo'), lin.get('D_um2_s_hi'))} µm²/s"]
        if p["offset"] == "fit" and D is not None:
            sigma = _with_interval(_nm(lin.get("localization_sd_um")), _nm(lin.get("localization_sd_um_lo")),
                                   _nm(lin.get("localization_sd_um_hi")), ".0f")
            text.append(f"σ = {sigma} nm")
        check = p.get("intercept_check") or {}
        if check.get("D_um2_s") is not None and check.get("offset_um2") is not None:
            # Drawn from τ = 0, where it meets its intercept: the offset it fitted.
            t = np.concatenate([[0.0], tau])
            _fit_line(ax_lin, t, 4 * check["D_um2_s"] * t + check["offset_um2"], k + 1, _MUTED_INK, ":",
                      "linear, fitted offset (check)")
            sigmas = [f"SDs {_nm(check['sigma_sds_um']):.0f} nm"] if check.get("sigma_sds_um") is not None else []
            sigmas.append(f"intercept {_nm(check['localization_sd_um']):.0f} nm"
                          if check.get("localization_sd_um") is not None else "intercept < 0")
            text.append("σ: " + " · ".join(sigmas))
        text += _fit_problem(lin)
        ax_lin.text(0.02, 0.97, "\n".join(text), transform=ax_lin.transAxes, va="top", ha="left", fontsize=8,
                    color=_INK)
        ax_lin.set_xlim(left=0)
        ax_lin.set_ylim(bottom=0)
        ax_lin.set_ylabel("MSD (µm²)", fontsize=9)
        size = f"{p['n_frames']} frames" if per_track else f"{p['n_tracks']} tracks"
        ax_lin.set_title(f"{p['name']} · {size} · D: first {p['n_points']} lags", fontsize=9, loc="left",
                         color=_INK)

        # Log-log axes: α, net of the offset. Lags with nothing left after it
        # cannot be logged, and are left out here as the fit leaves them out.
        net = msd - off
        keep = net > 0
        # The lower bar stops short of zero, which a log axis cannot show.
        _msd_points(ax_log, tau[keep], net[keep], [np.minimum(se, 0.999 * net)[keep], se[keep]], inside_log[keep],
                    curve)
        K, alpha = pw.get("K_um2_s_alpha"), pw.get("alpha")
        if K is not None and alpha is not None:
            _fit_line(ax_log, tau, 4 * K * tau**alpha, m, _DECONVOLVED_COLOR, "--", "power law (α)")
        ax_log.set_xscale("log")
        ax_log.set_yscale("log")
        _plain_log_ticks(ax_log)
        ax_log.set_ylabel("MSD − offset (µm²)", fontsize=9)
        n_alpha = p.get("alpha_points", p["n_points"])
        span = f" · {np.log10(tau[n_alpha - 1] / tau[0]):.1f} decades of τ" if n_alpha <= len(tau) else ""
        ax_log.set_title(f"α: first {n_alpha} lags{span}", fontsize=9, loc="left", color=_INK)
        text = [f"α = {_with_interval(alpha, pw.get('alpha_lo'), pw.get('alpha_hi'), '.2f')}"]
        if alpha is not None and per_track:
            text.append(f"{int(p['n_units'][m - 1])} displacement pairs at lag {m}")
        elif alpha is not None:
            text.append(f"{int(p['n_units'][m - 1])} of {p['n_tracks']} tracks reach lag {m}")
        text += _fit_problem(pw)
        ax_log.text(0.02, 0.97, "\n".join(text), transform=ax_log.transAxes, va="top", ha="left", fontsize=8,
                    color=_INK)
        if row == 0:
            for ax in (ax_lin, ax_log):
                ax.legend(fontsize=7, frameon=False, loc="lower right")
    for ax in axes[-1]:
        ax.set_xlabel("lag τ (s)", fontsize=9)
    fig.suptitle(title, fontsize=8, x=0.01, ha="left", color=_MUTED_INK)
    return fig


# Samples are identities compared on one axis, so -- unlike region classes,
# which get a panel each -- they are hues, in this fixed order (validated
# for adjacent lines; the first three carry the posterior figures' roles).
_SAMPLE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")


def plot_pooled_populations(
    distributions: pl.DataFrame, populations: pl.DataFrame, samples: dict[str, list[str]], level: float
) -> Figure:
    """`pooling.population_tables` drawn, one sample per hue (`samples`
    maps each to its bundles; past the eighth color they are left to the
    tables, with a note).

    Left, each sample's population over log10 D: the log-normal of all its
    movies' tracks in one fit, with its `level` band, the deconvolved
    distribution dotted (the check on its shape), and each movie's own
    log-normal thin. Right, the log-normal's median D and spread sigma with
    their intervals, the sample's (filled) above each of its movies'
    (hollow): one fit per sample assumes its movies share a population,
    and the movies' rows show whether they do. Samples are drawn side by
    side, not tested against each other -- that comparison, with the movie
    as the replicate, belongs outside the GUI (`pooled_lognormal_draws_D.csv`)."""
    by_sample = distributions.filter(pl.col("by") == "sample")
    names = list(dict.fromkeys(by_sample["group"].to_list()))
    shown = names[: len(_SAMPLE_COLORS)]
    color = dict(zip(shown, _SAMPLE_COLORS))
    sample_of = {bundle: sample for sample, bundles in samples.items() for bundle in bundles}
    per_movie = distributions.filter(pl.col("by") == "experiment")

    rows = []  # (label, row, filled, color) in drawing order, top to bottom
    for name in shown:
        rows.append((f"{name}", populations.filter((pl.col("by") == "sample") & (pl.col("group") == name)).row(0, named=True),
                     True, color[name]))
        for bundle in samples.get(name, []):
            movie = populations.filter((pl.col("by") == "experiment") & (pl.col("group") == bundle))
            if movie.height:
                rows.append((f"   {bundle}", movie.row(0, named=True), False, color[name]))

    fig = Figure(figsize=(11.0, max(3.8, 1.6 + 0.26 * len(rows))), layout="constrained")
    ax_dens, ax_med, ax_sig = fig.subplots(1, 3, width_ratios=(1.35, 1, 0.75))
    for ax in (ax_dens, ax_med, ax_sig):
        _style_axis(ax)

    spans = []
    for name in shown:
        rows_s = by_sample.filter(pl.col("group") == name)
        x = np.log10(rows_s["D_um2_s"].to_numpy())
        band = (rows_s["lognormal_low"].to_numpy(), rows_s["lognormal_high"].to_numpy())
        _draw_band_curve(ax_dens, x, rows_s["lognormal"].to_numpy(), band, color[name],
                         f"{name} · {rows_s['n_tracks'][0]} tracks", lw=2.2)
        ax_dens.plot(x, _density(rows_s["deconvolved"].to_numpy(), x), ":", color=color[name], lw=1.2)
        for key in ("lognormal", "deconvolved"):
            spans.append(_mass_range(rows_s[key].to_numpy(), x, tail=0.005))
    for (bundle,), rows_m in per_movie.group_by("group", maintain_order=True):
        if sample_of.get(bundle) in color:
            x = np.log10(rows_m["D_um2_s"].to_numpy())
            ax_dens.plot(x, _density(rows_m["lognormal"].to_numpy(), x), color=color[sample_of[bundle]], lw=0.8,
                         alpha=0.7)
    ax_dens.plot([], [], ":", color=_MUTED_INK, lw=1.2, label="dotted: deconvolved (any shape)")
    if per_movie.height:
        ax_dens.plot([], [], color=_MUTED_INK, lw=0.8, label="thin: each movie's own log-normal")
    if spans:
        ax_dens.set_xlim(min(lo for lo, _ in spans) - _LOG_D_PAD, max(hi for _, hi in spans) + _LOG_D_PAD)
    ax_dens.set_ylim(bottom=0)
    ax_dens.set_xlabel(units.mpl_log_label("D_um2_s"), fontsize=9)
    ax_dens.set_ylabel(r"density per $\log_{10} D$", fontsize=8, color=_MUTED_INK)
    note = f"each sample's movies in one log-normal fit, {level:.0%} band"
    if len(names) > len(shown):
        note += f" · {len(names) - len(shown)} more samples in the tables only"
    ax_dens.set_title(f"D by sample\n{note}", fontsize=9, loc="left", color=_INK)
    ax_dens.legend(fontsize=7, frameon=False, loc="upper left")

    ys = np.arange(len(rows))
    for y, (_label, row, filled, tone) in zip(ys, rows):
        for ax, key in ((ax_med, "lognormal_D_median"), (ax_sig, "lognormal_sigma_ln_D")):
            unit = "_um2_s" if key.endswith("median") else ""
            mid, lo, hi = row[f"{key}{unit}"], row[f"{key}_low{unit}"], row[f"{key}_high{unit}"]
            if mid is None:
                continue
            ax.plot([lo, hi], [y, y], color=tone, lw=2 if filled else 1.2, solid_capstyle="round")
            ax.plot([mid], [y], "o", ms=6 if filled else 5, color=tone, mfc=tone if filled else "white", mew=1.4,
                    zorder=3)
    ax_med.set_xscale("log")
    _plain_log_ticks(ax_med)
    ax_med.set_yticks(ys, [label for label, *_ in rows], fontsize=8)
    ax_sig.set_yticks(ys, [""] * len(rows))
    for ax in (ax_med, ax_sig):
        ax.set_ylim(len(rows) - 0.5, -0.5)
    ax_sig.set_xlim(left=0)
    ax_med.set_xlabel("median D (µm²/s)", fontsize=9)
    ax_sig.set_xlabel("spread σ (ln D)", fontsize=9)
    ax_med.set_title(f"log-normal, {level:.0%} intervals\nfilled: sample · hollow: one movie", fontsize=9,
                     loc="left", color=_INK)
    ax_sig.set_title("σ near 0: one D\ndescribes the tracks", fontsize=9, loc="left", color=_INK)
    return fig


def plot_track_posterior(
    track_id: int,
    d_grid: np.ndarray,
    d_weights: np.ndarray,
    d_interval: tuple[float, float, float],
    n_frames: int | None = None,
    d_floor: float | None = None,
    d_grid_edge: str | None = None,
) -> Figure:
    """One track's posterior over log10 D, with its mean E[D] and the shaded
    90% interval and its localization floor. A short track's posterior is
    wide, and that width is the answer, not a defect. A posterior cut by a
    grid edge (`d_grid_edge`) is read as a bound, and the title says so."""
    fig = Figure(figsize=(4.2, 2.8), layout="constrained")
    ax = fig.subplots()
    _style_axis(ax)
    x = np.log10(d_grid)
    low, mean, high = (np.log10(v) for v in d_interval)
    dens = _density(d_weights, x)
    inside = (x >= low) & (x <= high)
    ax.fill_between(x, dens, where=inside, color=_DECONVOLVED_COLOR, alpha=0.18, lw=0)
    ax.plot(x, dens, color=_DECONVOLVED_COLOR, lw=1.8)
    ax.axvline(mean, color=_INK, lw=1)
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
    low, mean, high = d_interval
    head = f"track {track_id}" + (f" · {n_frames} frames" if n_frames else "")
    reading = {"low": f"D < {high:.3g} µm²/s (cut at the grid's low edge: an upper bound)",
               "high": f"D > {low:.3g} µm²/s (cut at the grid's high edge: a lower bound)",
               "both": "no information about D (cut at both grid edges)"}.get(d_grid_edge)
    line = reading or f"E[D] = {mean:.3g} µm²/s [{low:.3g}, {high:.3g}]"
    ax.set_title(f"{head}\n{line}", fontsize=9, loc="left", color=_INK)
    return fig
