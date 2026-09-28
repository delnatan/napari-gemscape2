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

The posterior figures (`plot_d_ensemble`, `plot_track_posterior`) follow
one color assignment by role, not by series (per-track medians of alpha and
of the D ratio are the same neutral histogram): per-track medians are a
neutral histogram, the deconvolved distribution is blue, the summed
(shared-value) posterior is orange, the pooled (averaged) posterior is aqua,
and the per-track heat map is one blue ramp. Region classes are separate panels
stacked on one shared x axis rather than more hues, so every panel reads
the same way.

Axis labels default to `napari_gemscape2.units.mpl_label(column)` rather than
to the raw column name: since the axes here are picked at runtime from
whatever the tracks pane holds, a plot could otherwise put
`radius_of_gyration_um` (µm) against `se_x_max` (px) with nothing on
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
) -> Figure:
    """`style` is "points" (scatter over a KDE), "density" (2D histogram,
    sparse bins as points), or "auto": density from `DENSITY_MIN_POINTS`
    tracks up."""
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
_SUMMED_COLOR = "#eb6834"
_POOLED_COLOR = "#1baf7a"
# One hue, light to dark, its zero end receding into the figure's surface:
# the per-track heat map is magnitude, not identity.
_POSTERIOR_CMAP = LinearSegmentedColormap.from_list(
    "posterior", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
)
# The joint plot's density: the same ramp without its surface-colored end,
# so the emptiest drawn bin still reads against the empty ones.
_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "density", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
)
_INK = "#0b0b0b"
_MUTED_INK = "#52514e"


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
    """One log10-D range for every panel: wherever any panel has medians
    or deconvolved mass, rather than the whole grid, which spans five
    decades and would squeeze the data into a sliver."""
    spans = []
    for panel in panels:
        # Not the pooled posterior: short tracks give it a tail decades long.
        spans.append(_mass_range(panel["deconvolved"], x, tail=0.01))
        medians = np.asarray(panel["medians"], dtype=float)
        medians = medians[medians > 0]
        if len(medians):
            spans.append((np.log10(medians.min()), np.log10(medians.max())))
    x_lo = max(min(lo for lo, _ in spans) - _LOG_D_PAD, x[0])
    x_hi = min(max(hi for _, hi in spans) + _LOG_D_PAD, x[-1])
    return x_lo, x_hi


def plot_d_ensemble(
    d_grid: np.ndarray,
    panels: list[dict],
    alpha_grid: np.ndarray | None = None,
    title: str | None = None,
    ratio_label: str | None = None,
) -> Figure:
    """The population read of a posterior run: one row per panel (a region
    class, or "all"), with log10 D on the left, then alpha when alpha was
    computed, then how D changes with timescale when that was (panels with
    `ratio_medians`; `ratio_label` names the two timescales on its axis).

    Each panel dict holds `name`, `n_tracks`, `medians` (per-track D
    posterior medians), `deconvolved` and `summed` (weights on `d_grid`),
    `summed_interval` (low, median, high), and optionally
    `alpha_medians`, `alpha_summed_interval`.

    On one density axis: the histogram of per-track medians (what the
    typical track says), and the deconvolved distribution (how D is spread
    across tracks, with each track's own uncertainty removed -- widths
    are resolution-limited). The summed posterior -- one D shared by every
    track -- is far narrower than either, so rather than a curve that
    would flatten the other two it is a marker with its 90% interval
    above them.
    """
    has_alpha = alpha_grid is not None and any(p.get("alpha_medians") is not None for p in panels)
    has_ratio = any(p.get("ratio_medians") is not None for p in panels)
    n, n_cols = len(panels), 1 + has_alpha + has_ratio
    fig = Figure(figsize=(6.0 + 2.6 * (n_cols - 1), 1.2 + 2.3 * n), layout="constrained")
    axes = fig.subplots(n, n_cols, squeeze=False, sharex="col")
    ratio_col = n_cols - 1
    x = np.log10(d_grid)
    x_lo, x_hi = _log_d_range(panels, x)
    bins = np.arange(x_lo, x_hi + _LOG_D_BIN, _LOG_D_BIN)
    for row, panel in enumerate(panels):
        ax = axes[row][0]
        _style_axis(ax)
        medians = np.asarray(panel["medians"], dtype=float)
        medians = np.log10(medians[medians > 0])
        if len(medians):
            ax.hist(medians, bins=bins, density=True, color=_HIST_COLOR, edgecolor="white",
                    lw=0.5, label="per-track medians")
        dens = _density(panel["deconvolved"], x)
        ax.plot(x, dens, color=_DECONVOLVED_COLOR, lw=1.8, label="deconvolved")
        top = max(dens.max(), ax.get_ylim()[1])
        low, median, high = (np.log10(v) for v in panel["summed_interval"])
        _interval_marker(ax, low, median, high, top * 1.08, _SUMMED_COLOR)
        ax.plot([], [], "o-", color=_SUMMED_COLOR, lw=2, ms=5, label="summed (shared D), 90%")
        ax.set_ylim(0, top * 1.18)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylabel("density", fontsize=8, color=_MUTED_INK)
        ax.set_title(
            f"{panel['name']} · {panel['n_tracks']} tracks · shared D = "
            f"{panel['summed_interval'][1]:.3g} µm²/s",
            fontsize=9, loc="left", color=_INK,
        )
        if row == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper left")
        if has_alpha:
            ax_a = axes[row][1]
            _style_axis(ax_a)
            alpha_medians = panel.get("alpha_medians")
            if alpha_medians is not None and len(alpha_medians):
                ax_a.hist(alpha_medians, bins=np.linspace(0, 2, 26), density=True,
                          color=_HIST_COLOR, edgecolor="white", lw=0.8)
                ax_a.axvline(1.0, color=_MUTED_INK, lw=0.8, ls="--", zorder=0)
                a_top = ax_a.get_ylim()[1]
                a_low, a_med, a_high = panel["alpha_summed_interval"]
                _interval_marker(ax_a, a_low, a_med, a_high, a_top * 1.08, _SUMMED_COLOR)
                ax_a.set_ylim(0, a_top * 1.18)
                ax_a.set_title(f"shared α = {a_med:.2f}", fontsize=9, loc="left", color=_INK)
            ax_a.set_xlim(0, 2)
        if has_ratio:
            _ratio_panel(axes[row][ratio_col], panel)
    axes[-1][0].set_xlabel(units.mpl_log_label("D_um2_s"), fontsize=9)
    if has_alpha:
        axes[-1][1].set_xlabel(r"$\alpha$ (1 = Brownian, dashed)", fontsize=9)
    if has_ratio:
        axes[-1][ratio_col].set_xlabel(ratio_label or "D ratio", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    return fig


# The ratio panel's range: a tenth to ten times, log-spaced.
_RATIO_BINS = np.logspace(-1, 1, 41)
_RATIO_TICKS = (.25, .5, 1, 2, 4)


def _ratio_panel(ax, panel: dict) -> None:
    """Per-track D(long) / D(short) medians on a log axis, with how many
    tracks are confidently slower or faster at the longer timescale."""
    _style_axis(ax)
    ratios = np.asarray(panel.get("ratio_medians", []), dtype=float)
    ratios = ratios[ratios > 0]
    if len(ratios):
        ax.hist(np.clip(ratios, _RATIO_BINS[0], _RATIO_BINS[-1]), bins=_RATIO_BINS, color=_HIST_COLOR,
                edgecolor="white", lw=0.5)
        ax.set_title(f"median ratio {np.median(ratios):.2f}", fontsize=9, loc="left", color=_INK)
        evidence = panel.get("evidence")
        ax.text(
            0.98, 0.97,
            f"{panel.get('n_slower', 0)} slower\n{panel.get('n_faster', 0)} faster\nof {len(ratios)}"
            + (f"\n(P > {evidence:g})" if evidence else ""),
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=_MUTED_INK,
        )
    else:
        ax.set_title("no track long enough", fontsize=9, loc="left", color=_MUTED_INK)
    ax.axvline(1.0, color=_MUTED_INK, lw=0.8, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(_RATIO_BINS[0], _RATIO_BINS[-1])
    ax.set_xticks(_RATIO_TICKS, [f"{t:g}" for t in _RATIO_TICKS])
    ax.minorticks_off()


def plot_d_posteriors(d_grid: np.ndarray, panels: list[dict], title: str | None = None) -> Figure:
    """Every track's posterior next to what the population makes of them:
    one row per panel (`ensemble_panels(..., track_posteriors=True)`).

    Left, a heat map with one row per track, sorted by its posterior
    median: a well-determined track is a short bright streak, a short or
    noisy one a long faint smear. Right, on one density axis: the pooled
    posterior (the tracks' posteriors averaged -- where they put D,
    blurred by each one's uncertainty), the histogram of per-track
    medians, and the deconvolved distribution (that blur removed). The
    summed log posterior -- one D shared by every track -- is far
    narrower than any of them, so it is the marker with its 90% interval
    above the curves, as in `plot_d_ensemble`."""
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
        colorbar = fig.colorbar(image, ax=ax_map, pad=0.01, shrink=0.85)
        colorbar.set_label(r"density per $\log_{10} D$", fontsize=8, color=_MUTED_INK)
        colorbar.ax.tick_params(colors=_MUTED_INK, labelsize=7)
        colorbar.outline.set_visible(False)

        _style_axis(ax)
        medians = np.asarray(panel["medians"], dtype=float)
        medians = np.log10(medians[medians > 0])
        if len(medians):
            ax.hist(medians, bins=bins, density=True, color=_HIST_COLOR, edgecolor="white",
                    lw=0.5, label="per-track medians")
        pooled = _density(panel["pooled"], x)
        deconvolved = _density(panel["deconvolved"], x)
        ax.plot(x, pooled, color=_POOLED_COLOR, lw=2, label="pooled posterior")
        ax.plot(x, deconvolved, color=_DECONVOLVED_COLOR, lw=2, label="deconvolved")
        top = max(pooled.max(), deconvolved.max(), ax.get_ylim()[1])
        low, median, high = (np.log10(v) for v in panel["summed_interval"])
        _interval_marker(ax, low, median, high, top * 1.08, _SUMMED_COLOR)
        ax.plot([], [], "o-", color=_SUMMED_COLOR, lw=2, ms=5, label="summed log posterior (shared D), 90%")
        ax.set_ylim(0, top * 1.18)
        ax.set_ylabel(r"density per $\log_{10} D$", fontsize=8, color=_MUTED_INK)
        ax.set_title(
            f"shared D = {panel['summed_interval'][1]:.3g} µm²/s", fontsize=9, loc="left", color=_INK
        )
        if row == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[-1][0].set_xlim(x_lo, x_hi)
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
    alpha_grid: np.ndarray | None = None,
    alpha_weights: np.ndarray | None = None,
    alpha_interval: tuple[float, float, float] | None = None,
    n_frames: int | None = None,
) -> Figure:
    """One track's posterior over log10 D (and alpha, when computed), with
    its median and the shaded 90% interval. A short track's posterior is
    wide, and that width is the answer, not a defect."""
    has_alpha = alpha_weights is not None
    fig = Figure(figsize=(7.0 if has_alpha else 4.2, 2.8), layout="constrained")
    axes = fig.subplots(1, 2 if has_alpha else 1, squeeze=False)[0]
    specs = [(axes[0], np.log10(d_grid), d_weights, tuple(np.log10(v) for v in d_interval),
              units.mpl_log_label("D_um2_s"))]
    if has_alpha:
        specs.append((axes[1], alpha_grid, alpha_weights, alpha_interval, r"$\alpha$"))
    for ax, x, w, (low, median, high), label in specs:
        _style_axis(ax)
        dens = _density(w, x)
        inside = (x >= low) & (x <= high)
        ax.fill_between(x, dens, where=inside, color=_DECONVOLVED_COLOR, alpha=0.18, lw=0)
        ax.plot(x, dens, color=_DECONVOLVED_COLOR, lw=1.8)
        ax.axvline(median, color=_INK, lw=1)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylim(bottom=0)
    lo, hi = _mass_range(d_weights, specs[0][1])
    axes[0].set_xlim(lo - _LOG_D_PAD, hi + _LOG_D_PAD)
    if has_alpha:
        axes[1].axvline(1.0, color=_MUTED_INK, lw=0.8, ls="--", zorder=0)
    axes[0].set_ylabel("posterior density", fontsize=8, color=_MUTED_INK)
    low, median, high = d_interval
    head = f"track {track_id}" + (f" · {n_frames} frames" if n_frames else "")
    axes[0].set_title(
        f"{head}\nD = {median:.3g} µm²/s [{low:.3g}, {high:.3g}]",
        fontsize=9, loc="left", color=_INK,
    )
    if has_alpha:
        a_low, a_med, a_high = alpha_interval
        axes[1].set_title(f"α = {a_med:.2f} [{a_low:.2f}, {a_high:.2f}]", fontsize=9, loc="left",
                          color=_INK)
    return fig
