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

The posterior figures (`plot_d_ensemble`, `plot_d_by_length`,
`plot_track_posterior`) follow one color assignment by role, not by series:
the log-normal population is aqua, the deconvolved distribution is blue, and
the shared D (or alpha) is orange. The population is always the tracks' log-likelihoods combined
under a model of it -- never a histogram of their medians or an average of
their posteriors (see `diffusion`). Track-length groups are ordered, so they are steps of one ramp,
light for short tracks and dark for long ones. Every D axis carries the localization floor -- the D at which a track's
motion per frame equals its localization noise -- as a thin dotted line at
the tracks' median floor. It marks accuracy, not resolution: information on
D accumulates over a track's steps (and over the tracks a population pools),
so D below the floor is resolved all the same, but most of each step's
variance there is the localization noise being subtracted, so D leans on
the SDs being calibrated -- a bias more frames do not average away. Hence
a line and no shaded band, which read as a limit D cannot go below. Region classes are separate panels stacked on one
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
    band whose mid is drawn as a line across the plot at that axis's
    value -- the localization floor on a D axis
    (`diffusion.floor_references`)."""
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
    (already log10 on a log10-D axis): a thin dotted line at the band's
    mid (the tracks' median). Its 10-90% spread is left out on purpose:
    a shaded band reads as a region D cannot be resolved in, which it is
    not (see the module docstring)."""
    line = ax.axvline if vertical else ax.axhline
    line(band[1], color=_MUTED_INK, lw=0.8, ls=":", zorder=1, label=label)


def _floor_label(band: tuple[float, float, float]) -> str:
    return f"localization floor {band[1]:.2g} µm²/s · below it, D rests on the SDs"


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
    have mass, or its median localization floor is, rather than the whole grid,
    which spans five decades and would squeeze the data into a sliver."""
    spans = []
    for panel in panels:
        for key in ("lognormal", "deconvolved"):
            spans.append(_mass_range(panel[key], x, tail=0.005))
        spans.append(tuple(np.log10([panel["shared_interval"][0], panel["shared_interval"][2]])))
        if panel.get("floor") is not None:  # always in view: it is what D is read against
            spans.append((np.log10(panel["floor"][1]),) * 2)
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
    is a thin line behind all three."""
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


def plot_d_by_length(panels: list[dict], title: str | None = None) -> Figure:
    """Which tracks each part of the distribution of D comes from: one row
    per panel (`diffusion.length_panels`: `name`, `composition` -- a
    diffusionkit `LengthComposition` -- and `floor`).

    Left, each track-length group's deconvolved population, stacked: each
    group's band is its share of all the tracks (or detections, as the
    composition was weighted), so the stack is a population. Right, each
    group's own population, scaled to its peak, with its tracks and
    detections at the right: short tracks come mostly from fast particles,
    which leave the focal depth within a few frames. A group with few tracks
    is mostly the deconvolution's smoothness prior."""
    n = len(panels)
    fig = Figure(figsize=(8.5, 1.0 + 3.0 * n), layout="constrained")
    axes = fig.subplots(n, 2, squeeze=False, sharex=True, gridspec_kw={"width_ratios": [1, 1.15]})
    x_lo = x_hi = None
    for row, panel in enumerate(panels):
        comp = panel["composition"]
        x = comp.u / np.log(10)
        dx = float(np.mean(np.diff(x)))
        contrib = comp.deconvolved
        lo, hi = _mass_range(contrib.sum(axis=0), x, tail=0.005)
        if panel.get("floor") is not None:
            lo, hi = min(lo, np.log10(panel["floor"][1])), max(hi, np.log10(panel["floor"][1]))
        x_lo = lo if x_lo is None else min(x_lo, lo)
        x_hi = hi if x_hi is None else max(x_hi, hi)
        labels = comp.labels()
        colors = _LENGTH_CMAP(np.linspace(0.15, 1.0, len(labels)))
        ax = axes[row][0]
        _style_axis(ax)
        bottom = np.zeros(len(x))
        for j, label in enumerate(labels):
            upper = bottom + contrib[j] / dx
            ax.fill_between(x, bottom, upper, color=colors[j], lw=0, label=label)
            bottom = upper
        if panel.get("floor") is not None:
            _draw_floor(ax, tuple(np.log10(panel["floor"])))
        ax.set_title(f"{panel['name']} · deconvolved, by track length", fontsize=9, loc="left", color=_INK)
        ax.set_ylabel(f"{comp.weight} per $\\log_{{10}} D$", fontsize=8, color=_MUTED_INK)
        ax.set_ylim(0, bottom.max() * 1.05)
        ax.legend(title="frames", fontsize=7, title_fontsize=7, frameon=False, loc="upper left", reverse=True)

        ax = axes[row][1]
        _style_axis(ax)
        ax.grid(False)
        own = contrib / np.maximum(contrib.max(axis=1, keepdims=True), 1e-300)
        ax.imshow(own, aspect="auto", origin="lower", cmap=_POSTERIOR_CMAP, interpolation="nearest",
                  extent=[x[0] - dx / 2, x[-1] + dx / 2, -0.5, len(labels) - 0.5], vmin=0, vmax=1)
        if panel.get("floor") is not None:
            ax.axvline(np.log10(panel["floor"][1]), color=_INK, lw=0.8, ls=":")
        ax.set_yticks(range(len(labels)), labels)
        for j in range(len(labels)):
            ax.text(1.01, j, f"{comp.n_tracks[j]} / {comp.n_detections[j]}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7, color=_MUTED_INK)
        ax.set_ylabel("track length (frames)", fontsize=8, color=_MUTED_INK)
        ax.set_title("each length's own population (peak = 1) · tracks / detections", fontsize=9, loc="left",
                     color=_INK)
    axes[-1][0].set_xlim(x_lo - _LOG_D_PAD, x_hi + _LOG_D_PAD)
    for ax in axes[-1]:
        ax.set_xlabel(units.mpl_log_label("D_um2_s"), fontsize=9)
    if title is None:
        title = (
            "shared per track: each length group's population, stacked by its share of the tracks"
            if panels[0]["composition"].weight == "tracks"
            else "shared per detection: each group weighed by its frames, so long (often slow) tracks weigh more -- "
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
    ax.set_ylabel("likelihood (normalized)", fontsize=8, color=_MUTED_INK)
    low, mean, high = d_interval
    head = f"track {track_id}" + (f" · {n_frames} frames" if n_frames else "")
    reading = {"low": f"D < {high:.3g} µm²/s (cut at the grid's low edge: an upper bound)",
               "high": f"D > {low:.3g} µm²/s (cut at the grid's high edge: a lower bound)",
               "both": "no information about D (cut at both grid edges)"}.get(d_grid_edge)
    line = reading or f"E[D] = {mean:.3g} µm²/s [{low:.3g}, {high:.3g}]"
    ax.set_title(f"{head}\n{line}", fontsize=9, loc="left", color=_INK)
    return fig


# --- alpha and D_ref (fBm profile fits, `napari_gemscape2.fbm`) -----------

# A track whose alpha interval reaches both ends of the grid: the data do not bound alpha.
_UNINFORMATIVE_COLOR = "#b9b8b3"


def _d_ref_label(tau_ref_s: float) -> str:
    return rf"$\log_{{10}}$ $D_{{ref}}$ at $\tau_{{ref}}$ = {tau_ref_s:g} s ($\mu$m$^2$/s) · not a Brownian D"


def plot_alpha_D(panels: list[dict], max_bars: int = 150, seed: int = 0, title: str | None = None) -> Figure:
    """Every fitted track's alpha_hat against its D_ref, one panel per group
    (`fbm.scatter_panels`: `name`, `fits`, `floor`, `tau_ref_s`), with each
    track's likelihood-ratio intervals as error bars.

    Bars are drawn for a random `max_bars` of a panel's tracks, so a large
    movie stays readable. Tracks whose alpha interval spans the whole grid
    are grey points without bars: they say nothing about alpha. A track that
    only bounds D_ref from above is a grey "<" at its upper bound. Tracks with
    no motion above their noise have no alpha_hat and are only counted. The
    dotted line is the median D_ref floor. Per-track fits side by side -- not
    a population."""
    n = len(panels)
    fig = Figure(figsize=(5.6, 0.6 + 3.4 * n), layout="constrained")
    axes = fig.subplots(n, 1, squeeze=False, sharex=True, sharey=True)[:, 0]
    rng = np.random.default_rng(seed)
    measured = [p["fits"].filter(pl.col("alpha_hat").is_not_null() & (pl.col("D_ref_low_um2_s") > 0))
                for p in panels]
    xs = np.concatenate([m["log10_D_ref_hat"].to_numpy() for m in measured])
    x_lo, x_hi = (np.quantile(xs, [0.005, 0.995]) if len(xs) else (-3.0, 0.0))
    for ax, panel in zip(axes, panels):
        _style_axis(ax)
        fits = panel["fits"]
        fitted = fits.filter(pl.col("alpha_hat").is_not_null())
        bound = fitted.filter(pl.col("D_ref_low_um2_s") == 0)
        ok = fitted.filter(pl.col("D_ref_low_um2_s") > 0)
        blank = (ok["alpha_edge"] == "both").fill_null(False).to_numpy()
        x, y = ok["log10_D_ref_hat"].to_numpy(), ok["alpha_hat"].to_numpy()
        if panel.get("floor") is not None:
            _draw_floor(ax, tuple(np.log10(panel["floor"])), label=f"D_ref floor {panel['floor'][1]:.2g} µm²/s (median)")
            x_lo, x_hi = min(x_lo, np.log10(panel["floor"][1])), max(x_hi, np.log10(panel["floor"][1]))
        ax.axhline(1, color=_MUTED_INK, lw=0.6, ls=(0, (2, 3)), zorder=1)
        ax.scatter(x[blank], y[blank], s=4, color=_UNINFORMATIVE_COLOR, lw=0, zorder=2,
                   label="α interval spans the grid")
        ax.scatter(x[~blank], y[~blank], s=4, color=_DECONVOLVED_COLOR, alpha=0.5, lw=0, zorder=2, label="α̂")
        if bound.height:
            ax.scatter(np.log10(bound["D_ref_high_um2_s"].to_numpy()), bound["alpha_hat"].to_numpy(), s=9,
                       marker=4, color=_UNINFORMATIVE_COLOR, lw=0.6, zorder=2,
                       label="D_ref upper bound only (at its bound)")
        sub = ok.filter(~pl.Series(blank))
        pick = np.arange(sub.height) if sub.height <= max_bars else rng.choice(sub.height, max_bars, replace=False)
        sx, sy = sub["log10_D_ref_hat"].to_numpy()[pick], sub["alpha_hat"].to_numpy()[pick]
        dlo = sx - np.log10(sub["D_ref_low_um2_s"].to_numpy()[pick])
        dhi = np.log10(sub["D_ref_high_um2_s"].to_numpy()[pick]) - sx
        alo = sy - sub["alpha_low"].to_numpy()[pick]
        ahi = sub["alpha_high"].to_numpy()[pick] - sy
        ax.errorbar(sx, sy, xerr=[dlo, dhi], yerr=[alo, ahi], fmt="none", ecolor=_DECONVOLVED_COLOR,
                    elinewidth=0.4, alpha=0.3, zorder=3)
        unidentified = fits.height - fitted.height
        ax.set_title(f"{panel['name']} · {fits.height} tracks"
                     + (f", {bound.height} bounding D_ref from above" if bound.height else "")
                     + (f", {unidentified} without motion above the noise (no α̂)" if unidentified else ""),
                     fontsize=9, loc="left", color=_INK)
        ax.set_ylabel(r"$\hat\alpha$ (maximum likelihood)", fontsize=9)
        ax.set_ylim(0, 2)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right", markerscale=2.5)
    axes[-1].set_xlim(x_lo - _LOG_D_PAD, x_hi + _LOG_D_PAD)
    axes[-1].set_xlabel(_d_ref_label(panels[0]["tau_ref_s"]), fontsize=9)
    fig.suptitle(title or "per-track fits (fBm), likelihood-ratio intervals — not a population",
                 fontsize=9, color=_MUTED_INK, x=0.01, ha="left")
    return fig


def plot_fbm_track(panel: dict) -> Figure:
    """One track's likelihood over (log10 D_ref, alpha) (`fbm.track_panel`), relative
    to its maximum, with the likelihood-ratio region outlined and its profiles in
    alpha (right) and D_ref (top): each interval is where its profile stays above
    the threshold. No prior is involved."""
    fit = panel["fit"]
    fig = Figure(figsize=(5.4, 4.4), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.3], height_ratios=[1.2, 4])
    ax = fig.add_subplot(gs[1, 0])
    top = fig.add_subplot(gs[0, 0], sharex=ax)
    side = fig.add_subplot(gs[1, 1], sharey=ax)
    x, a = panel["log10_D_ref"], panel["alphas"]
    dx, da = x[1] - x[0], a[1] - a[0]
    thr = panel["threshold"]
    ax.imshow(np.exp(panel["loglik"]), origin="lower", aspect="auto", cmap=_POSTERIOR_CMAP,
              interpolation="nearest", vmin=0, vmax=1,
              extent=[x[0] - dx / 2, x[-1] + dx / 2, a[0] - da / 2, a[-1] + da / 2])
    ax.contour(x, a, panel["loglik"], levels=[thr], colors=[_INK], linewidths=0.8)
    if fit["alpha_hat"] is not None:
        ax.plot([fit["log10_D_ref_hat"]], [fit["alpha_hat"]], "o", color=_INK, ms=4, mec="white", mew=0.8)
    ax.axhline(1, color=_MUTED_INK, lw=0.6, ls=(0, (2, 3)))
    floor = fit.get("D_ref_floor_um2_s")
    for axis in (ax, top):
        _style_axis(axis)
        axis.grid(False)
        if floor:
            axis.axvline(np.log10(floor), color=_MUTED_INK, lw=0.8, ls=":")
    if floor and x[0] <= np.log10(floor) <= x[-1]:
        top.text(np.log10(floor), 0.98, " floor", transform=top.get_xaxis_transform(), ha="left", va="top",
                 fontsize=7, color=_MUTED_INK)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 2)
    ax.set_xlabel(_d_ref_label(panel["tau_ref_s"]), fontsize=8)
    ax.set_ylabel(r"$\alpha$", fontsize=9)
    low = 3 * thr
    top.plot(x, np.maximum(panel["profile_D_ref"], low), color=_DECONVOLVED_COLOR, lw=1.6)
    top.axhline(thr, color=_MUTED_INK, lw=0.8, ls="--")
    top.set_ylim(low, 0.3)
    top.set_ylabel("Δ log-lik.", fontsize=7, color=_MUTED_INK)
    top.tick_params(labelbottom=False)
    _style_axis(side)
    side.grid(False)
    side.plot(np.maximum(panel["profile_alpha"], low), a, color=_DECONVOLVED_COLOR, lw=1.6, label="profile")
    side.axvline(thr, color=_MUTED_INK, lw=0.8, ls="--", label=f"{panel['level']:.0%} threshold")
    side.set_xlim(low, 0.3)
    side.tick_params(labelleft=False)
    side.set_xlabel("Δ log-lik.", fontsize=7, color=_MUTED_INK)
    side.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.add_subplot(gs[0, 1]).axis("off")
    if fit["alpha_hat"] is None:
        alpha_text = "no motion above the noise: α not identified"
    else:
        alpha_text = f"α {fit['alpha_hat']:.2f} [{fit['alpha_low']:.2f}, {fit['alpha_high']:.2f}]"
    d_text = (f"D_ref ≤ {fit['D_ref_high_um2_s']:.3g} µm²/s" if not fit["D_ref_low_um2_s"] else
              f"D_ref {fit['D_ref_hat_um2_s']:.3g} µm²/s [{fit['D_ref_low_um2_s']:.3g}, {fit['D_ref_high_um2_s']:.3g}]")
    fig.suptitle(
        f"track {panel['track_id']} · {panel['n_frames']} frames · p(α = 1) {fit['alpha_p_brownian']:.2g}\n"
        f"{alpha_text} · {d_text}",
        fontsize=9, color=_INK, x=0.01, ha="left",
    )
    return fig


def _fmt_interval(v: list[float], digits: int = 2) -> str:
    return f"{v[0]:.{digits}f} [{v[1]:.{digits}f}, {v[2]:.{digits}f}]"


def plot_alpha_population(panels: list[dict]) -> Figure:
    """The population of (alpha, D_ref), one row per group (`fbm.population_panels`):
    left, the fitted distribution g over (log10 D_ref, alpha) with the D_ref
    floor; right, alpha's distribution across tracks with its pointwise band at
    the run's credible level, and alpha's distribution above the floor
    (dashed), where the tracks tell D_ref values apart. Below the floor g's
    alpha is mostly the smoothness prior's continuation of alpha above."""
    n = len(panels)
    fig = Figure(figsize=(7.2, 0.4 + 3.3 * n), layout="constrained")
    rows = fig.subfigures(n, 1, squeeze=False)[:, 0]
    for i, (sub, panel) in enumerate(zip(rows, panels)):
        pop, sm = panel["population"], panel["summary"]
        a, x = pop.alphas, pop.log10_D_ref
        da, dx = a[1] - a[0], x[1] - x[0]
        ax, side = sub.subplots(1, 2, width_ratios=[3, 2], sharey=True)
        for axis in (ax, side):
            _style_axis(axis)
            axis.grid(False)
        ax.imshow(pop.weights / (da * dx), origin="lower", aspect="auto", cmap=_POSTERIOR_CMAP,
                  interpolation="nearest", extent=[x[0] - dx / 2, x[-1] + dx / 2, a[0] - da / 2, a[-1] + da / 2])
        ax.axvline(pop.log10_floor, color=_MUTED_INK, lw=0.8, ls=":")
        ax.text(pop.log10_floor, 1.96, " floor", ha="left", va="top", fontsize=7, color=_MUTED_INK)
        ax.axhline(1, color=_MUTED_INK, lw=0.6, ls=(0, (2, 3)))
        seen = x[pop.weights.sum(axis=0) > 1e-3 * pop.weights.sum(axis=0).max()]
        ax.set_xlim(min(seen.min(), pop.log10_floor) - 2 * _LOG_D_PAD, max(seen.max(), pop.log10_floor) + 2 * _LOG_D_PAD)
        ax.set_ylim(0, 2)
        ax.set_ylabel(r"$\alpha$", fontsize=9)
        ax.set_xlabel(_d_ref_label(panel["tau_ref_s"]), fontsize=8)
        lo, hi = pop.alpha_band(panel["level"])
        side.fill_betweenx(a, lo / da, hi / da, color=_DECONVOLVED_COLOR, alpha=0.18, lw=0,
                           label=f"{panel['level']:.0%} band")
        side.plot(pop.alpha_weights() / da, a, color=_DECONVOLVED_COLOR, lw=1.6, label="all tracks")
        side.plot(pop.alpha_weights(above_floor=True) / da, a, color=_DECONVOLVED_COLOR, lw=1.1, ls="--",
                  label="above the floor")
        shared = panel.get("shared")
        if shared is not None:
            side.axhspan(shared["low"], shared["high"], color=_SHARED_COLOR, alpha=0.18, lw=0)
            side.axhline(shared["alpha"], color=_SHARED_COLOR, lw=1.4, label=f"shared α, {shared['level']:.0%}")
        side.axhline(1, color=_MUTED_INK, lw=0.6, ls=(0, (2, 3)))
        side.set_xlim(left=0)
        side.tick_params(labelleft=False)
        side.set_xlabel("density across tracks", fontsize=7, color=_MUTED_INK)
        if i == 0:
            side.legend(fontsize=6.5, frameon=False, loc="upper right")
        up = sm["above_floor"]
        sub.suptitle(
            f"{panel['name']} · {sm['n_tracks']} tracks\n"
            f"P(α<1) {_fmt_interval(sm['p_alpha_below_1'])} · mean α {_fmt_interval(sm['alpha_mean'])}"
            f" · SD of α {_fmt_interval(sm['alpha_sd'])}\n"
            f"above the D_ref floor: P(α<1) {_fmt_interval(up['p_alpha_below_1'])} · mean α "
            f"{_fmt_interval(up['alpha_mean'])} · {sm['share_below_floor'][0]:.0%} of the population below it"
            + ("" if panel.get("shared") is None else
               f"\nshared α {panel['shared']['alpha']:.2f} [{panel['shared']['low']:.2f}, {panel['shared']['high']:.2f}]"
               f" (one α for every track)"),
            fontsize=8, x=0.01, ha="left", color=_INK,
        )
    fig.suptitle("population of α and D_ref: 2D deconvolution of the tracks' joint likelihoods (fBm)",
                 fontsize=9, color=_MUTED_INK, x=0.01, ha="left")
    return fig
