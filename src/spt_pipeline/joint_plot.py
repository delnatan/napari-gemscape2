"""Generic 2D scatter+KDE joint-distribution plot for any pair of per-track
properties. diffusionkit ships this exact diagnostic for D vs alpha
(`bayes.plot_K_joint`), but hardcodes a log10 transform on the D
column and always draws an alpha=1 Brownian reference line -- both specific
to that one comparison. This generalizes it to arbitrary columns/labels/log
scaling so any two trajectory properties (classical vs Bayesian D, alpha,
r2, track length, anisotropy epsilon, ...) can be compared the same way.

Axis labels default to `spt_pipeline.units.mpl_label(column)` rather than
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
from matplotlib.figure import Figure

from spt_pipeline import units
from spt_pipeline.diffusion import resolved_mle_rows


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
) -> Figure:
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

    g = sns.JointGrid(data=pdf, x="x", y="y", height=6, ratio=4)
    sns.kdeplot(
        data=pdf, x="x", y="y", ax=g.ax_joint,
        fill=True, cmap="Blues", alpha=0.6, thresh=0.05, levels=12, zorder=0,
    )
    sns.scatterplot(
        data=pdf, x="x", y="y", ax=g.ax_joint,
        s=18, alpha=0.6, color="0.15", edgecolor="none", zorder=1,
    )
    g.ax_marg_x.hist(pdf["x"], bins=30, color="steelblue", edgecolor="white")
    g.ax_marg_y.hist(pdf["y"], bins=30, color="steelblue", edgecolor="white", orientation="horizontal")

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
    g.ax_marg_x.set_title(subtitle, fontsize=10, loc="left")
    return g.figure


def numeric_columns(df: pl.DataFrame, exclude: tuple[str, ...] = ("track_id",)) -> list[str]:
    return [c for c, dtype in zip(df.columns, df.dtypes) if c not in exclude and dtype.is_numeric()]


def _join_groups(rows: pl.DataFrame, groups: pl.DataFrame | None) -> tuple[pl.DataFrame, str | None]:
    """`rows` with a `group` column joined on, and "group" as the seaborn
    hue when it names more than one group -- else `rows` as given and no
    hue, since one group is the same as none."""
    if groups is None:
        return rows, None
    rows = rows.join(groups.select("track_id", "group"), on="track_id", how="left")
    return rows, ("group" if rows["group"].drop_nulls().n_unique() > 1 else None)


def plot_d_z_joint(
    mle_rows: pl.DataFrame,
    summary: dict,
    title: str | None = None,
    groups: pl.DataFrame | None = None,
) -> Figure:
    """The Brownian-MLE population view: log10 D (x) against the
    non-Brownian score z (y), one point per track that resolved motion.

    What makes it more than `plot_property_joint`:

      - the z marginal is a density with the N(0,1) curve over it -- what
        z would look like if every track were Brownian with correctly
        calibrated localization SDs -- and the joint axis carries the
        ±1.96 band, so "how many fall outside by chance" is visible;
      - mean z ± its standard error is drawn on the marginal and written
        in the title, since z is read at the population level (one short
        track's z says little; a shift of the mean across ~100 does);
      - `unresolved` tracks (D̂ = 0) have no log D and no z, so they can't
        be plotted; the title counts them, with the median of their upper
        limits, rather than letting them vanish.

    `mle_rows` are the `brownian_mle` rows of `ClassicAnalysis.fits`;
    `summary` is `diffusion.summarize_mle` over the same rows.

    `groups` (`track_id`, `group`), when it names more than one group --
    the ROIs a run was split into -- colors the scatter and the D marginal
    by group, so whether the regions' populations separate is read off
    the same figure. The z marginal and its N(0,1) reference stay pooled:
    the reference is the same for every group.
    """
    resolved, hue = _join_groups(
        mle_rows.filter(pl.col("z_nonbrownian").is_not_null() & (pl.col("D_um2_s") > 0)), groups
    )
    x = np.log10(resolved["D_um2_s"].to_numpy().astype(float))
    y = resolved["z_nonbrownian"].to_numpy().astype(float)
    pdf = pd.DataFrame({"x": x, "y": y})
    if hue is not None:
        pdf["group"] = resolved["group"].fill_null("(none)").to_list()

    g = sns.JointGrid(data=pdf, x="x", y="y", height=6, ratio=4)
    # Recessive references first, so the data sits on top of them.
    g.ax_joint.axhspan(-1.96, 1.96, color="0.95", zorder=0, lw=0)
    for level in (-1.96, 1.96):
        g.ax_joint.axhline(level, color="0.55", lw=0.8, ls="--", zorder=0)
    g.ax_joint.axhline(0.0, color="0.55", lw=0.8, zorder=0)
    if len(pdf) >= 3 and pdf["x"].nunique() > 1 and pdf["y"].nunique() > 1:
        sns.kdeplot(
            data=pdf, x="x", y="y", ax=g.ax_joint,
            fill=True, cmap="Blues", alpha=0.6, thresh=0.05, levels=12, zorder=1,
        )
    if hue is None:
        sns.scatterplot(
            data=pdf, x="x", y="y", ax=g.ax_joint,
            s=18, alpha=0.6, color="0.15", edgecolor="none", zorder=2,
        )
        g.ax_marg_x.hist(pdf["x"], bins=30, color="steelblue", edgecolor="white")
    else:
        sns.scatterplot(
            data=pdf, x="x", y="y", hue=hue, ax=g.ax_joint,
            s=18, alpha=0.75, edgecolor="none", zorder=2,
        )
        sns.histplot(
            data=pdf, x="x", hue=hue, ax=g.ax_marg_x, bins=30,
            element="step", fill=False, legend=False,
        )
        g.ax_marg_x.set_xlabel("")
        g.ax_marg_x.set_ylabel("")
        g.ax_joint.legend(title="ROI", fontsize=8, title_fontsize=8, loc="lower right")

    # z is shown whole (it is a score, not a quantity with outliers to
    # trim), with the axis never narrower than ±4 so the N(0,1) reference
    # reads the same from one run to the next.
    z_lo = min(-4.0, float(np.min(y)) - 0.25) if len(y) else -4.0
    z_hi = max(4.0, float(np.max(y)) + 0.25) if len(y) else 4.0
    bins = np.linspace(z_lo, z_hi, 41)
    g.ax_marg_y.hist(
        pdf["y"], bins=bins, density=True, color="steelblue", edgecolor="white",
        orientation="horizontal",
    )
    grid = np.linspace(z_lo, z_hi, 400)
    g.ax_marg_y.plot(np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi), grid, color="0.2", lw=1.2)
    mean_z, se_z = summary.get("mean_z"), summary.get("se_mean_z")
    if mean_z is not None:
        g.ax_marg_y.axhline(mean_z, color="0.2", lw=1.2)
        if se_z is not None:
            g.ax_marg_y.axhspan(mean_z - se_z, mean_z + se_z, color="0.2", alpha=0.25, lw=0)
    g.ax_joint.set_ylim(z_lo, z_hi)

    g.ax_joint.set_xlabel(units.mpl_log_label("D_um2_s") + " — Brownian MLE")
    g.ax_joint.set_ylabel(r"non-Brownian score $z$ (0 if Brownian; $-$ confined, $+$ directed)")

    lines = [title or r"Brownian MLE: $D$ vs. $z$"]
    stats = f"n = {len(pdf)}"
    if mean_z is not None:
        stats += f" · mean z = {mean_z:+.2f}"
        if se_z is not None:
            stats += f" ± {se_z:.2f} (SE)"
    if summary.get("sd_z") is not None:
        stats += f" · SD {summary['sd_z']:.2f}"
    if summary.get("frac_abs_z_gt_1_96") is not None:
        stats += f" · |z| > 1.96: {100 * summary['frac_abs_z_gt_1_96']:.1f}% (5% if Brownian)"
    lines.append(stats)
    n_unresolved = summary.get("n_unresolved", 0)
    if n_unresolved:
        upper = summary.get("unresolved_D_upper_median_um2_s")
        note = f"{n_unresolved} unresolved (D̂ = 0, no z) not shown"
        if upper is not None:
            note += f"; median upper limit D < {upper:.3g} µm²/s"
        lines.append(note)
    g.ax_marg_x.set_title("\n".join(lines), fontsize=9, loc="left")
    g.ax_marg_y.set_xlabel("density", fontsize=8)
    return g.figure


def plot_d_histogram(
    mle_rows: pl.DataFrame,
    summary: dict,
    title: str | None = None,
    groups: pl.DataFrame | None = None,
) -> Figure:
    """The routine read of a Brownian MLE run: the per-track D
    distribution as a histogram of log10 D.

    Log bins because D spans decades across a field of tracks -- on a
    linear axis it is one spike at the low end. The median and the
    interquartile range (`summary`, from `diffusion.summarize_mle`) are
    drawn on it and written in the title. `unresolved` tracks (D̂ = 0)
    have no log D, so they can't be binned; the title counts them with
    the median of their upper limits instead of letting them vanish.

    `groups` (`track_id`, `group`), when it names more than one group --
    the ROIs a run was split into -- draws one step histogram per group
    on shared bins, so whether the regions' D distributions separate is
    read off the same axes.
    """
    resolved, hue = _join_groups(resolved_mle_rows(mle_rows), groups)
    x = np.log10(resolved["D_um2_s"].to_numpy().astype(float))

    fig = Figure(figsize=(6.5, 4.2), layout="constrained")
    ax = fig.add_subplot()
    if len(x):
        bins = np.histogram_bin_edges(x, bins="fd") if len(x) > 3 else 10
        if not np.isscalar(bins) and len(bins) - 1 > 60:
            bins = 60
        if hue is None:
            ax.hist(x, bins=bins, color="steelblue", edgecolor="white")
        else:
            pdf = pd.DataFrame({"x": x, "group": resolved["group"].fill_null("(none)").to_list()})
            sns.histplot(data=pdf, x="x", hue="group", ax=ax, bins=bins, element="step", fill=False)
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title("ROI")
    median = summary.get("median_D_um2_s")
    q25, q75 = summary.get("q25_D_um2_s"), summary.get("q75_D_um2_s")
    if q25 and q75:
        ax.axvspan(np.log10(q25), np.log10(q75), color="0.2", alpha=0.12, lw=0, zorder=0)
    if median:
        ax.axvline(np.log10(median), color="0.2", lw=1.2)
    ax.set_xlabel(units.mpl_log_label("D_um2_s") + " — Brownian MLE")
    ax.set_ylabel("tracks")

    lines = [title or r"Brownian MLE: per-track $D$"]
    stats = f"n = {len(x)}"
    if median is not None:
        stats += f" · median D = {median:.3g} µm²/s"
        if q25 is not None and q75 is not None:
            stats += f" (IQR {q25:.3g}–{q75:.3g})"
    lines.append(stats)
    n_unresolved = summary.get("n_unresolved", 0)
    if n_unresolved:
        upper = summary.get("unresolved_D_upper_median_um2_s")
        note = f"{n_unresolved} unresolved (D̂ = 0) not shown"
        if upper is not None:
            note += f"; median upper limit D < {upper:.3g} µm²/s"
        lines.append(note)
    ax.set_title("\n".join(lines), fontsize=9, loc="left")
    return fig
