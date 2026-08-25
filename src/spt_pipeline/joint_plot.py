"""Generic 2D scatter+KDE joint-distribution plot for any pair of per-track
properties. diffusionkit ships this exact diagnostic for D vs alpha
(`bayes.plot_D_alpha_joint`), but hardcodes a log10 transform on the D
column and always draws an alpha=1 Brownian reference line -- both specific
to that one comparison. This generalizes it to arbitrary columns/labels/log
scaling so any two trajectory properties (classical vs Bayesian D, alpha,
r2, track length, anisotropy epsilon, ...) can be compared the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure


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

    xlabel = x_label or x_col
    ylabel = y_label or y_col
    g.ax_joint.set_xlabel(f"log10({xlabel})" if log_x else xlabel)
    g.ax_joint.set_ylabel(f"log10({ylabel})" if log_y else ylabel)

    subtitle = f"{title or f'{ylabel} vs {xlabel}'} (r={r:.2f}, n={len(x)}"
    if n_dropped:
        pct = int(100 * (display_quantiles[1] - display_quantiles[0]))
        subtitle += f", {n_dropped} outside {pct}% display range"
    subtitle += ")"
    g.ax_marg_x.set_title(subtitle, fontsize=10, loc="left")
    return g.figure


def numeric_columns(df: pl.DataFrame, exclude: tuple[str, ...] = ("track_id",)) -> list[str]:
    return [c for c, dtype in zip(df.columns, df.dtypes) if c not in exclude and dtype.is_numeric()]
