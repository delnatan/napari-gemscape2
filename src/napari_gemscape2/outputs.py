"""What "Save analysis" (and `gemscape2 diffusion`) writes into a bundle, built in one place for both.

`analysis_outputs` turns the runs on screen -- the D likelihoods (`diffusion.PosteriorAnalysis`), the α · D fits
(`fbm.FbmAnalysis`) and the spatial classes -- into `diffusion_summary.json`'s record and the tables
`results.write_diffusion_results` writes. Two of those tables are this module's own, the ones made for plotting
outside the widget:

- `populations_table` (`populations.csv`): one row per group of tracks -- "all" (the tracks the filters pass),
  each region class, each spatial class -- with the D population's numbers (the log-normal's median D, spread
  and mean D, each with its interval, and the deconvolved distribution's mode and median) and the alpha
  population's (mean and SD of alpha across tracks, P(alpha < 1), each with its interval, and the shared alpha).
  Concatenated across bundles (each row carries `result_id`), it is the table to compare movies and conditions
  from, with the movie as the replicate.
- `alpha_distributions_table` (`distributions_alpha.csv`): each group's alpha population on the alpha grid,
  the alpha counterpart of `distributions_D.csv`.

Intervals are `_low`/`_high` at the analysis's credible level (`diffusion.LEVEL`, 90%), except the shared
alpha's, a likelihood-ratio interval at the α · D run's level (95%).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl
from diffusionkit.spatial import SpatialAnalysis

from napari_gemscape2.diffusion import (
    LEVEL,
    MSDWindow,
    PosteriorAnalysis,
    analysis_summary,
    analysis_tables,
    ensemble,
    lognormal_summary_of,
)
from napari_gemscape2.fbm import (
    FbmAnalysis,
    alpha_population,
    fbm_record,
    population_summary,
    profile_long_table,
    shared_alpha_summary,
)
from napari_gemscape2.fbm import loglik_long_table as fbm_loglik_table
from napari_gemscape2.spatial import class_groups, class_map, spatial_record

# `populations.csv`'s columns, in order; a group without a read leaves its columns empty.
POPULATION_SCHEMA = {
    "result_id": pl.String,
    "group": pl.String,
    # The D population: the group's fitted tracks, their log-likelihoods combined.
    "n_tracks": pl.Int64,
    "lognormal_D_median_um2_s": pl.Float64,
    "lognormal_D_median_low_um2_s": pl.Float64,
    "lognormal_D_median_high_um2_s": pl.Float64,
    "lognormal_sigma_ln_D": pl.Float64,
    "lognormal_sigma_ln_D_low": pl.Float64,
    "lognormal_sigma_ln_D_high": pl.Float64,
    "lognormal_D_mean_um2_s": pl.Float64,
    "lognormal_D_mean_low_um2_s": pl.Float64,
    "lognormal_D_mean_high_um2_s": pl.Float64,
    # Why the log-normal's numbers lean on its prior's bounds, when they do.
    "lognormal_problem": pl.String,
    "deconvolved_D_mode_um2_s": pl.Float64,
    "deconvolved_D_median_um2_s": pl.Float64,
    "D_floor_median_um2_s": pl.Float64,
    # The alpha population: the group's tracks with an α · D fit (`alpha_n_tracks`).
    "alpha_n_tracks": pl.Int64,
    "alpha_mean": pl.Float64,
    "alpha_mean_low": pl.Float64,
    "alpha_mean_high": pl.Float64,
    "alpha_sd": pl.Float64,
    "alpha_sd_low": pl.Float64,
    "alpha_sd_high": pl.Float64,
    "p_alpha_below_1": pl.Float64,
    "p_alpha_below_1_low": pl.Float64,
    "p_alpha_below_1_high": pl.Float64,
    # The share of the population below the D_ref floor, where its alpha is mostly the smoothness prior's.
    "alpha_share_below_floor": pl.Float64,
    # One alpha for every track: the maximum of their summed profiles, its likelihood-ratio interval and test
    # of alpha = 1. There even when the group has too few tracks for a population.
    "shared_alpha": pl.Float64,
    "shared_alpha_low": pl.Float64,
    "shared_alpha_high": pl.Float64,
    "shared_alpha_p_brownian": pl.Float64,
}


def _d_reads(analysis: Optional[PosteriorAnalysis], ids: Optional[set]) -> dict:
    ens = None if analysis is None else ensemble(analysis, ids)
    if ens is None:
        return {}
    out = {"n_tracks": ens.n_tracks, **lognormal_summary_of(ens.lognormal, analysis.level)}
    grid, weights = analysis.D_grid_um2_s, ens.deconvolved_D
    cdf = np.cumsum(weights) - weights / 2
    out["deconvolved_D_mode_um2_s"] = float(grid[int(np.argmax(weights))])
    out["deconvolved_D_median_um2_s"] = float(np.exp(np.interp(0.5, cdf, np.log(grid))))
    fits = analysis.fits.filter(pl.col("posterior_status") == "ok")
    if ids is not None:
        fits = fits.filter(pl.col("track_id").is_in(list(ids)))
    floors = fits["D_floor_um2_s"].drop_nulls()
    if floors.len():
        out["D_floor_median_um2_s"] = float(floors.median())
    return out


def _alpha_reads(fbm: Optional[FbmAnalysis], ids: Optional[set]) -> dict:
    if fbm is None:
        return {}
    out = {}
    shared = shared_alpha_summary(fbm, ids)
    if shared is not None:
        out.update(alpha_n_tracks=shared["n_tracks"], shared_alpha=shared["alpha"], shared_alpha_low=shared["low"],
                   shared_alpha_high=shared["high"], shared_alpha_p_brownian=shared["p_brownian"])
    pop = alpha_population(fbm, ids)
    if pop is not None:
        summary = population_summary(pop, LEVEL)
        out["alpha_n_tracks"] = summary["n_tracks"]
        for name in ("alpha_mean", "alpha_sd", "p_alpha_below_1"):
            if summary[name] is not None:
                out[name], out[f"{name}_low"], out[f"{name}_high"] = summary[name]
        if summary["share_below_floor"] is not None:
            out["alpha_share_below_floor"] = summary["share_below_floor"][0]
    return out


def populations_table(
    analysis: Optional[PosteriorAnalysis],
    fbm: Optional[FbmAnalysis],
    groups: dict[str, Optional[set]],
    result_id: Optional[str],
) -> Optional[pl.DataFrame]:
    """`populations.csv`: one row per group of `groups` (name: track ids, None for every fitted track) with a D
    or an alpha read, in `POPULATION_SCHEMA`'s columns. The deconvolutions are cached on the analyses: warm
    them off the GUI thread first. None when no group has a read."""
    rows = []
    for name, ids in groups.items():
        reads = {**_d_reads(analysis, ids), **_alpha_reads(fbm, ids)}
        if reads:
            rows.append({"result_id": result_id, "group": name, **reads})
    if not rows:
        return None
    return pl.DataFrame([{c: row.get(c) for c in POPULATION_SCHEMA} for row in rows], schema=POPULATION_SCHEMA)


def alpha_distributions_table(fbm: Optional[FbmAnalysis], groups: dict[str, Optional[set]]) -> Optional[pl.DataFrame]:
    """`distributions_alpha.csv`: each group's alpha population (`fbm.alpha_population`), its margin over
    D_ref on the alpha grid, long by group: `group`, `n_tracks`, `alpha`, `deconvolved` (weights, summing to 1
    over the grid within a group) with `_low`/`_high`, its pointwise band at `diffusion.LEVEL`, and
    `deconvolved_above_floor`, the population above the D_ref floor alone, renormalized. None without a group
    of enough tracks (`fbm.MIN_POPULATION_TRACKS`)."""
    if fbm is None:
        return None
    parts = []
    for name, ids in groups.items():
        pop = alpha_population(fbm, ids)
        if pop is None:
            continue
        lo, hi = pop.alpha_band(LEVEL)
        n = len(pop.alphas)
        parts.append(pl.DataFrame({
            "group": [name] * n,
            "n_tracks": [pop.n_tracks] * n,
            "alpha": pop.alphas,
            "deconvolved": pop.alpha_weights(),
            "deconvolved_low": lo,
            "deconvolved_high": hi,
            "deconvolved_above_floor": pop.alpha_weights(above_floor=True),
        }))
    return pl.concat(parts) if parts else None


def output_groups(ids: Optional[set], by_class: Optional[dict], spatial: Optional[SpatialAnalysis]) -> dict:
    """The groups every saved population is read over: "all" (the tracks the filters pass, `ids`), each region
    class (`by_class`) and each spatial class (`spatial.class_groups`)."""
    return {"all": ids, **(by_class or {}), **({} if spatial is None else class_groups(spatial))}


def analysis_outputs(
    analysis: Optional[PosteriorAnalysis],
    fbm: Optional[FbmAnalysis],
    spatial: Optional[SpatialAnalysis],
    ids: Optional[set],
    by_class: Optional[dict],
    *,
    msd: Optional[MSDWindow],
    result_id: Optional[str],
    pixel_size_um: float,
    frame_shape_px: Optional[tuple[int, int]],
) -> tuple[dict, dict]:
    """`(summary, tables)`: `diffusion_summary.json`'s record of the runs (the caller adds the filters and the
    packages) and `results.write_diffusion_results`' tables, but the per-track summary. Every deconvolution
    they read is computed here if it is not cached yet."""
    summary: dict = {}
    tables: dict = {}
    groups = output_groups(ids, by_class, spatial)
    if analysis is not None:
        summary = analysis_summary(analysis, ids, by_class, msd=msd)
        tables = analysis_tables(analysis, ids, by_class, None if spatial is None else class_groups(spatial))
    if spatial is not None:
        summary["spatial"] = spatial_record(spatial)
        tables["spatial_classes"] = spatial.probabilities()
        if spatial.patterned:
            tables["spatial_class_map"] = class_map(spatial, pixel_size_um, frame_shape_px)
    if fbm is not None:
        summary["alpha_D"] = fbm_record(fbm, ids, by_class)
        tables["loglik_alpha_D"] = fbm_loglik_table(fbm)
        tables["profile_alpha"] = profile_long_table(fbm)
        tables["distributions_alpha"] = alpha_distributions_table(fbm, groups)
    tables["populations"] = populations_table(analysis, fbm, groups, result_id)
    return summary, tables
