"""GUI-free pooling of saved diffusion analyses across bundles (movies): what `gemscape2 pool` does.

Each bundle that `gemscape2 diffusion` (or the widget's Save analysis) wrote holds its tracks' posteriors
over D (`posterior_D.parquet`) and which tracks passed the filters. Pooling reads those back -- nothing is
refitted -- and hands them to diffusionkit's batch layer:

- the D populations (`GridPostBatch.populations`): one per sample (the bundles that are replicates of each
  other), and one per bundle when a sample has several, with the distances between them;
- optionally the ensemble-averaged MSD (`classic.ensemble_msd`) of each sample, recomputed from the tracks,
  with its fit over an explicit number of lags.

Only tracks that passed the filters when each bundle was saved enter either. The posteriors must share one D
grid. The ensemble MSD runs with the exposure treated as 0, as the widget's MSD comparison does: the MSD
estimators have no blur model, so it is labelled as that wherever it shows.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from diffusionkit import Acquisition, Experiment
from diffusionkit.classic import EnsembleMSD, MSDOptions, analyze_experiments, ensemble_msd
from diffusionkit.gridpost import GridPosteriorAnalysis, GridPostBatch, GridPosteriors, cdf_distance
from diffusionkit.gridpost.workflow import FIT_SCHEMA as DK_FIT_SCHEMA

from napari_gemscape2.diffusion import PosteriorAnalysis, restore_analysis, tracks_to_diffusionkit_df
from napari_gemscape2.diffusion_batch import bundle_track_table
from napari_gemscape2.results import load_diffusion_results, load_result
from napari_gemscape2.viewer import layer_units_metadata

OFFSETS = ("provided", "fit")
# The unit of draws the pooled distributions and distances are read from.
POPULATION_DRAWS = 1000


@dataclass(frozen=True)
class PooledBundle:
    """One bundle's saved analysis, restored, and the tracks it was run on."""

    result_id: str
    sample: str
    analysis: PosteriorAnalysis
    passing_ids: Optional[set]  # None: every track passed
    tracks: pl.DataFrame  # `tracks_to_diffusionkit_df`'s table


@dataclass(frozen=True)
class PoolSettings:
    """`[pool]` in the config. The ensemble MSD is off unless asked for, and then its windows are explicit:
    `ensemble_max_lag` is how far each track's MSD is computed, `ensemble_n_points` how many of the averaged
    curve's first lags the fit uses (the window changes D and alpha, so it has no default)."""

    ensemble_msd: bool = False
    ensemble_max_lag: Optional[int] = None
    ensemble_n_points: Optional[int] = None
    ensemble_offset: str = "provided"  # where the localization offset comes from: the tracks' SDs, or the fit's intercept
    n_boot: int = 200

    def __post_init__(self):
        if self.ensemble_offset not in OFFSETS:
            raise ValueError(f"ensemble_offset must be one of {OFFSETS}, got {self.ensemble_offset!r}")
        if self.ensemble_msd:
            if self.ensemble_max_lag is None or self.ensemble_n_points is None:
                raise ValueError("ensemble_msd needs ensemble_max_lag and ensemble_n_points (the fit's lag window is explicit)")
            if not 3 <= self.ensemble_n_points <= self.ensemble_max_lag:
                raise ValueError("need 3 <= ensemble_n_points <= ensemble_max_lag")

    @classmethod
    def names(cls) -> set[str]:
        return {f.name for f in fields(cls)}


def load_pooled_bundle(result_dir: str | Path, sample: Optional[str] = None) -> PooledBundle:
    """Restore the saved analysis of one bundle. Raises ValueError if it has none, or it is stale (its tracks
    differ from the bundle's now); `sample` defaults to the bundle's name."""
    result_dir = Path(result_dir)
    saved = load_diffusion_results(result_dir)
    if saved is None:
        raise ValueError("no saved diffusion analysis here -- run `gemscape2 diffusion` first")
    _points, tracks_df, manifest, _labels, regions = load_result(result_dir)
    params = manifest.get("params", {})
    units = layer_units_metadata(params.get("pixel_size_um"), params.get("dt_s"), result_dir, params.get("exposure_s"))
    track_points = bundle_track_table(tracks_df, units["pixel_size_um"], units["dt_s"], regions)
    tracks = tracks_to_diffusionkit_df(track_points, units["pixel_size_um"], units["dt_s"])
    restored = restore_analysis(tracks, **saved)
    passes = saved["tracks_summary"].filter(pl.col("passes_filters"))["track_id"] if (
        "passes_filters" in saved["tracks_summary"].columns
    ) else None
    ids = None if passes is None else set(passes.to_list())
    return PooledBundle(result_dir.name, sample or result_dir.name, restored.analysis, ids, tracks)


def to_gridpost(analysis: PosteriorAnalysis, ids: Optional[set]) -> GridPosteriorAnalysis:
    """diffusionkit's `GridPosteriorAnalysis` of this analysis's tracks `ids` (None: all): the GUI's column names
    mapped back to diffusionkit's, the posteriors of the fitted ones."""
    fits = analysis.fits if ids is None else analysis.fits.filter(pl.col("track_id").is_in(list(ids)))
    dk_fits = fits.select(
        "track_id", "n_frames",
        pl.lit("posterior_D").alias("model"), pl.lit("grid_posterior").alias("method"),
        pl.col("posterior_status").alias("status"), "message",
        pl.lit("credible_interval").alias("uncertainty_method"),
        pl.col("D_median_um2_s").alias("D_post_median_um2_s"), pl.col("D_low_um2_s").alias("D_post_lo_um2_s"),
        pl.col("D_high_um2_s").alias("D_post_hi_um2_s"), pl.col("D_info_bits").alias("D_post_info_bits"),
        "D_floor_um2_s",
    ).select(DK_FIT_SCHEMA.keys()).cast(DK_FIT_SCHEMA)
    rows = analysis.rows_for(ids)
    return GridPosteriorAnalysis(
        dk_fits, analysis.acquisition, analysis.options,
        GridPosteriors(analysis.fitted_ids[rows], analysis.fitted_frames[rows], analysis.log_post_D[rows]),
    )


def pooled_batch(bundles: list[PooledBundle]) -> GridPostBatch:
    """diffusionkit's batch over the bundles' passing tracks (bundle names must be unique)."""
    names = [b.result_id for b in bundles]
    if len(set(names)) != len(names):
        raise ValueError(f"bundle names must be unique, repeated: {sorted({n for n in names if names.count(n) > 1})}")
    return GridPostBatch.from_analyses(
        {b.result_id: to_gridpost(b.analysis, b.passing_ids) for b in bundles}, {b.result_id: b.sample for b in bundles}
    )


def population_tables(batch: GridPostBatch, level: float, n_samples: int = POPULATION_DRAWS) -> tuple[pl.DataFrame, pl.DataFrame]:
    """`(distributions, distances)` of the batch's D populations.

    distributions: long by `by` ("sample", plus "experiment" when some sample has several) and `group`:
    `n_tracks`, `n_excluded`, `D_um2_s`, `deconvolved` (weights summing to 1 over the grid, the same quantity
    as `distributions_D.csv`) and its pointwise `level` band.
    distances: the W1 distance in ln D between every pair of groups of one kind, as the median of its draws
    with the `level` interval (`gridpost.cdf_distance`); `same_sample` is set for experiment pairs. Two draws
    of one population are still apart, so read a sample pair against the same-sample experiment pairs."""
    kinds = ["sample"] + (["experiment"] if len(batch.acquisitions) > len(set(batch.samples.values())) else [])
    q = [(1 - level) / 2, .5, (1 + level) / 2]
    parts, rows, grid = [], [], np.exp(batch.options.u_D())
    for by in kinds:
        pops = batch.populations(by, n_samples=n_samples)
        for name, pop in pops.items():
            lo, hi = pop.band(level)
            parts.append(pl.DataFrame({
                "by": by, "group": name, "n_tracks": pop.n_tracks, "n_excluded": pop.n_excluded,
                "D_um2_s": grid, "deconvolved": pop.weights, "deconvolved_low": lo, "deconvolved_high": hi,
            }))
        for a, b in combinations(pops, 2):
            lo_d, med, hi_d = np.quantile(cdf_distance(pops[a], pops[b]), q)
            rows.append({"by": by, "a": a, "b": b,
                         "same_sample": batch.samples[a] == batch.samples[b] if by == "experiment" else None,
                         "W1_ln_D_median": float(med), "W1_ln_D_low": float(lo_d), "W1_ln_D_high": float(hi_d)})
    distances = pl.DataFrame(rows, schema={
        "by": pl.String, "a": pl.String, "b": pl.String, "same_sample": pl.Boolean,
        "W1_ln_D_median": pl.Float64, "W1_ln_D_low": pl.Float64, "W1_ln_D_high": pl.Float64})
    return pl.concat(parts), distances


def pooled_ensemble_msd(bundles: list[PooledBundle], settings: PoolSettings) -> tuple[EnsembleMSD, pl.DataFrame]:
    """The ensemble-averaged MSD of each sample, and its fit over `settings.ensemble_n_points` lags.

    Tracks are the bundles' passing ones, with the exposure treated as 0 (blur-free); each track's MSD is
    computed to `ensemble_max_lag`, then averaged, pair-weighted, per sample. Intervals resample tracks."""
    experiments = [
        Experiment(
            b.result_id,
            b.tracks if b.passing_ids is None else b.tracks.filter(pl.col("track_id").is_in(list(b.passing_ids))),
            Acquisition(dt_s=b.analysis.acquisition.dt_s),
            b.sample,
        )
        for b in bundles
    ]
    options = MSDOptions(
        max_lag=settings.ensemble_max_lag, min_frames=max(5, *(b.analysis.min_frames for b in bundles)),
        localization="provided",
    )
    ens = ensemble_msd(analyze_experiments(experiments, options), "sample", n_boot=settings.n_boot)
    return ens, ens.fit(settings.ensemble_n_points, settings.ensemble_offset, level=bundles[0].analysis.level)
