"""GUI-free pooling of saved diffusion analyses across bundles (movies): what `gemscape2 pool` and the
experiment list's "Pool analyses…" do, through `run_pooling` and `write_pool`.

Each bundle that `gemscape2 diffusion` (or the widget's Save analysis) wrote holds its tracks' posteriors
over D (`loglik_D.parquet`) and which tracks passed the filters. Pooling reads those back -- nothing is
refitted -- and hands them to diffusionkit's batch layer:

- the D populations (`GridPostBatch.populations`): one per sample (the bundles that are replicates of each
  other), and one per bundle when a sample has several, with the distances between them;
- optionally the ensemble-averaged MSD (`classic.ensemble_msd`) of each sample, recomputed from the tracks,
  with D and alpha each fitted over an explicit number of lags.

Only tracks that passed the filters when each bundle was saved enter either. The posteriors must share one D
grid. The ensemble MSD runs with the exposure treated as 0, as the widget's MSD comparison does: the MSD
estimators have no blur model, so it is labelled as that wherever it shows.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from itertools import combinations
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import polars as pl
from diffusionkit import Acquisition, Experiment
from diffusionkit.classic import EnsembleMSD
from diffusionkit.gridpost import GridLikelihoods, GridPosteriorAnalysis, GridPostBatch, cdf_distance
from diffusionkit.gridpost.workflow import FIT_SCHEMA as DK_FIT_SCHEMA

from napari_gemscape2.batch import _relative
from napari_gemscape2.diffusion import (
    ENSEMBLE_MSD_BOOT,
    ENSEMBLE_MSD_OFFSETS,
    PosteriorAnalysis,
    ensemble_msd_blur_free,
    ensemble_msd_fits,
    grid_record,
    restore_analysis,
    tracks_to_diffusionkit_df,
)
from napari_gemscape2.diffusion_batch import bundle_track_table
from napari_gemscape2.results import load_diffusion_results, load_result, package_provenance, write_pooled_results
from napari_gemscape2.viewer import layer_units_metadata

OFFSETS = ENSEMBLE_MSD_OFFSETS
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
    curve's first lags the linear fit (D) uses -- the window changes D, so it has no default -- and
    `ensemble_alpha_points` how many the log-log fit (alpha) uses, the whole curve when left out."""

    ensemble_msd: bool = False
    ensemble_max_lag: Optional[int] = None
    ensemble_n_points: Optional[int] = None
    ensemble_alpha_points: Optional[int] = None
    ensemble_offset: str = "provided"  # where the localization offset comes from: the tracks' SDs, or the fit's intercept
    n_boot: int = ENSEMBLE_MSD_BOOT

    def __post_init__(self):
        if self.ensemble_offset not in OFFSETS:
            raise ValueError(f"ensemble_offset must be one of {OFFSETS}, got {self.ensemble_offset!r}")
        if self.ensemble_msd:
            if self.ensemble_max_lag is None or self.ensemble_n_points is None:
                raise ValueError("ensemble_msd needs ensemble_max_lag and ensemble_n_points (the fit's lag window is explicit)")
            if not 3 <= self.ensemble_n_points <= self.ensemble_max_lag:
                raise ValueError("need 3 <= ensemble_n_points <= ensemble_max_lag")
            if self.ensemble_alpha_points is None:
                # Recorded resolved, so the summary and a written config say the window used.
                object.__setattr__(self, "ensemble_alpha_points", self.ensemble_max_lag)
            if not 3 <= self.ensemble_alpha_points <= self.ensemble_max_lag:
                raise ValueError("need 3 <= ensemble_alpha_points <= ensemble_max_lag")

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


def n_pooled(bundle: PooledBundle) -> int:
    """How many of the bundle's fitted tracks enter the pooling: those that passed its saved filters."""
    fitted = bundle.analysis.fitted_ids
    return len(fitted) if bundle.passing_ids is None else len(set(fitted.tolist()) & bundle.passing_ids)


def guess_sample(name: str) -> str:
    """A bundle name with a trailing replicate number taken off ("wt_2" -> "wt", "mut-03" -> "mut"); the name
    itself when nothing is left. A starting guess to edit, never applied unasked."""
    return re.sub(r"[ _.\-]*\d+$", "", name) or name


def to_gridpost(analysis: PosteriorAnalysis, ids: Optional[set]) -> GridPosteriorAnalysis:
    """diffusionkit's `GridPosteriorAnalysis` of this analysis's tracks `ids` (None: all): the GUI's column names
    mapped back to diffusionkit's, the log-likelihoods of the fitted ones."""
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
        GridLikelihoods(analysis.fitted_ids[rows], analysis.fitted_frames[rows], analysis.loglik_D[rows]),
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
    as `distributions_D.csv`) and its pointwise `level` band, and `cumulative` (the CDF) with its own band --
    a CDF's band is not the running sum of the density's, so it is drawn from the draws too.
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
            c_lo, c_hi = pop.band(level, cumulative=True)
            parts.append(pl.DataFrame({
                "by": by, "group": name, "n_tracks": pop.n_tracks, "n_excluded": pop.n_excluded,
                "D_um2_s": grid, "deconvolved": pop.weights, "deconvolved_low": lo, "deconvolved_high": hi,
                "cumulative": np.cumsum(pop.weights), "cumulative_low": c_lo, "cumulative_high": c_hi,
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
    """The ensemble-averaged MSD of each sample, and its fits: D over `settings.ensemble_n_points` lags, alpha
    over `settings.ensemble_alpha_points` (a sample whose curve stops short gets `insufficient_data` rows).

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
    min_frames = max(b.analysis.min_frames for b in bundles)
    ens = ensemble_msd_blur_free(experiments, min_frames, settings.ensemble_max_lag, settings.n_boot)
    fits = ensemble_msd_fits(
        ens, settings.ensemble_n_points, settings.ensemble_offset, bundles[0].analysis.level,
        settings.ensemble_alpha_points,
    )
    return ens, fits


@dataclass(frozen=True)
class PoolResult:
    """Everything one pooling found: what `write_pool` saves and the pooling dialog draws."""

    distributions: pl.DataFrame  # `population_tables`
    distances: pl.DataFrame
    summary: dict  # pooled_summary.json
    ensemble: Optional[EnsembleMSD] = None  # with `settings.ensemble_msd`
    ensemble_fits: Optional[pl.DataFrame] = None


def run_pooling(
    bundles: list[PooledBundle], settings: PoolSettings, progress: Optional[Callable[[str], None]] = None
) -> PoolResult:
    """Pool `bundles` (from `load_pooled_bundle`) by their samples: the D populations and distances, and the
    ensemble MSD when `settings` ask for it. `progress` is told each stage. Raises ValueError for bundles that
    cannot be pooled (repeated names, different D grids)."""
    report = progress or (lambda _stage: None)
    report("deconvolving the D populations")
    level = bundles[0].analysis.level
    distributions, distances = population_tables(pooled_batch(bundles), level)
    ens = fits = None
    if settings.ensemble_msd:
        report("averaging the MSD over tracks")
        ens, fits = pooled_ensemble_msd(bundles, settings)

    import diffusionkit
    import napari_gemscape2

    samples: dict[str, list[str]] = {}
    for b in bundles:
        samples.setdefault(b.sample, []).append(b.result_id)
    summary = {
        "analysis": "diffusionkit.gridpost populations pooled over saved posteriors (flat prior in ln D)",
        "samples": samples,
        "tracks_pooled": {b.result_id: n_pooled(b) for b in bundles},
        "credible_level": level,
        "grid": grid_record(bundles[0].analysis.options),
        "tracks_selection": "those that passed each bundle's saved filters",
        "ensemble_msd": (
            {**{k: v for k, v in vars(settings).items() if k != "ensemble_msd"},
             "exposure": "treated as 0 (no blur model)", "weight": "pairs", "resample": "track"}
            if settings.ensemble_msd
            else None
        ),
        "packages": package_provenance(diffusionkit, napari_gemscape2),
    }
    return PoolResult(distributions, distances, summary, ens, fits)


def write_pool(out_dir: str | Path, result: PoolResult) -> None:
    """`result`'s tables and summary to `out_dir` (`results.write_pooled_results`)."""
    write_pooled_results(
        out_dir,
        distributions_D=result.distributions,
        distances_D=result.distances,
        summary=result.summary,
        ensemble_msd=None if result.ensemble is None else result.ensemble.curves,
        ensemble_msd_fits=result.ensemble_fits,
    )


def write_pool_config(
    config_path: Path,
    *,
    results_root: Path,
    inputs: list[tuple[Path, str, str]],
    settings: PoolSettings,
    output: Optional[Path] = None,
) -> None:
    """The TOML config `gemscape2 pool` re-runs this pooling from: `inputs` are `(image_path, result_id,
    sample)`, `output` the pooled folder (None: `<results_root>/pooled`). Paths are relative to the config's
    own folder, as in `batch.write_batch_config`."""
    base = config_path.parent
    s = json.dumps  # a JSON string is a valid TOML basic string
    lines = [
        "# Written by the experiment list's \"Pool analyses…\"; re-run with",
        "#   gemscape2 pool <this file>",
        f"results_root = {s(_relative(results_root, base))}",
        "",
        "[pool]",
    ]
    if output is not None:
        lines.append(f"output = {s(_relative(output, base))}")
    lines.append(f"ensemble_msd = {'true' if settings.ensemble_msd else 'false'}")
    if settings.ensemble_msd:
        lines += [
            f"ensemble_max_lag = {settings.ensemble_max_lag}",
            f"ensemble_n_points = {settings.ensemble_n_points}",
            f"ensemble_alpha_points = {settings.ensemble_alpha_points}",
            f"ensemble_offset = {s(settings.ensemble_offset)}",
            f"n_boot = {settings.n_boot}",
        ]
    for image_path, result_id, sample in inputs:
        lines += [
            "",
            "[[inputs]]",
            f"path = {s(_relative(image_path, base))}",
            f"result_id = {s(result_id)}",
            f"sample = {s(sample)}",
        ]
    config_path.write_text("\n".join(lines) + "\n")
