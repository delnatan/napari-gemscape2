"""GUI-free diffusion analysis: this project's tracks in, diffusionkit's
grid posteriors out, plus the tables the widget shows and the bundle saves.

`tracks_to_diffusionkit_df` is the bridge. spotsolve's per-localization
position error is `se_x`/`se_y` (the CRLB from the fit's Fisher
information); diffusionkit calls the same quantity `sigma_x_um`/
`sigma_y_um`. Only the name and the unit differ, so it is renamed and
scaled here rather than duplicated upstream.

The analysis is `diffusionkit.gridpost`: for each track, the exact Gaussian
displacement likelihood evaluated on a grid in ln D, with the camera
exposure's motion blur modelled. `analyze_posteriors`
is diffusionkit's own `gridpost.analyze_tracks`, asked to keep each track's
log-likelihood on the D grid (what every population read is built from and
what the bundle saves), and every grid comes from the one `GridPostOptions`
the run is given -- the same object a diffusionkit script would pass, so the
two agree number for number. The D grid's range is the flat prior's support:
a posterior cut by an edge is flagged in `D_grid_edge` ("low": read the
track's numbers as upper bounds; "high": as lower bounds; "both": no
information).

Tracks are combined by adding their log-likelihoods under a model of the
population (diffusionkit's docs/gridpost.md, "Combining tracks"), never by
averaging their posteriors or histogramming their medians:

  - **per track** (no pooling): one point estimate, the posterior mean
    E[D] (`D_mean`, the summary a grid edge barely moves), and the
    equal-tailed 90% interval (`D_low`/`D_high` = the 5% and 95% quantiles).
    E[D] leans high for short tracks under the flat prior, so the column is
    not averaged: averages come from the population models below, and
    `partially_pooled_table` gives each track's E[D] under the population;
  - **shared** (complete pooling, `gridpost.fit_shared_D`): one D for every
    track. When the tracks differ it lands near their mean D, with an
    interval far too narrow;
  - **log-normal** (partial pooling, `gridpost.fit_lognormal`): ln D ~
    N(mu, sigma) across tracks -- the population's median D, its spread
    sigma in ln D and its mean D, each with an interval. sigma near 0 means
    one shared D describes the tracks;
  - **deconvolved** (partial pooling, any shape, `gridpost.deconvolve`): a
    smooth log density whose smoothness the data choose (Laplace evidence),
    with a pointwise band from posterior draws. A second view, against
    which the log-normal's shape is checked: two modes here mean the
    log-normal's numbers describe the wrong shape;
  - **by track length**: `gridpost.by_track_length`, the tracks' own
    posteriors (unpooled) and their posteriors under the deconvolved
    distribution (partially pooled) split into track-length groups, per
    track or per detection. Fast particles leave the focal depth within a
    few frames, so short tracks come mostly from fast particles and long
    tracks from slow ones;
  - **localization floor** (`D_floor_um2_s`, diffusionkit's
    `posterior.localization_floor`): the D at which a track's motion per
    frame equals its localization noise. Every D axis shows it as a
    reference scale (its median over the tracks, and their 10-90% band),
    not as a threshold anything is classified by.

There is no per-track shape metric (alpha, radius of gyration,
straightness, ...): short tracks cannot pin one down. alpha stays in the
MSD comparison (`diffusionkit.classic`), which is labelled as such.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np
import polars as pl
from diffusionkit import Acquisition, Experiment
from diffusionkit.classic import (
    EnsembleMSD,
    MSDOptions,
    analyze_experiments,
    compute_msd,
    ensemble_msd,
    fit_brownian_msd,
    fit_linear_msd,
    fit_loglog_msd,
    fit_windows,
    window_lags,
)
from diffusionkit.classic import analyze_tracks as analyze_msd
from diffusionkit.classic.batch import FIT_SCHEMA as ENSEMBLE_FIT_SCHEMA
from diffusionkit.gridpost import GridLikelihoods, GridPostOptions, LengthComposition, by_track_length
from diffusionkit.gridpost import analyze_tracks as dk_analyze_tracks
from diffusionkit.gridpost import deconvolve as dk_deconvolve
from diffusionkit.gridpost import lognormal as dk_lognormal

from napari_gemscape2.pipeline import filter_mask

# The credible-interval mass: `_low`/`_high` are its (1 - LEVEL)/2 and
# (1 + LEVEL)/2 quantiles, 5% and 95%.
LEVEL = 0.9
# diffusionkit's own hard minimum: 3 frames give the 2 displacements the
# whitening needs.
MIN_FRAMES = 3

# `GridPostOptions`' grid fields, the ones settable here (the widget's Grid
# section, the CLI's `[diffusion] grid`). Unset ones are diffusionkit's
# defaults; whichever were used are saved (`analysis_summary`'s "grid").
GRID_FIELDS = ("D_min_um2_s", "D_max_um2_s", "n_D")


def posterior_options(min_frames: int = MIN_FRAMES, grid: Optional[dict] = None) -> GridPostOptions:
    """The `GridPostOptions` a run here uses: `grid` holds any of
    `GRID_FIELDS`, and the credible level is fixed at `LEVEL`. Raises
    ValueError for an unknown key or an invalid grid."""
    unknown = set(grid or {}) - set(GRID_FIELDS)
    if unknown:
        raise ValueError(f"unknown grid settings: {', '.join(sorted(unknown))}")
    return GridPostOptions(min_frames=min_frames, level=LEVEL, **(grid or {}))


def grid_record(options: GridPostOptions) -> dict:
    """`options`' grid, for `diffusion_summary.json` (and back through
    `posterior_options(grid=...)`)."""
    return {name: getattr(options, name) for name in GRID_FIELDS}


def tracks_to_diffusionkit_df(tracks_df: pl.DataFrame, pixel_size_um: float, dt_s: float) -> pl.DataFrame:
    tracks = tracks_df.select(
        pl.col("track_id").cast(pl.Int64),
        pl.col("frame").cast(pl.Int64),
        (pl.col("frame") * dt_s).alias("t_s"),
        (pl.col("x") * pixel_size_um).alias("x_um"),
        (pl.col("y") * pixel_size_um).alias("y_um"),
        (pl.col("se_x") * pixel_size_um).alias("sigma_x_um"),
        (pl.col("se_y") * pixel_size_um).alias("sigma_y_um"),
    ).sort(["track_id", "frame"])
    track_lengths = tracks.group_by("track_id").agg(pl.len().alias("track_length"))
    return tracks.join(track_lengths, on="track_id", how="left").sort(["track_id", "frame"])


# What a track's posterior is computed from (`analyze_posteriors`): its
# frames, times and positions with their localization errors, all in the
# physical units the pixel size and frame interval give them.
_FINGERPRINT_COLUMNS = ("track_id", "frame", "t_s", "x_um", "y_um", "sigma_x_um", "sigma_y_um")


def tracks_fingerprint(tracks: pl.DataFrame) -> str:
    """SHA-256 of `tracks_to_diffusionkit_df`'s table, independent of row
    order: what a saved analysis records it was fitted on, so reopening
    the bundle can tell whether the tracks on screen are still those
    tracks (a re-link renumbers them; a new pixel size rescales them)."""
    ordered = tracks.select(_FINGERPRINT_COLUMNS).sort("track_id", "frame")
    digest = hashlib.sha256()
    for name in _FINGERPRINT_COLUMNS:
        dtype = np.int64 if name in ("track_id", "frame") else np.float64
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(ordered[name].to_numpy(), dtype=dtype).tobytes())
    return digest.hexdigest()


# --- the per-track table ------------------------------------------------

# Per-vertex columns that are position, identity, or already per-track --
# nothing to summarize. `track_length` is taken from the diffusionkit table
# instead (guaranteed present there), and y/x become the centroid.
# `loc_id` is a detection's serial number: its min/mean/max are three
# columns of pure noise in a table that already runs past fifty.
# `flags` is a bitmask: its min/mean/max are not quantities.
_QC_SKIP_COLUMNS = frozenset(
    {"track_id", "loc_id", "frame", "y", "x", "track_length", "region", "cell", "flags"}
)

# How each genuinely per-point column is collapsed to one number per track.
# min/max are what actually replaced the old per-point "Data Explorer": its
# rule was "keep a track only if EVERY one of its points passes this range",
# which is `col_min >= lo and col_max <= hi` -- the same cut, expressed as a
# track property, so it can sit in the same filter panel as `D_mean_um2_s` and be
# read against the same table. The mean is the plain descriptive statistic
# the min/max pair does not imply, and the one to filter on when the
# question is about the track's typical quality rather than its worst point.
_QC_AGGREGATIONS = (
    ("min", lambda col: pl.col(col).min()),
    ("mean", lambda col: pl.col(col).mean()),
    ("max", lambda col: pl.col(col).max()),
)


def qc_aggregate_table(tracks_df_px: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Every per-vertex detection-quality column on the Tracks layer
    (`flux`, `se_y`/`se_x`, `bg`, `fit_sigma`, ...), collapsed to one row
    per track, plus the names of the columns that produced aggregates.

    Columns that are already constant within each track are passed through
    under their own name rather than tripled: `duration_s` and
    `mean_step_um` are per-track numbers that `viewer.py` broadcast onto
    every vertex, and `duration_s_min == duration_s_mean == duration_s_max`
    would be three identical columns and two useless filter choices."""
    numeric = [
        col
        for col, dtype in zip(tracks_df_px.columns, tracks_df_px.dtypes)
        if col not in _QC_SKIP_COLUMNS and dtype.is_numeric()
    ]
    if not numeric:
        return tracks_df_px.select("track_id").unique().sort("track_id"), []

    # One pass to find out which of those are per-point and which are
    # per-track-broadcast, rather than hardcoding a list that would go
    # stale the moment the detector gains a column.
    varies = tracks_df_px.group_by("track_id").agg(
        pl.col(col).n_unique().alias(col) for col in numeric
    )
    per_point = [col for col in numeric if varies[col].max() is not None and varies[col].max() > 1]
    per_track = [col for col in numeric if col not in per_point]

    exprs = [pl.col(col).first().alias(col) for col in per_track]
    for col in per_point:
        exprs.extend(agg(col).alias(f"{col}_{suffix}") for suffix, agg in _QC_AGGREGATIONS)
    return tracks_df_px.group_by("track_id").agg(exprs).sort("track_id"), per_point


def base_track_table(
    diffkit_tracks: pl.DataFrame, tracks_df_px: pl.DataFrame
) -> tuple[pl.DataFrame, list[str]]:
    """One row per track: identity, position and detection-quality
    context columns that every
    tab's results get left-joined onto, plus the names of
    the aggregate QC columns (so the tracks pane can hide that group from
    the table when it is not being used -- there are three per detector
    field, and they are the widest group in the table).

    Position columns are pixel-space (`tracks_df_px`, the viewer's own
    coordinate system), not the physical-unit table."""
    centroids = tracks_df_px.group_by("track_id").agg(
        pl.col("y").mean().alias("y_px"), pl.col("x").mean().alias("x_px")
    )
    lengths = diffkit_tracks.group_by("track_id").agg(pl.col("track_length").first())
    qc, per_point = qc_aggregate_table(tracks_df_px)
    qc_columns = [c for c in qc.columns if c != "track_id"]
    table = (
        lengths.join(centroids, on="track_id", how="left")
        .join(qc, on="track_id", how="left")
        .sort("track_id")
    )
    if "region_class" in tracks_df_px.columns:
        # Which region the track was linked in (`regions.label_points`) --
        # one per track, since tracking links each region on its own.
        regions = tracks_df_px.group_by("track_id").agg(pl.col("region_class").first())
        table = table.join(regions, on="track_id", how="left")
    # Only the tripled columns are the hideable group; the passed-through
    # per-track ones (duration_s, mean_step_um) are core context.
    hideable = [c for c in qc_columns if any(c.startswith(f"{col}_") for col in per_point)]
    return table, hideable


# --- per-track grid posteriors -------------------------------------------

FIT_SCHEMA = {
    "track_id": pl.Int64,
    "n_frames": pl.Int64,
    "posterior_status": pl.String,
    "message": pl.String,
    "D_mean_um2_s": pl.Float64,
    "D_low_um2_s": pl.Float64,
    "D_high_um2_s": pl.Float64,
    # Which grid edge cuts the posterior: "low" (the data only bound D from
    # above -- read the row as upper bounds), "high", "both", or null.
    "D_grid_edge": pl.String,
    # What the track taught about D (KL from the flat prior, in bits --
    # relative to the grid's range, so comparable only on one grid).
    "D_info_bits": pl.Float64,
    # The D at which the track's motion per frame equals its localization
    # noise: a reference scale for D, from the track's own SDs.
    "D_floor_um2_s": pl.Float64,
}

# The localization floor's band on a plot: these quantiles of the tracks'
# own floors, around their median.
FLOOR_BAND = (0.1, 0.9)
# `D_grid_edge`'s values (null when the posterior is not cut).
GRID_EDGES = ("low", "high", "both")
# Columns whose scale is the localization floor, `D_floor_um2_s`.
FLOOR_COLUMNS = ("D_mean_um2_s", "D_low_um2_s", "D_high_um2_s", "D_msd_um2_s", "D_partially_pooled_um2_s")
# How the by-length split counts tracks (`gridpost.by_track_length`): once
# each, or once per frame (the composition of the spots seen in focus).
LENGTH_WEIGHTS = ("tracks", "detections")

# The per-track columns the widget shows and the summary saves.
POSTERIOR_COLUMNS = tuple(c for c in FIT_SCHEMA if c not in ("n_frames", "message"))


@dataclass(frozen=True)
class PosteriorAnalysis:
    """One run of `analyze_posteriors`.

    `fits` has a row for every input track, including the `excluded`
    (too short) and `invalid_input` ones, with the reason in `message`.
    The likelihood arrays hold only the tracks that were fitted, in the
    order of `fitted_ids`: `loglik_D`, each track's log-likelihood over
    `D_grid_um2_s` (`options`' grid), normalized so each row's logsumexp is
    0 -- under the flat prior that is also the track's posterior -- with
    each track's frames in `fitted_frames`. `tracks_sha256` is `tracks_fingerprint` of
    the tracks it was run on.
    """

    fits: pl.DataFrame
    acquisition: Acquisition
    options: GridPostOptions
    fitted_ids: np.ndarray
    fitted_frames: np.ndarray
    loglik_D: np.ndarray
    tracks_sha256: Optional[str] = None
    # `ensemble`'s results by track ids: the summary, the figures and the
    # saved tables all read the same few, and each costs a deconvolution
    # (~1-2 s up to a few thousand tracks).
    _ensembles: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def min_frames(self) -> int:
        return self.options.min_frames

    @property
    def level(self) -> float:
        return self.options.level

    @property
    def u_D(self) -> np.ndarray:
        return self.options.u_D()

    @property
    def D_grid_um2_s(self) -> np.ndarray:
        return np.exp(self.options.u_D())

    def rows_for(self, track_ids: Optional[set]) -> np.ndarray:
        """Row indices into the likelihood arrays for `track_ids`, or all
        rows for None."""
        if track_ids is None:
            return np.arange(len(self.fitted_ids))
        return np.flatnonzero(np.isin(self.fitted_ids, list(track_ids)))


def analyze_posteriors(
    tracks: pl.DataFrame,
    acquisition: Acquisition,
    options: GridPostOptions,
    *,
    progress: Optional[Callable[[int, int], None]] = None,
    threads: Optional[int] = None,
) -> PosteriorAnalysis:
    """Grid posteriors over D for every track in `tracks`, on `options`' grid (`posterior_options`).

    `tracks` is `tracks_to_diffusionkit_df`'s table. This is
    `diffusionkit.gridpost.analyze_tracks` itself -- same statuses, same
    numbers -- reshaped to one row per track, with each fitted track's
    log-likelihood row kept. Tracks are fitted on a pool of `threads`
    (default: one per CPU; diffusionkit's linear algebra releases the GIL),
    with the result identical to a serial run. `progress(done, total)` is
    called from the calling thread."""
    threads = threads or os.cpu_count() or 1
    if threads > 1:
        with ThreadPoolExecutor(threads) as pool:
            result = dk_analyze_tracks(
                tracks, acquisition, options, progress=progress, keep_likelihoods=True, map_fn=pool.map
            )
    else:
        result = dk_analyze_tracks(tracks, acquisition, options, progress=progress, keep_likelihoods=True)
    lik = result.likelihoods
    fits = (
        result.fits.select(
            "track_id",
            "n_frames",
            pl.col("status").alias("posterior_status"),
            "message",
            pl.col("D_post_mean_um2_s").alias("D_mean_um2_s"),
            pl.col("D_post_lo_um2_s").alias("D_low_um2_s"),
            pl.col("D_post_hi_um2_s").alias("D_high_um2_s"),
            "D_grid_edge",
            pl.col("D_post_info_bits").alias("D_info_bits"),
            "D_floor_um2_s",
        )
        .select(FIT_SCHEMA.keys())
        .cast(FIT_SCHEMA)
    )
    return PosteriorAnalysis(
        fits=fits,
        acquisition=acquisition,
        options=options,
        fitted_ids=lik.track_ids,
        fitted_frames=lik.n_frames,
        loglik_D=lik.loglik_D,
        tracks_sha256=tracks_fingerprint(tracks),
    )


# --- the ensemble --------------------------------------------------------


@dataclass(frozen=True)
class Ensemble:
    """The population read of one set of tracks, on the D grid: their
    log-likelihoods combined under three models of the population.

    `shared` is complete pooling (diffusionkit's `SharedD`): the posterior
    of one D for every track, resolved below the grid step. `shared_loglik_D`
    is the same thing on the grid -- the tracks' log-likelihoods summed,
    0 at the peak (its absolute level, a sum over n tracks, means nothing).
    `lognormal` is partial pooling with ln D ~ N(mu, sigma) across tracks
    (diffusionkit's `LogNormal`: the posterior over (mu, sigma), the
    distribution at its mode as `weights` and draws of it as `samples`).
    `deconvolved_D` is partial pooling with a smooth density of any shape
    (`gridpost.deconvolve`), `deconvolved_D_band` its pointwise
    equal-tailed band at the analysis's credible level and
    `deconvolved_D_lambda` the smoothness its evidence chose;
    `deconvolved_D_fit` is the whole fit, draws included (what
    `length_composition` splits by track length). Weights sum to 1 over the
    grid."""

    n_tracks: int
    shared: dk_lognormal.SharedD
    shared_loglik_D: np.ndarray
    lognormal: dk_lognormal.LogNormal
    deconvolved_D: np.ndarray
    deconvolved_D_band: tuple[np.ndarray, np.ndarray]
    deconvolved_D_lambda: float
    deconvolved_D_fit: dk_deconvolve.Deconvolution


_ENSEMBLE_CACHE_SIZE = 32


def _cached(analysis: PosteriorAnalysis, key, compute):
    if key in analysis._ensembles:
        return analysis._ensembles[key]
    result = compute()
    if len(analysis._ensembles) >= _ENSEMBLE_CACHE_SIZE:
        analysis._ensembles.clear()
    analysis._ensembles[key] = result
    return result


def _ids_key(track_ids: Optional[set]):
    return None if track_ids is None else frozenset(track_ids)


def ensemble(analysis: PosteriorAnalysis, track_ids: Optional[set] = None) -> Optional[Ensemble]:
    """The `Ensemble` over `track_ids` (all fitted tracks for None), or
    None when none of them was fitted. Cached on `analysis`."""
    return _cached(analysis, _ids_key(track_ids), lambda: _ensemble(analysis, track_ids))


def _ensemble(analysis: PosteriorAnalysis, track_ids: Optional[set]) -> Optional[Ensemble]:
    rows = analysis.rows_for(track_ids)
    if len(rows) == 0:
        return None
    lls = analysis.loglik_D[rows]
    total = lls.sum(axis=0)
    fit = dk_deconvolve.deconvolve(lls, analysis.u_D)
    return Ensemble(
        n_tracks=len(rows),
        shared=dk_lognormal.fit_shared_D(lls, analysis.u_D),
        shared_loglik_D=total - total.max(),
        lognormal=dk_lognormal.fit_lognormal(lls, analysis.u_D),
        deconvolved_D=fit.weights,
        deconvolved_D_band=fit.band(analysis.level),
        deconvolved_D_lambda=float(fit.lam),
        deconvolved_D_fit=fit,
    )


# Draws of the deconvolved distribution the by-length split is computed
# under: each costs one pass over every track's likelihood (~0.04 s for ten
# thousand tracks), and 50 are enough for a 90% band.
LENGTH_DRAWS = 50


def length_composition(
    analysis: PosteriorAnalysis, track_ids: Optional[set] = None, weight: str = "tracks"
) -> Optional[LengthComposition]:
    """`gridpost.by_track_length` over `track_ids` (all for None): the
    tracks' own posteriors (unpooled) and their posteriors under the
    deconvolved distribution (partially pooled) split into track-length
    groups, each track counted once (`weight="tracks"`) or once per frame
    ("detections"). None when no track was fitted. Cached on `analysis`."""

    def compute():
        ens = ensemble(analysis, track_ids)
        if ens is None:
            return None
        rows = analysis.rows_for(track_ids)
        likelihoods = GridLikelihoods(analysis.fitted_ids[rows], analysis.fitted_frames[rows], analysis.loglik_D[rows])
        return by_track_length(likelihoods, analysis.u_D, ens.deconvolved_D_fit, weight=weight, n_draws=LENGTH_DRAWS)

    return _cached(analysis, ("length", _ids_key(track_ids), weight), compute)


def _grid_summary(weights: np.ndarray, grid: np.ndarray, level: float, log_grid: bool) -> dict:
    """Median and equal-tailed interval of a grid distribution -- the same
    midpoint-CDF interpolation diffusionkit's `summary` uses."""
    x = np.log(grid) if log_grid else grid
    cdf = np.cumsum(weights) - weights / 2
    q = np.interp([(1 - level) / 2, 0.5, (1 + level) / 2], cdf, x)
    if log_grid:
        q = np.exp(q)
    return {"low": float(q[0]), "median": float(q[1]), "high": float(q[2])}


def _low_high(interval: dict) -> dict:
    """diffusionkit's `{median, lo, hi}` as this module's `{low, median, high}`."""
    return {"low": interval["lo"], "median": interval["median"], "high": interval["hi"]}


def _grid_mode(weights: np.ndarray, grid: np.ndarray) -> float:
    return float(grid[int(np.argmax(weights))])


# `summarize`'s log-normal keys: diffusionkit's summary names, each with its
# interval as `_low`/`_high` before the unit.
_LOGNORMAL_KEYS = (("D_median", "_um2_s"), ("sigma_ln_D", ""), ("D_mean", "_um2_s"))


def lognormal_summary_of(fit: dk_lognormal.LogNormal, level: float) -> dict:
    """A diffusionkit `LogNormal`'s numbers as flat keys: `lognormal_D_median_um2_s`
    (the population's median D) with `lognormal_D_median_low_um2_s` /
    `_high_um2_s`, and likewise `lognormal_sigma_ln_D` (its spread in ln D)
    and `lognormal_D_mean_um2_s`; `lognormal_problem`, the fit's reason its
    numbers depend on a prior bound, or None."""
    summary = fit.summary(level)
    out = {}
    for name, unit in _LOGNORMAL_KEYS:
        values = summary[name + unit]
        out[f"lognormal_{name}{unit}"] = values["median"]
        out[f"lognormal_{name}_low{unit}"] = values["lo"]
        out[f"lognormal_{name}_high{unit}"] = values["hi"]
    out["lognormal_problem"] = fit.problem or None
    return out


def lognormal_summary(ens: Ensemble, level: float) -> dict:
    """`lognormal_summary_of` the ensemble's log-normal, `lognormal_problem`
    left out when there is none."""
    out = lognormal_summary_of(ens.lognormal, level)
    if out["lognormal_problem"] is None:
        del out["lognormal_problem"]
    return out


def summarize(analysis: PosteriorAnalysis, track_ids: Optional[set] = None) -> dict:
    """Population-level numbers for `track_ids` (all for None), flat keys
    with units in their names so the saved JSON reads on its own.

    Counts are by diffusionkit's own statuses (`n_ok`, `n_excluded`,
    `n_invalid_input`), plus `n_D_grid_edge_low` / `_high` / `_both`: fitted
    tracks whose D posterior is cut by that end of the grid's range, so their
    numbers depend on it (`D_grid_edge`).
    `n_detections` counts the fitted tracks' frames. The population is the
    tracks' log-likelihoods combined (`Ensemble`), never a statistic of
    their per-track medians: `lognormal_*` (`lognormal_summary`: the
    population's median D, spread and mean D), `shared_D_*` (one D for
    every track: its median and interval), and `deconvolved_D_*` the
    deconvolved distribution's median and 90% range (a spread across
    tracks, not an uncertainty), its mode, and the smoothness (`lambda`) its
    evidence chose. `D_floor_*_um2_s` are the median and 10%/90% quantiles
    of the tracks' localization floors."""
    fits = analysis.fits
    if track_ids is not None:
        fits = fits.filter(pl.col("track_id").is_in(list(track_ids)))
    statuses = dict(fits.group_by("posterior_status").len().iter_rows())
    ok = fits.filter(pl.col("posterior_status") == "ok")
    out: dict = {
        "n_tracks": fits.height,
        **{f"n_{status}": int(count) for status, count in sorted(statuses.items())},
        "n_detections": int(ok["n_frames"].sum()),
        "n_frames_min": _scalar(ok["n_frames"].min()),
        "n_frames_median": _scalar(ok["n_frames"].median()),
        "n_frames_max": _scalar(ok["n_frames"].max()),
        **{f"n_D_grid_edge_{edge}": int((ok["D_grid_edge"] == edge).sum()) for edge in GRID_EDGES},
    }
    floors = ok["D_floor_um2_s"].drop_nulls()
    if floors.len():
        out.update(
            D_floor_median_um2_s=_scalar(floors.median()),
            D_floor_q10_um2_s=_scalar(floors.quantile(FLOOR_BAND[0])),
            D_floor_q90_um2_s=_scalar(floors.quantile(FLOOR_BAND[1])),
        )
    ens = ensemble(analysis, track_ids)
    if ens is not None:
        shared = _low_high(ens.shared.summary(analysis.level))
        spread = _grid_summary(ens.deconvolved_D, analysis.D_grid_um2_s, analysis.level, log_grid=True)
        out.update(
            lognormal_summary(ens, analysis.level),
            **{f"shared_D_{k}_um2_s": v for k, v in shared.items()},
            **{f"deconvolved_D_{k}_um2_s": v for k, v in spread.items()},
            deconvolved_D_mode_um2_s=_grid_mode(ens.deconvolved_D, analysis.D_grid_um2_s),
            deconvolved_D_lambda=ens.deconvolved_D_lambda,
        )
    return out


def summarize_by_group(analysis: PosteriorAnalysis, groups: pl.DataFrame) -> dict[str, dict]:
    """`summarize` per group -- per region class, when each region's tracks
    were linked on their own. `groups` has `track_id` and `group`; groups
    come back in their order of first appearance."""
    names = groups["group"].unique(maintain_order=True).drop_nulls().to_list()
    return {
        name: summarize(analysis, set(groups.filter(pl.col("group") == name)["track_id"].to_list()))
        for name in names
    }


def partially_pooled_table(
    analysis: PosteriorAnalysis, ids: Optional[set], by_class: Optional[dict[str, set]] = None
) -> pl.DataFrame:
    """`track_id`, `D_partially_pooled_um2_s`: each fitted track's E[D] with
    its population as the prior -- the log-normal (`Ensemble.lognormal`) of
    its region class when there are several (`by_class`), else of all the
    tracks passing the filters (`ids`, None for all). Tracks outside those
    are left out.

    It borrows from the population: it does not lean on the grid's edges or
    on the flat prior, and averages to the population's mean, but it moves
    with which tracks make up the population -- so it is written with a
    saved analysis, not offered as a column to filter on."""
    parts = []
    for gids in (by_class or {"all": ids}).values():
        ens = ensemble(analysis, gids)
        if ens is None:
            continue
        rows = analysis.rows_for(gids)
        parts.append(pl.DataFrame({
            "track_id": analysis.fitted_ids[rows],
            "D_partially_pooled_um2_s": ens.lognormal.partially_pooled_means(analysis.loglik_D[rows]),
        }, schema={"track_id": pl.Int64, "D_partially_pooled_um2_s": pl.Float64}))
    if not parts:
        return pl.DataFrame(schema={"track_id": pl.Int64, "D_partially_pooled_um2_s": pl.Float64})
    return pl.concat(parts)


def _scalar(value) -> Optional[float]:
    return None if value is None else float(value)


def ensemble_panels(
    analysis: PosteriorAnalysis,
    groups: Optional[dict[str, Optional[set]]] = None,
    *,
    track_posteriors: bool = False,
) -> list[dict]:
    """What `joint_plot.plot_d_ensemble` and `plot_d_posteriors` draw, one
    dict per group (default one group, "all"); groups with no fitted track
    are left out. Each holds the three population reads (`Ensemble`):
    `lognormal` (weights on the grid) with its pointwise `lognormal_band`
    and its numbers (`lognormal_summary`'s keys, under `summary`),
    `deconvolved` with its `deconvolved_band`, and `shared_interval`
    (low, median, high), all at credible `level`. `track_posteriors` adds
    every track's flat-prior posterior as rows sorted by where its
    likelihood peaks -- not by its median, which a wide posterior's prior
    sets -- for the heat map. `floor` is the tracks' localization floor as
    (10%, median, 90%), None when no track has one."""
    fits = analysis.fits.filter(pl.col("posterior_status") == "ok")
    panels = []
    for name, ids in (groups or {"all": None}).items():
        ens = ensemble(analysis, ids)
        if ens is None:
            continue
        rows = fits if ids is None else fits.filter(pl.col("track_id").is_in(list(ids)))
        shared = _low_high(ens.shared.summary(analysis.level))
        panel = {
            "name": name,
            "n_tracks": ens.n_tracks,
            "level": analysis.level,
            "lognormal": ens.lognormal.weights,
            "lognormal_band": ens.lognormal.band(analysis.level),
            "summary": lognormal_summary(ens, analysis.level),
            "deconvolved": ens.deconvolved_D,
            "deconvolved_band": ens.deconvolved_D_band,
            "shared_interval": (shared["low"], shared["median"], shared["high"]),
            "floor": floor_band(rows["D_floor_um2_s"]),
        }
        if track_posteriors:
            lls = analysis.loglik_D[analysis.rows_for(ids)]
            order = np.argsort(lls.argmax(axis=1), kind="stable")
            panel["track_posteriors"] = np.exp(lls[order])
        panels.append(panel)
    return panels


def length_panels(
    analysis: PosteriorAnalysis, groups: Optional[dict[str, Optional[set]]] = None, weight: str = "tracks"
) -> list[dict]:
    """What `joint_plot.plot_d_by_length` draws, one dict per group (as
    `ensemble_panels`): `name`, `composition` (`length_composition`) and
    `floor`."""
    fits = analysis.fits.filter(pl.col("posterior_status") == "ok")
    panels = []
    for name, ids in (groups or {"all": None}).items():
        comp = length_composition(analysis, ids, weight)
        if comp is None:
            continue
        rows = fits if ids is None else fits.filter(pl.col("track_id").is_in(list(ids)))
        panels.append({"name": name, "composition": comp, "floor": floor_band(rows["D_floor_um2_s"])})
    return panels


def floor_band(floors) -> Optional[tuple[float, float, float]]:
    """(10%, median, 90%) of per-track localization floors (`D_floor_um2_s`
    values, nulls skipped), or None when there are none."""
    values = np.asarray(pl.Series(floors).drop_nulls().to_numpy(), dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return None
    lo, mid, hi = np.quantile(values, [FLOOR_BAND[0], 0.5, FLOOR_BAND[1]])
    return float(lo), float(mid), float(hi)


def floor_references(table: pl.DataFrame, columns) -> dict[str, tuple[float, float, float]]:
    """`{column: floor_band}` for those of `columns` that are a D with a
    floor (`FLOOR_COLUMNS`), from `table`'s `D_floor_um2_s`, for the joint
    plot, over the tracks it draws."""
    if "D_floor_um2_s" not in table.columns:
        return {}
    band = floor_band(table["D_floor_um2_s"])
    if band is None:
        return {}
    return {c: band for c in columns if c in FLOOR_COLUMNS}


def track_posterior(analysis: PosteriorAnalysis, track_id: int) -> Optional[dict]:
    """One fitted track's flat-prior posterior weights and intervals, for
    `joint_plot.plot_track_posterior`; None if it wasn't fitted."""
    rows = np.flatnonzero(analysis.fitted_ids == track_id)
    if len(rows) == 0:
        return None
    fit = analysis.fits.filter(pl.col("track_id") == track_id).row(0, named=True)
    out = {
        "track_id": track_id,
        "n_frames": fit["n_frames"],
        "d_grid": analysis.D_grid_um2_s,
        "d_weights": np.exp(analysis.loglik_D[rows[0]]),
        "d_interval": (fit["D_low_um2_s"], fit["D_mean_um2_s"], fit["D_high_um2_s"]),
        "d_grid_edge": fit["D_grid_edge"],
        "d_floor": fit["D_floor_um2_s"],
    }
    return out


# --- tables the bundle saves --------------------------------------------


def loglik_long_table(analysis: PosteriorAnalysis) -> pl.DataFrame:
    """Every fitted track's log-likelihood over D, one row per (track, grid
    point): `track_id`, `D_um2_s`, `loglik` (normalized: logsumexp over a
    track's rows is 0, which makes it also the track's flat-prior posterior).
    What every population read is built from, so a bundle reopens -- and
    pools -- without refitting."""
    ids, grid = analysis.fitted_ids, analysis.D_grid_um2_s
    n_grid = len(grid)
    return pl.DataFrame(
        {
            "track_id": np.repeat(ids, n_grid),
            "D_um2_s": np.tile(grid, len(ids)),
            "loglik": analysis.loglik_D.reshape(-1),
        },
        schema={"track_id": pl.Int64, "D_um2_s": pl.Float64, "loglik": pl.Float64},
    )


def distributions_table(
    analysis: PosteriorAnalysis, groups: Optional[dict[str, Optional[set]]] = None
) -> Optional[pl.DataFrame]:
    """The population reads on the D grid, long by group: `group`,
    `n_tracks`, `D_um2_s`, and `Ensemble`'s three models --
    `shared_loglik` (the tracks' log-likelihoods summed, 0 at the peak: one
    D for every track), `lognormal` (the log-normal population at its
    posterior mode) and `deconvolved` (weights; each sums to 1 over the
    grid within a group), each population with `_low`/`_high`, its
    pointwise band at the credible level.

    `groups` maps a group name to its track ids; the default is one group,
    "all", over every fitted track."""
    parts = []
    for name, ids in (groups or {"all": None}).items():
        ens = ensemble(analysis, ids)
        if ens is None:
            continue
        n_grid = len(analysis.D_grid_um2_s)
        lognormal_band = ens.lognormal.band(analysis.level)
        parts.append(
            pl.DataFrame(
                {
                    "group": [name] * n_grid,
                    "n_tracks": [ens.n_tracks] * n_grid,
                    "D_um2_s": analysis.D_grid_um2_s,
                    "shared_loglik": ens.shared_loglik_D,
                    "lognormal": ens.lognormal.weights,
                    "lognormal_low": lognormal_band[0],
                    "lognormal_high": lognormal_band[1],
                    "deconvolved": ens.deconvolved_D,
                    "deconvolved_low": ens.deconvolved_D_band[0],
                    "deconvolved_high": ens.deconvolved_D_band[1],
                }
            )
        )
    return pl.concat(parts) if parts else None


def length_distributions_table(
    analysis: PosteriorAnalysis, groups: Optional[dict[str, Optional[set]]] = None
) -> Optional[pl.DataFrame]:
    """`length_composition` on the D grid, long by group, weight and
    track-length group: `group`, `weight` ("tracks" or "detections"),
    `length_min`/`length_max` (the group's track lengths in frames, both
    ends included; `length_max` empty for the open-ended last group),
    `n_tracks`, `n_detections`,
    `D_um2_s`, `unpooled` (the tracks' own flat-prior posteriors) and
    `partially_pooled` (each track's posterior under the deconvolved
    distribution, mean over its draws), with `partially_pooled_low`/`_high`
    the pointwise band at the credible level. Each is that length group's
    share of all the group's tracks (or detections): summed over lengths
    and the grid, a weight's rows add up to 1."""
    parts = []
    for name, ids in (groups or {"all": None}).items():
        for weight in LENGTH_WEIGHTS:
            comp = length_composition(analysis, ids, weight)
            if comp is None:
                continue
            lo, hi = comp.band(analysis.level)
            mean = comp.partially_pooled.mean(axis=0)
            n_grid = len(comp.u)
            upper = [int(e) - 1 for e in comp.edges[1:]] + [None]
            for j in range(len(comp.edges)):
                parts.append(
                    pl.DataFrame(
                        {
                            "group": [name] * n_grid,
                            "weight": [weight] * n_grid,
                            "length_min": [int(comp.edges[j])] * n_grid,
                            "length_max": pl.Series([upper[j]] * n_grid, dtype=pl.Int64),
                            "n_tracks": [int(comp.n_tracks[j])] * n_grid,
                            "n_detections": [int(comp.n_detections[j])] * n_grid,
                            "D_um2_s": np.exp(comp.u),
                            "unpooled": comp.unpooled[j],
                            "partially_pooled": mean[j],
                            "partially_pooled_low": lo[j],
                            "partially_pooled_high": hi[j],
                        }
                    )
                )
    return pl.concat(parts) if parts else None


# --- MSD comparison -----------------------------------------------------


# The per-track MSD fits' default windows. D over the first 3 lags
# (diffusionkit's own default, and the least: the fits need three points);
# alpha over the first 10, about a decade of tau -- a log-log slope over
# three lags spans half a decade and says little about curvature. The
# share-of-each-track form: D over 30% (the 25-40% rule), alpha over 50%,
# short of the last lags, where a single track's MSD rests on a pair or two.
MSD_MAX_LAG = 3
MSD_MIN_LAG = 3
MSD_ALPHA_MAX_LAG = 10
MSD_LAG_FRACTION = 0.3
MSD_ALPHA_LAG_FRACTION = 0.5
# What the per-track and ensemble MSD comparisons are, for the summaries.
MSD_METHOD = {
    "D_fit": "linear through the origin of the offset-corrected MSD (offset from the SDs)",
    "alpha_fit": "log-log OLS of the offset-corrected MSD (offset from the SDs)",
    "exposure": "treated as 0 (the MSD estimators have no blur model)",
}


@dataclass(frozen=True)
class MSDWindow:
    """The per-track MSD fits' windows: D over each track's first `max_lag`
    lags and alpha over its first `alpha_max_lag` -- or, when `lag_fraction`
    is set, shares of each track's longest lag (`lag_fraction` for D,
    `alpha_lag_fraction` for alpha), so longer tracks fit more of their
    curve. Each is capped at the track's own longest lag. The names are
    diffusionkit's `MSDOptions`'."""

    max_lag: int = MSD_MAX_LAG
    alpha_max_lag: int = MSD_ALPHA_MAX_LAG
    lag_fraction: Optional[float] = None
    alpha_lag_fraction: float = MSD_ALPHA_LAG_FRACTION

    def __post_init__(self):
        for name in ("max_lag", "alpha_max_lag"):
            if getattr(self, name) < MSD_MIN_LAG:
                raise ValueError(f"msd_{name} must be at least {MSD_MIN_LAG}, got {getattr(self, name)}")
        for name in ("lag_fraction", "alpha_lag_fraction"):
            value = getattr(self, name)
            if value is not None and not 0 < value <= 1:
                raise ValueError(f"msd_{name} must be in (0, 1], got {value}")

    @property
    def by_fraction(self) -> bool:
        return self.lag_fraction is not None

    def options(self, min_frames: int) -> MSDOptions:
        """diffusionkit's `MSDOptions` for these windows, alpha by log-log."""
        common = dict(min_frames=max(min_frames, 5), localization="provided", alpha_fit="loglog")
        if self.by_fraction:
            return MSDOptions(
                max_lag=None, lag_fraction=self.lag_fraction, alpha_lag_fraction=self.alpha_lag_fraction, **common
            )
        return MSDOptions(max_lag=self.max_lag, alpha_max_lag=self.alpha_max_lag, **common)

    def text(self) -> str:
        """As a reader sees it: "D: 3 lags · α: 10 lags", or by share."""
        if self.by_fraction:
            return f"D: {self.lag_fraction:.0%} · α: {self.alpha_lag_fraction:.0%} of each track"
        return f"D: {self.max_lag} lags · α: {self.alpha_max_lag} lags"

    def record(self) -> dict:
        """For `diffusion_summary.json`: the two windows in use, under
        their `[diffusion]` config names, and what the fits are."""
        windows = (
            {"msd_lag_fraction": self.lag_fraction, "msd_alpha_lag_fraction": self.alpha_lag_fraction}
            if self.by_fraction
            else {"msd_max_lag": self.max_lag, "msd_alpha_max_lag": self.alpha_max_lag}
        )
        return {**windows, **MSD_METHOD}

    @classmethod
    def from_settings(cls, settings: dict) -> "MSDWindow":
        """From `msd_*` keys (a `record()`, or a `[diffusion]` config); the
        ones left out keep their defaults."""
        keys = {"max_lag", "alpha_max_lag", "lag_fraction", "alpha_lag_fraction"}
        return cls(**{k: settings[f"msd_{k}"] for k in keys if settings.get(f"msd_{k}") is not None})


def msd_fits_blur_free(
    tracks: pl.DataFrame,
    dt_s: float,
    min_frames: int,
    window: MSDWindow,
    progress: Optional[Callable[[int, int], None]] = None,
) -> pl.DataFrame:
    """diffusionkit.classic's MSD fits of `tracks_to_diffusionkit_df`'s
    table over `window`, for comparison with the posteriors.

    The MSD fits have no blur model, so diffusionkit excludes them when
    `exposure_s > 0`; the comparison therefore runs them with the exposure
    treated as 0, which is exactly the assumption that biases them, and is
    labelled as such wherever it shows."""
    return analyze_msd(tracks, Acquisition(dt_s=dt_s), window.options(min_frames), progress=progress).fits


def msd_track_panel(
    tracks: pl.DataFrame, dt_s: float, track_id: int, min_frames: int, window: MSDWindow
) -> Optional[dict]:
    """One track's time-averaged MSD, as a panel `joint_plot.plot_track_msd`
    draws the way `plot_ensemble_msd` draws a group: the raw curve over the
    larger of the two windows, D's fit over its window (through the origin
    of the SD-corrected curve) and alpha's log-log fit over its own --
    the fits in the D_msd / alpha_msd columns. Also the linear fit with a
    free intercept over D's window, whose offset is read off the curve
    rather than the SDs (`intercept_check`): the two localization SDs
    disagreeing is a sign the SDs are miscalibrated -- or that a few noisy
    lags cannot pin an intercept, which is why D_msd does not use it. No
    error bars: a track's lags share their displacements, so its points are
    not independent and no per-point SD means what it would on an ensemble.
    None if the track is absent or too short."""
    track = tracks.filter(pl.col("track_id") == track_id).sort("frame")
    options = window.options(min_frames)
    if track.height < options.min_frames:
        return None
    curve = compute_msd(track, Acquisition(dt_s=dt_s), options)
    n_d, n_alpha = fit_windows(track.height, options)
    brownian = fit_brownian_msd(curve.head(n_d))
    power = fit_loglog_msd(curve.head(n_alpha))
    linear = fit_linear_msd(curve.head(n_d)).parameters
    # The SDs' offset is 4 sigma^2 for an isotropic 2-D error.
    offset_sds = float(np.mean(curve.localization_offset_um2[:n_d]))
    return {
        "name": f"track {track_id}",
        "n_frames": track.height,
        "unit": "pairs",
        "n_points": n_d,
        "alpha_points": n_alpha,
        "offset": "provided",
        "level": LEVEL,
        "tau_s": np.asarray(curve.tau_s, float),
        "msd_um2": np.asarray(curve.msd_um2, float),
        "se_um2": np.full(len(curve.lag), np.nan),
        "n_units": np.asarray(curve.n_pairs),
        "offset_um2": np.asarray(curve.localization_offset_um2, float),
        "linear": {"model": "brownian", "status": brownian.status, "message": brownian.message, **brownian.parameters},
        "power_law": {"model": "power_law", "status": power.status, "message": power.message, **power.parameters},
        "intercept_check": {
            "D_um2_s": linear.get("D_um2_s"),
            "offset_um2": linear.get("offset_um2"),
            "localization_sd_um": linear.get("localization_sd_um"),
            "sigma_sds_um": float(np.sqrt(offset_sds / 4)) if offset_sds >= 0 else None,
        },
    }


def msd_track_table(fits: pl.DataFrame) -> pl.DataFrame:
    """diffusionkit.classic's MSD fits, one row per track: `D_msd_um2_s`
    over D's window, `K_msd_um2_s_alpha`/`alpha_msd` from the log-log fit
    over alpha's. No uncertainties -- diffusionkit estimates none for one
    track's MSD fits."""
    brownian = fits.filter(pl.col("model") == "brownian").select(
        "track_id", pl.col("D_um2_s").alias("D_msd_um2_s")
    )
    power_law = fits.filter(pl.col("model") == "power_law").select(
        "track_id",
        pl.col("K_um2_s_alpha").alias("K_msd_um2_s_alpha"),
        pl.col("alpha").alias("alpha_msd"),
    )
    return brownian.join(power_law, on="track_id", how="full", coalesce=True)


# --- the ensemble-averaged MSD ------------------------------------------

# Where the ensemble fit's localization offset comes from: the tracks' SDs
# ("provided": D through the origin of the offset-corrected curve), or the
# intercept of a linear fit to the raw curve ("fit", no SDs used).
ENSEMBLE_MSD_OFFSETS = ("provided", "fit")
# Bootstrap resamples of whole tracks behind the ensemble MSD's intervals.
ENSEMBLE_MSD_BOOT = 200
# The ensemble MSD's windows. The curve runs to ENSEMBLE_MSD_MAX_LAG; D is
# fitted over its first 30% (the 25-40% rule: the best-measured lags), and
# alpha over all of it, since a log-log slope needs a span of lags to show
# curvature -- three lags cover well under a decade of tau.
ENSEMBLE_MSD_MAX_LAG = 10
ENSEMBLE_D_FRACTION = MSD_LAG_FRACTION
ENSEMBLE_ALPHA_FRACTION = 1.0


def ensemble_windows(max_lag: int, d_fraction: float, alpha_fraction: float) -> tuple[int, int]:
    """`(n_points, alpha_points)`: the first lags of a curve run to `max_lag`
    that D and alpha are fitted over, each a fraction of it (diffusionkit's
    `window_lags`: never under three lags, never more than `max_lag`). One
    window for every group, so their fits compare on the same lags."""
    return window_lags(max_lag, d_fraction), window_lags(max_lag, alpha_fraction)


def ensemble_msd_blur_free(
    experiments: list[Experiment], min_frames: int, max_lag: int, n_boot: int = ENSEMBLE_MSD_BOOT
) -> EnsembleMSD:
    """diffusionkit's ensemble-averaged MSD of each `sample` among
    `experiments`: every track's time-averaged MSD to `max_lag`, averaged
    so each squared displacement counts once (pair-weighted), with `n_boot`
    resamples of whole tracks for the intervals. The experiments'
    acquisitions carry no exposure: as in the per-track comparison, the MSD
    estimators have no blur model, so the exposure is treated as 0."""
    # Each track's own fits are not used here; log-log keeps them cheap.
    options = MSDOptions(max_lag=max_lag, min_frames=max(min_frames, 5), localization="provided", alpha_fit="loglog")
    return ensemble_msd(analyze_experiments(experiments, options), "sample", n_boot=n_boot)


def group_experiments(tracks: pl.DataFrame, dt_s: float, groups: dict[str, Optional[set]]) -> list[Experiment]:
    """One blur-free `Experiment` per group of one movie's tracks
    (`tracks_to_diffusionkit_df`'s table; ids None = all), named and
    sampled after the group, for `ensemble_msd_blur_free`. Empty groups
    are left out."""
    out = []
    for name, ids in groups.items():
        rows = tracks if ids is None else tracks.filter(pl.col("track_id").is_in(list(ids)))
        if rows.height:
            out.append(Experiment(str(name), rows, Acquisition(dt_s=dt_s), str(name)))
    return out


def ensemble_msd_fits(
    ens: EnsembleMSD, n_points: int, offset: str, level: float = LEVEL, alpha_points: Optional[int] = None
) -> pl.DataFrame:
    """`EnsembleMSD.fit` -- D over the first `n_points` lags, alpha over the
    first `alpha_points` (default `n_points`) -- one group at a time: a
    group whose curve stops short of a window (no track of it reaches that
    lag) gets rows with status `insufficient_data` saying so, and the other
    groups are still fitted."""
    alpha_points = n_points if alpha_points is None else alpha_points
    need = max(n_points, alpha_points)
    parts = []
    for name, curve in ens.means.items():
        if len(curve.lag) >= need:
            one = replace(
                ens,
                curves=ens.curves.filter(pl.col("group") == name),
                means={name: curve},
                resamples={name: ens.resamples[name]},
                n_units={name: ens.n_units[name]},
            )
            parts.append(one.fit(n_points, offset, level=level, alpha_points=alpha_points))
            continue
        message = f"no track reaches lag {need}: the curve ends at lag {len(curve.lag)}"
        rows = [
            {
                **{column: None for column in ENSEMBLE_FIT_SCHEMA},
                "group": name, "model": model, "status": "insufficient_data", "message": message,
                "n_lags": 0, "n_points": n, "n_units": ens.n_units[name], "uncertainty_method": "not_estimated",
            }
            for model, n in (("linear" if offset == "fit" else "brownian", n_points), ("power_law", alpha_points))
        ]
        parts.append(pl.DataFrame(rows, schema=ENSEMBLE_FIT_SCHEMA))
    return pl.concat(parts)


def ensemble_msd_panels(
    ens: EnsembleMSD, n_points: int, offset: str, level: float = LEVEL, alpha_points: Optional[int] = None
) -> list[dict]:
    """What `joint_plot.plot_ensemble_msd` draws, one dict per group: the
    averaged curve (`tau_s`, `msd_um2`, its SEM over tracks `se_um2` --
    the bootstrap SD over resampled tracks -- and `n_units`, the tracks
    reaching each lag), the offset the fits took off
    (`offset_um2`, per lag), and `ensemble_msd_fits`' rows (`linear` over
    the first `n_points` lags, `power_law` over the first `alpha_points`)
    with their `level` intervals."""
    alpha_points = n_points if alpha_points is None else alpha_points
    fits = ensemble_msd_fits(ens, n_points, offset, level, alpha_points)
    panels = []
    for name, curve in ens.means.items():
        rows = {r["model"]: r for r in fits.filter(pl.col("group") == name).iter_rows(named=True)}
        linear = rows.get("linear") or rows.get("brownian") or {}
        if offset == "fit":
            # The power law is fitted net of the intercept, clipped at 0.
            b = linear.get("offset_um2")
            offsets = np.full(len(curve.lag), max(b, 0.0) if b is not None else 0.0)
        else:
            offsets = np.asarray(curve.localization_offset_um2, float)
        table = ens.curves.filter(pl.col("group") == name).sort("lag")
        panels.append({
            "name": name,
            "n_tracks": ens.n_units[name],
            "n_points": n_points,
            "alpha_points": alpha_points,
            "offset": offset,
            "level": level,
            "tau_s": np.asarray(curve.tau_s, float),
            "msd_um2": np.asarray(curve.msd_um2, float),
            "se_um2": table["msd_se_um2"].fill_null(np.nan).to_numpy(),
            "n_units": table["n_units"].to_numpy(),
            "offset_um2": offsets,
            "linear": linear,
            "power_law": rows.get("power_law") or {},
        })
    return panels


# --- the per-track summary ----------------------------------------------

# Per-point QC columns reach the tracks pane as `<col>_min/_mean/_max`
# (`qc_aggregate_table`). The saved summary keeps
# the mean only: it says what a track's detections were typically like,
# at a third of the width.
_QC_EXTREME_SUFFIXES = ("_min", "_max")


def tracks_summary_table(
    base: pl.DataFrame,
    results: Optional[pl.DataFrame],
    *,
    result_id: Optional[str],
    pixel_size_um: float,
    passing_ids: Optional[set],
) -> pl.DataFrame:
    """One row per track, for `results.TRACKS_SUMMARY_FILENAME`: the table
    to read an experiment's tracks from, and to pool across experiments.

    `base` is the diffusion widget's per-track table (length, centroid in
    px, shape, `region_class`, per-point detection QC aggregated per
    track); `results` the per-track analysis columns (posterior means and
    bounds, and any MSD or NUTS columns), or None before a run. Kept from
    `base`: every column except the per-point min/max. Added: `result_id`
    (which experiment -- so pooling is a plain concat), the centroid in
    µm, and `passes_filters` (the tracks pane's cuts; `passing_ids` None
    means no cut is set).

    Every track is a row, including excluded ones and ones a filter
    rejects: the columns say which, so a later script can apply the same
    rule or another without a re-run."""
    means = {c[: -len("_mean")] for c in base.columns if c.endswith("_mean")}
    drop = [c for c in base.columns if c.endswith(_QC_EXTREME_SUFFIXES) and c[: -len("_min")] in means]
    table = base.drop(drop).with_columns(
        (pl.col("x_px") * pixel_size_um).alias("x_um"),
        (pl.col("y_px") * pixel_size_um).alias("y_um"),
        pl.lit(result_id, dtype=pl.String).alias("result_id"),
        (
            pl.col("track_id").is_in(list(passing_ids)) if passing_ids is not None else pl.lit(True)
        ).alias("passes_filters"),
    )
    if results is not None:
        table = table.join(
            results.with_columns(pl.col("track_id").cast(table.schema["track_id"])),
            on="track_id",
            how="left",
        )
    lead = [
        c
        for c in (
            "result_id", "track_id", "region_class", "passes_filters", "track_length",
            "duration_s", "mean_step_um", "x_um", "y_um", "x_px", "y_px",
        )
        if c in table.columns
    ]
    result_cols = [c for c in (results.columns if results is not None else []) if c != "track_id"]
    rest = [c for c in table.columns if c not in lead and c not in result_cols]
    return table.select(lead + result_cols + rest).sort("track_id")


# --- the saved analysis ---------------------------------------------------
#
# What "Save analysis" writes (`results.write_diffusion_results`), built
# here rather than in the widget so `gemscape2 diffusion` writes the same
# files from the same code -- a bundle analyzed headless reopens in the
# widget exactly like one analyzed there.

# Exposure longer than the frame interval by at most this fraction is read
# as a streaming acquisition (exposure == interval) whose measured median
# interval came out a hair short, and clamped to it -- with a note. Past
# it, the two numbers genuinely disagree and the run is refused.
EXPOSURE_CLAMP_FRACTION = 0.01


def passing_track_ids(
    joined: pl.DataFrame, min_track_length: int, filters: Optional[dict]
) -> Optional[set]:
    """Tracks of `joined` (the per-track table with results joined in)
    passing a minimum length and the `{column: (lo, hi)}` cuts -- None
    when no cut is set, meaning every track passes."""
    ids = None
    if min_track_length > 1:
        ids = set(joined.filter(pl.col("track_length") >= min_track_length)["track_id"].to_list())
    if filters:
        cut = set(joined.filter(filter_mask(joined, filters))["track_id"].to_list())
        ids = cut if ids is None else (ids & cut)
    return ids


def region_class_groups(base: pl.DataFrame, ids: Optional[set]) -> Optional[dict[str, set]]:
    """`{region class: its track ids}` restricted to `ids` (None = all), or
    None when `base`'s tracks don't span more than one class."""
    if "region_class" not in base.columns or base["region_class"].drop_nulls().n_unique() < 2:
        return None
    groups = base.select("track_id", "region_class")
    if ids is not None:
        groups = groups.filter(pl.col("track_id").is_in(list(ids)))
    names = sorted(groups["region_class"].drop_nulls().unique().to_list())
    return {
        name: set(groups.filter(pl.col("region_class") == name)["track_id"].to_list())
        for name in names
    } or None


def filter_record(min_track_length: int, filters: Optional[dict]) -> dict:
    """What `passes_filters` meant, for `diffusion_summary.json`. An open
    bound is null, as in `manifest.json`."""

    def bound(value) -> Optional[float]:
        return float(value) if value is not None and np.isfinite(value) else None

    return {
        "min_track_length": min_track_length,
        "ranges": {col: [bound(lo), bound(hi)] for col, (lo, hi) in (filters or {}).items()},
    }


def analysis_tables(
    analysis: PosteriorAnalysis,
    ids: Optional[set],
    by_class: Optional[dict],
) -> dict:
    """The likelihood and distribution tables, as `write_diffusion_results`
    keyword arguments. The ensemble is over the tracks passing the filters
    (`ids`), split by region class when there are several."""
    groups = {"all": ids, **(by_class or {})}
    return dict(
        loglik_D=loglik_long_table(analysis),
        distributions_D=distributions_table(analysis, groups),
        distributions_D_by_length=length_distributions_table(analysis, groups),
    )


def analysis_summary(
    analysis: PosteriorAnalysis,
    ids: Optional[set],
    by_class: Optional[dict],
    *,
    msd: Optional[MSDWindow] = None,
) -> dict:
    """`diffusion_summary.json`'s settings and population numbers (the
    caller adds the filter record and provenance). Flat keys with units in
    their names, so the JSON reads on its own and the widget can label
    each when it is reopened."""
    summary = {
        "analysis": "diffusionkit.gridpost grid posteriors, flat prior in ln D over the D grid; "
        "population: the tracks' log-likelihoods combined (shared D, log-normal, deconvolved)",
        "dt_s": analysis.acquisition.dt_s,
        "exposure_s": analysis.acquisition.exposure_s,
        "min_frames": analysis.min_frames,
        "credible_level": analysis.level,
        # Every grid the run used (`GridPostOptions`' fields), so a
        # diffusionkit script can repeat it: GridPostOptions(**grid, ...).
        "grid": grid_record(analysis.options),
        "D_grid_um2_s": [float(analysis.D_grid_um2_s[0]), float(analysis.D_grid_um2_s[-1]), analysis.options.n_D],
        # The per-track MSD fits' windows and estimators, when they ran.
        "msd": msd.record() if msd is not None else None,
        "tracks_sha256": analysis.tracks_sha256,
        **summarize(analysis, ids),
    }
    if by_class:
        summary["by_region_class"] = {
            name: summarize(analysis, gids) for name, gids in by_class.items()
        }
    return summary


def posterior_results_table(analysis: PosteriorAnalysis, msd: Optional[pl.DataFrame]) -> pl.DataFrame:
    """The per-track result columns a posterior run adds to the tracks
    table (and to `tracks_summary.csv`): the posterior's, plus
    `msd_track_table`'s when the MSD comparison ran."""
    display = analysis.fits.select(POSTERIOR_COLUMNS)
    if msd is not None:
        display = display.join(msd, on="track_id", how="left")
    return display


# --- reopening a saved analysis -------------------------------------------
#
# The inverse of the tables above: `results.load_diffusion_results`'s
# files back into the `PosteriorAnalysis` they were written from, so a
# bundle reopens with its plots, per-track columns and summary live --
# re-summarized as filters change -- without re-running any fit.


class StaleAnalysisError(ValueError):
    """The bundle has a saved analysis, but not of the tracks loaded now
    (or its likelihoods don't lie on the grid its summary records): it has
    to be re-run."""


@dataclass(frozen=True)
class SavedAnalysis:
    """A saved analysis, restored: the likelihoods, the MSD comparison's
    columns and the NUTS fits' rows when they were saved, and the
    `diffusion_summary.json` settings they were run and summarized with."""

    analysis: PosteriorAnalysis
    msd: Optional[pl.DataFrame]
    nuts_rows: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _loglik_matrix(table: pl.DataFrame, grid_col: str, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`loglik_long_table`'s table back to `(track ids, log-likelihood rows)`."""
    table = table.sort("track_id", grid_col)
    ids = table["track_id"].unique(maintain_order=True).to_numpy().astype(np.int64)
    n_grid = len(grid)
    if table.height != len(ids) * n_grid or not np.allclose(
        table[grid_col].to_numpy()[:n_grid], grid, rtol=1e-12, atol=0
    ):
        raise StaleAnalysisError(f"its {grid_col} likelihoods are not on the grid its summary records")
    return ids, table["loglik"].to_numpy().reshape(len(ids), n_grid)


def _saved_options(summary: dict) -> GridPostOptions:
    """The `GridPostOptions` a saved analysis ran with. Raises
    `StaleAnalysisError` for a grid this version has no field for."""
    if "alpha_grid" in summary:
        raise StaleAnalysisError("saved by a version with the alpha posterior, which is no longer computed")
    unknown = set(summary["grid"]) - set(GRID_FIELDS)
    if unknown:
        raise StaleAnalysisError(f"saved with grid settings this version doesn't have: {', '.join(sorted(unknown))}")
    return GridPostOptions(min_frames=summary["min_frames"], level=summary["credible_level"], **summary["grid"])


def restore_analysis(
    tracks: pl.DataFrame,
    *,
    summary: dict,
    tracks_summary: pl.DataFrame,
    loglik_D: pl.DataFrame,
) -> SavedAnalysis:
    """Rebuild the saved analysis (`results.load_diffusion_results`'s
    tables) over `tracks`, the `tracks_to_diffusionkit_df` table loaded
    now. Raises `StaleAnalysisError` unless the tracks it was fitted on
    are, vertex for vertex, among `tracks` (`tracks_fingerprint`): a
    track's posterior is only its own if its track_id still names it."""
    saved_sha = summary.get("tracks_sha256")
    if not saved_sha:
        raise StaleAnalysisError("saved without a fingerprint of its tracks")
    if "posterior_status" not in tracks_summary.columns:
        raise StaleAnalysisError("its per-track table has no posterior columns")

    # Every track the run was given has a status; the rest were outside a
    # "restrict fits to filtered tracks" run.
    ran = tracks_summary.filter(pl.col("posterior_status").is_not_null())
    fit_ids = ran["track_id"].cast(pl.Int64)
    fitted_tracks = tracks.filter(pl.col("track_id").is_in(fit_ids.implode()))
    if fitted_tracks["track_id"].n_unique() != fit_ids.n_unique() or tracks_fingerprint(fitted_tracks) != saved_sha:
        raise StaleAnalysisError("its tracks differ from the ones loaded (re-tracked or rescaled since)")

    options = _saved_options(summary)
    fitted_ids, loglik_D = _loglik_matrix(loglik_D, "D_um2_s", np.exp(options.u_D()))

    acquisition = Acquisition(dt_s=summary["dt_s"], exposure_s=summary["exposure_s"])
    fits = (
        ran.select(
            pl.col(c).cast(dtype) if c in ran.columns else pl.lit(None, dtype=dtype).alias(c)
            for c, dtype in FIT_SCHEMA.items()
            if c not in ("n_frames", "message")
        )
        .with_columns(
            ran["track_length"].cast(pl.Int64).alias("n_frames"), pl.lit("", dtype=pl.String).alias("message")
        )
        .select(FIT_SCHEMA.keys())
        .sort("track_id")
    )

    frames = dict(zip(fits["track_id"].to_list(), fits["n_frames"].to_list()))
    analysis = PosteriorAnalysis(
        fits=fits,
        acquisition=acquisition,
        options=options,
        fitted_ids=fitted_ids,
        fitted_frames=np.array([frames[i] for i in fitted_ids], dtype=np.int64),
        loglik_D=loglik_D,
        tracks_sha256=saved_sha,
    )

    msd_cols = [c for c in ("D_msd_um2_s", "K_msd_um2_s_alpha", "alpha_msd") if c in ran.columns]
    msd = None
    if summary.get("msd") and msd_cols:
        msd = ran.select(pl.col("track_id").cast(pl.Int64), *msd_cols).filter(
            pl.any_horizontal(pl.col(c).is_not_null() for c in msd_cols)
        )

    # NUTS fits of tracks the fingerprint covers -- of any other, nothing
    # says the track_id still names the same track.
    nuts_rows: list[dict] = []
    if "nuts_model" in ran.columns:
        nuts_cols = ["track_id", "nuts_model", *(c for c in ran.columns if "_nuts" in c)]
        for row in ran.filter(pl.col("nuts_model").is_not_null()).select(nuts_cols).iter_rows(named=True):
            nuts_rows.append({k: v for k, v in row.items() if v is not None})

    return SavedAnalysis(analysis=analysis, msd=msd, nuts_rows=nuts_rows, summary=summary)
