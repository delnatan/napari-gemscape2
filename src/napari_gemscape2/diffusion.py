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
posterior vector (what the ensemble is built from and what the bundle
saves), and every grid comes from the one `GridPostOptions` the run is
given -- the same object a diffusionkit script would pass, so the two agree
number for number. The D grid's range is the flat prior's support: a
posterior cut by an edge is flagged in `D_at_grid_edge`.

  - **per track**: the posterior median of D and its equal-tailed 90%
    interval (`D_low`/`D_high` = the 5% and 95% quantiles);
  - **shared**: the per-track log posteriors added up -- the posterior of
    one value shared by every track. Sharp, but only honest when the
    tracks really do share it;
  - **mean posterior**: the tracks' posteriors averaged -- where they put
    the value, blurred by each track's own uncertainty and prior;
  - **deconvolved**: `gridpost.deconvolve`, how D is
    distributed across tracks, with each track's own uncertainty taken out
    rather than averaged in: a smooth log density whose smoothness the data
    choose (Laplace evidence), with a pointwise band from posterior draws
    of the whole distribution. A peak narrower than the tracks can resolve
    comes out as wide as that resolution;
  - **by track length**: `gridpost.by_track_length`, the mean posterior
    and the deconvolved distribution split into track-length groups, per
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
from diffusionkit.gridpost import GridPosteriors, GridPostOptions, LengthComposition, by_track_length
from diffusionkit.gridpost import analyze_tracks as dk_analyze_tracks
from diffusionkit.gridpost import deconvolve as dk_deconvolve
from diffusionkit.gridpost import posterior as dk_post
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp

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
# track property, so it can sit in the same filter panel as `D_median_um2_s` and be
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
    "D_median_um2_s": pl.Float64,
    "D_low_um2_s": pl.Float64,
    "D_high_um2_s": pl.Float64,
    "D_at_grid_edge": pl.Boolean,
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
# Columns whose scale is the localization floor, `D_floor_um2_s`.
FLOOR_COLUMNS = ("D_median_um2_s", "D_low_um2_s", "D_high_um2_s", "D_msd_um2_s")
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
    The posterior arrays hold only the tracks that were fitted, in the
    order of `fitted_ids`, as normalized log posteriors (flat prior, so
    each row is also the track's log likelihood up to a constant):
    `log_post_D` over `D_grid_um2_s`, `options`' grid, with each track's
    frames in `fitted_frames`. `tracks_sha256` is `tracks_fingerprint` of
    the tracks it was run on.
    """

    fits: pl.DataFrame
    acquisition: Acquisition
    options: GridPostOptions
    fitted_ids: np.ndarray
    fitted_frames: np.ndarray
    log_post_D: np.ndarray
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
        """Row indices into the posterior arrays for `track_ids`, or all
        rows for None."""
        if track_ids is None:
            return np.arange(len(self.fitted_ids))
        return np.flatnonzero(np.isin(self.fitted_ids, list(track_ids)))


def _normalized_log(log_weights: np.ndarray) -> np.ndarray:
    return log_weights - logsumexp(log_weights)


def _at_grid_edge(log_post_D: np.ndarray) -> np.ndarray:
    """Per row of `log_post_D`: is that posterior cut by a grid edge
    (diffusionkit's `posterior.edge_ratios` over its threshold)?"""
    return np.array(
        [max(dk_post.edge_ratios(np.exp(lp))) > dk_post.EDGE_RATIO_WARN for lp in log_post_D], dtype=bool
    )


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
    posterior vector kept. Tracks are fitted on a pool of `threads`
    (default: one per CPU; diffusionkit's linear algebra releases the GIL),
    with the result identical to a serial run. `progress(done, total)` is
    called from the calling thread."""
    threads = threads or os.cpu_count() or 1
    if threads > 1:
        with ThreadPoolExecutor(threads) as pool:
            result = dk_analyze_tracks(
                tracks, acquisition, options, progress=progress, keep_posteriors=True, map_fn=pool.map
            )
    else:
        result = dk_analyze_tracks(tracks, acquisition, options, progress=progress, keep_posteriors=True)
    post = result.posteriors
    edge = pl.DataFrame(
        {"track_id": post.track_ids, "D_at_grid_edge": _at_grid_edge(post.log_post_D)},
        schema={"track_id": pl.Int64, "D_at_grid_edge": pl.Boolean},
    )
    fits = (
        result.fits.select(
            "track_id",
            "n_frames",
            pl.col("status").alias("posterior_status"),
            "message",
            pl.col("D_post_median_um2_s").alias("D_median_um2_s"),
            pl.col("D_post_lo_um2_s").alias("D_low_um2_s"),
            pl.col("D_post_hi_um2_s").alias("D_high_um2_s"),
            pl.col("D_post_info_bits").alias("D_info_bits"),
            "D_floor_um2_s",
        )
        .join(edge, on="track_id", how="left")
        .select(FIT_SCHEMA.keys())
        .cast(FIT_SCHEMA)
    )
    return PosteriorAnalysis(
        fits=fits,
        acquisition=acquisition,
        options=options,
        fitted_ids=post.track_ids,
        fitted_frames=post.n_frames,
        log_post_D=post.log_post_D,
        tracks_sha256=tracks_fingerprint(tracks),
    )


# --- the ensemble --------------------------------------------------------


@dataclass(frozen=True)
class Ensemble:
    """The population read of one set of tracks' posteriors, on the D grid.

    `shared_log_D` is the sum of the tracks' normalized log posteriors,
    shifted to 0 at its peak (its absolute level, a sum over n tracks, is
    thousands below 0 and means nothing), and `shared_D` the same thing
    as grid weights: the posterior of one value shared by every track.
    With many tracks it is narrower than a grid cell, so its weights sit
    in a cell or two; `_shared_summary` reads it below the grid step.
    `mean_posterior_D` is the average of the tracks' posteriors -- where
    the tracks put the value, blurred by each track's own uncertainty and
    prior. `deconvolved_D` is `gridpost.deconvolve`'s distribution across
    the tracks, that blur removed, with `deconvolved_D_band` its pointwise
    equal-tailed band at the analysis's credible level and
    `deconvolved_D_lambda` the smoothness its evidence chose;
    `deconvolved_D_fit` is the whole fit, its posterior draws included
    (what `length_composition` splits by track length). All weights sum
    to 1 over the grid."""

    n_tracks: int
    shared_log_D: np.ndarray
    shared_D: np.ndarray
    mean_posterior_D: np.ndarray
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
    log_post = analysis.log_post_D[rows]
    shared_log = log_post.sum(axis=0)
    shared_log = shared_log - shared_log.max()
    # Flat prior: each row is the track's log likelihood up to a constant.
    fit = dk_deconvolve.deconvolve(log_post, analysis.u_D)
    return Ensemble(
        n_tracks=len(rows),
        shared_log_D=shared_log,
        shared_D=np.exp(_normalized_log(shared_log)),
        mean_posterior_D=np.exp(log_post).mean(axis=0),
        deconvolved_D=fit.weights,
        deconvolved_D_band=fit.band(analysis.level),
        deconvolved_D_lambda=float(fit.lam),
        deconvolved_D_fit=fit,
    )


# Draws of the deconvolved distribution the by-length split is computed
# under: each costs one pass over every track's posterior (~0.04 s for ten
# thousand tracks), and 50 are enough for a 90% band.
LENGTH_DRAWS = 50


def length_composition(
    analysis: PosteriorAnalysis, track_ids: Optional[set] = None, weight: str = "tracks"
) -> Optional[LengthComposition]:
    """`gridpost.by_track_length` over `track_ids` (all for None): the mean
    posterior and the deconvolved distribution split into track-length
    groups, each track counted once (`weight="tracks"`) or once per frame
    ("detections"). None when no track was fitted. Cached on `analysis`."""

    def compute():
        ens = ensemble(analysis, track_ids)
        if ens is None:
            return None
        rows = analysis.rows_for(track_ids)
        posteriors = GridPosteriors(analysis.fitted_ids[rows], analysis.fitted_frames[rows], analysis.log_post_D[rows])
        return by_track_length(posteriors, analysis.u_D, ens.deconvolved_D_fit, weight=weight, n_draws=LENGTH_DRAWS)

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


# The shared posterior's summary is read off a cubic spline of its log
# through the cells within `_SHARED_WINDOW` (log units) of its peak, at
# `_SHARED_REFINE` points per grid cell.
_SHARED_WINDOW = 40.0
_SHARED_REFINE = 64


def _shared_summary(shared_log: np.ndarray, grid: np.ndarray, level: float, log_grid: bool) -> dict:
    """Median and equal-tailed interval of the shared posterior
    (`Ensemble.shared_log_D`), resolved below the grid step.

    With many tracks the shared posterior is narrower than a grid cell, so
    `_grid_summary` of its weights returns the peak cell and about one
    cell either side, whatever its real width. Its log is a sum of smooth
    per-track log likelihoods -- close to a parabola near the peak, which
    the spline's not-a-knot ends reproduce -- so it is interpolated on
    the grid it is smooth on (ln D) and summarized there."""
    x = np.log(grid) if log_grid else np.asarray(grid, dtype=float)
    near = np.flatnonzero(shared_log >= shared_log.max() - _SHARED_WINDOW)
    lo, hi = max(near[0] - 2, 0), min(near[-1] + 2, len(x) - 1)
    while hi - lo < 3 and (lo > 0 or hi < len(x) - 1):  # a cubic needs four points
        lo, hi = max(lo - 1, 0), min(hi + 1, len(x) - 1)
    if hi - lo < 3:
        return _grid_summary(np.exp(_normalized_log(shared_log)), grid, level, log_grid)
    fine = np.linspace(x[lo], x[hi], (hi - lo) * _SHARED_REFINE + 1)
    log_w = CubicSpline(x[lo : hi + 1], shared_log[lo : hi + 1])(fine)
    out = _grid_summary(np.exp(_normalized_log(log_w)), fine, level, log_grid=False)
    return {k: float(np.exp(v)) for k, v in out.items()} if log_grid else out


def _grid_mode(weights: np.ndarray, grid: np.ndarray) -> float:
    return float(grid[int(np.argmax(weights))])


def summarize(analysis: PosteriorAnalysis, track_ids: Optional[set] = None) -> dict:
    """Population-level numbers for `track_ids` (all for None), flat keys
    with units in their names so the saved JSON reads on its own.

    Counts are by diffusionkit's own statuses (`n_ok`, `n_excluded`,
    `n_invalid_input`), plus `n_D_at_grid_edge`: fitted tracks whose D
    posterior is cut by the grid's range, so their numbers depend on it.
    `n_detections` counts the fitted tracks' frames. `median_D_um2_s`/`q25`/`q75` are over the per-track
    posterior medians -- the typical track. `shared_D_*` is the shared-D
    posterior's median and 90% interval (resolved below the grid step,
    `_shared_summary`), and `deconvolved_D_*` the deconvolved
    distribution's median and 90% range (a spread across tracks, not an
    uncertainty), its mode, and the smoothness (`lambda`) its evidence
    chose. `D_floor_*_um2_s` are the median and 10%/90% quantiles of the
    tracks' localization floors."""
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
        "median_D_um2_s": _scalar(ok["D_median_um2_s"].median()),
        "q25_D_um2_s": _scalar(ok["D_median_um2_s"].quantile(0.25)),
        "q75_D_um2_s": _scalar(ok["D_median_um2_s"].quantile(0.75)),
        "n_D_at_grid_edge": int(ok["D_at_grid_edge"].sum()),
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
        shared = _shared_summary(ens.shared_log_D, analysis.D_grid_um2_s, analysis.level, log_grid=True)
        spread = _grid_summary(ens.deconvolved_D, analysis.D_grid_um2_s, analysis.level, log_grid=True)
        out.update(
            {f"shared_D_{k}_um2_s": v for k, v in shared.items()},
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
    are left out. `track_posteriors` adds every track's posterior weights
    as rows sorted by their median (`track_posteriors`), for the heat map.
    `floor` is the tracks' localization floor as (10%, median, 90%), None
    when no track has one."""
    fits = analysis.fits.filter(pl.col("posterior_status") == "ok")
    panels = []
    for name, ids in (groups or {"all": None}).items():
        ens = ensemble(analysis, ids)
        if ens is None:
            continue
        rows = fits if ids is None else fits.filter(pl.col("track_id").is_in(list(ids)))
        shared = _shared_summary(ens.shared_log_D, analysis.D_grid_um2_s, analysis.level, log_grid=True)
        panel = {
            "name": name,
            "n_tracks": ens.n_tracks,
            "medians": rows["D_median_um2_s"].to_numpy(),
            "deconvolved": ens.deconvolved_D,
            "deconvolved_band": ens.deconvolved_D_band,
            "level": analysis.level,
            "mean_posterior": ens.mean_posterior_D,
            "shared_interval": (shared["low"], shared["median"], shared["high"]),
            "floor": floor_band(rows["D_floor_um2_s"]),
        }
        if track_posteriors:
            weights = np.exp(analysis.log_post_D[analysis.rows_for(ids)])
            # Each row's median grid cell: where its CDF first reaches 1/2.
            order = np.argsort((np.cumsum(weights, axis=1) >= 0.5).argmax(axis=1), kind="stable")
            panel["track_posteriors"] = weights[order]
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
    """One fitted track's posterior weights and intervals, for
    `joint_plot.plot_track_posterior`; None if it wasn't fitted."""
    rows = np.flatnonzero(analysis.fitted_ids == track_id)
    if len(rows) == 0:
        return None
    fit = analysis.fits.filter(pl.col("track_id") == track_id).row(0, named=True)
    out = {
        "track_id": track_id,
        "n_frames": fit["n_frames"],
        "d_grid": analysis.D_grid_um2_s,
        "d_weights": np.exp(analysis.log_post_D[rows[0]]),
        "d_interval": (fit["D_low_um2_s"], fit["D_median_um2_s"], fit["D_high_um2_s"]),
        "d_floor": fit["D_floor_um2_s"],
    }
    return out


# --- tables the bundle saves --------------------------------------------


def posterior_long_table(analysis: PosteriorAnalysis) -> pl.DataFrame:
    """Every fitted track's log posterior, one row per (track, grid point):
    `track_id`, `D_um2_s`, `log_posterior` (normalized: logsumexp over a
    track's rows is 0)."""
    ids, grid = analysis.fitted_ids, analysis.D_grid_um2_s
    n_grid = len(grid)
    return pl.DataFrame(
        {
            "track_id": np.repeat(ids, n_grid),
            "D_um2_s": np.tile(grid, len(ids)),
            "log_posterior": analysis.log_post_D.reshape(-1),
        },
        schema={"track_id": pl.Int64, "D_um2_s": pl.Float64, "log_posterior": pl.Float64},
    )


def distributions_table(
    analysis: PosteriorAnalysis, groups: Optional[dict[str, Optional[set]]] = None
) -> Optional[pl.DataFrame]:
    """The ensemble distributions on the D grid, long by group: `group`,
    `n_tracks`, `D_um2_s`, and `Ensemble`'s four reads --
    `shared_log_posterior` (the summed log posteriors, 0 at their peak),
    `shared_posterior` (the same as weights: one value shared by every
    track, often within a cell or two), `mean_posterior` (the tracks'
    posteriors averaged) and `deconvolved` (weights; each sums to 1 over
    the grid within a group), with `deconvolved_low`/`deconvolved_high` its
    pointwise band at the credible level.

    `groups` maps a group name to its track ids; the default is one group,
    "all", over every fitted track."""
    parts = []
    for name, ids in (groups or {"all": None}).items():
        ens = ensemble(analysis, ids)
        if ens is None:
            continue
        n_grid = len(analysis.D_grid_um2_s)
        parts.append(
            pl.DataFrame(
                {
                    "group": [name] * n_grid,
                    "n_tracks": [ens.n_tracks] * n_grid,
                    "D_um2_s": analysis.D_grid_um2_s,
                    "shared_log_posterior": ens.shared_log_D,
                    "shared_posterior": ens.shared_D,
                    "mean_posterior": ens.mean_posterior_D,
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
    `D_um2_s`, `mean_posterior` (the flat-prior posteriors) and
    `deconvolved` (each track's posterior under the deconvolved
    distribution, mean over its draws), with `deconvolved_low`/`_high` the
    pointwise band at the credible level. Each is that length group's share
    of all the group's tracks (or detections): summed over lengths and the
    grid, a weight's rows add up to 1."""
    parts = []
    for name, ids in (groups or {"all": None}).items():
        for weight in LENGTH_WEIGHTS:
            comp = length_composition(analysis, ids, weight)
            if comp is None:
                continue
            lo, hi = comp.band(analysis.level)
            mean = comp.deconvolved.mean(axis=0)
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
                            "mean_posterior": comp.pooled[j],
                            "deconvolved": mean[j],
                            "deconvolved_low": lo[j],
                            "deconvolved_high": hi[j],
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
    track); `results` the per-track analysis columns (posterior medians and
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
    """The posterior and distribution tables, as `write_diffusion_results`
    keyword arguments. The ensemble is over the tracks passing the filters
    (`ids`), split by region class when there are several."""
    groups = {"all": ids, **(by_class or {})}
    return dict(
        posterior_D=posterior_long_table(analysis),
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
        "analysis": "diffusionkit.gridpost grid posteriors, flat prior in ln D over the D grid",
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
    (or its posteriors don't lie on the grid its summary records): it has
    to be re-run."""


@dataclass(frozen=True)
class SavedAnalysis:
    """A saved analysis, restored: the posteriors, the MSD comparison's
    columns and the NUTS fits' rows when they were saved, and the
    `diffusion_summary.json` settings they were run and summarized with."""

    analysis: PosteriorAnalysis
    msd: Optional[pl.DataFrame]
    nuts_rows: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _posterior_matrix(table: pl.DataFrame, grid_col: str, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`posterior_long_table`'s table back to `(track ids, log posteriors)`."""
    table = table.sort("track_id", grid_col)
    ids = table["track_id"].unique(maintain_order=True).to_numpy().astype(np.int64)
    n_grid = len(grid)
    if table.height != len(ids) * n_grid or not np.allclose(
        table[grid_col].to_numpy()[:n_grid], grid, rtol=1e-12, atol=0
    ):
        raise StaleAnalysisError(f"its {grid_col} posteriors are not on the grid its summary records")
    return ids, table["log_posterior"].to_numpy().reshape(len(ids), n_grid)


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
    posterior_D: pl.DataFrame,
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
    fitted_ids, log_post_D = _posterior_matrix(posterior_D, "D_um2_s", np.exp(options.u_D()))

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
        log_post_D=log_post_D,
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
