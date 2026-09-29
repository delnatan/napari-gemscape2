"""GUI-free diffusion analysis: this project's tracks in, diffusionkit's
grid posteriors out, plus the tables the widget shows and the bundle saves.

`tracks_to_diffusionkit_df` is the bridge. spotsolve's per-localization
position error is `se_x`/`se_y` (the CRLB from the fit's Fisher
information); diffusionkit calls the same quantity `sigma_x_um`/
`sigma_y_um`. Only the name and the unit differ, so it is renamed and
scaled here rather than duplicated upstream.

The analysis is `diffusionkit.gridpost`: for each track, the exact Gaussian
displacement likelihood evaluated on a grid in ln D (and the fBm exponent
alpha with its scale, D at one frame, integrated out over the same grid),
both with the camera exposure's motion blur modelled. `analyze_posteriors`
is diffusionkit's own `gridpost.analyze_tracks`, asked to keep each track's
posterior vector (what the ensemble is built from and what the bundle
saves), and every grid comes from the one `GridPostOptions` the run is
given -- the same object a diffusionkit script would pass, so the two agree
number for number. The D grid's range is the flat prior's support: a
posterior cut by an edge is flagged in `D_at_grid_edge`.

  - **per track**: the posterior median of D and its equal-tailed 90%
    interval (`D_low`/`D_high` = the 5% and 95% quantiles), likewise alpha;
  - **shared**: the per-track log posteriors added up -- the posterior of
    one value shared by every track. Sharp, but only honest when the
    tracks really do share it;
  - **mean posterior**: the tracks' posteriors averaged -- where they put
    the value, blurred by each track's own uncertainty and prior;
  - **deconvolved**: `gridpost.deconvolve`, the smoothed nonparametric
    MLE of how D (and alpha) is distributed across tracks, with each
    track's own uncertainty taken out rather than averaged in. Peak
    locations and the mass in each mode are robust; peak widths are
    resolution-limited;
  - **localization floor** (`D_floor_um2_s`, diffusionkit's
    `posterior.localization_floor`): the D at which a track's motion per
    frame equals its localization noise. Every D axis shows it as a
    reference scale (its median over the tracks, and their 10-90% band),
    not as a threshold anything is classified by.

There are no per-track shape metrics (radius of gyration, straightness,
...): they measure the same bending of the MSD that alpha does, with less
information, and are easily computed from the tracks when wanted.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np
import polars as pl
from diffusionkit import Acquisition
from diffusionkit.classic import MSDOptions
from diffusionkit.classic import analyze_tracks as analyze_msd
from diffusionkit.gridpost import GridPostOptions
from diffusionkit.gridpost import analyze_tracks as dk_analyze_tracks
from diffusionkit.gridpost import deconvolve as dk_deconvolve
from diffusionkit.gridpost import posterior as dk_post
from diffusionkit.gridpost import posterior_alpha as dk_post_alpha
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp

from napari_gemscape2.pipeline import filter_mask

# The credible-interval mass: `_low`/`_high` are its (1 - LEVEL)/2 and
# (1 + LEVEL)/2 quantiles, 5% and 95%.
LEVEL = 0.9
# diffusionkit's own hard minimum: 3 frames give the 2 displacements the
# whitening needs.
MIN_FRAMES = 3
# `gridpost.deconvolve`'s defaults: EM iterations, and the Gaussian
# smoothing per iteration in grid cells (~2.3% of D per cell).
DECONVOLVE_ITERS = 500
DECONVOLVE_SMOOTH = 0.5

# `GridPostOptions`' grid fields, the ones settable here (the widget's Grid
# section, the CLI's `[diffusion] grid`). Unset ones are diffusionkit's
# defaults; whichever were used are saved (`analysis_summary`'s "grid").
GRID_FIELDS = ("D_min_um2_s", "D_max_um2_s", "n_D", "alpha_min", "alpha_max", "n_alpha")


def posterior_options(
    min_frames: int = MIN_FRAMES,
    alpha: bool = True,
    grid: Optional[dict] = None,
) -> GridPostOptions:
    """The `GridPostOptions` a run here uses: `grid` holds any of
    `GRID_FIELDS`; the credible level is fixed at `LEVEL`, and alpha's
    likelihood is diffusionkit's default ("auto": exact for short tracks,
    debiased Whittle for long ones). Raises ValueError for an unknown key
    or an invalid grid."""
    unknown = set(grid or {}) - set(GRID_FIELDS)
    if unknown:
        raise ValueError(f"unknown grid settings: {', '.join(sorted(unknown))}")
    return GridPostOptions(min_frames=min_frames, level=LEVEL, compute_alpha=alpha, **(grid or {}))


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
# track property, so it can sit in the same filter panel as `alpha` and be
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
    "alpha_status": pl.String,
    "alpha_median": pl.Float64,
    "alpha_low": pl.Float64,
    "alpha_high": pl.Float64,
    # What the track taught about alpha (bits from the flat alpha prior):
    # near 0, alpha_median is the prior's midpoint, not a measurement.
    "alpha_info_bits": pl.Float64,
}

_ALPHA_COLUMNS = ("alpha_status", "alpha_median", "alpha_low", "alpha_high", "alpha_info_bits")

# The localization floor's band on a plot: these quantiles of the tracks'
# own floors, around their median.
FLOOR_BAND = (0.1, 0.9)
# Columns whose scale is the localization floor, `D_floor_um2_s`.
FLOOR_COLUMNS = ("D_median_um2_s", "D_low_um2_s", "D_high_um2_s", "D_msd_um2_s")
# The alpha medians are drawn in these strata of alpha_info_bits, one gray
# step each (a display, not a cut: every track stays in).
ALPHA_BITS_STRATA = (0.5, 1.5)

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
    `log_post_D` over `D_grid_um2_s`, `log_post_alpha` over `alpha_grid`
    for `alpha_ids` (None when alpha was not computed) -- both grids
    `options`'. `tracks_sha256` is `tracks_fingerprint` of the tracks it
    was run on.
    """

    fits: pl.DataFrame
    acquisition: Acquisition
    options: GridPostOptions
    fitted_ids: np.ndarray
    log_post_D: np.ndarray
    alpha_ids: Optional[np.ndarray]
    log_post_alpha: Optional[np.ndarray]
    tracks_sha256: Optional[str] = None
    # `ensemble`'s results by (track ids, `Deconvolution`): the summary, the
    # figures and the saved tables all read the same few, and each costs
    # a deconvolution (~0.2 ms per track per 100 iterations).
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

    @property
    def alpha_grid(self) -> np.ndarray:
        return self.options.alphas()

    @property
    def has_alpha(self) -> bool:
        return self.log_post_alpha is not None and len(self.alpha_ids) > 0

    def rows_for(self, track_ids: Optional[set], *, alpha: bool = False) -> np.ndarray:
        """Row indices into the D (or alpha) arrays for `track_ids`, or all
        rows for None."""
        ids = self.alpha_ids if alpha else self.fitted_ids
        if track_ids is None:
            return np.arange(len(ids))
        return np.flatnonzero(np.isin(ids, list(track_ids)))


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
    """Grid posteriors over D (and alpha) for every track in `tracks`, on `options`' grids (`posterior_options`).

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
    by_model = {
        model: result.fits.filter(pl.col("model") == model).drop("model", "n_frames")
        for model in ("posterior_D", "posterior_alpha")
    }
    D = by_model["posterior_D"].select(
        "track_id",
        pl.col("status").alias("posterior_status"),
        pl.col("message").alias("_D_message"),
        pl.col("D_post_median_um2_s").alias("D_median_um2_s"),
        pl.col("D_post_lo_um2_s").alias("D_low_um2_s"),
        pl.col("D_post_hi_um2_s").alias("D_high_um2_s"),
        pl.col("D_post_info_bits").alias("D_info_bits"),
        "D_floor_um2_s",
    )
    alpha = by_model["posterior_alpha"].select(
        "track_id",
        pl.col("status").alias("alpha_status"),
        pl.col("message").alias("_alpha_message"),
        pl.col("alpha_post_median").alias("alpha_median"),
        pl.col("alpha_post_lo").alias("alpha_low"),
        pl.col("alpha_post_hi").alias("alpha_high"),
        pl.col("alpha_post_info_bits").alias("alpha_info_bits"),
    )
    edge = pl.DataFrame(
        {"track_id": post.D_track_ids, "D_at_grid_edge": _at_grid_edge(post.log_post_D)},
        schema={"track_id": pl.Int64, "D_at_grid_edge": pl.Boolean},
    )
    n_frames = result.fits.filter(pl.col("model") == "posterior_D").select("track_id", "n_frames")
    fits = (
        n_frames.join(D, on="track_id", how="left")
        .join(alpha, on="track_id", how="left")
        .join(edge, on="track_id", how="left")
        .with_columns(
            # One message per track: the D posterior's, then the alpha
            # posterior's when it says something else.
            pl.when((pl.col("_alpha_message") != "") & (pl.col("_alpha_message") != pl.col("_D_message")))
            .then(pl.concat_str(["_D_message", "_alpha_message"], separator="; ").str.strip_chars("; "))
            .otherwise(pl.col("_D_message"))
            .alias("message")
        )
        .select(FIT_SCHEMA.keys())
        .cast(FIT_SCHEMA)
    )
    alpha_ran = options.compute_alpha
    return PosteriorAnalysis(
        fits=fits,
        acquisition=acquisition,
        options=options,
        fitted_ids=post.D_track_ids,
        log_post_D=post.log_post_D,
        alpha_ids=post.alpha_track_ids if alpha_ran else None,
        log_post_alpha=post.log_post_alpha if alpha_ran else None,
        tracks_sha256=tracks_fingerprint(tracks),
    )


# --- the ensemble --------------------------------------------------------


@dataclass(frozen=True)
class Deconvolution:
    """`gridpost.deconvolve`'s settings. `iters` EM iterations: one from
    the flat start is just the mean posterior, more remove the blur the
    tracks' own uncertainty adds. `smooth` is the Gaussian smoothing per
    iteration in grid cells (a cell is ~2.3% of D on the default grid):
    less lets peaks sharpen toward spikes, more widens them -- read a width
    as resolution-limited either way. D is spread over the analysis's own
    D grid, the same range each track's prior has (`GridPostOptions`)."""

    iters: int = DECONVOLVE_ITERS
    smooth: float = DECONVOLVE_SMOOTH

    def record(self) -> dict:
        """For `diffusion_summary.json` (and back through `from_record`)."""
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict) -> "Deconvolution":
        """A saved record."""
        return cls(**record)


@dataclass(frozen=True)
class Ensemble:
    """The population read of one set of tracks' posteriors, on the D grid
    (and the alpha grid, when alpha was computed).

    `shared_log_*` is the sum of the tracks' normalized log posteriors,
    shifted to 0 at its peak (its absolute level, a sum over n tracks, is
    thousands below 0 and means nothing), and `shared_*` the same thing
    as grid weights: the posterior of one value shared by every track.
    With many tracks it is narrower than a grid cell, so its weights sit
    in a cell or two; `_shared_summary` reads it below the grid step. `mean_posterior_*` is the average
    of the tracks' posteriors -- where the tracks put the value, blurred
    by each track's own uncertainty and prior; it is the deconvolution's
    starting point. `deconvolved_*` is `gridpost.deconvolve`'s
    distribution across the tracks, that blur removed: for alpha, of the
    K-marginalized likelihoods (flat alpha prior), so a track with a flat
    alpha posterior adds nothing, where it adds mass at the prior's
    midpoint to the mean posterior and the histogram of medians. All
    weights sum to 1 over their grid."""

    n_tracks: int
    shared_log_D: np.ndarray
    shared_D: np.ndarray
    mean_posterior_D: np.ndarray
    deconvolved_D: np.ndarray
    n_tracks_alpha: int = 0
    shared_log_alpha: Optional[np.ndarray] = None
    shared_alpha: Optional[np.ndarray] = None
    mean_posterior_alpha: Optional[np.ndarray] = None
    deconvolved_alpha: Optional[np.ndarray] = None


_ENSEMBLE_CACHE_SIZE = 32


def ensemble(
    analysis: PosteriorAnalysis,
    track_ids: Optional[set] = None,
    deconvolution: Deconvolution = Deconvolution(),
) -> Optional[Ensemble]:
    """The `Ensemble` over `track_ids` (all fitted tracks for None), or
    None when none of them was fitted."""
    key = (None if track_ids is None else frozenset(track_ids), deconvolution)
    if key in analysis._ensembles:
        return analysis._ensembles[key]
    result = _ensemble(analysis, track_ids, deconvolution)
    if len(analysis._ensembles) >= _ENSEMBLE_CACHE_SIZE:
        analysis._ensembles.clear()
    analysis._ensembles[key] = result
    return result


def _ensemble(
    analysis: PosteriorAnalysis, track_ids: Optional[set], deconvolution: Deconvolution
) -> Optional[Ensemble]:
    rows = analysis.rows_for(track_ids)
    if len(rows) == 0:
        return None
    log_post = analysis.log_post_D[rows]
    shared_log = log_post.sum(axis=0)
    shared_log = shared_log - shared_log.max()
    deconvolved = dk_deconvolve.deconvolve(
        log_post, dk_post.flat(analysis.u_D), iters=deconvolution.iters, smooth=deconvolution.smooth
    )
    result = dict(
        n_tracks=len(rows),
        shared_log_D=shared_log,
        shared_D=np.exp(_normalized_log(shared_log)),
        mean_posterior_D=np.exp(log_post).mean(axis=0),
        deconvolved_D=deconvolved,
    )
    if analysis.has_alpha:
        alpha_rows = analysis.rows_for(track_ids, alpha=True)
        if len(alpha_rows):
            # Flat alpha prior: each row is also the track's K-marginalized
            # log likelihood up to a constant, what `deconvolve` takes.
            log_post_alpha = analysis.log_post_alpha[alpha_rows]
            shared_alpha = log_post_alpha.sum(axis=0)
            shared_alpha = shared_alpha - shared_alpha.max()
            result.update(
                n_tracks_alpha=len(alpha_rows),
                shared_log_alpha=shared_alpha,
                shared_alpha=np.exp(_normalized_log(shared_alpha)),
                mean_posterior_alpha=np.exp(log_post_alpha).mean(axis=0),
                deconvolved_alpha=dk_deconvolve.deconvolve(
                    log_post_alpha,
                    dk_post_alpha.flat_alpha(analysis.alpha_grid),
                    iters=deconvolution.iters,
                    smooth=deconvolution.smooth,
                ),
            )
    return Ensemble(**result)


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
    (`Ensemble.shared_log_*`), resolved below the grid step.

    With many tracks the shared posterior is narrower than a grid cell, so
    `_grid_summary` of its weights returns the peak cell and about one
    cell either side, whatever its real width. Its log is a sum of smooth
    per-track log likelihoods -- close to a parabola near the peak, which
    the spline's not-a-knot ends reproduce -- so it is interpolated on
    the grid it is smooth on (ln D, or alpha) and summarized there."""
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


def summarize(
    analysis: PosteriorAnalysis,
    track_ids: Optional[set] = None,
    deconvolution: Deconvolution = Deconvolution(),
) -> dict:
    """Population-level numbers for `track_ids` (all for None), flat keys
    with units in their names so the saved JSON reads on its own.

    Counts are by diffusionkit's own statuses (`n_ok`, `n_excluded`,
    `n_invalid_input`), plus `n_D_at_grid_edge`: fitted tracks whose D
    posterior is cut by the grid's range, so their numbers depend on it. `median_D_um2_s`/`q25`/`q75` are over the per-track
    posterior medians -- the typical track. `shared_D_*` is the shared-D
    posterior's median and 90% interval (resolved below the grid step,
    `_shared_summary`), and `deconvolved_D_*` the deconvolved
    distribution's median and 90% range (a spread across tracks, not an
    uncertainty) plus its mode; likewise `shared_alpha_*` and
    `deconvolved_alpha_*`. `D_floor_*_um2_s` are the median and 10%/90%
    quantiles of the tracks' localization floors, and
    `median_alpha_info_bits` what the typical track taught about alpha."""
    fits = analysis.fits
    if track_ids is not None:
        fits = fits.filter(pl.col("track_id").is_in(list(track_ids)))
    statuses = dict(fits.group_by("posterior_status").len().iter_rows())
    ok = fits.filter(pl.col("posterior_status") == "ok")
    out: dict = {
        "n_tracks": fits.height,
        **{f"n_{status}": int(count) for status, count in sorted(statuses.items())},
        "n_frames_min": _scalar(ok["n_frames"].min()),
        "n_frames_median": _scalar(ok["n_frames"].median()),
        "n_frames_max": _scalar(ok["n_frames"].max()),
        "median_D_um2_s": _scalar(ok["D_median_um2_s"].median()),
        "q25_D_um2_s": _scalar(ok["D_median_um2_s"].quantile(0.25)),
        "q75_D_um2_s": _scalar(ok["D_median_um2_s"].quantile(0.75)),
        "n_D_at_grid_edge": int(ok["D_at_grid_edge"].sum()),
    }
    alpha_ok = fits.filter(pl.col("alpha_status") == "ok")
    if alpha_ok.height:
        out.update(
            n_alpha=alpha_ok.height,
            median_alpha=_scalar(alpha_ok["alpha_median"].median()),
            q25_alpha=_scalar(alpha_ok["alpha_median"].quantile(0.25)),
            q75_alpha=_scalar(alpha_ok["alpha_median"].quantile(0.75)),
            median_alpha_info_bits=_scalar(alpha_ok["alpha_info_bits"].median()),
        )
    floors = ok["D_floor_um2_s"].drop_nulls()
    if floors.len():
        out.update(
            D_floor_median_um2_s=_scalar(floors.median()),
            D_floor_q10_um2_s=_scalar(floors.quantile(FLOOR_BAND[0])),
            D_floor_q90_um2_s=_scalar(floors.quantile(FLOOR_BAND[1])),
        )
    ens = ensemble(analysis, track_ids, deconvolution)
    if ens is not None:
        shared = _shared_summary(ens.shared_log_D, analysis.D_grid_um2_s, analysis.level, log_grid=True)
        spread = _grid_summary(ens.deconvolved_D, analysis.D_grid_um2_s, analysis.level, log_grid=True)
        out.update(
            {f"shared_D_{k}_um2_s": v for k, v in shared.items()},
            **{f"deconvolved_D_{k}_um2_s": v for k, v in spread.items()},
            deconvolved_D_mode_um2_s=_grid_mode(ens.deconvolved_D, analysis.D_grid_um2_s),
        )
        if ens.shared_log_alpha is not None:
            shared_a = _shared_summary(ens.shared_log_alpha, analysis.alpha_grid, analysis.level, log_grid=False)
            spread_a = _grid_summary(ens.deconvolved_alpha, analysis.alpha_grid, analysis.level, log_grid=False)
            out.update(
                {f"shared_alpha_{k}": v for k, v in shared_a.items()},
                **{f"deconvolved_alpha_{k}": v for k, v in spread_a.items()},
                deconvolved_alpha_mode=_grid_mode(ens.deconvolved_alpha, analysis.alpha_grid),
            )
    return out


def summarize_by_group(
    analysis: PosteriorAnalysis, groups: pl.DataFrame, deconvolution: Deconvolution = Deconvolution()
) -> dict[str, dict]:
    """`summarize` per group -- per region class, when each region's tracks
    were linked on their own. `groups` has `track_id` and `group`; groups
    come back in their order of first appearance."""
    names = groups["group"].unique(maintain_order=True).drop_nulls().to_list()
    return {
        name: summarize(
            analysis, set(groups.filter(pl.col("group") == name)["track_id"].to_list()), deconvolution
        )
        for name in names
    }


def _scalar(value) -> Optional[float]:
    return None if value is None else float(value)


def ensemble_panels(
    analysis: PosteriorAnalysis,
    groups: Optional[dict[str, Optional[set]]] = None,
    deconvolution: Deconvolution = Deconvolution(),
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
        ens = ensemble(analysis, ids, deconvolution)
        if ens is None:
            continue
        rows = fits if ids is None else fits.filter(pl.col("track_id").is_in(list(ids)))
        shared = _shared_summary(ens.shared_log_D, analysis.D_grid_um2_s, analysis.level, log_grid=True)
        panel = {
            "name": name,
            "n_tracks": ens.n_tracks,
            "medians": rows["D_median_um2_s"].to_numpy(),
            "deconvolved": ens.deconvolved_D,
            "mean_posterior": ens.mean_posterior_D,
            "shared_interval": (shared["low"], shared["median"], shared["high"]),
            "floor": floor_band(rows["D_floor_um2_s"]),
        }
        if track_posteriors:
            weights = np.exp(analysis.log_post_D[analysis.rows_for(ids)])
            # Each row's median grid cell: where its CDF first reaches 1/2.
            order = np.argsort((np.cumsum(weights, axis=1) >= 0.5).argmax(axis=1), kind="stable")
            panel["track_posteriors"] = weights[order]
        if ens.shared_log_alpha is not None:
            shared_a = _shared_summary(ens.shared_log_alpha, analysis.alpha_grid, analysis.level, log_grid=False)
            alpha_rows = rows.filter(pl.col("alpha_median").is_not_null())
            panel["alpha_medians"] = alpha_rows["alpha_median"].to_numpy()
            panel["alpha_info_bits"] = alpha_rows["alpha_info_bits"].to_numpy()
            panel["alpha_bits_strata"] = ALPHA_BITS_STRATA
            panel["alpha_deconvolved"] = ens.deconvolved_alpha
            panel["alpha_shared_interval"] = (shared_a["low"], shared_a["median"], shared_a["high"])
        panels.append(panel)
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
    if analysis.has_alpha:
        alpha_rows = np.flatnonzero(analysis.alpha_ids == track_id)
        if len(alpha_rows):
            out.update(
                alpha_grid=analysis.alpha_grid,
                alpha_weights=np.exp(analysis.log_post_alpha[alpha_rows[0]]),
                alpha_interval=(fit["alpha_low"], fit["alpha_median"], fit["alpha_high"]),
                alpha_info_bits=fit["alpha_info_bits"],
            )
    return out


# --- tables the bundle saves --------------------------------------------


def posterior_long_table(analysis: PosteriorAnalysis, *, alpha: bool = False) -> Optional[pl.DataFrame]:
    """Every fitted track's log posterior, one row per (track, grid point):
    `track_id`, `D_um2_s` (or `alpha`), `log_posterior` (normalized:
    logsumexp over a track's rows is 0). None for alpha when it wasn't
    computed."""
    if alpha:
        if not analysis.has_alpha:
            return None
        ids, log_post, grid, name = analysis.alpha_ids, analysis.log_post_alpha, analysis.alpha_grid, "alpha"
    else:
        ids, log_post, grid, name = analysis.fitted_ids, analysis.log_post_D, analysis.D_grid_um2_s, "D_um2_s"
    n_grid = len(grid)
    return pl.DataFrame(
        {
            "track_id": np.repeat(ids, n_grid),
            name: np.tile(grid, len(ids)),
            "log_posterior": log_post.reshape(-1),
        },
        schema={"track_id": pl.Int64, name: pl.Float64, "log_posterior": pl.Float64},
    )


def distributions_table(
    analysis: PosteriorAnalysis,
    groups: Optional[dict[str, Optional[set]]] = None,
    *,
    alpha: bool = False,
    deconvolution: Deconvolution = Deconvolution(),
) -> Optional[pl.DataFrame]:
    """The ensemble distributions on their grid, long by group: `group`,
    `n_tracks`, the grid column (`D_um2_s` or `alpha`), and
    `Ensemble`'s four reads -- `shared_log_posterior` (the summed log
    posteriors, 0 at their peak), `shared_posterior` (the same as
    weights: one value shared by every track, often within a cell or
    two), `mean_posterior` (the tracks' posteriors averaged) and
    `deconvolved` (weights; each sums to 1 over the grid within a group).

    `groups` maps a group name to its track ids; the default is one group,
    "all", over every fitted track."""
    groups = groups or {"all": None}
    parts = []
    for name, ids in groups.items():
        ens = ensemble(analysis, ids, deconvolution)
        if ens is None:
            continue
        if alpha:
            if ens.shared_log_alpha is None:
                continue
            part = {
                "alpha": analysis.alpha_grid,
                "shared_log_posterior": ens.shared_log_alpha,
                "shared_posterior": ens.shared_alpha,
                "mean_posterior": ens.mean_posterior_alpha,
                "deconvolved": ens.deconvolved_alpha,
            }
            n = ens.n_tracks_alpha
            grid_len = len(analysis.alpha_grid)
        else:
            part = {
                "D_um2_s": analysis.D_grid_um2_s,
                "shared_log_posterior": ens.shared_log_D,
                "shared_posterior": ens.shared_D,
                "mean_posterior": ens.mean_posterior_D,
                "deconvolved": ens.deconvolved_D,
            }
            n = ens.n_tracks
            grid_len = len(analysis.D_grid_um2_s)
        parts.append(
            pl.DataFrame({"group": [name] * grid_len, "n_tracks": [n] * grid_len, **part})
        )
    return pl.concat(parts) if parts else None


# --- MSD comparison -----------------------------------------------------


# The MSD comparison's lag window: diffusionkit's own default. Not exposed
# -- the MSD fits are a cross-check here, not an analysis to tune.
MSD_MAX_LAG = 3


def msd_fits_blur_free(tracks: pl.DataFrame, dt_s: float, min_frames: int) -> pl.DataFrame:
    """diffusionkit.classic's MSD fits of `tracks_to_diffusionkit_df`'s
    table, for comparison with the posteriors.

    The MSD fits have no blur model, so diffusionkit excludes them when
    `exposure_s > 0`; the comparison therefore runs them with the exposure
    treated as 0, which is exactly the assumption that biases them, and is
    labelled as such wherever it shows."""
    options = MSDOptions(max_lag=MSD_MAX_LAG, min_frames=max(min_frames, 5), localization="provided")
    return analyze_msd(tracks, Acquisition(dt_s=dt_s), options).fits


def msd_track_table(fits: pl.DataFrame) -> pl.DataFrame:
    """diffusionkit.classic's MSD fits, one row per track: `D_msd_um2_s`
    from the linear fit, `K_msd_um2_s_alpha`/`alpha_msd` from the power
    law. No uncertainties -- diffusionkit estimates none for MSD fits."""
    brownian = fits.filter(pl.col("model") == "brownian").select(
        "track_id", pl.col("D_um2_s").alias("D_msd_um2_s")
    )
    power_law = fits.filter(pl.col("model") == "power_law").select(
        "track_id",
        pl.col("K_um2_s_alpha").alias("K_msd_um2_s_alpha"),
        pl.col("alpha").alias("alpha_msd"),
    )
    return brownian.join(power_law, on="track_id", how="full", coalesce=True)


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
    deconvolution: Deconvolution = Deconvolution(),
) -> dict:
    """The posterior and distribution tables, as `write_diffusion_results`
    keyword arguments. The ensemble is over the tracks passing the filters
    (`ids`), split by region class when there are several."""
    groups = {"all": ids, **(by_class or {})}
    return dict(
        posterior_D=posterior_long_table(analysis),
        posterior_alpha=posterior_long_table(analysis, alpha=True),
        distributions_D=distributions_table(analysis, groups, deconvolution=deconvolution),
        distributions_alpha=distributions_table(analysis, groups, alpha=True, deconvolution=deconvolution),
    )


def analysis_summary(
    analysis: PosteriorAnalysis,
    ids: Optional[set],
    by_class: Optional[dict],
    *,
    msd_comparison: bool,
    deconvolution: Deconvolution = Deconvolution(),
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
        "alpha_grid": (
            [float(analysis.alpha_grid[0]), float(analysis.alpha_grid[-1]), analysis.options.n_alpha]
            if analysis.has_alpha
            else None
        ),
        # alpha's likelihood: exact below `whittle_min_frames`, debiased
        # Whittle from there on (diffusionkit's `alpha_method`).
        "alpha_likelihood": (
            {"method": analysis.options.alpha_method, "whittle_min_frames": analysis.options.alpha_whittle_min_frames}
            if analysis.has_alpha
            else None
        ),
        "msd_comparison": msd_comparison,
        "tracks_sha256": analysis.tracks_sha256,
        "deconvolution": deconvolution.record(),
        **summarize(analysis, ids, deconvolution),
    }
    if by_class:
        summary["by_region_class"] = {
            name: summarize(analysis, gids, deconvolution) for name, gids in by_class.items()
        }
    return summary


def posterior_results_table(analysis: PosteriorAnalysis, msd: Optional[pl.DataFrame]) -> pl.DataFrame:
    """The per-track result columns a posterior run adds to the tracks
    table (and to `tracks_summary.csv`): the posterior's, without the alpha
    group when alpha wasn't computed, plus `msd_track_table`'s when the
    MSD comparison ran."""
    display = analysis.fits.select(POSTERIOR_COLUMNS)
    if not analysis.has_alpha:
        display = display.drop(_ALPHA_COLUMNS)
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

    @property
    def deconvolution(self) -> Deconvolution:
        return Deconvolution.from_record(self.summary["deconvolution"])


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
    unknown = set(summary["grid"]) - set(GRID_FIELDS)
    if unknown:
        raise StaleAnalysisError(f"saved with grid settings this version doesn't have: {', '.join(sorted(unknown))}")
    alpha_likelihood = summary["alpha_likelihood"] or {}
    return GridPostOptions(
        min_frames=summary["min_frames"],
        level=summary["credible_level"],
        compute_alpha=summary["alpha_grid"] is not None,
        **({"alpha_method": alpha_likelihood["method"],
            "alpha_whittle_min_frames": alpha_likelihood["whittle_min_frames"]} if alpha_likelihood else {}),
        **summary["grid"],
    )


def restore_analysis(
    tracks: pl.DataFrame,
    *,
    summary: dict,
    tracks_summary: pl.DataFrame,
    posterior_D: pl.DataFrame,
    posterior_alpha: Optional[pl.DataFrame],
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
    alpha_ids = log_post_alpha = None
    if options.compute_alpha:
        if posterior_alpha is not None:
            alpha_ids, log_post_alpha = _posterior_matrix(posterior_alpha, "alpha", options.alphas())
        else:  # alpha ran, and no track got one
            alpha_ids, log_post_alpha = np.empty(0, dtype=np.int64), np.empty((0, options.n_alpha))

    acquisition = Acquisition(dt_s=summary["dt_s"], exposure_s=summary["exposure_s"])
    # The alpha columns are absent from a run without alpha
    # (`posterior_results_table`).
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

    analysis = PosteriorAnalysis(
        fits=fits,
        acquisition=acquisition,
        options=options,
        fitted_ids=fitted_ids,
        log_post_D=log_post_D,
        alpha_ids=alpha_ids,
        log_post_alpha=log_post_alpha,
        tracks_sha256=saved_sha,
    )

    msd_cols = [c for c in ("D_msd_um2_s", "K_msd_um2_s_alpha", "alpha_msd") if c in ran.columns]
    msd = None
    if summary.get("msd_comparison") and msd_cols:
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
