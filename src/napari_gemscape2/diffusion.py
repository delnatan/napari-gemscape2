"""GUI-free diffusion analysis: this project's tracks in, diffusionkit's
grid posteriors out, plus the tables the widget shows and the bundle saves.

`tracks_to_diffusionkit_df` is the bridge. spotsolve's per-localization
position error is `se_x`/`se_y` (the CRLB from the fit's Fisher
information); diffusionkit calls the same quantity `sigma_x_um`/
`sigma_y_um`. Only the name and the unit differ, so it is renamed and
scaled here rather than duplicated upstream.

The analysis is `diffusionkit.gridpost`: for each track, the exact Gaussian
displacement likelihood evaluated on a grid in ln D (and, with no exposure
blur, the fBm exponent alpha with K integrated out). `analyze_posteriors`
is diffusionkit's own `gridpost.analyze_tracks`, asked to keep each track's
posterior vector (what the ensemble is built from and what the bundle
saves), and every grid comes from the one `GridPostOptions` the run is
given -- the same object a diffusionkit script would pass, so the two agree
number for number. The D grid's range is the flat prior's support: a
posterior cut by an edge is flagged in `D_at_grid_edge`.

  - **per track**: the posterior median of D and its equal-tailed 90%
    interval (`D_low`/`D_high` = the 5% and 95% quantiles), likewise alpha;
  - **summed**: the per-track log posteriors added up -- the posterior of
    one D shared by every track. Sharp, but only honest when the tracks
    really do share a D;
  - **deconvolved**: `gridpost.deconvolve`, the smoothed nonparametric
    MLE of how D is distributed across tracks, with each track's own
    uncertainty taken out rather than averaged in. Peak locations and the
    mass in each mode are robust; peak widths are resolution-limited.

`track_geometry` (radius of gyration, straightness, ...) is model-free
shape, ported from diffusionkit when that package narrowed to analysis.
"""

from __future__ import annotations

import hashlib
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
GRID_FIELDS = ("D_min_um2_s", "D_max_um2_s", "n_D", "alpha_min", "alpha_max", "n_alpha", "n_K")


def posterior_options(min_frames: int = MIN_FRAMES, alpha: bool = True, grid: Optional[dict] = None) -> GridPostOptions:
    """The `GridPostOptions` a run here uses: `grid` holds any of
    `GRID_FIELDS`; the credible level is fixed at `LEVEL`. Raises
    ValueError for an unknown key or an invalid grid."""
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


def track_geometry(tracks: pl.DataFrame) -> pl.DataFrame:
    """One row per track of `tracks_to_diffusionkit_df`'s table.

    - `radius_of_gyration_um`: RMS distance of the positions from their own
      centroid -- the track's spatial extent.
    - `net_displacement_um`: straight-line distance from first to last position.
    - `straightness`: net displacement over the summed step lengths, in
      [0, 1]; near 1 for directed motion.
    - `gyration_asymmetry`: `(l1 - l2)^2 / (l1 + l2)^2` from the gyration
      tensor's eigenvalues, in [0, 1]; 0 for an isotropic cloud, 1 for a line.

    Raw geometry, so localization error inflates all of it, most on short
    or slow tracks. Null where undefined: everything for a single-point
    track, the two ratios for a track that never moved.
    """
    ordered = tracks.sort(["track_id", "frame"]).with_columns(
        (pl.col("x_um").diff().over("track_id") ** 2 + pl.col("y_um").diff().over("track_id") ** 2)
        .sqrt()
        .alias("_step_um")
    )
    per_track = ordered.group_by("track_id").agg(
        pl.len().alias("_n"),
        pl.col("x_um").var(ddof=0).alias("_txx"),
        pl.col("y_um").var(ddof=0).alias("_tyy"),
        ((pl.col("x_um") - pl.col("x_um").mean()) * (pl.col("y_um") - pl.col("y_um").mean()))
        .mean()
        .alias("_txy"),
        (
            (pl.col("x_um").last() - pl.col("x_um").first()) ** 2
            + (pl.col("y_um").last() - pl.col("y_um").first()) ** 2
        )
        .sqrt()
        .alias("net_displacement_um"),
        pl.col("_step_um").sum().alias("_path_um"),
    )
    trace = pl.col("_txx") + pl.col("_tyy")
    multi = pl.col("_n") > 1
    return per_track.select(
        "track_id",
        pl.when(multi).then(trace.sqrt()).alias("radius_of_gyration_um"),
        pl.when(multi).then(pl.col("net_displacement_um")).alias("net_displacement_um"),
        pl.when(multi & (pl.col("_path_um") > 0))
        .then(pl.col("net_displacement_um") / pl.col("_path_um"))
        .alias("straightness"),
        pl.when(multi & (trace > 0))
        .then(((pl.col("_txx") - pl.col("_tyy")) ** 2 + 4 * pl.col("_txy") ** 2) / trace**2)
        .alias("gyration_asymmetry"),
    ).sort("track_id")


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
    """One row per track: identity, position, shape
    (`track_geometry` -- radius of gyration,
    straightness, ...) and detection-quality context columns that every
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
    geometry = track_geometry(diffkit_tracks)
    qc, per_point = qc_aggregate_table(tracks_df_px)
    qc_columns = [c for c in qc.columns if c != "track_id"]
    table = (
        lengths.join(centroids, on="track_id", how="left")
        .join(geometry, on="track_id", how="left")
        .join(qc, on="track_id", how="left")
        .sort("track_id")
    )
    region_cols = [c for c in ("region_class", "cell") if c in tracks_df_px.columns]
    if region_cols:
        # Which region the track was linked in (`regions.label_points`) --
        # one per track, since tracking links each region on its own.
        regions = tracks_df_px.group_by("track_id").agg(pl.col(c).first() for c in region_cols)
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
    "alpha_status": pl.String,
    "alpha_median": pl.Float64,
    "alpha_low": pl.Float64,
    "alpha_high": pl.Float64,
}

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
) -> PosteriorAnalysis:
    """Grid posteriors over D (and alpha) for every track in `tracks`, on
    `options`' grids (`posterior_options`).

    `tracks` is `tracks_to_diffusionkit_df`'s table. This is
    `diffusionkit.gridpost.analyze_tracks` itself -- same statuses, same
    numbers -- reshaped to one row per track, with each fitted track's
    posterior vector kept. The alpha posterior has no exposure-blur model,
    so it is computed only when `options.compute_alpha` and
    `acquisition.exposure_s == 0`; otherwise every row's `alpha_status`
    says why not. `progress(done, total)` is called from the calling
    thread."""
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
    )
    alpha = by_model["posterior_alpha"].select(
        "track_id",
        pl.col("status").alias("alpha_status"),
        pl.col("message").alias("_alpha_message"),
        pl.col("alpha_post_median").alias("alpha_median"),
        pl.col("alpha_post_lo").alias("alpha_low"),
        pl.col("alpha_post_hi").alias("alpha_high"),
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
    alpha_ran = options.compute_alpha and acquisition.exposure_s == 0
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
    the flat start is just the pooled posterior, more remove the blur the
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
    def from_record(cls, record: Optional[dict]) -> "Deconvolution":
        """A saved record; absent (a summary from before these were
        settable) means the defaults it was run with."""
        names = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (record or {}).items() if k in names})


@dataclass(frozen=True)
class Ensemble:
    """The population read of one set of tracks' posteriors, on the D grid
    (and the alpha grid, when alpha was computed).

    `summed_log_*` is the sum of the tracks' normalized log posteriors --
    unnormalized, so it keeps its scale (a sum over n tracks) --
    and `summed_*` the same thing normalized to grid weights: the
    posterior of one value shared by every track. `pooled_D` is the
    average of the tracks' posteriors (their sum, normalized) -- where
    the tracks put D, blurred by each track's own uncertainty; it is
    the deconvolution's starting point. `deconvolved_D` is
    `gridpost.deconvolve`'s distribution of D across the tracks, that
    blur removed. All weights sum to 1 over their grid."""

    n_tracks: int
    summed_log_D: np.ndarray
    summed_D: np.ndarray
    pooled_D: np.ndarray
    deconvolved_D: np.ndarray
    n_tracks_alpha: int = 0
    summed_log_alpha: Optional[np.ndarray] = None
    summed_alpha: Optional[np.ndarray] = None


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
    summed_log = log_post.sum(axis=0)
    deconvolved = dk_deconvolve.deconvolve(
        log_post, dk_post.flat(analysis.u_D), iters=deconvolution.iters, smooth=deconvolution.smooth
    )
    result = dict(
        n_tracks=len(rows),
        summed_log_D=summed_log,
        summed_D=np.exp(_normalized_log(summed_log)),
        pooled_D=np.exp(log_post).mean(axis=0),
        deconvolved_D=deconvolved,
    )
    if analysis.has_alpha:
        alpha_rows = analysis.rows_for(track_ids, alpha=True)
        if len(alpha_rows):
            summed_alpha = analysis.log_post_alpha[alpha_rows].sum(axis=0)
            result.update(
                n_tracks_alpha=len(alpha_rows),
                summed_log_alpha=summed_alpha,
                summed_alpha=np.exp(_normalized_log(summed_alpha)),
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
    posterior medians -- the typical track. `summed_D_*` is the shared-D
    posterior's median and 90% interval, and `deconvolved_D_*` the
    deconvolved distribution's median and 90% range (a spread across
    tracks, not an uncertainty) plus its mode."""
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
        )
    ens = ensemble(analysis, track_ids, deconvolution)
    if ens is not None:
        summed = _grid_summary(ens.summed_D, analysis.D_grid_um2_s, analysis.level, log_grid=True)
        spread = _grid_summary(ens.deconvolved_D, analysis.D_grid_um2_s, analysis.level, log_grid=True)
        out.update(
            {f"summed_D_{k}_um2_s": v for k, v in summed.items()},
            **{f"deconvolved_D_{k}_um2_s": v for k, v in spread.items()},
            deconvolved_D_mode_um2_s=_grid_mode(ens.deconvolved_D, analysis.D_grid_um2_s),
        )
        if ens.summed_alpha is not None:
            summed_a = _grid_summary(ens.summed_alpha, analysis.alpha_grid, analysis.level, log_grid=False)
            out.update({f"summed_alpha_{k}": v for k, v in summed_a.items()})
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
    as rows sorted by their median (`track_posteriors`), for the heat map."""
    fits = analysis.fits.filter(pl.col("posterior_status") == "ok")
    panels = []
    for name, ids in (groups or {"all": None}).items():
        ens = ensemble(analysis, ids, deconvolution)
        if ens is None:
            continue
        rows = fits if ids is None else fits.filter(pl.col("track_id").is_in(list(ids)))
        summed = _grid_summary(ens.summed_D, analysis.D_grid_um2_s, analysis.level, log_grid=True)
        panel = {
            "name": name,
            "n_tracks": ens.n_tracks,
            "medians": rows["D_median_um2_s"].to_numpy(),
            "deconvolved": ens.deconvolved_D,
            "pooled": ens.pooled_D,
            "summed": ens.summed_D,
            "summed_interval": (summed["low"], summed["median"], summed["high"]),
        }
        if track_posteriors:
            weights = np.exp(analysis.log_post_D[analysis.rows_for(ids)])
            # Each row's median grid cell: where its CDF first reaches 1/2.
            order = np.argsort((np.cumsum(weights, axis=1) >= 0.5).argmax(axis=1), kind="stable")
            panel["track_posteriors"] = weights[order]
        if ens.summed_alpha is not None:
            summed_a = _grid_summary(ens.summed_alpha, analysis.alpha_grid, analysis.level, log_grid=False)
            panel["alpha_medians"] = rows["alpha_median"].drop_nulls().to_numpy()
            panel["alpha_summed_interval"] = (summed_a["low"], summed_a["median"], summed_a["high"])
        panels.append(panel)
    return panels


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
    }
    if analysis.has_alpha:
        alpha_rows = np.flatnonzero(analysis.alpha_ids == track_id)
        if len(alpha_rows):
            out.update(
                alpha_grid=analysis.alpha_grid,
                alpha_weights=np.exp(analysis.log_post_alpha[alpha_rows[0]]),
                alpha_interval=(fit["alpha_low"], fit["alpha_median"], fit["alpha_high"]),
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
    """The ensemble distributions on their shared grid, long by group:
    `group`, `n_tracks`, the grid column (`D_um2_s` or `alpha`),
    `summed_log_posterior`, `summed_posterior` and -- for D --
    `pooled_posterior` and `deconvolved` (weights; each sums to 1 over
    the grid within a group).

    `groups` maps a group name to its track ids; the default is one group,
    "all", over every fitted track."""
    groups = groups or {"all": None}
    parts = []
    for name, ids in groups.items():
        ens = ensemble(analysis, ids, deconvolution)
        if ens is None:
            continue
        if alpha:
            if ens.summed_alpha is None:
                continue
            part = {
                "alpha": analysis.alpha_grid,
                "summed_log_posterior": ens.summed_log_alpha,
                "summed_posterior": ens.summed_alpha,
            }
            n = ens.n_tracks_alpha
            grid_len = len(analysis.alpha_grid)
        else:
            part = {
                "D_um2_s": analysis.D_grid_um2_s,
                "summed_log_posterior": ens.summed_log_D,
                "summed_posterior": ens.summed_D,
                "pooled_posterior": ens.pooled_D,
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
    px, shape, `region_class`/`cell`, per-point detection QC aggregated per
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
            "result_id", "track_id", "region_class", "cell", "passes_filters", "track_length",
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
        distributions_alpha=distributions_table(analysis, groups, alpha=True),
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
        display = display.drop("alpha_status", "alpha_median", "alpha_low", "alpha_high")
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
        return Deconvolution.from_record(self.summary.get("deconvolution"))


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
    """The `GridPostOptions` a saved analysis ran with. A summary from
    before the grid was settable has no "grid", and ran on diffusionkit's
    default grids, whose D and alpha ranges it recorded."""
    grid = summary.get("grid")
    if grid is None:
        grid = {}
        if summary.get("D_grid_um2_s"):
            lo, hi, n = summary["D_grid_um2_s"]
            grid.update(D_min_um2_s=lo, D_max_um2_s=hi, n_D=int(n))
        if summary.get("alpha_grid"):
            lo, hi, n = summary["alpha_grid"]
            grid.update(alpha_min=lo, alpha_max=hi, n_alpha=int(n))
    return GridPostOptions(
        min_frames=summary["min_frames"],
        level=summary["credible_level"],
        compute_alpha=summary.get("alpha_grid") is not None,
        **{k: v for k, v in grid.items() if k in GRID_FIELDS},
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

    # `D_at_grid_edge` from the posteriors themselves -- a bundle saved
    # before the column existed gets it too.
    edge = pl.DataFrame(
        {"track_id": fitted_ids, "D_at_grid_edge": _at_grid_edge(log_post_D)},
        schema={"track_id": pl.Int64, "D_at_grid_edge": pl.Boolean},
    )
    fits = (
        ran.select(
            pl.col(c).cast(dtype) if c in ran.columns else pl.lit(None, dtype=dtype).alias(c)
            for c, dtype in FIT_SCHEMA.items()
            if c not in ("n_frames", "message", "D_at_grid_edge")
        )
        .with_columns(
            ran["track_length"].cast(pl.Int64).alias("n_frames"), pl.lit("", dtype=pl.String).alias("message")
        )
        .join(edge, on="track_id", how="left")
        .select(FIT_SCHEMA.keys())
        .sort("track_id")
    )

    analysis = PosteriorAnalysis(
        fits=fits,
        acquisition=Acquisition(dt_s=summary["dt_s"], exposure_s=summary["exposure_s"]),
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
