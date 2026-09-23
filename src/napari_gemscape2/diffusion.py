"""GUI-free diffusion analysis: this project's tracks in, diffusionkit's
grid posteriors out, plus the tables the widget shows and the bundle saves.

`tracks_to_diffusionkit_df` is the bridge. spotsolve's per-localization
position error is `se_x`/`se_y` (the CRLB from the fit's Fisher
information); diffusionkit calls the same quantity `sigma_x_um`/
`sigma_y_um`. Only the name and the unit differ, so it is renamed and
scaled here rather than duplicated upstream.

The analysis is `diffusionkit.gridpost`: for each track, the exact Gaussian
displacement likelihood evaluated on a fixed grid in ln D (and, with no
exposure blur, the fBm exponent alpha with K integrated out). diffusionkit's
own `gridpost.analyze_tracks` reports only each posterior's median and
interval; `analyze_posteriors` below calls the same public functions but
keeps the per-track vectors too, because they are what the ensemble is
built from and what the bundle saves:

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

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import polars as pl
from diffusionkit import Acquisition, validated_track_frame
from diffusionkit.gridpost import deconvolve as dk_deconvolve
from diffusionkit.gridpost import posterior as dk_post
from diffusionkit.gridpost import posterior_alpha as dk_alpha
from scipy.special import logsumexp

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

D_GRID_UM2_S = np.exp(dk_post.U)
ALPHA_GRID = dk_alpha.ALPHA


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


# --- per-track grid posteriors -------------------------------------------

FIT_SCHEMA = {
    "track_id": pl.Int64,
    "n_frames": pl.Int64,
    "posterior_status": pl.String,
    "message": pl.String,
    "D_median_um2_s": pl.Float64,
    "D_low_um2_s": pl.Float64,
    "D_high_um2_s": pl.Float64,
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
    `log_post_D` over `D_GRID_UM2_S`, `log_post_alpha` over `ALPHA_GRID`
    for `alpha_ids` (None when alpha was not computed).
    """

    fits: pl.DataFrame
    acquisition: Acquisition
    min_frames: int
    level: float
    fitted_ids: np.ndarray
    log_post_D: np.ndarray
    alpha_ids: Optional[np.ndarray]
    log_post_alpha: Optional[np.ndarray]

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


def analyze_posteriors(
    tracks: pl.DataFrame,
    acquisition: Acquisition,
    *,
    min_frames: int = MIN_FRAMES,
    level: float = LEVEL,
    alpha: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> PosteriorAnalysis:
    """Grid posteriors over D (and alpha) for every track in `tracks`.

    `tracks` is `tracks_to_diffusionkit_df`'s table. Mirrors
    `diffusionkit.gridpost.analyze_tracks` -- same functions, same flat
    priors, same statuses -- but keeps each track's posterior vector.
    The alpha posterior has no exposure-blur model, so it is computed
    only when `alpha` and `acquisition.exposure_s == 0`; otherwise every
    row's `alpha_status` says why not. `progress(done, total)` is called
    from the calling thread.
    """
    groups = tracks.sort("track_id", "frame").partition_by("track_id", maintain_order=True)
    alpha_blocked = None
    if not alpha:
        alpha_blocked = "not requested"
    elif acquisition.exposure_s > 0:
        alpha_blocked = "The alpha posterior has no exposure-blur model"
    rows, fitted_ids, log_post_D, alpha_ids, log_post_alpha = [], [], [], [], []
    flat_D = dk_post.flat()
    flat_K = dk_alpha.flat_K(dk_alpha.U)  # its own K grid, not the D grid

    def fit_one(group: pl.DataFrame) -> dict:
        track_id = int(group["track_id"][0])
        row = {"track_id": track_id, "n_frames": group.height, "message": ""}
        try:
            track = validated_track_frame(group, acquisition, require_localization=True)
        except ValueError as exc:
            return {**row, "posterior_status": "invalid_input", "alpha_status": "invalid_input",
                    "message": str(exc)}
        if track.height < min_frames:
            return {**row, "posterior_status": "excluded", "alpha_status": "excluded",
                    "message": f"{track.height} frames; min_frames={min_frames}"}
        try:
            lp = _normalized_log(dk_post.track_loglik(track, acquisition) + flat_D)
        except ValueError as exc:  # e.g. a zero localization SD
            return {**row, "posterior_status": "invalid_input", "alpha_status": "invalid_input",
                    "message": str(exc)}
        s = dk_post.summary(np.exp(lp), level=level)
        row.update(
            posterior_status="ok",
            D_median_um2_s=s["median"],
            D_low_um2_s=s["lo"],
            D_high_um2_s=s["hi"],
        )
        fitted_ids.append(track_id)
        log_post_D.append(lp)
        if alpha_blocked is not None:
            row.update(alpha_status="excluded" if alpha else "not_run", message=alpha_blocked)
            return row
        try:
            p_alpha = dk_alpha.alpha_posterior(dk_alpha.joint_loglik(track, acquisition), flat_K)
        except ValueError as exc:
            row.update(alpha_status="invalid_input", message=str(exc))
            return row
        sa = dk_alpha.summary(p_alpha, level=level)
        row.update(
            alpha_status="ok", alpha_median=sa["median"], alpha_low=sa["lo"], alpha_high=sa["hi"]
        )
        alpha_ids.append(track_id)
        with np.errstate(divide="ignore"):
            log_post_alpha.append(np.log(p_alpha))
        return row

    if progress is not None:
        progress(0, len(groups))
    for done, group in enumerate(groups, 1):
        rows.append(fit_one(group))
        if progress is not None:
            progress(done, len(groups))
    n_grid = len(D_GRID_UM2_S)
    return PosteriorAnalysis(
        fits=pl.DataFrame(rows, schema=FIT_SCHEMA),
        acquisition=acquisition,
        min_frames=min_frames,
        level=level,
        fitted_ids=np.array(fitted_ids, dtype=np.int64),
        log_post_D=np.array(log_post_D) if log_post_D else np.empty((0, n_grid)),
        alpha_ids=None if alpha_blocked is not None else np.array(alpha_ids, dtype=np.int64),
        log_post_alpha=(
            None
            if alpha_blocked is not None
            else (np.array(log_post_alpha) if log_post_alpha else np.empty((0, len(ALPHA_GRID))))
        ),
    )


# --- the ensemble --------------------------------------------------------


@dataclass(frozen=True)
class Ensemble:
    """The population read of one set of tracks' posteriors, on the D grid
    (and the alpha grid, when alpha was computed).

    `summed_log_*` is the sum of the tracks' normalized log posteriors --
    unnormalized, so it keeps its scale (a sum over n tracks) --
    and `summed_*` the same thing normalized to grid weights: the
    posterior of one value shared by every track. `deconvolved_D` is
    `gridpost.deconvolve`'s distribution of D across the tracks. All
    weights sum to 1 over their grid."""

    n_tracks: int
    summed_log_D: np.ndarray
    summed_D: np.ndarray
    deconvolved_D: np.ndarray
    n_tracks_alpha: int = 0
    summed_log_alpha: Optional[np.ndarray] = None
    summed_alpha: Optional[np.ndarray] = None


def ensemble(analysis: PosteriorAnalysis, track_ids: Optional[set] = None) -> Optional[Ensemble]:
    """The `Ensemble` over `track_ids` (all fitted tracks for None), or
    None when none of them was fitted."""
    rows = analysis.rows_for(track_ids)
    if len(rows) == 0:
        return None
    log_post = analysis.log_post_D[rows]
    summed_log = log_post.sum(axis=0)
    deconvolved = dk_deconvolve.deconvolve(
        log_post, dk_post.flat(), iters=DECONVOLVE_ITERS, smooth=DECONVOLVE_SMOOTH
    )
    result = dict(
        n_tracks=len(rows),
        summed_log_D=summed_log,
        summed_D=np.exp(_normalized_log(summed_log)),
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


def summarize(analysis: PosteriorAnalysis, track_ids: Optional[set] = None) -> dict:
    """Population-level numbers for `track_ids` (all for None), flat keys
    with units in their names so the saved JSON reads on its own.

    Counts are by diffusionkit's own statuses (`n_ok`, `n_excluded`,
    `n_invalid_input`). `median_D_um2_s`/`q25`/`q75` are over the per-track
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
    }
    alpha_ok = fits.filter(pl.col("alpha_status") == "ok")
    if alpha_ok.height:
        out.update(
            n_alpha=alpha_ok.height,
            median_alpha=_scalar(alpha_ok["alpha_median"].median()),
            q25_alpha=_scalar(alpha_ok["alpha_median"].quantile(0.25)),
            q75_alpha=_scalar(alpha_ok["alpha_median"].quantile(0.75)),
        )
    ens = ensemble(analysis, track_ids)
    if ens is not None:
        summed = _grid_summary(ens.summed_D, D_GRID_UM2_S, analysis.level, log_grid=True)
        spread = _grid_summary(ens.deconvolved_D, D_GRID_UM2_S, analysis.level, log_grid=True)
        out.update(
            {f"summed_D_{k}_um2_s": v for k, v in summed.items()},
            **{f"deconvolved_D_{k}_um2_s": v for k, v in spread.items()},
            deconvolved_D_mode_um2_s=_grid_mode(ens.deconvolved_D, D_GRID_UM2_S),
        )
        if ens.summed_alpha is not None:
            summed_a = _grid_summary(ens.summed_alpha, ALPHA_GRID, analysis.level, log_grid=False)
            out.update({f"summed_alpha_{k}": v for k, v in summed_a.items()})
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
    analysis: PosteriorAnalysis, groups: Optional[dict[str, Optional[set]]] = None
) -> list[dict]:
    """What `joint_plot.plot_d_ensemble` draws, one dict per group (default
    one group, "all"); groups with no fitted track are left out."""
    fits = analysis.fits.filter(pl.col("posterior_status") == "ok")
    panels = []
    for name, ids in (groups or {"all": None}).items():
        ens = ensemble(analysis, ids)
        if ens is None:
            continue
        rows = fits if ids is None else fits.filter(pl.col("track_id").is_in(list(ids)))
        summed = _grid_summary(ens.summed_D, D_GRID_UM2_S, analysis.level, log_grid=True)
        panel = {
            "name": name,
            "n_tracks": ens.n_tracks,
            "medians": rows["D_median_um2_s"].to_numpy(),
            "deconvolved": ens.deconvolved_D,
            "summed": ens.summed_D,
            "summed_interval": (summed["low"], summed["median"], summed["high"]),
        }
        if ens.summed_alpha is not None:
            summed_a = _grid_summary(ens.summed_alpha, ALPHA_GRID, analysis.level, log_grid=False)
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
        "d_grid": D_GRID_UM2_S,
        "d_weights": np.exp(analysis.log_post_D[rows[0]]),
        "d_interval": (fit["D_low_um2_s"], fit["D_median_um2_s"], fit["D_high_um2_s"]),
    }
    if analysis.has_alpha:
        alpha_rows = np.flatnonzero(analysis.alpha_ids == track_id)
        if len(alpha_rows):
            out.update(
                alpha_grid=ALPHA_GRID,
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
        ids, log_post, grid, name = analysis.alpha_ids, analysis.log_post_alpha, ALPHA_GRID, "alpha"
    else:
        ids, log_post, grid, name = analysis.fitted_ids, analysis.log_post_D, D_GRID_UM2_S, "D_um2_s"
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
) -> Optional[pl.DataFrame]:
    """The ensemble distributions on their shared grid, long by group:
    `group`, `n_tracks`, the grid column (`D_um2_s` or `alpha`),
    `summed_log_posterior`, `summed_posterior` and -- for D --
    `deconvolved` (weights; each sums to 1 over the grid within a group).

    `groups` maps a group name to its track ids; the default is one group,
    "all", over every fitted track."""
    groups = groups or {"all": None}
    parts = []
    for name, ids in groups.items():
        ens = ensemble(analysis, ids)
        if ens is None:
            continue
        if alpha:
            if ens.summed_alpha is None:
                continue
            part = {
                "alpha": ALPHA_GRID,
                "summed_log_posterior": ens.summed_log_alpha,
                "summed_posterior": ens.summed_alpha,
            }
            n = ens.n_tracks_alpha
            grid_len = len(ALPHA_GRID)
        else:
            part = {
                "D_um2_s": D_GRID_UM2_S,
                "summed_log_posterior": ens.summed_log_D,
                "summed_posterior": ens.summed_D,
                "deconvolved": ens.deconvolved_D,
            }
            n = ens.n_tracks
            grid_len = len(D_GRID_UM2_S)
        parts.append(
            pl.DataFrame({"group": [name] * grid_len, "n_tracks": [n] * grid_len, **part})
        )
    return pl.concat(parts) if parts else None


# --- MSD comparison -----------------------------------------------------


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
# (`widgets/diffusion_panel._qc_aggregate_table`). The saved summary keeps
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
