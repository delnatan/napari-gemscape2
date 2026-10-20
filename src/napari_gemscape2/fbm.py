"""GUI-free alpha-D analysis: diffusionkit's fBm profile-likelihood fits (`diffusionkit.fbmpost`)
for this project's tracks, and the tables and plot data the widget's α · D tab shows
and the bundle saves.

`diffusion.py` holds the standard quantification, the Brownian D posterior
(`gridpost`). This is a second, separate fit: per track, the anomalous exponent
alpha and an amplitude under fractional Brownian motion, with the same exposure
blur and per-frame localization SDs. Its likelihood is gridpost's at alpha = 1.

**Per track, no prior.** The amplitude is maximized at each alpha (the profile
likelihood), so each track gets `alpha_hat`, its likelihood-ratio interval
(`alpha_low`, `alpha_high`, 95% by default), `alpha_edge` when that interval
reaches an end of the alpha grid, and a likelihood-ratio test of alpha = 1
(`alpha_p_brownian`). A track with no motion above its localization noise has
no `alpha_hat`, and its D_ref is only bounded above (`D_ref_low_um2_s` = 0).

**D_ref is not a Brownian D.** It is the fBm amplitude expressed as
D_ref = MSD_1D(tau_ref) / (2 tau_ref), the true motion's apparent D at one lag
`tau_ref_s` (default 0.1 s). It equals a Brownian D only at alpha = 1, and
otherwise changes with tau_ref. So it is never compared with `D_mean_um2_s` or an
MSD D, and `D_mean_um2_s` is never paired with `alpha_hat`.

**Across tracks**, the two reads the D tab has for D:

  - **shared alpha** (`shared_alpha_summary`): one alpha for every track, the
    maximum of the tracks' summed profiles, each track keeping its own amplitude.
    Profiles add because each is a likelihood;
  - **the population** (`alpha_population`): diffusionkit's 2D deconvolution of
    the tracks' joint (alpha, log10 D_ref) likelihoods (`fbmpost.deconvolve_tracks`),
    saved in `loglik_alpha_D.parquet`. How alpha varies across tracks has to be
    fitted jointly with D_ref. Below the D_ref floor the tracks barely tell D_ref
    values apart, and alpha there is mostly the smoothness prior's continuation
    of alpha above it, so the summary also gives the reads above the floor and
    the share of the population below it.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import polars as pl
from diffusionkit import Acquisition
from diffusionkit.fbmpost import AlphaDPopulation, AlphaProfiles, FbmPostOptions, JointGrid, JointLikelihoods
from diffusionkit.fbmpost import analyze_tracks as dk_analyze_fbm
from diffusionkit.fbmpost import deconvolve_tracks as dk_deconvolve_alpha_D
from diffusionkit.fbmpost import likelihood as fbm_likelihood
from diffusionkit.fbmpost import shared_alpha as dk_shared_alpha
from diffusionkit.fbmpost.population import informative_range
from scipy.stats import chi2

from napari_gemscape2.diffusion import LEVEL, MIN_FRAMES, StaleAnalysisError, floor_band, tracks_fingerprint

# The per-track columns a run adds to the tracks pane and `tracks_summary.csv`.
FBM_SCHEMA = {
    "track_id": pl.Int64,
    # diffusionkit's status for this fit ("ok", "excluded", "invalid_input"),
    # apart from the D posterior's `posterior_status`.
    "fbm_status": pl.String,
    # alpha's maximum-likelihood value (null when the track shows no motion above
    # its noise) and its likelihood-ratio interval.
    "alpha_hat": pl.Float64,
    "alpha_low": pl.Float64,
    "alpha_high": pl.Float64,
    # Which end of the alpha grid the interval reaches: "low", "high", "both" (no information on alpha).
    "alpha_edge": pl.String,
    # The likelihood-ratio test of alpha = 1: the log-likelihood gained over it, and its p-value.
    "alpha_dll_brownian": pl.Float64,
    "alpha_p_brownian": pl.Float64,
    # D_ref at the run's tau_ref, at alpha_hat, and its profile-likelihood interval -- not a Brownian D.
    # D_ref_low_um2_s = 0: the track only bounds D_ref from above.
    "D_ref_hat_um2_s": pl.Float64,
    "D_ref_low_um2_s": pl.Float64,
    "D_ref_high_um2_s": pl.Float64,
    "log10_D_ref_hat": pl.Float64,
    # The D_ref at which the motion's MSD at tau_ref equals the localization noise's.
    "D_ref_floor_um2_s": pl.Float64,
}
FBM_COLUMNS = tuple(c for c in FBM_SCHEMA if c != "track_id")

# The default reference lag, diffusionkit's.
TAU_REF_S = FbmPostOptions().tau_ref_s
# A track rejects alpha = 1 below this p-value (`alpha_p_brownian`), for the summary's count.
BROWNIAN_P = 0.05

_RENAME = {
    "status": "fbm_status",
    "alpha_lo": "alpha_low",
    "alpha_hi": "alpha_high",
    "D_ref_lo_um2_s": "D_ref_low_um2_s",
    "D_ref_hi_um2_s": "D_ref_high_um2_s",
}


# The grids a run evaluates the likelihood on, the settings the tab and a batch expose:
# - n_alpha: the per-track alpha grid's points over 0.02-1.98 (`FbmPostOptions.n_alpha`). Each costs
#   one eigendecomposition per track; alpha_hat is refined between them.
# - alpha_group, d_log10_D, cell_points: the population's cells (`JointGrid`): alpha_group alpha
#   rows by d_log10_D decades of D, each cell the likelihood averaged over cell_points D points. A
#   run's time with the population kept is mostly these cells: in proportion to cell_points / d_log10_D.
# cell_points is 4 here, not diffusionkit's 8: on 0.05-decade cells 4 points stay within 0.08 nats of 8
# on GEM-like tracks, at two thirds of the run time.
FBM_GRID_DEFAULTS = {
    "n_alpha": FbmPostOptions().n_alpha,
    "alpha_group": JointGrid().alpha_group,
    "d_log10_D": JointGrid().d_log10_D_ref,
    "cell_points": 4,
}


def fbm_options(min_frames: int = MIN_FRAMES, tau_ref_s: float = TAU_REF_S, grid: Optional[dict] = None) -> FbmPostOptions:
    """The `FbmPostOptions` a run here uses: diffusionkit's alpha range and its
    likelihood-ratio level (95%), `grid`'s n_alpha (`FBM_GRID_DEFAULTS` for any key
    left out), which must make at least 3 population cells of `alpha_group`. Raises
    ValueError for an invalid value."""
    g = fbm_grid_settings(grid)
    n_alpha = int(g["n_alpha"])
    if n_alpha < 3:
        raise ValueError(f"n_alpha must be at least 3, got {n_alpha}")
    if n_alpha // max(int(g["alpha_group"]), 1) < 3:
        raise ValueError(f"{n_alpha} α points make fewer than 3 population cells of {g['alpha_group']}")
    return FbmPostOptions(min_frames=min_frames, tau_ref_s=tau_ref_s, n_alpha=n_alpha)


def fbm_grid_settings(grid: Optional[dict] = None) -> dict:
    """`grid` over `FBM_GRID_DEFAULTS`. Raises ValueError for an unknown key."""
    grid = dict(grid or {})
    unknown = set(grid) - set(FBM_GRID_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown alpha-D grid setting(s): {', '.join(sorted(unknown))}")
    return {**FBM_GRID_DEFAULTS, **grid}


def joint_grid(grid: Optional[dict] = None) -> JointGrid:
    """The population's cells (`JointGrid`) from `grid` (`FBM_GRID_DEFAULTS` for any key
    left out). Raises ValueError for an invalid value."""
    g = fbm_grid_settings(grid)
    return JointGrid(alpha_group=int(g["alpha_group"]), d_log10_D_ref=float(g["d_log10_D"]),
                     cell_points=int(g["cell_points"]))


def grid_record(options: FbmPostOptions, joint: Optional[JointGrid]) -> dict:
    """The grid settings a run used (`FBM_GRID_DEFAULTS`' keys); a run restored from a
    bundle saved before they were recorded has diffusionkit's population cells then."""
    joint = joint or JointGrid()
    return {"n_alpha": options.n_alpha, "alpha_group": joint.alpha_group, "d_log10_D": joint.d_log10_D_ref,
            "cell_points": joint.cell_points}


@dataclass(frozen=True)
class FbmAnalysis:
    """One run of `analyze_fbm`: a row per input track (`FBM_SCHEMA`, plus
    `n_frames`), what it ran with, `tracks_fingerprint` of its tracks, each
    fitted track's profile of alpha (`profiles`, what `shared_alpha_summary`
    sums) and its joint likelihood (`likelihoods`, what `alpha_population`
    deconvolves); either is None for a run saved without it."""

    fits: pl.DataFrame
    acquisition: Acquisition
    options: FbmPostOptions
    tracks_sha256: Optional[str] = None
    # The population's cells (`joint_grid`); None for a run restored from a bundle that did not record them.
    joint: Optional[JointGrid] = None
    likelihoods: Optional[JointLikelihoods] = field(default=None, compare=False, repr=False)
    profiles: Optional[AlphaProfiles] = field(default=None, compare=False, repr=False)
    # `alpha_population`'s results by track ids: ~10 s each for a thousand tracks.
    _populations: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def tau_ref_s(self) -> float:
        return self.options.tau_ref_s

    @property
    def ok(self) -> pl.DataFrame:
        return self.fits.filter(pl.col("fbm_status") == "ok")

    @property
    def grid(self) -> dict:
        """The grid settings it ran with (`grid_record`)."""
        return grid_record(self.options, self.joint)

    def columns(self) -> pl.DataFrame:
        """The per-track columns (`FBM_SCHEMA`) for the tracks pane."""
        return self.fits.select(FBM_SCHEMA.keys())


def analyze_fbm(
    tracks: pl.DataFrame,
    acquisition: Acquisition,
    options: FbmPostOptions,
    *,
    joint: Optional[JointGrid] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    threads: Optional[int] = None,
) -> FbmAnalysis:
    """`diffusionkit.fbmpost.analyze_tracks` on `tracks` (`tracks_to_diffusionkit_df`'s
    table), on a pool of `threads` (default: one per CPU), reshaped to `FBM_SCHEMA`,
    with each fitted track's profile of alpha and its joint likelihood on `joint`'s
    cells (default `joint_grid()`), the latter kept over the D cells some track tells
    apart (`fbmpost.population.informative_range`)."""
    joint = joint_grid() if joint is None else joint
    threads = threads or os.cpu_count() or 1
    kw = dict(progress=progress, keep_likelihoods=True, keep_profiles=True, joint=joint)
    if threads > 1:
        with ThreadPoolExecutor(threads) as pool:
            result = dk_analyze_fbm(tracks, acquisition, options, map_fn=pool.map, **kw)
    else:
        result = dk_analyze_fbm(tracks, acquisition, options, **kw)
    fits = result.fits.rename(_RENAME).select("n_frames", *FBM_SCHEMA.keys()).cast(FBM_SCHEMA)
    profiles = result.profiles if len(result.profiles.track_ids) else None
    return FbmAnalysis(fits, acquisition, options, tracks_fingerprint(tracks), joint, _cropped(result.likelihoods),
                       profiles)


def _cropped(lik: JointLikelihoods) -> Optional[JointLikelihoods]:
    """`lik` over its informative D cells: a subset's range lies inside it, and the rest is
    either flat (below detection) or negligible for every track."""
    if not len(lik.track_ids):
        return None
    keep = informative_range(lik)
    return JointLikelihoods(lik.track_ids, lik.n_frames, lik.D_ref_floor_um2_s, lik.alphas,
                            lik.log10_D_ref[keep], lik.loglik[:, :, keep], lik.tau_s, lik.tau_ref_s)


def _rows(analysis: FbmAnalysis, ids: Optional[set]) -> pl.DataFrame:
    rows = analysis.fits
    return rows if ids is None else rows.filter(pl.col("track_id").is_in(list(ids)))


def summarize_fbm(analysis: FbmAnalysis, ids: Optional[set] = None) -> dict:
    """Counts over the tracks `ids` (None: all): by status, how many have an
    alpha_hat, how many of those pin alpha inside the grid, how many reject
    alpha = 1 at `BROWNIAN_P`, how many only bound D_ref from above, and the
    D_ref floor. The across-track numbers are `shared_alpha_summary`'s and
    `population_summary`'s."""
    rows = _rows(analysis, ids)
    ok = rows.filter(pl.col("fbm_status") == "ok")
    identified = ok.filter(pl.col("alpha_hat").is_not_null())
    out = {
        "n_tracks": rows.height,
        "n_ok": ok.height,
        **{f"n_{s}": n for s, n in rows["fbm_status"].value_counts().iter_rows() if s != "ok"},
        "n_alpha_identified": identified.height,
        "n_alpha_bounded": int(identified["alpha_edge"].is_null().sum()),
        "n_non_brownian": int((ok["alpha_p_brownian"] < BROWNIAN_P).sum()),
        "n_D_ref_upper_bound": int((ok["D_ref_low_um2_s"] == 0).sum()),
    }
    band = floor_band(ok["D_ref_floor_um2_s"])
    if band is not None:
        out["D_ref_floor_median_um2_s"] = band[1]
    return out


def shared_alpha_summary(analysis: FbmAnalysis, ids: Optional[set] = None) -> Optional[dict]:
    """One alpha for every fitted track in `ids` (None: all): `fbmpost.shared_alpha` of
    their profiles -- the maximum of the summed profiles, its likelihood-ratio interval
    at the run's level and the test of alpha = 1. None without profiles or tracks."""
    prof = analysis.profiles
    if prof is None:
        return None
    rows = np.arange(len(prof.track_ids)) if ids is None else np.flatnonzero(np.isin(prof.track_ids, list(ids)))
    if not len(rows):
        return None
    s = dk_shared_alpha(prof.rows(rows), analysis.options.level)
    return {"n_tracks": s.n_tracks, "alpha": s.alpha_hat, "low": s.alpha_lo, "high": s.alpha_hi,
            "dll_brownian": s.dll_brownian, "p_brownian": s.p_brownian, "level": s.level,
            "at_grid_edge": s.at_grid_edge}


# A population needs this many fitted tracks; fewer say little beyond their own fits.
MIN_POPULATION_TRACKS = 20
_POPULATION_CACHE_SIZE = 16


def alpha_population(analysis: FbmAnalysis, ids: Optional[set] = None,
                     progress: Optional[Callable[[str, int, int], None]] = None) -> Optional[AlphaDPopulation]:
    """The population of (alpha, log10 D_ref) over the tracks `ids` (None: all
    fitted), `fbmpost.deconvolve_tracks` of their joint likelihoods -- or None
    without likelihoods or with fewer than `MIN_POPULATION_TRACKS` fitted
    tracks. ~10 s for a thousand tracks (less for a subset of tracks already fitted together); cached on
    `analysis`. `progress` is diffusionkit's `(stage, done, total)` (`population_stage` makes one)."""
    lik = analysis.likelihoods
    if lik is None:
        return None
    key = None if ids is None else frozenset(ids)
    if key in analysis._populations:
        return analysis._populations[key]
    rows = np.arange(len(lik.track_ids)) if ids is None else np.flatnonzero(np.isin(lik.track_ids, list(ids)))
    # A population already fitted over more of these tracks (all of them, or a spatial run's before its
    # classes) starts this one's search for its smoothness where it found its own.
    parent = None if key is None else next(
        (p for k, p in analysis._populations.items() if p is not None and (k is None or k > key)), None)
    lam_start = None if parent is None else parent.lam
    pop = (dk_deconvolve_alpha_D(lik.rows(rows), lam_start=lam_start, progress=progress)
           if len(rows) >= MIN_POPULATION_TRACKS else None)
    if len(analysis._populations) >= _POPULATION_CACHE_SIZE:
        analysis._populations.clear()
    analysis._populations[key] = pop
    return pop


def population_stage(prefix: str, report: Callable[[str, int, int], None]) -> Callable[[str, int, int], None]:
    """diffusionkit's deconvolution progress, `(stage, done, total)`, as `report(text, done, total)` for a
    progress bar: the Laplace fits counted as they go, since how many the smoothness needs is not known
    ahead (total 0: a busy bar), then the draws as a fraction. `prefix` names the population."""

    def stage(name: str, done: int, total: int) -> None:
        if name == "sampling":
            report(f"{prefix}sampling", done, total)
        else:
            report(f"{prefix}{'support' if name == 'coarse' else 'smoothness'}, fit {done}", 0, 0)

    return stage


def population_is_cached(analysis: FbmAnalysis, ids: Optional[set] = None) -> bool:
    """Whether `alpha_population(analysis, ids)` would return without computing."""
    return analysis.likelihoods is None or (None if ids is None else frozenset(ids)) in analysis._populations


def _interval(draws: np.ndarray, level: float) -> Optional[list[float]]:
    """[median, low, high] over the draws where the read is defined (the alpha-D_ref correlation is not
    in a draw whose alpha is one cell), or None in none."""
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return None
    tail = (1 - level) / 2
    return [float(v) for v in np.quantile(draws, [0.5, tail, 1 - tail])]


def population_summary(pop: AlphaDPopulation, level: float = LEVEL) -> dict:
    """An `AlphaDPopulation`'s numbers, each [median, low, high] over its draws
    at `level`: alpha's mean and SD across tracks, P(alpha < 1), the
    alpha-D_ref correlation, the same reads above the D_ref floor, and the
    share of the population below it."""
    return {
        "n_tracks": pop.n_tracks,
        "alpha_mean": _interval(pop.alpha_mean(), level),
        "alpha_sd": _interval(pop.alpha_sd(), level),
        "p_alpha_below_1": _interval(pop.p_alpha_below(1.0), level),
        "alpha_D_ref_corr": _interval(pop.corr(), level),
        "share_below_floor": _interval(pop.mass_below_floor(), level),
        "above_floor": {
            "alpha_mean": _interval(pop.alpha_mean(above_floor=True), level),
            "alpha_sd": _interval(pop.alpha_sd(above_floor=True), level),
            "p_alpha_below_1": _interval(pop.p_alpha_below(1.0, above_floor=True), level),
        },
        "D_ref_floor_um2_s": float(10**pop.log10_floor),
        "lambda": list(pop.lam),
    }


def population_panels(analysis: FbmAnalysis, groups: dict[str, Optional[set]],
                      report: Optional[Callable[[str, int, int], None]] = None) -> list[dict]:
    """What `joint_plot.plot_alpha_population` draws, one dict per group with
    a population: `name`, the `AlphaDPopulation` (`population`), its summary
    and band at `diffusion.LEVEL`, the shared alpha (`shared_alpha_summary`, or
    None) and tau_ref. `report(text, done, total)` follows the deconvolutions."""
    panels = []
    for i, (name, ids) in enumerate(groups.items()):
        prefix = f"{name} ({i + 1}/{len(groups)}) · " if len(groups) > 1 else ""
        pop = alpha_population(analysis, ids, progress=None if report is None else population_stage(prefix, report))
        if pop is not None:
            panels.append({"name": name, "population": pop, "summary": population_summary(pop, LEVEL),
                           "level": LEVEL, "shared": shared_alpha_summary(analysis, ids),
                           "tau_ref_s": analysis.tau_ref_s})
    return panels


def fbm_record(analysis: FbmAnalysis, ids: Optional[set], by_class: Optional[dict]) -> dict:
    """`diffusion_summary.json`'s "alpha_D" entry: what the run was, its
    settings, `summarize_fbm` over the tracks the filters pass, "shared_alpha"
    (`shared_alpha_summary`) and, under "population", `population_summary` of
    their `alpha_population` (each region class's under "by_region_class"). A
    population not yet computed is computed here: warm the cache off the GUI
    thread first."""
    o = analysis.options

    def reads(entry: dict, group: Optional[set]) -> dict:
        shared = shared_alpha_summary(analysis, group)
        if shared is not None:
            entry["shared_alpha"] = shared
        pop = alpha_population(analysis, group)
        if pop is not None:
            entry["population"] = population_summary(pop, LEVEL)
        return entry

    record = {
        "analysis": "diffusionkit.fbmpost profile likelihood under blurred fBm: the amplitude maximized at each "
        "alpha, likelihood-ratio intervals, no prior; D_ref = MSD_1D(tau_ref) / (2 tau_ref) is not a Brownian D",
        "dt_s": analysis.acquisition.dt_s,
        "exposure_s": analysis.acquisition.exposure_s,
        "min_frames": o.min_frames,
        "lr_level": o.level,
        "population_level": LEVEL,
        "tau_ref_s": o.tau_ref_s,
        "alpha_grid": [o.alpha_min, o.alpha_max, o.n_alpha],
        # The grid settings (`FBM_GRID_DEFAULTS`' keys), what a batch template reruns with.
        "grid": analysis.grid,
        "tracks_sha256": analysis.tracks_sha256,
        # The lag D is set at on the joint likelihood's cells (`loglik_long_table`'s `log10_D`).
        "population_tau_s": None if analysis.likelihoods is None else analysis.likelihoods.tau_s,
        **summarize_fbm(analysis, ids),
    }
    reads(record, ids)
    if by_class:
        record["by_region_class"] = {name: reads(summarize_fbm(analysis, gids), gids) for name, gids in by_class.items()}
    return record


def profile_long_table(analysis: FbmAnalysis) -> Optional[pl.DataFrame]:
    """Every fitted track's profile log-likelihood of alpha, one row per (track, alpha):
    `track_id`, `n_frames`, `alpha`, `loglik` (each track's maximum 0) -- what `shared_alpha_summary`
    sums, so a bundle reopens without refitting. None without profiles."""
    prof = analysis.profiles
    if prof is None:
        return None
    n, n_a = prof.loglik.shape
    return pl.DataFrame(
        {"track_id": np.repeat(prof.track_ids, n_a), "n_frames": np.repeat(prof.n_frames, n_a),
         "alpha": np.tile(prof.alphas, n), "loglik": prof.loglik.reshape(-1)},
        schema={"track_id": pl.Int64, "n_frames": pl.Int64, "alpha": pl.Float64, "loglik": pl.Float64},
    )


def _profiles_from(table: pl.DataFrame) -> AlphaProfiles:
    """`profile_long_table`'s table back to `AlphaProfiles`."""
    table = table.sort("track_id", "alpha")
    alphas = np.unique(table["alpha"].to_numpy())
    per_track = table.group_by("track_id", maintain_order=True).agg(pl.col("n_frames").first(),
                                                                   pl.len().alias("n_alpha"))
    if (per_track["n_alpha"] != len(alphas)).any():
        raise ValueError("profile_alpha: every track needs one row per alpha")
    return AlphaProfiles(per_track["track_id"].to_numpy(), per_track["n_frames"].to_numpy(), alphas,
                         table["loglik"].to_numpy().reshape(per_track.height, len(alphas)))


def loglik_long_table(analysis: FbmAnalysis) -> Optional[pl.DataFrame]:
    """Every fitted track's joint likelihood, one row per (track, cell):
    `track_id`, `alpha`, `log10_D` (D at the cells' lag, `fbm_record`'s
    "population_tau_s"), `loglik` (each track's maximum 0) --
    what `alpha_population` deconvolves, so a bundle reopens without
    refitting. Also the track's frames and D_ref floor, repeated per row.
    None without likelihoods."""
    lik = analysis.likelihoods
    if lik is None:
        return None
    n, n_a, n_d = lik.loglik.shape
    return pl.DataFrame(
        {
            "track_id": np.repeat(lik.track_ids, n_a * n_d),
            "n_frames": np.repeat(lik.n_frames, n_a * n_d),
            "D_ref_floor_um2_s": np.repeat(lik.D_ref_floor_um2_s, n_a * n_d),
            "alpha": np.tile(np.repeat(lik.alphas, n_d), n),
            "log10_D": np.tile(lik.log10_D_ref, n * n_a),
            "loglik": lik.loglik.reshape(-1).astype(np.float32),
        },
        schema={"track_id": pl.Int64, "n_frames": pl.Int64, "D_ref_floor_um2_s": pl.Float64, "alpha": pl.Float64,
                "log10_D": pl.Float64, "loglik": pl.Float32},
    )


def _likelihoods_from(table: pl.DataFrame, tau_s: float, tau_ref_s: float) -> JointLikelihoods:
    """`loglik_long_table`'s table back to `JointLikelihoods`, its cells' D at the lag `tau_s`."""
    if "log10_D" not in table.columns:
        raise StaleAnalysisError("its joint likelihoods were saved by an earlier α · D version: Run to re-fit")
    table = table.sort("track_id", "alpha", "log10_D")
    alphas = np.unique(table["alpha"].to_numpy())
    D = np.unique(table["log10_D"].to_numpy())
    per_track = table.group_by("track_id", maintain_order=True).agg(
        pl.col("n_frames").first(), pl.col("D_ref_floor_um2_s").first(), pl.len().alias("n_cells"))
    if (per_track["n_cells"] != len(alphas) * len(D)).any():
        raise ValueError("loglik_alpha_D: every track needs one row per (alpha, log10_D) cell")
    return JointLikelihoods(
        per_track["track_id"].to_numpy(), per_track["n_frames"].to_numpy(),
        per_track["D_ref_floor_um2_s"].to_numpy(), alphas, D,
        table["loglik"].cast(pl.Float64).to_numpy().reshape(per_track.height, len(alphas), len(D)),
        tau_s, tau_ref_s,
    )


def restore_fbm(tracks: pl.DataFrame, record: dict, tracks_summary: pl.DataFrame,
                loglik: Optional[pl.DataFrame] = None, profiles: Optional[pl.DataFrame] = None) -> FbmAnalysis:
    """A saved alpha-D run back from `diffusion_summary.json`'s "alpha_D" entry,
    `tracks_summary.csv`'s columns and, when saved, `loglik_alpha_D.parquet`
    (`loglik_long_table`; without it the run has no population) and
    `profile_alpha.parquet` (`profile_long_table`; without it, no shared alpha). Raises
    `StaleAnalysisError` unless its tracks are, vertex for vertex, among
    `tracks` (as `diffusion.restore_analysis`)."""
    if "fbm_status" not in tracks_summary.columns:
        raise StaleAnalysisError("its per-track table has no alpha-D columns")
    if "alpha_grid" not in record:
        raise StaleAnalysisError("it was saved by an earlier α · D version (grid posteriors): Run to re-fit")
    ran = tracks_summary.filter(pl.col("fbm_status").is_not_null())
    fit_ids = ran["track_id"].cast(pl.Int64)
    fitted = tracks.filter(pl.col("track_id").is_in(fit_ids.implode()))
    if fitted["track_id"].n_unique() != fit_ids.n_unique() or tracks_fingerprint(fitted) != record.get("tracks_sha256"):
        raise StaleAnalysisError("its tracks differ from the ones loaded (re-tracked or rescaled since)")
    a_min, a_max, n_a = record["alpha_grid"]
    options = FbmPostOptions(min_frames=record["min_frames"], level=record["lr_level"], tau_ref_s=record["tau_ref_s"],
                             alpha_min=a_min, alpha_max=a_max, n_alpha=n_a)
    fits = ran.select(
        pl.col("track_length").cast(pl.Int64).alias("n_frames"),
        *(pl.col(c).cast(dtype) if c in ran.columns else pl.lit(None, dtype=dtype).alias(c)
          for c, dtype in FBM_SCHEMA.items()),
    ).sort("track_id")
    acquisition = Acquisition(dt_s=record["dt_s"], exposure_s=record["exposure_s"])
    ok_ids = set(fits.filter(pl.col("fbm_status") == "ok")["track_id"].to_list())
    likelihoods = None
    if loglik is not None and loglik.height:
        if record.get("population_tau_s") is None:
            raise StaleAnalysisError("it was saved by an earlier α · D version (no population lag): Run to re-fit")
        likelihoods = _likelihoods_from(loglik, record["population_tau_s"], record["tau_ref_s"])
        if set(likelihoods.track_ids.tolist()) != ok_ids:
            raise StaleAnalysisError("its joint likelihoods are not those of its fitted tracks")
    prof = None
    if profiles is not None and profiles.height:
        prof = _profiles_from(profiles)
        if set(prof.track_ids.tolist()) != ok_ids:
            raise StaleAnalysisError("its alpha profiles are not those of its fitted tracks")
    saved = record.get("grid")
    joint = joint_grid(saved) if saved else None
    return FbmAnalysis(fits, acquisition, options, record["tracks_sha256"], joint, likelihoods, prof)


def scatter_panels(analysis: FbmAnalysis, groups: dict[str, Optional[set]]) -> list[dict]:
    """What `joint_plot.plot_alpha_D` draws, one dict per group with fitted tracks:
    `name`, the group's "ok" rows (`fits`), its D_ref floor band and tau_ref."""
    panels = []
    for name, ids in groups.items():
        ok = _rows(analysis, ids).filter(pl.col("fbm_status") == "ok")
        if ok.height:
            panels.append({"name": name, "fits": ok, "floor": floor_band(ok["D_ref_floor_um2_s"]),
                           "tau_ref_s": analysis.tau_ref_s})
    return panels


def track_panel(analysis: FbmAnalysis, track: pl.DataFrame, n_cells: int = 241) -> Optional[dict]:
    """The selected track's likelihood over (alpha, log10 D_ref) for `joint_plot.plot_fbm_track`,
    recomputed (tens of ms for a short track, ~2 s at 300 frames): the log-likelihood relative
    to its maximum on a common grid, its profiles in alpha and in log10 D_ref, the likelihood-ratio
    threshold that makes the run's intervals, and the track's fit row. None unless the track was
    fitted "ok"."""
    track_id = int(track["track_id"][0])
    row = analysis.fits.filter(pl.col("track_id") == track_id)
    if row.height == 0 or row["fbm_status"][0] != "ok":
        return None
    fit = row.row(0, named=True)
    options = analysis.options
    hi = np.log10(fit["D_ref_high_um2_s"])
    lo = np.log10(fit["D_ref_low_um2_s"]) if fit["D_ref_low_um2_s"] else hi - 3.0
    floor = fit["D_ref_floor_um2_s"]
    if floor is not None and floor > 0:  # always in view: it is what D_ref is read against
        lo, hi = min(lo, np.log10(floor)), max(hi, np.log10(floor))
    pad = max(0.4, 0.15 * (hi - lo))
    axis = np.linspace(lo - pad, hi + pad, n_cells)
    ll = fbm_likelihood.log_likelihood_surface(track, analysis.acquisition, options, axis)
    ll = ll - ll.max()
    return {
        "track_id": track_id,
        "n_frames": int(row["n_frames"][0]),
        "alphas": options.alphas(),
        "log10_D_ref": axis,
        "loglik": ll,
        "profile_alpha": ll.max(axis=1),
        "profile_D_ref": ll.max(axis=0),
        "threshold": -chi2.ppf(options.level, 1) / 2,
        "level": options.level,
        "fit": fit,
        "tau_ref_s": options.tau_ref_s,
    }
