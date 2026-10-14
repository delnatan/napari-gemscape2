"""GUI-free alpha-D analysis: diffusionkit's fBm grid posteriors (`diffusionkit.fbmpost`)
for this project's tracks, and the tables and plot data the widget's α · D tab shows
and the bundle saves.

`diffusion.py` holds the standard quantification, the Brownian D posterior
(`gridpost`). This is a second, separate posterior: per track, the joint grid
posterior over the anomalous exponent alpha and an amplitude, under fractional
Brownian motion with the same exposure blur and per-frame localization SDs. Its
likelihood is gridpost's at alpha = 1.

Every per-track number here is a posterior under stated priors, and the priors
matter (diffusionkit's docs/fbmpost.md, "Prior and likelihood"):

  - **alpha** is flat on [0.02, 1.98]. A track with no information on alpha
    returns that prior (mean 1, SD 0.57), and `alpha_info_bits` is then ~0;
  - **the amplitude** is flat in ln SNR. Summing it out is what makes a track's
    alpha curve, so near the localization floor that prior shapes the curve.

**D_ref is not a Brownian D.** It is the fBm amplitude expressed as
D_ref = MSD_1D(tau_ref) / (2 tau_ref), the true motion's apparent D at one lag
`tau_ref_s` (default 0.1 s). It equals a Brownian D only at alpha = 1, and
otherwise changes with tau_ref (at alpha = 0.3, from 0.1 s to 1 s divides it by
5). Its margin inherits the alpha prior: below its floor (`D_ref_floor_um2_s`)
read it as an upper bound. So it is never compared with `D_mean_um2_s` or an
MSD D, and `D_mean_um2_s` is never paired with `alpha_mean`.

**The population of alpha** (`alpha_population`) is diffusionkit's 2D
deconvolution of the tracks' joint (alpha, log10 D_ref) likelihoods
(`fbmpost.deconvolve_tracks`), kept per track by the run (`JointLikelihoods`)
and saved in `loglik_alpha_D.parquet`. A track's alpha curve is a posterior,
not a likelihood: its amplitude prior tilts it near the floor, so tracks do not
combine by adding those curves (a 1D deconvolution of them was biased in
simulation). The joint likelihoods carry no prior, and the D_ref population the
alpha population is marginalized under is fitted with it. Below the D_ref floor
the tracks barely tell D_ref values apart, and alpha there is mostly the
smoothness prior's continuation of alpha above it, so the summary also gives
the reads above the floor and the share of the population below it.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import polars as pl
from diffusionkit import Acquisition
from diffusionkit.fbmpost import AlphaDPopulation, FbmPostOptions, JointLikelihoods
from diffusionkit.fbmpost import analyze_tracks as dk_analyze_fbm
from diffusionkit.fbmpost import deconvolve_tracks as dk_deconvolve_alpha_D
from diffusionkit.fbmpost import posterior as fbm_posterior
from diffusionkit.fbmpost.population import informative_range

from napari_gemscape2.diffusion import LEVEL, MIN_FRAMES, StaleAnalysisError, floor_band, tracks_fingerprint

# The per-track columns a run adds to the tracks pane and `tracks_summary.csv`.
FBM_SCHEMA = {
    "track_id": pl.Int64,
    # diffusionkit's status for this posterior ("ok", "excluded",
    # "invalid_input"), apart from the D posterior's `posterior_status`.
    "fbm_status": pl.String,
    # alpha's posterior mean, 90% interval, SD, information (bits gained over
    # the flat prior) and probability of alpha < 1.
    "alpha_mean": pl.Float64,
    "alpha_low": pl.Float64,
    "alpha_high": pl.Float64,
    "alpha_sd": pl.Float64,
    "alpha_info_bits": pl.Float64,
    "alpha_p_below_1": pl.Float64,
    # D_ref at the run's tau_ref: E[D_ref] and its 90% interval -- not a Brownian D.
    "D_ref_mean_um2_s": pl.Float64,
    "D_ref_low_um2_s": pl.Float64,
    "D_ref_high_um2_s": pl.Float64,
    # The ellipse: log10 D_ref's mean and SD and its correlation with alpha.
    "log10_D_ref_mean": pl.Float64,
    "log10_D_ref_sd": pl.Float64,
    "alpha_D_ref_corr": pl.Float64,
    # The D_ref at which the motion's MSD at tau_ref equals the localization noise's.
    "D_ref_floor_um2_s": pl.Float64,
    # Which edge of the amplitude prior cuts the posterior: "low" (D_ref an upper bound), "high", "both".
    "D_ref_grid_edge": pl.String,
}
FBM_COLUMNS = tuple(c for c in FBM_SCHEMA if c != "track_id")

# Below this many bits a track's alpha is mostly its prior: the scatter
# draws it grey, without an ellipse.
MIN_BITS = 0.5
# The default reference lag, diffusionkit's.
TAU_REF_S = FbmPostOptions().tau_ref_s

_RENAME = {
    "status": "fbm_status",
    "alpha_post_mean": "alpha_mean",
    "alpha_post_lo": "alpha_low",
    "alpha_post_hi": "alpha_high",
    "alpha_post_sd": "alpha_sd",
    "alpha_post_info_bits": "alpha_info_bits",
    "alpha_post_p_below_1": "alpha_p_below_1",
    "D_ref_post_mean_um2_s": "D_ref_mean_um2_s",
    "D_ref_post_lo_um2_s": "D_ref_low_um2_s",
    "D_ref_post_hi_um2_s": "D_ref_high_um2_s",
    "log10_D_ref_post_mean": "log10_D_ref_mean",
    "log10_D_ref_post_sd": "log10_D_ref_sd",
    "alpha_log10_D_ref_corr": "alpha_D_ref_corr",
    "D_grid_edge": "D_ref_grid_edge",
}


def fbm_options(min_frames: int = MIN_FRAMES, tau_ref_s: float = TAU_REF_S) -> FbmPostOptions:
    """The `FbmPostOptions` a run here uses: diffusionkit's priors and grids, the
    credible level fixed at `diffusion.LEVEL`. Raises ValueError for an invalid value."""
    return FbmPostOptions(min_frames=min_frames, level=LEVEL, tau_ref_s=tau_ref_s)


@dataclass(frozen=True)
class FbmAnalysis:
    """One run of `analyze_fbm`: a row per input track (`FBM_SCHEMA`, plus
    `n_frames`), what it ran with, `tracks_fingerprint` of its tracks, and
    each fitted track's joint likelihood (`likelihoods`, None for a run saved
    without them), what `alpha_population` deconvolves."""

    fits: pl.DataFrame
    acquisition: Acquisition
    options: FbmPostOptions
    tracks_sha256: Optional[str] = None
    likelihoods: Optional[JointLikelihoods] = field(default=None, compare=False, repr=False)
    # `alpha_population`'s results by track ids: ~10 s each for a thousand tracks.
    _populations: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def tau_ref_s(self) -> float:
        return self.options.tau_ref_s

    @property
    def ok(self) -> pl.DataFrame:
        return self.fits.filter(pl.col("fbm_status") == "ok")

    def columns(self) -> pl.DataFrame:
        """The per-track columns (`FBM_SCHEMA`) for the tracks pane."""
        return self.fits.select(FBM_SCHEMA.keys())


def analyze_fbm(
    tracks: pl.DataFrame,
    acquisition: Acquisition,
    options: FbmPostOptions,
    *,
    progress: Optional[Callable[[int, int], None]] = None,
    threads: Optional[int] = None,
) -> FbmAnalysis:
    """`diffusionkit.fbmpost.analyze_tracks` on `tracks` (`tracks_to_diffusionkit_df`'s
    table), on a pool of `threads` (default: one per CPU), reshaped to `FBM_SCHEMA`,
    with each fitted track's joint likelihood kept over the D_ref cells some
    track tells apart (`fbmpost.population.informative_range`)."""
    threads = threads or os.cpu_count() or 1
    if threads > 1:
        with ThreadPoolExecutor(threads) as pool:
            result = dk_analyze_fbm(tracks, acquisition, options, progress=progress, keep_likelihoods=True,
                                    map_fn=pool.map)
    else:
        result = dk_analyze_fbm(tracks, acquisition, options, progress=progress, keep_likelihoods=True)
    fits = result.fits.rename(_RENAME).select("n_frames", *FBM_SCHEMA.keys()).cast(FBM_SCHEMA)
    return FbmAnalysis(fits, acquisition, options, tracks_fingerprint(tracks), _cropped(result.likelihoods))


def _cropped(lik: JointLikelihoods) -> Optional[JointLikelihoods]:
    """`lik` over its informative D_ref cells: a subset's range lies inside it, and the rest is
    either flat (below detection) or negligible for every track."""
    if not len(lik.track_ids):
        return None
    keep = informative_range(lik)
    return JointLikelihoods(lik.track_ids, lik.n_frames, lik.D_ref_floor_um2_s, lik.alphas,
                            lik.log10_D_ref[keep], lik.loglik[:, :, keep])


def _rows(analysis: FbmAnalysis, ids: Optional[set]) -> pl.DataFrame:
    rows = analysis.fits
    return rows if ids is None else rows.filter(pl.col("track_id").is_in(list(ids)))


def summarize_fbm(analysis: FbmAnalysis, ids: Optional[set] = None) -> dict:
    """Counts over the tracks `ids` (None: all): by status, how many carry at
    least `MIN_BITS` on alpha, how many only bound D_ref from above, and the
    D_ref floor. The population's numbers are `population_summary`'s."""
    rows = _rows(analysis, ids)
    ok = rows.filter(pl.col("fbm_status") == "ok")
    out = {
        "n_tracks": rows.height,
        "n_ok": ok.height,
        **{f"n_{s}": n for s, n in rows["fbm_status"].value_counts().iter_rows() if s != "ok"},
        "n_alpha_informative": int((ok["alpha_info_bits"] >= MIN_BITS).sum()),
        "n_D_ref_upper_bound": int((ok["D_ref_grid_edge"] == "low").sum()),
    }
    band = floor_band(ok["D_ref_floor_um2_s"])
    if band is not None:
        out["D_ref_floor_median_um2_s"] = band[1]
    return out


# A population needs this many fitted tracks; fewer say little beyond their own posteriors.
MIN_POPULATION_TRACKS = 20
_POPULATION_CACHE_SIZE = 16


def alpha_population(analysis: FbmAnalysis, ids: Optional[set] = None) -> Optional[AlphaDPopulation]:
    """The population of (alpha, log10 D_ref) over the tracks `ids` (None: all
    fitted), `fbmpost.deconvolve_tracks` of their joint likelihoods -- or None
    without likelihoods or with fewer than `MIN_POPULATION_TRACKS` fitted
    tracks. ~10 s for a thousand tracks; cached on `analysis`."""
    lik = analysis.likelihoods
    if lik is None:
        return None
    key = None if ids is None else frozenset(ids)
    if key in analysis._populations:
        return analysis._populations[key]
    rows = np.arange(len(lik.track_ids)) if ids is None else np.flatnonzero(np.isin(lik.track_ids, list(ids)))
    pop = dk_deconvolve_alpha_D(lik.rows(rows)) if len(rows) >= MIN_POPULATION_TRACKS else None
    if len(analysis._populations) >= _POPULATION_CACHE_SIZE:
        analysis._populations.clear()
    analysis._populations[key] = pop
    return pop


def population_is_cached(analysis: FbmAnalysis, ids: Optional[set] = None) -> bool:
    """Whether `alpha_population(analysis, ids)` would return without computing."""
    return analysis.likelihoods is None or (None if ids is None else frozenset(ids)) in analysis._populations


def _interval(draws: np.ndarray, level: float) -> list[float]:
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


def population_panels(analysis: FbmAnalysis, groups: dict[str, Optional[set]]) -> list[dict]:
    """What `joint_plot.plot_alpha_population` draws, one dict per group with
    a population: `name`, the `AlphaDPopulation` (`population`), its summary,
    the credible level and tau_ref."""
    panels = []
    for name, ids in groups.items():
        pop = alpha_population(analysis, ids)
        if pop is not None:
            panels.append({"name": name, "population": pop, "summary": population_summary(pop, analysis.options.level),
                           "level": analysis.options.level, "tau_ref_s": analysis.tau_ref_s})
    return panels


def fbm_record(analysis: FbmAnalysis, ids: Optional[set], by_class: Optional[dict]) -> dict:
    """`diffusion_summary.json`'s "alpha_D" entry: what the run was, its
    settings, `summarize_fbm` over the tracks the filters pass and, under
    "population", `population_summary` of their `alpha_population` (each
    region class's under "by_region_class"). A population not yet computed
    is computed here: warm the cache off the GUI thread first."""
    o = analysis.options
    record = {
        "analysis": "diffusionkit.fbmpost grid posteriors under blurred fBm: flat prior on alpha, "
        "flat prior in ln SNR on the amplitude; D_ref = MSD_1D(tau_ref) / (2 tau_ref) is not a Brownian D",
        "dt_s": analysis.acquisition.dt_s,
        "exposure_s": analysis.acquisition.exposure_s,
        "min_frames": o.min_frames,
        "credible_level": o.level,
        "tau_ref_s": o.tau_ref_s,
        "alpha_prior": [o.alpha_min, o.alpha_max, o.n_alpha],
        "snr_prior": [o.snr_min, o.snr_max, o.n_snr],
        "tracks_sha256": analysis.tracks_sha256,
        **summarize_fbm(analysis, ids),
    }
    pop = alpha_population(analysis, ids)
    if pop is not None:
        record["population"] = population_summary(pop, o.level)
    if by_class:
        record["by_region_class"] = {}
        for name, gids in by_class.items():
            entry = summarize_fbm(analysis, gids)
            pop = alpha_population(analysis, gids)
            if pop is not None:
                entry["population"] = population_summary(pop, o.level)
            record["by_region_class"][name] = entry
    return record


def loglik_long_table(analysis: FbmAnalysis) -> Optional[pl.DataFrame]:
    """Every fitted track's joint likelihood, one row per (track, cell):
    `track_id`, `alpha`, `log10_D_ref`, `loglik` (each track's maximum 0) --
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
            "log10_D_ref": np.tile(lik.log10_D_ref, n * n_a),
            "loglik": lik.loglik.reshape(-1).astype(np.float32),
        },
        schema={"track_id": pl.Int64, "n_frames": pl.Int64, "D_ref_floor_um2_s": pl.Float64, "alpha": pl.Float64,
                "log10_D_ref": pl.Float64, "loglik": pl.Float32},
    )


def _likelihoods_from(table: pl.DataFrame) -> JointLikelihoods:
    """`loglik_long_table`'s table back to `JointLikelihoods`."""
    table = table.sort("track_id", "alpha", "log10_D_ref")
    alphas = np.unique(table["alpha"].to_numpy())
    D = np.unique(table["log10_D_ref"].to_numpy())
    per_track = table.group_by("track_id", maintain_order=True).agg(
        pl.col("n_frames").first(), pl.col("D_ref_floor_um2_s").first(), pl.len().alias("n_cells"))
    if (per_track["n_cells"] != len(alphas) * len(D)).any():
        raise ValueError("loglik_alpha_D: every track needs one row per (alpha, log10_D_ref) cell")
    return JointLikelihoods(
        per_track["track_id"].to_numpy(), per_track["n_frames"].to_numpy(),
        per_track["D_ref_floor_um2_s"].to_numpy(), alphas, D,
        table["loglik"].cast(pl.Float64).to_numpy().reshape(per_track.height, len(alphas), len(D)),
    )


def restore_fbm(tracks: pl.DataFrame, record: dict, tracks_summary: pl.DataFrame,
                loglik: Optional[pl.DataFrame] = None) -> FbmAnalysis:
    """A saved alpha-D run back from `diffusion_summary.json`'s "alpha_D" entry,
    `tracks_summary.csv`'s columns and, when saved, `loglik_alpha_D.parquet`
    (`loglik_long_table`; without it the run has no population). Raises
    `StaleAnalysisError` unless its tracks are, vertex for vertex, among
    `tracks` (as `diffusion.restore_analysis`)."""
    if "fbm_status" not in tracks_summary.columns:
        raise StaleAnalysisError("its per-track table has no alpha-D columns")
    ran = tracks_summary.filter(pl.col("fbm_status").is_not_null())
    fit_ids = ran["track_id"].cast(pl.Int64)
    fitted = tracks.filter(pl.col("track_id").is_in(fit_ids.implode()))
    if fitted["track_id"].n_unique() != fit_ids.n_unique() or tracks_fingerprint(fitted) != record.get("tracks_sha256"):
        raise StaleAnalysisError("its tracks differ from the ones loaded (re-tracked or rescaled since)")
    a_min, a_max, n_a = record["alpha_prior"]
    s_min, s_max, n_s = record["snr_prior"]
    options = FbmPostOptions(
        min_frames=record["min_frames"], level=record["credible_level"], tau_ref_s=record["tau_ref_s"],
        alpha_min=a_min, alpha_max=a_max, n_alpha=n_a, snr_min=s_min, snr_max=s_max, n_snr=n_s,
    )
    fits = ran.select(
        pl.col("track_length").cast(pl.Int64).alias("n_frames"),
        *(pl.col(c).cast(dtype) if c in ran.columns else pl.lit(None, dtype=dtype).alias(c)
          for c, dtype in FBM_SCHEMA.items()),
    ).sort("track_id")
    acquisition = Acquisition(dt_s=record["dt_s"], exposure_s=record["exposure_s"])
    likelihoods = None
    if loglik is not None and loglik.height:
        likelihoods = _likelihoods_from(loglik)
        ok_ids = set(fits.filter(pl.col("fbm_status") == "ok")["track_id"].to_list())
        if set(likelihoods.track_ids.tolist()) != ok_ids:
            raise StaleAnalysisError("its joint likelihoods are not those of its fitted tracks")
    return FbmAnalysis(fits, acquisition, options, record["tracks_sha256"], likelihoods)


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
    """The selected track's joint posterior for `joint_plot.plot_fbm_track`,
    recomputed (tens of ms for a short track, ~1 s at 300 frames): the posterior
    density over (alpha, log10 D_ref) on a common grid, both margins, the flat
    alpha prior and the track's fit row. None unless the track was fitted "ok"."""
    track_id = int(track["track_id"][0])
    row = analysis.fits.filter(pl.col("track_id") == track_id)
    if row.height == 0 or row["fbm_status"][0] != "ok":
        return None
    options = analysis.options
    grid = fbm_posterior.log_likelihood_grid(track, analysis.acquisition, options)
    p = np.exp(fbm_posterior.log_posterior(grid.loglik))  # (n_alpha, n_snr), sums to 1
    x = fbm_posterior.log10_D_ref(grid.ln_K, options)  # each cell's log10 D_ref
    alphas = options.alphas()
    p_alpha = p.sum(axis=1)
    # Each alpha row is the same ln SNR grid shifted in log10 D_ref: its mass per
    # unit log10 D_ref, interpolated onto one axis spanning where the mass is.
    order = np.argsort(x, axis=None)
    cdf = np.cumsum(p.ravel()[order])
    lo, hi = np.interp([1e-4, 1 - 1e-4], cdf, x.ravel()[order])
    floor = row["D_ref_floor_um2_s"][0]
    if floor is not None and floor > 0:  # always in view: it is what D_ref is read against
        lo, hi = min(lo, np.log10(floor)), max(hi, np.log10(floor))
    pad = max(0.4, 0.15 * (hi - lo))
    axis = np.linspace(lo - pad, hi + pad, n_cells)
    dx = np.diff(x, axis=1).mean(axis=1)  # the row's log10 D_ref step
    density = np.array([np.interp(axis, xr, pr / d, left=0.0, right=0.0) for xr, pr, d in zip(x, p, dx)])
    return {
        "track_id": track_id,
        "n_frames": int(row["n_frames"][0]),
        "alphas": alphas,
        "log10_D_ref": axis,
        "density": density / (alphas[1] - alphas[0]),  # per unit alpha and log10 D_ref
        "p_alpha": p_alpha / (alphas[1] - alphas[0]),
        "p_D_ref": density.sum(axis=0),
        "prior_alpha": 1.0 / (options.alpha_max - options.alpha_min + (alphas[1] - alphas[0])),
        "fit": row.row(0, named=True),
        "tau_ref_s": options.tau_ref_s,
    }
