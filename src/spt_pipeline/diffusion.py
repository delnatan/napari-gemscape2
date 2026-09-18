"""Bridge from this project's tracks_df to diffusionkit's tidy schema.

diffusionkit.classic.io.load_tracks builds its tidy table (track_id,
frame, t_s, x_um, y_um, sigma_x_um, sigma_y_um, track_length) from a raw
CSV with track_id/frame/x/y/sigma_x/sigma_y in pixels -- essentially what
pipeline.run_track_step's tracks_df already has in memory
(spotsolve.tracking.link is the localization table plus an added track_id
column, so every detector column survives linking).
tracks_to_diffusionkit_df is that same conversion, minus the CSV
round-trip, kept column-for-column identical to load_tracks so
diffusionkit code downstream sees the same thing either way.

The one rename: what diffusionkit calls `sigma_x`/`sigma_y` -- the
per-localization position error it builds its noise term from -- is
`se_x`/`se_y` in spotsolve's table, where it is the CRLB from the fit's
Fisher information. Same quantity, and the reason a dim detection
correctly gets a wider noise term than a bright one; only the name
differs, so it is renamed here rather than duplicated upstream.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl


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


# --- diffusionkit.classic.analyze_tracks results -----------------------
#
# `ClassicAnalysis.fits` has one row per (track_id, model): the Brownian
# displacement MLE (`brownian_mle`, the default analysis here) and two MSD
# fits (`brownian`, `power_law`) kept only as a comparison. The helpers
# below turn it into what the diffusion widget shows -- per-track columns
# under names that stay distinct from every other fit's, and the
# population summary -- without Qt, so the numbers can be checked against
# a direct diffusionkit call.

MLE_MODEL = "brownian_mle"

# Two-sided 5% cut on z. Under Brownian motion z ~ N(0,1), so about 5% of
# tracks exceed it by chance alone -- the reference the summary reads the
# observed fraction against.
Z_CRITICAL = 1.96


def mle_rows(fits: pl.DataFrame) -> pl.DataFrame:
    return fits.filter(pl.col("model") == MLE_MODEL)


def mle_track_table(fits: pl.DataFrame) -> pl.DataFrame:
    """One row per track: the MLE's D, its upper limit, and the
    non-Brownian score, plus `mle_status` -- which matters more here than
    for most fits, since `unresolved` (D̂ = 0: localization noise explains
    all the motion) has a D of exactly 0 and no z, and is not a failure."""
    return mle_rows(fits).select(
        "track_id",
        pl.col("status").alias("mle_status"),
        pl.col("D_um2_s").alias("D_mle_um2_s"),
        pl.col("D_upper_um2_s").alias("D_upper_mle_um2_s"),
        "z_nonbrownian",
        "p_nonbrownian",
        "alpha_1step",
        "p_motion",
    )


def msd_track_table(fits: pl.DataFrame) -> pl.DataFrame:
    """The MSD comparison, one row per track: `D_msd_um2_s` from the
    linear fit, `K_msd_um2_s_alpha`/`alpha_msd` from the power law. No
    uncertainties -- diffusionkit estimates none for MSD fits -- and
    nothing at all (status `excluded`) for a run with `exposure_s > 0`,
    since MSD fits have no motion-blur model."""
    brownian = fits.filter(pl.col("model") == "brownian").select(
        "track_id", pl.col("D_um2_s").alias("D_msd_um2_s")
    )
    power_law = fits.filter(pl.col("model") == "power_law").select(
        "track_id",
        pl.col("K_um2_s_alpha").alias("K_msd_um2_s_alpha"),
        pl.col("alpha").alias("alpha_msd"),
    )
    return brownian.join(power_law, on="track_id", how="full", coalesce=True)


def summarize_mle(fits: pl.DataFrame) -> dict:
    """The population-level read of the MLE rows of `fits`.

    D is a per-track estimate; z is read here, across tracks. At ~5
    frames one track's z can't classify it (a single α = 0.5 track is
    flagged only 4-8% of the time), but a mean-z shift of 0.4-0.7 is
    plain across ~100 tracks -- hence mean z with its standard error, the
    SD (≈1 if the noise model is calibrated), and the fraction beyond
    ±1.96 against the 5% expected by chance.

    `median_D_um2_s` is over tracks that resolved motion; `unresolved`
    ones (D̂ = 0) have no log D and would pull a median to 0, so they are
    counted separately with the median of their upper limits instead
    ("D < x"), rather than dropped.

    Every count is keyed by diffusionkit's own `status` (`n_ok`,
    `n_unresolved`, `n_excluded`, `n_invalid_input`, ...), so a track
    rejected for a frame gap or a zero localization SD shows up by name.
    """
    mle = mle_rows(fits)
    statuses = dict(mle.group_by("status").len().iter_rows())
    fitted = mle.filter(pl.col("status").is_in(["ok", "unresolved"]))
    resolved = mle.filter(pl.col("z_nonbrownian").is_not_null() & (pl.col("D_um2_s") > 0))
    unresolved = mle.filter(pl.col("status") == "unresolved")
    z = resolved["z_nonbrownian"].to_numpy().astype(float)
    n_z = len(z)
    sd_z = float(np.std(z, ddof=1)) if n_z > 1 else None

    summary = {
        "n_tracks": mle.height,
        **{f"n_{status}": int(count) for status, count in sorted(statuses.items())},
        "n_frames_min": _scalar(fitted["n_frames"].min()),
        "n_frames_median": _scalar(fitted["n_frames"].median()),
        "n_frames_max": _scalar(fitted["n_frames"].max()),
        "median_D_um2_s": _scalar(resolved["D_um2_s"].median()),
        "unresolved_D_upper_median_um2_s": _scalar(unresolved["D_upper_um2_s"].median()),
        "n_z": n_z,
        "mean_z": float(np.mean(z)) if n_z else None,
        "se_mean_z": sd_z / np.sqrt(n_z) if sd_z is not None else None,
        "sd_z": sd_z,
        "frac_abs_z_gt_1_96": float(np.mean(np.abs(z) > Z_CRITICAL)) if n_z else None,
    }
    return summary


def summarize_mle_by_group(fits: pl.DataFrame, groups: pl.DataFrame) -> dict[str, dict]:
    """`summarize_mle` per group -- e.g. per ROI, when each region's
    tracks were linked on their own and the question is whether their
    motion differs. `groups` has `track_id` plus a `group` column; tracks
    it doesn't list are left out, and groups come back in `groups`' own
    order of first appearance."""
    labeled = fits.join(groups.select("track_id", "group"), on="track_id", how="inner")
    names = groups["group"].unique(maintain_order=True).drop_nulls().to_list()
    return {
        name: summarize_mle(labeled.filter(pl.col("group") == name).drop("group"))
        for name in names
    }


def _scalar(value) -> Optional[float]:
    return None if value is None else float(value)
