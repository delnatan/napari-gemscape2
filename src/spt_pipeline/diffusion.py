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
