"""Bridge from this project's tracks_df to diffusionkit's tidy schema.

diffusionkit.analysis.io.load_tracks builds its tidy table (track_id,
frame, t_s, x_um, y_um, sigma_x_um, sigma_y_um, track_length) from a raw
CSV with track_id/frame/x/y/sigma_x/sigma_y in pixels -- exactly what
pipeline.run_track_step's tracks_df already has in memory (sfwloc's
link_tracks_df is points_df plus an added track_id column, so sigma_x/
sigma_y survive linking). tracks_to_diffusionkit_df is that same
conversion, minus the CSV round-trip, kept column-for-column identical to
load_tracks so diffusionkit code downstream sees the same thing either way.
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
        (pl.col("sigma_x") * pixel_size_um).alias("sigma_x_um"),
        (pl.col("sigma_y") * pixel_size_um).alias("sigma_y_um"),
    ).sort(["track_id", "frame"])
    track_lengths = tracks.group_by("track_id").agg(pl.len().alias("track_length"))
    return tracks.join(track_lengths, on="track_id", how="left").sort(["track_id", "frame"])
