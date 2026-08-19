"""The one shared detect+track entry point.

Both the headless CLI (`cli.py`) and the interactive napari widget
(`widgets/experiment_list.py`) call `run_detect_track` -- the logic lives
here exactly once, unlike napari-gemscape where the interactive handlers
and its batch subprocess script each reimplemented the pipeline.

This mirrors sfwloc/scripts/track_beads_timelapse.py, which remains the
reference implementation: find_spots -> calibrate sigma -> bootstrap link
(fixed generous gate) -> estimate D from single-step MSD -> final link
(auto-derived gate). See that script's module docstring for why the gate
is derived rather than hand-picked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import polars as pl

from sfwloc.report import (
    calibrate_sigma_df,
    find_spots_df,
    find_spots_stack_df,
    link_tracks_df,
    recommended_gate_px,
)
from sfwloc.tracking_diagnostics import check_resolvability
from spt_pipeline.io_formats import load_stack

DEFAULT_SOLVER_KWARGS = dict(
    lam=0.15,
    refine_lam=0.0,
    n_iter=200,
    fista_iter=20,
    n_refine=2,
    refine_iter=10,
    varpro_fista_iter=50,
    prune_tol=1e-4,
    delta_dev_tol=1e-2,
    delta_dev_patience=3,
    delta_dev_min_iter=5,
    birth_test=True,
    split_test=True,
)

# ProgressCallback(done, total, stage) -- called from whatever thread
# run_detect_track executes on; the interactive widget wraps this in a
# QObject signal to cross back onto the Qt event-loop thread safely.
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class DetectTrackParams:
    sigma_init: float = 1.3
    bootstrap_gate_px: float = 3.0
    solver_kwargs: dict = field(default_factory=lambda: dict(DEFAULT_SOLVER_KWARGS))


def estimate_D_um2_s(linked_df: pl.DataFrame, dt_s: float, pixel_size_um: float, sigma_loc_um: float):
    """Single-step MSD estimate of D, corrected for localization noise:
    mean(r^2) = 4*D*dt + 4*sigma_loc_um^2."""
    df = linked_df.sort(["track_id", "frame"]).with_columns(
        pl.col("frame").diff().over("track_id").alias("dframe"),
        pl.col("y").diff().over("track_id").alias("dy_px"),
        pl.col("x").diff().over("track_id").alias("dx_px"),
    )
    valid = df.filter(pl.col("dframe") == 1)
    if valid.height == 0:
        return 0.0, 0
    dy_um = valid["dy_px"].to_numpy() * pixel_size_um
    dx_um = valid["dx_px"].to_numpy() * pixel_size_um
    mean_r2_um2 = float(np.mean(dy_um**2 + dx_um**2))
    D_est = max(0.0, (mean_r2_um2 - 4.0 * sigma_loc_um**2) / (4.0 * dt_s))
    return D_est, valid.height


def run_detect_track(
    image_path: str | Path,
    pixel_size_um: Optional[float] = None,
    dt_s: Optional[float] = None,
    channel: int = 0,
    z_index: int = 0,
    params: Optional[DetectTrackParams] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Run the full detect+track pipeline on one timelapse.

    `image_path` can be .tif/.tiff, .nd2, or .ims (see `io_formats.load_stack`).
    `channel`/`z_index` pick which plane to track for files with more than
    one (both default to 0).

    `pixel_size_um`/`dt_s` fall back to the file's own metadata if not
    given explicitly. If `progress_callback` is given, spot-finding runs
    frame-by-frame (reporting progress each frame) instead of the faster
    rayon-parallel `find_spots_stack_df` -- the same tradeoff
    `track_beads_timelapse.py` makes for an interactively-watched run.

    Returns (points_df, tracks_df, manifest_extra) -- `manifest_extra` is
    meant to be passed as `experiment.build_manifest`'s `params`.
    """
    params = params or DetectTrackParams()
    im, file_pixel_size_um, file_dt_s = load_stack(image_path, channel=channel, z_index=z_index)
    pixel_size_um = pixel_size_um if pixel_size_um is not None else file_pixel_size_um
    dt_s = dt_s if dt_s is not None else file_dt_s
    if pixel_size_um is None or dt_s is None:
        raise ValueError(
            f"{image_path}: pixel_size_um/dt_s not found in file metadata "
            "and not given explicitly"
        )
    t, h, w = im.shape

    def report(done: int, total: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(done, total, stage)

    total_steps = t + 3

    report(0, total_steps, "calibrating sigma")
    _, calib_summary = calibrate_sigma_df(im[0], sigma_init=params.sigma_init)
    sigma = calib_summary["sigma_estimate"]

    bg = np.median(im, axis=(1, 2))

    if progress_callback is not None:
        frames = []
        for i in range(t):
            frames.append(
                find_spots_df(im[i], sigma, bg[i], frame_idx=i, **params.solver_kwargs)
            )
            report(i + 1, total_steps, "finding spots")
        points_df = pl.concat(frames)
    else:
        points_df = find_spots_stack_df(im, sigma, bg, **params.solver_kwargs)

    report(t + 1, total_steps, "bootstrap linking")
    bootstrap = link_tracks_df(points_df, params.bootstrap_gate_px)
    sigma_y_um = points_df["sigma_y"].to_numpy() * pixel_size_um
    sigma_x_um = points_df["sigma_x"].to_numpy() * pixel_size_um
    sigma_loc_um = float(np.median(np.sqrt((sigma_y_um**2 + sigma_x_um**2) / 2.0)))
    D_est, n_links = estimate_D_um2_s(bootstrap, dt_s, pixel_size_um, sigma_loc_um)

    active_area_um2 = (h * pixel_size_um) * (w * pixel_size_um)
    mean_n_per_frame = points_df.group_by("frame").len()["len"].mean() if points_df.height else 0.0
    density_um2 = (mean_n_per_frame / active_area_um2) if mean_n_per_frame else 0.0
    resolvability = check_resolvability(D_est, dt_s, density_um2)

    report(t + 2, total_steps, "final linking")
    final_gate_px = recommended_gate_px(D_est, dt_s, pixel_size_um, sigma_loc_um=sigma_loc_um)
    tracks_df = link_tracks_df(points_df, final_gate_px)
    report(total_steps, total_steps, "done")

    manifest_extra = {
        "pixel_size_um": pixel_size_um,
        "dt_s": dt_s,
        "channel": channel,
        "z_index": z_index,
        "sigma_px": sigma,
        "sigma_loc_um": sigma_loc_um,
        "D_est_um2_s": D_est,
        "n_bootstrap_links": n_links,
        "final_gate_px": final_gate_px,
        "density_um2": density_um2,
        "resolvability_message": resolvability["message"],
        "n_points": points_df.height,
        "n_tracks": tracks_df["track_id"].n_unique() if tracks_df.height else 0,
        "solver_kwargs": params.solver_kwargs,
    }
    return points_df, tracks_df, manifest_extra
