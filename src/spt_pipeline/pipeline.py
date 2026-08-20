"""The shared detect+track pipeline, as three composable stages plus one
convenience wrapper that runs all of them.

Both the headless CLI (`cli.py`) and the interactive napari widget
(`widgets/experiment_list.py`) build on this module -- the logic lives
here exactly once, unlike napari-gemscape where the interactive handlers
and its batch subprocess script each reimplemented the pipeline.

This mirrors sfwloc/scripts/track_beads_timelapse.py, which remains the
reference implementation: find_spots -> calibrate sigma -> bootstrap link
(fixed generous gate) -> estimate D from single-step MSD -> final link
(auto-derived gate). See that script's module docstring for why the gate
is derived rather than hand-picked.

`PipelineSession` + `load_session`/`run_calibration_step`/
`run_detect_step`/`run_track_step` let each stage run independently and be
re-run after tweaking that stage's own knobs, without redoing earlier
stages -- this is what backs the dock widget's per-tab "Run calibration" /
"Run detect" / "Run tracking" buttons (each stage builds on whatever the
session already has: `run_detect_step` uses `session.sigma` from a prior
calibration if the caller doesn't pass one explicitly; `run_track_step`
needs `session.points_df` from a prior detect). `run_detect_track` composes
all three in one call for the headless CLI and the widget's batch/multi-
select "Run" action, where stepwise control isn't needed.
"""

from __future__ import annotations

import threading
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
    source_refine_steps=5,
    source_refine_step=1.0,
    pos_bound=1.0,
    refine_iter=10,
    amp_upper=1e5,
    varpro_fista_iter=50,
    prune_tol=1e-4,
    delta_dev_tol=1e-2,
    delta_dev_patience=3,
    delta_dev_min_iter=5,
    birth_test=True,
    split_test=False,
    split_init_sep=0.75,
    glrt_footprint_sigma=4.0,
    glrt_alpha=1e-3,
    glrt_lm_iter=30,
)

# Forwarded to sfwloc.report.calibrate_sigma_df (-> sfwloc_py's
# calibrate_sigma_from_image) as **kwargs, alongside sigma_init -- these
# mirror that binding's own defaults (rust/sfwloc-py/src/lib.rs).
DEFAULT_CALIBRATION_KWARGS = dict(
    box_size=11,
    sigma_lo=0.7,
    sigma_hi=5.0,
    delta_bound=1.5,
    amp_upper=1e6,
    bg_upper=1e4,
    peak_threshold_rel=0.35,
    peak_footprint=3,
    peak_border=0,
    peak_top_k=None,
    n_iter=30,
    mu0=1.0,
    mu_decrease=0.5,
    mu_increase=4.0,
    step_tol=1e-5,
    loss_tol=1e-7,
    gtol=1e-5,
)

# ProgressCallback(done, total, stage) -- called from whatever thread the
# stage function executes on; the interactive widget wraps this in a
# QObject signal to cross back onto the Qt event-loop thread safely.
ProgressCallback = Callable[[int, int, str], None]


class PipelineCancelled(Exception):
    """Raised at a `cancel_event` checkpoint (see `run_detect_step`/
    `run_track_step`/`run_calibration_step`'s `cancel_event` argument).
    Cooperative cancellation only -- takes effect at the next frame (detect)
    or stage boundary (calibration/track), not instantly, since
    `calibrate_sigma_df`/`link_tracks_df` are each one opaque Rust call with
    no interruption point of their own. Propagates like any other exception
    through `napari.qt.threading`'s `errored` signal; the widget is
    responsible for telling this apart from a real error."""


def _check_cancelled(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise PipelineCancelled("cancelled")


@dataclass
class DetectTrackParams:
    sigma_init: float = 1.3
    calibration_frame_index: int = 0
    bootstrap_gate_px: float = 3.0
    solver_kwargs: dict = field(default_factory=lambda: dict(DEFAULT_SOLVER_KWARGS))
    calibration_kwargs: dict = field(default_factory=lambda: dict(DEFAULT_CALIBRATION_KWARGS))
    # (start, end) frame slice, Python-slice semantics; None, or end <= 0,
    # means through the real last frame (see _resolve_frame_range).
    # Not `mask` -- that's an interactive-only concept (built from a live
    # napari Shapes layer), not something a headless/serialized
    # DetectTrackParams can carry. See run_detect_step's docstring.
    frame_range: Optional[tuple[int, int]] = None


@dataclass
class PipelineSession:
    """Mutable state threaded through the Calibrate -> Detect -> Track
    stages for one image. `*_used` fields record what each completed
    stage actually ran with, so `session_manifest_extra` reflects reality
    even if a UI knob was changed after that stage last ran."""

    image_path: Path
    image: np.ndarray  # (T, H, W)
    pixel_size_um: float
    dt_s: float
    channel: int = 0
    z_index: int = 0
    bg: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (T,) per-frame median background

    sigma: Optional[float] = None
    calib_summary: Optional[dict] = None
    calib_points_df: Optional[pl.DataFrame] = None
    calibration_kwargs_used: Optional[dict] = None
    calibration_frame_used: Optional[int] = None

    points_df: Optional[pl.DataFrame] = None
    solver_kwargs_used: Optional[dict] = None
    frame_range_used: Optional[tuple[int, int]] = None
    # Polygon ROI record(s) (see spt_pipeline.rois.shapes_layer_to_roi) for
    # whatever napari Shapes layer backed run_detect_step's `mask`, if any
    # -- set by the widget (not pipeline.py itself, which stays napari-
    # agnostic), carried through to session_manifest_extra's caller so
    # write_experiment can persist it alongside points/tracks.
    roi: Optional[list[dict]] = None

    tracks_df: Optional[pl.DataFrame] = None
    track_summary: Optional[dict] = None
    bootstrap_gate_px_used: Optional[float] = None


def load_session(
    image_path: str | Path,
    pixel_size_um: Optional[float] = None,
    dt_s: Optional[float] = None,
    channel: int = 0,
    z_index: int = 0,
) -> PipelineSession:
    """Load a timelapse and start a fresh (un-calibrated, un-detected,
    un-tracked) `PipelineSession`."""
    im, file_pixel_size_um, file_dt_s = load_stack(image_path, channel=channel, z_index=z_index)
    pixel_size_um = pixel_size_um if pixel_size_um is not None else file_pixel_size_um
    dt_s = dt_s if dt_s is not None else file_dt_s
    if pixel_size_um is None or dt_s is None:
        raise ValueError(
            f"{image_path}: pixel_size_um/dt_s not found in file metadata "
            "and not given explicitly"
        )
    return PipelineSession(
        image_path=Path(image_path),
        image=im,
        pixel_size_um=pixel_size_um,
        dt_s=dt_s,
        channel=channel,
        z_index=z_index,
        bg=np.median(im, axis=(1, 2)),
    )


def run_calibration_step(
    session: PipelineSession,
    sigma_init: float,
    calibration_kwargs: Optional[dict] = None,
    frame_index: int = 0,
    cancel_event: Optional[threading.Event] = None,
) -> PipelineSession:
    """PSF-sigma calibration against `session.image[frame_index]` (default:
    the first frame). Sets `session.sigma`/`session.calib_summary` in place
    (and returns `session`, for chaining).

    `frame_index` matters when the default frame isn't a good calibration
    reference -- e.g. sparser or better-focused elsewhere in the stack.

    `session.calib_points_df` gets the full per-spot fit table
    (`calibrate_sigma_df`'s first return value: one row per candidate
    peak, columns incl. `y`/`x`/`sigma`/`se_sigma`/`nll`/`converged`/
    `laplace_ok`/`at_bound`) -- meant to be shown as a napari Points
    layer with `features=` set to it (see
    `widgets/experiment_list.py::_on_calibrate_finished`), so every
    calibration spot's fit quality is inspectable, not just the
    aggregate `sigma_estimate`.

    `cancel_event`, if given, is only checked before this stage starts --
    `calibrate_sigma_df` is one opaque Rust call with no interruption point
    of its own, so a cancellation requested mid-fit still runs to completion
    (see `PipelineCancelled`'s docstring).
    """
    _check_cancelled(cancel_event)
    kwargs = dict(calibration_kwargs) if calibration_kwargs is not None else dict(DEFAULT_CALIBRATION_KWARGS)
    calib_points_df, calib_summary = calibrate_sigma_df(session.image[frame_index], sigma_init=sigma_init, **kwargs)
    session.sigma = calib_summary["sigma_estimate"]
    session.calib_summary = calib_summary
    session.calib_points_df = calib_points_df
    session.calibration_kwargs_used = kwargs
    session.calibration_frame_used = frame_index
    return session


def _resolve_frame_range(frame_range: Optional[tuple[int, int]], t: int) -> tuple[int, int]:
    """`frame_range` as a concrete `(start, end)` pair against a stack of
    length `t`. `None`, or an `end <= 0` (the params-panel frame-range
    spinbox's "last" sentinel value, `QSpinBox.setSpecialValueText`), both
    mean "through the real last frame" -- resolved here, against the
    actual stack length, rather than by whichever caller built the tuple
    guessing at it (the params panel doesn't always know `t` yet when the
    user sets `start` with `end` left at its default)."""
    if frame_range is None:
        return (0, t)
    start, end = frame_range
    return (start, end if end > 0 else t)


def run_detect_step(
    session: PipelineSession,
    sigma: Optional[float] = None,
    solver_kwargs: Optional[dict] = None,
    frame_range: Optional[tuple[int, int]] = None,
    mask: Optional[np.ndarray] = None,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> PipelineSession:
    """`find_spots` over `session.image[start:end]` (default: every frame).

    `sigma`, if given, is used as-is (a literal PSF sigma, skipping
    calibration entirely). If omitted, falls back to `session.sigma` from
    a prior `run_calibration_step` call -- raises if neither is available.

    `frame_range`, if given, is a `(start, end)` pair (Python-slice
    semantics: `end` exclusive) restricting which frames are processed --
    lets a single image be explored incrementally (a quick look at a few
    frames) rather than always committing to the whole stack. `end <= 0`
    (or the whole tuple `None`) means through the real last frame --
    see `_resolve_frame_range`. The resulting `points_df`'s `frame` column
    still holds true stack indices (`start`..`end-1`), not a re-based 0..n
    range.

    `mask`, if given, is a full-frame `(H, W)` boolean array restricting
    where `find_spots` may place new spikes (see
    `sfwloc_py.find_spots`'s docstring) -- typically built from a napari
    Shapes layer (`widgets/experiment_list.py`'s ROI handling). Only
    `find_spots` (the single-frame binding) accepts a mask, not the
    batched `find_spots_stack`, so a mask forces the frame-by-frame path
    below regardless of whether `progress_callback` is given.

    Frame-by-frame (reporting progress each frame, if `progress_callback`
    is given) is otherwise only used when explicitly requested via
    `progress_callback` -- the same tradeoff `track_beads_timelapse.py`
    makes for an interactively-watched run vs. the faster rayon-parallel
    `find_spots_stack_df`.

    `cancel_event`, if given, is checked before this stage starts and --
    only on the frame-by-frame path -- again before each frame, so a
    cancellation lands within one frame rather than only between stages
    (see `PipelineCancelled`'s docstring). The batched `find_spots_stack_df`
    path has no per-frame checkpoint of its own, so on that path a
    cancellation still only takes effect before this stage starts.
    """
    _check_cancelled(cancel_event)
    if sigma is None:
        if session.sigma is None:
            raise ValueError(
                "No sigma available -- run calibration first, or pass sigma explicitly."
            )
        sigma = session.sigma

    kwargs = dict(solver_kwargs) if solver_kwargs is not None else dict(DEFAULT_SOLVER_KWARGS)
    start, end = _resolve_frame_range(frame_range, session.image.shape[0])
    if end <= start:
        raise ValueError(f"empty frame_range: start={start} >= end={end}")

    if mask is not None or progress_callback is not None:
        frame_indices = range(start, end)
        n = len(frame_indices)
        frames = []
        for done, i in enumerate(frame_indices, start=1):
            _check_cancelled(cancel_event)
            frames.append(
                find_spots_df(session.image[i], sigma, session.bg[i], frame_idx=i, mask=mask, **kwargs)
            )
            if progress_callback is not None:
                progress_callback(done, n, "finding spots")
        points_df = pl.concat(frames)
    else:
        points_df = find_spots_stack_df(session.image[start:end], sigma, session.bg[start:end], **kwargs)
        if start != 0:
            points_df = points_df.with_columns((pl.col("frame") + start).alias("frame"))

    session.sigma = sigma
    session.points_df = points_df
    session.solver_kwargs_used = kwargs
    session.frame_range_used = (start, end)
    return session


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


def run_track_step(
    session: PipelineSession,
    bootstrap_gate_px: float,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> PipelineSession:
    """Bootstrap link (fixed `bootstrap_gate_px` gate) -> estimate D from
    single-step MSD -> final link (auto-derived gate). Requires
    `session.points_df` from a prior `run_detect_step` call.

    `progress_callback`, if given, is called twice: `(0, 2, "bootstrap
    linking")` before the bootstrap pass, `(1, 2, "final linking")` before
    the final pass -- these stages don't have finer-grained progress of
    their own.

    `cancel_event`, if given, is checked before each of those two passes --
    `link_tracks_df` is one opaque Rust call with no interruption point of
    its own, so this stage can only be skipped before it starts, not
    interrupted mid-call (see `PipelineCancelled`'s docstring).
    """
    if session.points_df is None:
        raise ValueError("No points available -- run detect first.")
    _check_cancelled(cancel_event)

    def report(done: int, total: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(done, total, stage)

    points_df = session.points_df
    h, w = session.image.shape[1:]

    report(0, 2, "bootstrap linking")
    bootstrap = link_tracks_df(points_df, bootstrap_gate_px)
    sigma_y_um = points_df["sigma_y"].to_numpy() * session.pixel_size_um
    sigma_x_um = points_df["sigma_x"].to_numpy() * session.pixel_size_um
    sigma_loc_um = float(np.median(np.sqrt((sigma_y_um**2 + sigma_x_um**2) / 2.0)))
    D_est, n_links = estimate_D_um2_s(bootstrap, session.dt_s, session.pixel_size_um, sigma_loc_um)

    active_area_um2 = (h * session.pixel_size_um) * (w * session.pixel_size_um)
    mean_n_per_frame = points_df.group_by("frame").len()["len"].mean() if points_df.height else 0.0
    density_um2 = (mean_n_per_frame / active_area_um2) if mean_n_per_frame else 0.0
    resolvability = check_resolvability(D_est, session.dt_s, density_um2)

    _check_cancelled(cancel_event)
    report(1, 2, "final linking")
    final_gate_px = recommended_gate_px(D_est, session.dt_s, session.pixel_size_um, sigma_loc_um=sigma_loc_um)
    tracks_df = link_tracks_df(points_df, final_gate_px)
    report(2, 2, "done")

    session.tracks_df = tracks_df
    session.bootstrap_gate_px_used = bootstrap_gate_px
    session.track_summary = {
        "sigma_loc_um": sigma_loc_um,
        "D_est_um2_s": D_est,
        "n_bootstrap_links": n_links,
        "final_gate_px": final_gate_px,
        "density_um2": density_um2,
        "resolvability_message": resolvability["message"],
    }
    return session


def session_manifest_extra(session: PipelineSession) -> dict:
    """Assemble the `manifest.json` `params` payload from a session that's
    been through all three stages -- meant to be passed as
    `experiment.build_manifest`'s `params`."""
    ts = session.track_summary or {}
    return {
        "pixel_size_um": session.pixel_size_um,
        "dt_s": session.dt_s,
        "channel": session.channel,
        "z_index": session.z_index,
        "sigma_px": session.sigma,
        "sigma_loc_um": ts.get("sigma_loc_um"),
        "D_est_um2_s": ts.get("D_est_um2_s"),
        "n_bootstrap_links": ts.get("n_bootstrap_links"),
        "bootstrap_gate_px": session.bootstrap_gate_px_used,
        "final_gate_px": ts.get("final_gate_px"),
        "density_um2": ts.get("density_um2"),
        "resolvability_message": ts.get("resolvability_message"),
        "n_points": session.points_df.height if session.points_df is not None else 0,
        "n_tracks": session.tracks_df["track_id"].n_unique() if session.tracks_df is not None and session.tracks_df.height else 0,
        "solver_kwargs": session.solver_kwargs_used,
        "calibration_kwargs": session.calibration_kwargs_used,
        "calibration_frame": session.calibration_frame_used,
        "frame_range": list(session.frame_range_used) if session.frame_range_used is not None else None,
    }


def run_detect_track(
    image_path: str | Path,
    pixel_size_um: Optional[float] = None,
    dt_s: Optional[float] = None,
    channel: int = 0,
    z_index: int = 0,
    params: Optional[DetectTrackParams] = None,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Run the full calibrate+detect+track pipeline on one timelapse in
    one call, composing `load_session`/`run_calibration_step`/
    `run_detect_step`/`run_track_step` -- for the headless CLI and the
    widget's batch/multi-select Run action. See this module's docstring
    for the stepwise alternative.

    `image_path` can be .tif/.tiff, .nd2, or .ims (see `io_formats.load_stack`).
    `channel`/`z_index` pick which plane to track for files with more than
    one (both default to 0). `pixel_size_um`/`dt_s` fall back to the
    file's own metadata if not given explicitly.

    `cancel_event`, if given, is forwarded to each stage -- see
    `PipelineCancelled`'s docstring for what "cancelled" actually means per
    stage (cooperative, frame-granular for detect, stage-boundary-only for
    calibration/track).

    Returns (points_df, tracks_df, manifest_extra) -- `manifest_extra` is
    meant to be passed as `experiment.build_manifest`'s `params`.
    """
    params = params or DetectTrackParams()
    session = load_session(image_path, pixel_size_um=pixel_size_um, dt_s=dt_s, channel=channel, z_index=z_index)
    start, end = _resolve_frame_range(params.frame_range, session.image.shape[0])
    n_frames = end - start
    total_steps = n_frames + 3

    def report(done: int, total: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(done, total, stage)

    report(0, total_steps, "calibrating sigma")
    run_calibration_step(
        session,
        params.sigma_init,
        params.calibration_kwargs,
        frame_index=params.calibration_frame_index,
        cancel_event=cancel_event,
    )

    detect_progress = (
        (lambda done, total, stage: report(done, total_steps, stage))
        if progress_callback is not None
        else None
    )
    run_detect_step(
        session,
        solver_kwargs=params.solver_kwargs,
        frame_range=params.frame_range,
        progress_callback=detect_progress,
        cancel_event=cancel_event,
    )

    track_progress = lambda done, total, stage: report(n_frames + 1 + done, total_steps, stage)
    run_track_step(session, params.bootstrap_gate_px, progress_callback=track_progress, cancel_event=cancel_event)

    return session.points_df, session.tracks_df, session_manifest_extra(session)
