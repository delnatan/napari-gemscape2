"""The shared detect+track pipeline, as three composable stages plus one
convenience wrapper that runs all of them.

Both the headless CLI (`cli.py`) and the interactive napari widget
(`widgets/experiment_list.py`) build on this module -- the logic lives
here exactly once, unlike napari-gemscape where the interactive handlers
and its batch subprocess script each reimplemented the pipeline.

Detection and linking are `spotsolve`'s: `spotsolve.localize`/
`localize_stack` (each frame fitted as one joint Poisson model, counts
decided by likelihood ratios at a threshold set by `fp_per_mpx`, the
expected false emitters per 10^6 pixels of pure noise -- so "how many are
there" and "where are they" are answered as one problem) and
`spotsolve.link` (Crocker-Grier: frame to frame, least summed squared
displacement within a required `max_step`).
`spotsolve.loctable` defines the table that passes between them, and this
module emits it verbatim -- `se_y`/`se_x` (per-detection CRLB), `flux`,
`flags` (`spotsolve.FitFlag`) and the rest -- so nothing downstream has to
learn a second schema.

Two things the previous sfwloc-based pipeline did are gone because
`spotsolve` makes them unnecessary rather than because they were dropped:

  - There is one *default* detector, not a per-acquisition choice of
    three. `spotsolve.localize`/`localize_stack` (multi-emitter, the
    default `DetectTrackParams.detector`) fits the frame jointly and
    decides how many emitters it holds by likelihood ratio -- the same
    rule handles a crowded field and a sparse one, so there is no
    dense/sparse/DAOPHOT algorithm to hand-pick per file. `spotsolve`
    also ships `localize_aguet`/`localize_aguet_stack`
    (`detector="aguet"`): independent single-emitter fits behind a LoG
    screen, with no multi-emitter search -- the spotfitlm-compatible
    baseline for genuinely sparse fields, kept on as an option rather
    than the default because the joint fit is the more general rule when
    in doubt.
  - Linking takes one number, `max_step` (px): the largest step a
    particle may take between consecutive frames. It is a setting, not a
    measurement -- spotsolve found that estimating it from the movie's
    own links creeps upward as each wider radius admits wrong links --
    so the old bootstrap-link -> estimate-D -> relink dance is gone, and
    nothing is fitted before linking.

`PipelineSession` + `load_session`/`run_detect_step`/`run_track_step` let
each stage run independently and be re-run after tweaking that stage's own
knobs, without redoing earlier stages -- this is what backs the dock
widget's per-tab "Run detect" / "Run tracking" buttons (`run_track_step`
needs `session.points_df` from a prior detect). `run_detect_track`
composes them in one call for the headless CLI, where stepwise control
isn't needed.

There is no sigma calibration step. The PSF width is a setting: run
detect over a few frames, read the `fit_sigma` histogram, adjust, run
again -- the distribution is on screen the whole time, so a bimodal or
ragged width (two focal planes, junk fitted as signal) is seen rather than
averaged into one number. The headless runner takes `sigma` explicitly.

spotsolve reports every fit it makes; nothing is rejected at detection.
What linking sees is decided in `run_track_step`, in three layers:
`loctable.filter_quality` (finite coordinates and positive SEs -- always,
since the linker and diffusionkit both need them), `exclude_flags` (fits
carrying chosen `FitFlag`s), and `point_filters`. `point_filters`/
`track_filters` are `{column: (lo, hi)}` cuts (see `filter_mask`) on
per-detection columns and per-track metrics. None of these touch
`points_df`, which keeps every detection so the rejected population stays
auditable, and all are recorded in `session_manifest_extra`.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import polars as pl
import spotsolve
from spotsolve import loctable

from napari_gemscape2 import regions as region_tools
from napari_gemscape2.regions import Regions
from napari_gemscape2.io_formats import StackMetadata, load_stack
from napari_gemscape2.tracking_diagnostics import check_resolvability

# The camera's own calibration, forwarded to every `spotsolve.localize`/
# `localize_stack` call. `offset` (ADU) is subtracted
# before fitting; everything else about the camera -- gain, read noise --
# is measured from each frame's own noise (`spotsolve` no longer takes
# either as an input), so `offset` is the only knob left here.
DEFAULT_CAMERA_KWARGS = dict(
    offset=0.0,
)

# Forwarded to `spotsolve.localize`/`localize_stack` as **kwargs, mirroring
# that function's own defaults (`spotsolve.native`).
#
# `fp_per_mpx` is the detector's one threshold, stated as what it costs:
# the expected number of false emitters per 10^6 pixels of pure noise
# (calibrated to within 3% on simulated noise for sigma 1.0-1.45). Lower
# it for fewer false positives, raise it for dim data.
#
# `slack` is the width range a fit is allowed to take, as multiples of
# `sigma`. Every fit is reported; one that ends on a width bound carries
# `FitFlag.AT_BOUND` instead of being dropped.
DEFAULT_DETECT_KWARGS = dict(
    fp_per_mpx=spotsolve.FP_PER_MPX,
    slack=spotsolve.SLACK,
)

# Keys a manifest written against spotsolve 0.1 may carry in its
# `detect_kwargs` that no current detector accepts: the box search's
# emitter cap and LoG cut, its count rule, and an older reporting band.
# Dropped on the way back in, so an old bundle still works as a template.
_OBSOLETE_DETECT_KWARGS = frozenset({"k_max", "threshold", "selection", "count_penalty", "band"})

# Forwarded to `spotsolve.localize_aguet`/`localize_aguet_stack` as **kwargs
# when `DetectTrackParams.detector == "aguet"`, mirroring their own defaults
# (`spotsolve.aguet`). None of `DEFAULT_DETECT_KWARGS`' keys apply here --
# Aguet fits one emitter per LoG-screened candidate independently rather
# than fitting the frame jointly, so there is no `slack` (its width is
# fitted free) and no `fp_per_mpx` (the screening cut is `significance`, a
# per-pixel level rather than a frame-wide rate). `boxsize` is the
# odd fit-crop size around each candidate and `itermax` its optimizer's
# iteration budget -- both rarely need changing.
DEFAULT_SPARSE_KWARGS = dict(
    significance=0.05,
    boxsize=9,
    itermax=50,
)

# ProgressCallback(done, total, stage) -- called from whatever thread the
# stage function executes on; the interactive widget wraps this in a
# QObject signal to cross back onto the Qt event-loop thread safely.
ProgressCallback = Callable[[int, int, str], None]


class PipelineCancelled(Exception):
    """Raised at a `cancel_event` checkpoint (see `run_detect_step`/
    `run_track_step`'s `cancel_event` argument).
    Cooperative cancellation only -- takes effect at the next chunk
    (detect, when a `progress_callback` puts it on the chunked
    `localize_stack` path) or stage boundary, not instantly, since
    `localize_stack` and `link` are each one opaque
    Rust call with no interruption point of their own. Propagates like any
    other exception through `napari.qt.threading`'s `errored` signal; the
    widget is responsible for telling this apart from a real error."""


def _check_cancelled(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise PipelineCancelled("cancelled")


# A filter spec: {column name -> (lo, hi)}, inclusive on both ends, ANDed
# across columns, either side None for unbounded (`qtkit.FilterSpec`; JSON
# writes it as null, and TOML, which has no null, takes a number past the
# data instead). This is what the Detect/Track tabs' histogram filters
# serialize to, what `manifest.json` records, and what a headless config
# can set directly -- one shape for all three, so a range dragged in the
# UI and a range typed into a TOML mean exactly the same thing. See
# `filter_mask`.
FilterSpec = dict[str, tuple[Optional[float], Optional[float]]]


def filter_mask(df: pl.DataFrame, filters: Optional[FilterSpec]) -> pl.Series:
    """Per-row boolean: does this row pass every filter in `filters`?
    Inclusive on both ends, ANDed across columns, and a column the table
    doesn't have is skipped rather than failing the whole run -- a filter
    spec outlives the table it was dragged on (it rides in `manifest.json`
    and can be re-applied to a re-detected movie whose columns differ).

    The one place the filter rule is written. The napari widgets call it
    to preview what a range does, `run_track_step` calls it to decide what
    linking sees, and `apply_track_filters` calls it on per-track metrics
    -- the same rule in all three, so what the histogram showed is what
    the saved bundle got."""
    if df.height == 0:
        return pl.Series("pass", [], dtype=pl.Boolean)
    expr = None
    for col, (lo, hi) in (filters or {}).items():
        if col not in df.columns:
            continue
        # A None side is unbounded -- the same rule as qtkit.filter_mask,
        # which the filter panels preview with.
        for cond in (
            (pl.col(col) >= lo) if lo is not None else None,
            (pl.col(col) <= hi) if hi is not None else None,
        ):
            if cond is not None:
                expr = cond if expr is None else expr & cond
    if expr is None:
        return pl.Series("pass", np.ones(df.height, dtype=bool))
    return df.select(expr.fill_null(False).alias("pass"))["pass"]


def apply_filters(df: pl.DataFrame, filters: Optional[FilterSpec]) -> pl.DataFrame:
    """`df` keeping only the rows `filter_mask` passes."""
    if not filters or df.height == 0:
        return df
    return df.filter(filter_mask(df, filters))


@dataclass
class DetectTrackParams:
    # The in-focus PSF width (px) the search runs at -- required, and
    # settled by eye: detect a few frames, read the `fit_sigma` histogram
    # (see this module's docstring).
    sigma: float
    # `spotsolve.link`'s `max_step`, px: the largest step linked between
    # consecutive frames. Required for the same reason `sigma` is -- it is
    # not estimated from the movie (see this module's docstring).
    max_step: float
    min_track_length: int = 2
    # `spotsolve.FitFlag` bits whose fits linking does not see (a bitmask;
    # 0 keeps every fit). Off by default, following spotsolve: a flag is a
    # diagnostic, not a verdict that the detection is wrong.
    exclude_flags: int = 0
    # Which spotsolve detector `run_detect_step` runs: "multi_emitter"
    # (default -- `spotsolve.localize`/`localize_stack`, one joint model
    # per frame) or "aguet" (`localize_aguet`/`localize_aguet_stack`, the
    # independent-fit sparse baseline).
    detector: str = "multi_emitter"
    camera_kwargs: dict = field(default_factory=lambda: dict(DEFAULT_CAMERA_KWARGS))
    # None picks `DEFAULT_DETECT_KWARGS` or `DEFAULT_SPARSE_KWARGS` to match
    # `detector` (see `run_detect_step`) -- left as None rather than always
    # defaulting to the multi-emitter dict, which would silently hand
    # `localize_aguet_stack` keyword arguments (`fp_per_mpx`, `slack`) it
    # doesn't accept.
    detect_kwargs: Optional[dict] = None
    # Worker threads the chosen `detector`'s stack function hands frames to,
    # in `run_detect_step`. None means every core (os.cpu_count()) --
    # spotsolve's own default for either detector.
    n_threads: Optional[int] = None
    # (start, end) frame slice, Python-slice semantics; None, or end <= 0,
    # means through the real last frame (see _resolve_frame_range).
    # Not `mask` -- that's an interactive-only concept (built from a live
    # napari Labels layer), not something a headless/serialized
    # DetectTrackParams can carry. See run_detect_step's docstring.
    frame_range: Optional[tuple[int, int]] = None
    # QC cuts on per-detection columns (`flux`, `fit_sigma`, `se_pos`, ...)
    # and on per-track metrics (`track_length`, `mean_step_um`,
    # `duration_s`). Like `exclude_flags`, these are applied to what linking
    # sees and to the final track set -- never to `points_df` itself, which
    # keeps every detection so the rejected population stays auditable.
    point_filters: FilterSpec = field(default_factory=dict)
    track_filters: FilterSpec = field(default_factory=dict)


@dataclass
class PipelineSession:
    """Mutable state threaded through the Detect -> Track
    stages for one image. `*_used` fields record what each completed
    stage actually ran with, so `session_manifest_extra` reflects reality
    even if a UI knob was changed after that stage last ran."""

    image_path: Path
    image: np.ndarray  # (T, H, W)
    pixel_size_um: float
    dt_s: float
    channel: int = 0
    z_index: int = 0
    # What the file itself said (`io_formats.StackMetadata`), kept beside
    # the two numbers actually in force: `pixel_size_um`/`dt_s` above may
    # be a caller's explicit override, and the difference between "read
    # from the file" and "supplied by hand" is exactly what the UI shows
    # and `session_manifest_extra` records.
    metadata: Optional[StackMetadata] = None
    # Camera exposure, carried for the diffusion analysis's motion-blur
    # model only -- no stage here reads it. None means "not known", which
    # is different from 0 ("instantaneous"): see `io_formats`.
    exposure_s: Optional[float] = None

    # The sigma the last detect ran at.
    sigma: Optional[float] = None

    points_df: Optional[pl.DataFrame] = None
    # One row per frame (`loctable.FRAME_SCHEMA`): detection and flagged
    # counts, median flux and precision, background and the measured
    # dispersion. Kept on the session rather than folded into `points_df`
    # because it's a fact about the frame, not about any one detection --
    # and it's what makes "why did this frame find nothing" answerable
    # after the fact.
    frames_df: Optional[pl.DataFrame] = None
    camera_kwargs_used: Optional[dict] = None
    detect_kwargs_used: Optional[dict] = None
    detector_used: Optional[str] = None
    frame_range_used: Optional[tuple[int, int]] = None
    # The painted regions image (`(H, W)` uint16, 0 = background) whose
    # `labels > 0` was run_detect_step's `mask`, and the table naming each
    # label (see `napari_gemscape2.regions`) -- set by the widget (not
    # pipeline.py itself, which stays napari-agnostic), carried through so
    # write_result can persist them alongside points/tracks.
    labels: Optional[np.ndarray] = None
    regions: Optional[Regions] = None
    # The saved manifest's `params`, when this session was rebuilt from a
    # bundle (`session_from_bundle`) rather than run here -- what
    # `session_manifest_extra` falls back on for anything the session
    # never recomputed (the detect settings, say), so a re-save of a
    # reopened bundle doesn't erase its own provenance.
    source_params: Optional[dict] = None

    tracks_df: Optional[pl.DataFrame] = None
    track_summary: Optional[dict] = None
    max_step_used: Optional[float] = None
    exclude_flags_used: Optional[int] = None
    min_track_length_used: Optional[int] = None
    point_filters_used: Optional[FilterSpec] = None
    track_filters_used: Optional[FilterSpec] = None


def load_session(
    image_path: str | Path,
    pixel_size_um: Optional[float] = None,
    dt_s: Optional[float] = None,
    channel: int = 0,
    z_index: int = 0,
    stack: Optional[tuple[np.ndarray, StackMetadata]] = None,
    exposure_s: Optional[float] = None,
) -> PipelineSession:
    """Load a timelapse and start a fresh (un-detected, un-tracked) `PipelineSession`. `stack` is an already-read
    `load_stack(image_path, channel, z_index)` result, so a caller that
    has the image in memory (the UI, which loaded it to display it) does
    not read it a second time.

    Raises if the file records no pixel size / frame interval and none was
    passed: every physical column downstream is those two numbers
    multiplied through, so there is no safe default to fall back on. The
    message names which one is missing and what the file did say about
    it (`StackMetadata.detail`), since "this .tif has no calibration" is
    a fixable problem and "pixel_size_um/dt_s not found" was not.

    `exposure_s` falls back to the file's the same way, but is never
    required: a session with no known exposure can still be detected and
    linked, and carries None for the diffusion analysis to ask about."""
    if stack is None:
        stack = load_stack(image_path, channel=channel, z_index=z_index)
    im, metadata = stack
    pixel_size_um = pixel_size_um if pixel_size_um is not None else metadata.pixel_size_um
    dt_s = dt_s if dt_s is not None else metadata.dt_s
    if pixel_size_um is None or dt_s is None:
        missing = " and ".join(
            name
            for name, value in (("pixel size", pixel_size_um), ("frame interval", dt_s))
            if value is None
        )
        raise ValueError(
            f"{Path(image_path).name}: no {missing} available — not in the file's own "
            f"metadata, and not given explicitly.\n{metadata.detail()}"
        )
    return PipelineSession(
        image_path=Path(image_path),
        image=im,
        pixel_size_um=pixel_size_um,
        dt_s=dt_s,
        channel=channel,
        z_index=z_index,
        metadata=metadata,
        exposure_s=exposure_s if exposure_s is not None else metadata.exposure_s,
    )


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
    camera_kwargs: Optional[dict] = None,
    detect_kwargs: Optional[dict] = None,
    frame_range: Optional[tuple[int, int]] = None,
    mask: Optional[np.ndarray] = None,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
    n_threads: Optional[int] = None,
    detector: str = "multi_emitter",
) -> PipelineSession:
    """Localize every spot over `session.image[start:end]` (default: every
    frame) with `spotsolve`, and assemble the result into the standard
    localization table.

    `detector` picks which spotsolve function does the work: the default
    "multi_emitter" (`spotsolve.localize_stack`, one joint model per
    frame) or "aguet" (`spotsolve.localize_aguet_stack`,
    independent single-emitter fits behind a LoG screen -- the sparse
    baseline). `detect_kwargs` must match whichever is chosen (`None` picks
    `DEFAULT_DETECT_KWARGS`/`DEFAULT_SPARSE_KWARGS` accordingly) -- the two
    detectors take disjoint keyword arguments, so a dict built for one
    raises a `TypeError` if forwarded to the other.

    Sets `session.points_df` (one row per detection,
    `loctable.LOCALIZATION_SCHEMA`) and `session.frames_df` (one row per
    frame, `loctable.FRAME_SCHEMA`). `loc_id` is unique over the whole
    run, and `frame` holds true stack indices (`start`..`end-1`), not a
    re-based 0..n range, on either path below.

    `sigma` is the in-focus PSF width. If omitted, falls back to
    `session.sigma` (the last run's) -- raises if neither is available.
    Note this is the width the search runs AT; each emitter still gets its
    own fitted width (`fit_sigma`); for `detector="multi_emitter"`,
    `detect_kwargs`' `slack` bounds how far it may stray, and a fit ending
    on that bound is flagged `AT_BOUND` rather than dropped.

    `frame_range`, if given, is a `(start, end)` pair (Python-slice
    semantics: `end` exclusive) restricting which frames are processed --
    lets a single image be explored incrementally (a quick look at a few
    frames) rather than always committing to the whole stack. `end <= 0`
    (or the whole tuple `None`) means through the real last frame -- see
    `_resolve_frame_range`.

    `mask`, if given, is a full-frame `(H, W)` boolean array restricting
    where emitters may be placed, forwarded as `spotsolve`'s `roi` --
    typically `labels > 0` of a napari Labels layer (`widgets/
    experiment_list.py`'s regions handling).

    Every fit spotsolve makes is kept, with its `FitFlag` diagnostics in
    `flags`; which ones linking sees is `run_track_step`'s decision, so
    "how much of this movie was junk" stays an auditable fact about the
    run rather than a silent deletion.

    Both paths run every frame through whichever rayon-parallel stack
    function `detector` selects -- `n_threads` (default: every core,
    `os.cpu_count()`) is how many native threads it hands frames to. They
    differ only in chunk size: with no `progress_callback`, the whole range
    goes through in one call; with one, the range is split into chunks of
    `n_threads` frames each -- small enough that progress and cancellation
    still land often, large enough that every thread stays busy within a
    chunk.

    `cancel_event`, if given, is checked before this stage starts and --
    on the `progress_callback` path only -- again before each chunk, so a
    cancellation lands within one chunk rather than only between stages
    (see `PipelineCancelled`'s docstring).
    """
    _check_cancelled(cancel_event)
    if sigma is None:
        if session.sigma is None:
            raise ValueError(
                "No sigma available -- pass sigma explicitly."
            )
        sigma = session.sigma

    camera = dict(camera_kwargs) if camera_kwargs is not None else dict(DEFAULT_CAMERA_KWARGS)
    localize_stack_fn = spotsolve.localize_aguet_stack if detector == "aguet" else spotsolve.localize_stack
    if detect_kwargs is not None:
        detect = dict(detect_kwargs)
    else:
        detect = dict(DEFAULT_SPARSE_KWARGS if detector == "aguet" else DEFAULT_DETECT_KWARGS)

    start, end = _resolve_frame_range(frame_range, session.image.shape[0])
    if end <= start:
        raise ValueError(f"empty frame_range: start={start} >= end={end}")

    n = end - start
    threads = int(n_threads or os.cpu_count() or 1)

    if progress_callback is not None:
        results = []
        i = start
        while i < end:
            _check_cancelled(cancel_event)
            j = min(i + threads, end)
            results.extend(
                localize_stack_fn(
                    session.image[i:j], sigma, roi=mask, images=False,
                    n_threads=threads, **camera, **detect
                )
            )
            progress_callback(len(results), n, "finding spots")
            i = j
    else:
        results = localize_stack_fn(
            session.image[start:end], sigma, roi=mask, images=False,
            n_threads=threads, **camera, **detect
        )

    # `frame_tables` is per-frame; stitch them with a running `loc_id0` so
    # `loc_id` is unique across the movie (its documented contract), and
    # with the true stack index as `frame`.
    locs_parts, frame_parts = [], []
    loc_id0 = 0
    for offset, result in enumerate(results):
        i = start + offset
        locs, frame_row = loctable.frame_tables(
            result,
            frame=i,
            t=i * session.dt_s,
            pixel_size=session.pixel_size_um,
            loc_id0=loc_id0,
        )
        loc_id0 += locs.height
        locs_parts.append(locs)
        frame_parts.append(frame_row)

    session.sigma = sigma
    session.points_df = loctable.concat(locs_parts)
    session.frames_df = loctable.concat(frame_parts)
    session.camera_kwargs_used = camera
    session.detect_kwargs_used = detect
    session.detector_used = detector
    session.frame_range_used = (start, end)
    # A fresh detection table supersedes whatever bundle this session was
    # rebuilt from, so none of that bundle's recorded numbers still apply.
    session.source_params = None
    return session


def estimate_D_um2_s(linked_df: pl.DataFrame, dt_s: float, pixel_size_um: float, sigma_loc_um: float):
    """Single-step MSD estimate of D, corrected for localization noise:
    mean(r^2) = 4*D*dt + 4*sigma_loc_um^2.

    A direct moment of the finished trajectories in physical units: the
    one number the Track status line and the crowding check read D off.
    It inherits `max_step`'s truncation (no longer step was linked), so a
    `max_step` set too tight shows up here as a D biased low."""
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


# The per-track metrics `track_metrics_df` computes, and so the columns a
# track filter can name. Kept as a constant because the Track tab's filter
# panel has to offer them before any track has been linked -- before
# there's a table to read column names off.
TRACK_METRIC_COLUMNS = ("track_length", "duration_s", "mean_step_um")


def track_metrics_df(tracks_df: pl.DataFrame, pixel_size_um: float, dt_s: float) -> pl.DataFrame:
    """One row per `track_id`, carrying `TRACK_METRIC_COLUMNS`:
    `track_length` (points survived by linking), `duration_s` (span between
    its first and last frame), `mean_step_um` (mean single-step
    displacement -- the same quantity `estimate_D_um2_s` pools across every
    track, kept per-track here instead).

    The per-track shape is what a filter needs (a track either passes or
    doesn't -- one verdict, not one per vertex); `track_features_df`
    broadcasts the same numbers back onto every vertex for napari's
    Tracks-layer `properties=`.

    A track with a single point has no step to average, so its
    `mean_step_um` is 0.0 rather than null (`color_by` on a napari Tracks
    layer requires a fully-populated numeric column)."""
    if tracks_df.height == 0 or "track_id" not in tracks_df.columns:
        return pl.DataFrame(
            schema={
                "track_id": pl.UInt32,
                "track_length": pl.UInt32,
                "mean_step_um": pl.Float64,
                "duration_s": pl.Float64,
            }
        )
    df = tracks_df.sort(["track_id", "frame"]).with_columns(
        pl.col("frame").diff().over("track_id").alias("_dframe"),
        (pl.col("y").diff().over("track_id") * pixel_size_um).alias("_dy_um"),
        (pl.col("x").diff().over("track_id") * pixel_size_um).alias("_dx_um"),
    )
    df = df.with_columns(
        pl.when(pl.col("_dframe") == 1)
        .then((pl.col("_dy_um") ** 2 + pl.col("_dx_um") ** 2).sqrt())
        .alias("_step_um")
    )
    agg = df.group_by("track_id").agg(
        pl.len().alias("track_length"),
        (pl.col("frame").max() - pl.col("frame").min()).alias("_span_frames"),
        pl.col("_step_um").mean().fill_null(0.0).alias("mean_step_um"),
    )
    return agg.with_columns((pl.col("_span_frames") * dt_s).alias("duration_s")).drop("_span_frames")


def track_features_df(tracks_df: pl.DataFrame, pixel_size_um: float, dt_s: float) -> pl.DataFrame:
    """`tracks_df` with `track_metrics_df`'s per-track features broadcast
    onto every one of that track's own rows. Presentation-layer only
    (napari's Tracks layer `properties=`, for `color_by`/hover) -- not used
    by the pipeline itself."""
    if tracks_df.height == 0 or "track_id" not in tracks_df.columns:
        return tracks_df.with_columns(
            pl.lit(0, dtype=pl.UInt32).alias("track_length"),
            pl.lit(0.0).alias("duration_s"),
            pl.lit(0.0).alias("mean_step_um"),
        )
    metrics = track_metrics_df(tracks_df, pixel_size_um, dt_s)
    return tracks_df.sort(["track_id", "frame"]).join(metrics, on="track_id", how="left")


def apply_track_filters(
    tracks_df: pl.DataFrame,
    filters: Optional[FilterSpec],
    pixel_size_um: float,
    dt_s: float,
) -> pl.DataFrame:
    """`tracks_df` keeping only whole tracks whose `track_metrics_df` row
    passes `filters` -- a track is kept or dropped entire, never trimmed
    to its passing vertices, since half a trajectory is not a shorter
    trajectory.

    Columns are matched against the metrics table, so a filter may also
    name a column already on `tracks_df` itself (a per-track fit result
    joined in, e.g. diffusionkit's `D` or `alpha`) -- those ride along
    through the `first()` below."""
    if not filters or tracks_df.height == 0 or "track_id" not in tracks_df.columns:
        return tracks_df
    metrics = track_metrics_df(tracks_df, pixel_size_um, dt_s)
    extra = [c for c in filters if c in tracks_df.columns and c not in metrics.columns]
    if extra:
        metrics = metrics.join(
            tracks_df.group_by("track_id").agg(pl.col(c).first() for c in extra),
            on="track_id",
            how="left",
        )
    keep = apply_filters(metrics, filters)["track_id"]
    return tracks_df.filter(pl.col("track_id").is_in(keep))


def run_track_step(
    session: PipelineSession,
    max_step: float,
    min_track_length: int = 2,
    exclude_flags: int = 0,
    point_filters: Optional[FilterSpec] = None,
    track_filters: Optional[FilterSpec] = None,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> PipelineSession:
    """Link `session.points_df` into trajectories with `spotsolve.link`,
    then drop tracks shorter than `min_track_length`. Requires
    `session.points_df` from a prior `run_detect_step` call.

    `max_step` (px) is the linker's one setting: between consecutive
    frames it minimizes the summed squared displacement, ending a track
    costs `max_step`^2, and no longer step is linked. About three times the
    rms step of the fastest particles of interest is spotsolve's advice;
    `track_summary`'s `rms_step_px` reports what the linked steps came to,
    to check the setting against.

    Frame-to-frame only: a missed detection ENDS a track rather than being
    bridged, so trajectories fragment instead of swapping identity. That
    is why `min_track_length` matters more here than a gap-closing linker
    would need it to.

    What the linker sees is `session.points_df` narrowed three times, and
    `session.points_df` itself is left whole each time -- the cuts apply to
    what the linker sees, not to what was saved:

      1. `loctable.filter_quality`: finite coordinates and positive
         `se_y`/`se_x`. Always applied -- the linker itself reads only
         positions, but diffusionkit needs each point's position error.
      2. `exclude_flags`: fits carrying any of these `spotsolve.FitFlag`
         bits (default 0, none excluded).
      3. `point_filters`: the Detect tab's histogram cuts
         (`{column: (lo, hi)}` -- see `filter_mask`).

    `track_filters` applies to the linked result, keeping whole tracks by
    their `track_metrics_df` row (see `apply_track_filters`). Both filter
    specs are recorded in `session_manifest_extra`, so what the run kept is
    readable off the bundle rather than only off whoever dragged the handle.

    When the table is labeled with more than one region, each region is
    linked on its own, which is what guarantees no track crosses a region
    boundary (nucleus -> cytoplasm, cell -> cell); `track_summary`'s
    `by_class` then reports each region class's own numbers.

    `min_track_length` defaults to 2 (drop singletons only): a length-1
    "track" is just an unlinked detection with no displacement of its own
    -- it contributes nothing to `estimate_D_um2_s` (needs a
    frame-to-frame pair) or to any diffusionkit fit downstream, so keeping
    it around is pure clutter in the saved bundle and the tracks layer.

    `progress_callback`, if given, is called with `(0, 1, "linking")`
    before the link and `(1, 1, "done")` after; `cancel_event` is checked
    before it starts (linking is one opaque Rust call -- see
    `PipelineCancelled`'s docstring).
    """
    if session.points_df is None:
        raise ValueError("No points available -- run detect first.")
    max_step = float(max_step)
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError(f"max_step must be positive and finite, got {max_step}")
    _check_cancelled(cancel_event)

    def report(done: int, total: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(done, total, stage)

    points_df = session.points_df
    if not points_df.height:
        raise ValueError("No detections to link -- run detect first.")
    link_input = loctable.filter_quality(points_df)
    n_valid = link_input.height
    if exclude_flags:
        link_input = link_input.filter((pl.col("flags") & exclude_flags) == 0)
    n_after_flags = link_input.height
    link_input = apply_filters(link_input, point_filters)
    if link_input.height == 0:
        if n_valid == 0:
            raise ValueError(
                "No linkable detections -- none has a finite position and a positive "
                "position standard error."
            )
        if n_after_flags == 0:
            raise ValueError(
                f"No linkable detections -- every usable fit carries an excluded flag "
                f"({flag_names(exclude_flags)}). Exclude fewer flags."
            )
        raise ValueError(
            f"No linkable detections -- the point filters "
            f"({', '.join(sorted(point_filters or {}))}) rejected all "
            f"{n_after_flags} of them. Widen or clear them."
        )

    def keep_long(df: pl.DataFrame) -> pl.DataFrame:
        if min_track_length > 1 and df.height:
            return df.filter(pl.len().over("track_id") >= min_track_length)
        return df

    report(0, 1, "linking")
    class_groups = _region_groups(link_input)
    by_class = None
    if class_groups is None:
        tracks_df = spotsolve.link(link_input, max_step)
    else:
        class_areas = _class_areas_px(session)
        pieces, by_class, next_id = [], {}, 0
        for name, class_rows, instances in class_groups:
            class_pieces = []
            for rows in instances:
                linked = spotsolve.link(rows, max_step)
                if linked.height:
                    linked = linked.with_columns(
                        (pl.col("track_id").cast(pl.Int64) + next_id).alias("track_id")
                    )
                    next_id = int(linked["track_id"].max()) + 1
                class_pieces.append(linked)
            linked = pl.concat(class_pieces, how="vertical_relaxed")
            kept = keep_long(linked)
            by_class[name] = {
                "n_regions": len(instances),
                **_link_summary(session, class_rows, kept, class_areas.get(name)),
                "n_tracks": kept["track_id"].n_unique() if kept.height else 0,
            }
            pieces.append(linked)
        tracks_df = pl.concat(pieces, how="vertical_relaxed") if pieces else link_input.head(0)
    n_tracks_linked = tracks_df["track_id"].n_unique() if tracks_df.height else 0
    tracks_df = keep_long(tracks_df)
    tracks_df = apply_track_filters(tracks_df, track_filters, session.pixel_size_um, session.dt_s)
    report(1, 1, "done")

    area_px = None
    if session.labels is not None:
        area_px = int((session.labels > 0).sum()) or None
    summary = _link_summary(session, link_input, tracks_df, area_px)

    if session.source_params:
        # Linking was just redone, so a reopened bundle's recorded link
        # numbers and cuts no longer describe this session -- only its
        # detect provenance still does.
        session.source_params = {
            key: value for key, value in session.source_params.items() if key not in _TRACK_KEYS
        }
    session.tracks_df = tracks_df
    session.max_step_used = max_step
    session.exclude_flags_used = exclude_flags
    session.min_track_length_used = min_track_length
    session.point_filters_used = dict(point_filters) if point_filters else None
    session.track_filters_used = dict(track_filters) if track_filters else None
    session.track_summary = {
        **summary,
        "n_points_linked": link_input.height,
        "n_points_dropped_invalid": points_df.height - n_valid,
        "n_points_dropped_flagged": n_valid - n_after_flags,
        "n_points_dropped_by_filter": n_after_flags - link_input.height,
        "n_tracks_linked": n_tracks_linked,
        "by_class": by_class,
    }
    return session


def flag_names(bits: int) -> str:
    """`spotsolve.FitFlag` bits as `"EDGE|STALLED"`, for messages and
    the manifest -- a bare integer mask says nothing to a reader."""
    return "|".join(flag.name for flag in spotsolve.FitFlag if flag and bits & flag) or "none"


OUTSIDE_CLASS = "(outside)"
UNASSIGNED_CLASS = "(unassigned)"


def _region_groups(
    link_input: pl.DataFrame,
) -> Optional[list[tuple[str, pl.DataFrame, list[pl.DataFrame]]]]:
    """`[(class, class_rows, [rows of each region of that class]), ...]`
    when `link_input` is labeled with more than one region
    (`regions.label_points`), else None -- one region or none links as a
    single field. Rows outside every region (`region == 0`, possible only
    for a table labeled after an unrestricted detect) form their own
    "(outside)" class rather than being dropped."""
    if "region" not in link_input.columns:
        return None
    if link_input["region"].drop_nulls().n_unique() < 2:
        return None
    classed = link_input.with_columns(
        pl.when(pl.col("region") == 0)
        .then(pl.lit(OUTSIDE_CLASS))
        .otherwise(pl.col("region_class").fill_null(UNASSIGNED_CLASS))
        .alias("_class")
    )
    groups = []
    for (name,), class_rows in classed.group_by("_class", maintain_order=True):
        class_rows = class_rows.drop("_class")
        instances = [rows for _, rows in class_rows.group_by("region", maintain_order=True)]
        groups.append((name, class_rows, instances))
    return sorted(groups, key=lambda g: g[0])


def _class_areas_px(session: PipelineSession) -> dict[str, int]:
    """Pixel area of each region class in the session's labels image."""
    if session.labels is None or session.regions is None:
        return {}
    return region_tools.class_areas_px(session.labels, session.regions)


def _link_summary(
    session: PipelineSession,
    link_input: pl.DataFrame,
    tracks_df: pl.DataFrame,
    area_px: Optional[int] = None,
) -> dict:
    """The per-run linking numbers `track_summary` reports -- for the
    whole field, or for one region class's slice of it. `area_px` is the
    area the detections could have come from: the region(s) when there are any,
    else the full frame, so a density (and the crowding verdict built on
    it) isn't diluted by area nothing could be detected in."""
    # The detector's own CRLB, pooled, as one localization precision in
    # physical units -- what `estimate_D_um2_s` subtracts off.
    se_y_um = link_input["se_y"].to_numpy() * session.pixel_size_um
    se_x_um = link_input["se_x"].to_numpy() * session.pixel_size_um
    sigma_loc_um = float(np.nanmedian(np.sqrt((se_y_um**2 + se_x_um**2) / 2.0)))
    D_est, n_links = estimate_D_um2_s(tracks_df, session.dt_s, session.pixel_size_um, sigma_loc_um)

    if area_px is None:
        h, w = session.image.shape[1:]
        area_px = h * w
    active_area_um2 = area_px * session.pixel_size_um**2
    mean_n_per_frame = (
        link_input.group_by("frame").len()["len"].mean() if link_input.height else 0.0
    )
    density_um2 = (mean_n_per_frame / active_area_um2) if mean_n_per_frame else 0.0
    resolvability = check_resolvability(D_est, session.dt_s, density_um2)
    return {
        "sigma_loc_um": sigma_loc_um,
        "D_est_um2_s": D_est,
        "n_linked_steps": n_links,
        # The 2-D rms of the linked steps, px -- what `max_step` is judged
        # against (spotsolve: about 3x the fastest particles' rms step).
        # Truncated by `max_step` itself, so read it as a floor.
        "rms_step_px": rms_step_px(tracks_df),
        "density_um2": density_um2,
        "crowding_ratio": resolvability["ratio"],
        "resolvability_verdict": resolvability["verdict"],
        "resolvability_message": resolvability["message"],
    }


def rms_step_px(tracks_df: pl.DataFrame) -> Optional[float]:
    """2-D rms displacement over every consecutive-frame step in
    `tracks_df`, px; None when there is no such step."""
    if tracks_df.height == 0 or "track_id" not in tracks_df.columns:
        return None
    steps = (
        tracks_df.sort(["track_id", "frame"])
        .select(
            pl.col("frame").diff().over("track_id").alias("dframe"),
            (pl.col("y").diff().over("track_id") ** 2 + pl.col("x").diff().over("track_id") ** 2).alias("r2"),
        )
        .filter(pl.col("dframe") == 1)
    )
    return float(np.sqrt(steps["r2"].mean())) if steps.height else None


def session_manifest_extra(session: PipelineSession) -> dict:
    """Assemble the `manifest.json` `params` payload from a session that's
    been through all three stages -- meant to be passed as
    `results.build_manifest`'s `params`."""
    ts = session.track_summary or {}
    params = {
        "pixel_size_um": session.pixel_size_um,
        "dt_s": session.dt_s,
        # Not used by any stage here; recorded so the diffusion analysis of
        # this bundle can model motion blur without asking again. None when
        # neither the file nor the user said.
        "exposure_s": session.exposure_s,
        # Where those two came from, and anything the reader flagged about
        # them (non-square pixels, irregular frame timing, an
        # unconvertible unit) -- see `io_formats.StackMetadata`. The
        # values alone don't say whether they were read off the file or
        # supplied by hand, and every physical column in this bundle is
        # them multiplied through.
        **_metadata_provenance(session),
        "channel": session.channel,
        "z_index": session.z_index,
        "sigma_px": session.sigma,
        "sigma_loc_um": ts.get("sigma_loc_um"),
        "D_est_um2_s": ts.get("D_est_um2_s"),
        "n_linked_steps": ts.get("n_linked_steps"),
        "rms_step_px": ts.get("rms_step_px"),
        "max_step_px": session.max_step_used,
        "min_track_length": session.min_track_length_used,
        "exclude_flags": (
            flag_names(session.exclude_flags_used)
            if session.exclude_flags_used is not None
            else None
        ),
        "n_points_dropped_invalid": ts.get("n_points_dropped_invalid"),
        "n_points_dropped_flagged": ts.get("n_points_dropped_flagged"),
        # What the histogram filters were set to, as plain
        # {column: [lo, hi]} -- the record of which detections and tracks
        # this bundle's results were computed from. points.parquet still
        # holds every detection, so these are what makes the difference
        # between it and tracks.parquet readable after the fact.
        "point_filters": _jsonable_filters(session.point_filters_used),
        "track_filters": _jsonable_filters(session.track_filters_used),
        "n_points_dropped_by_filter": ts.get("n_points_dropped_by_filter"),
        "n_tracks_linked": ts.get("n_tracks_linked"),
        "density_um2": ts.get("density_um2"),
        "crowding_ratio": ts.get("crowding_ratio"),
        "resolvability_verdict": ts.get("resolvability_verdict"),
        "resolvability_message": ts.get("resolvability_message"),
        "n_points": session.points_df.height if session.points_df is not None else 0,
        "n_tracks": session.tracks_df["track_id"].n_unique() if session.tracks_df is not None and session.tracks_df.height else 0,
        "camera_kwargs": session.camera_kwargs_used,
        "detector": session.detector_used,
        "detect_kwargs": _jsonable_detect_kwargs(session.detect_kwargs_used),
        "frame_range": list(session.frame_range_used) if session.frame_range_used is not None else None,
        "spotsolve_version": spotsolve.__version__,
        # Per-class linking numbers (`run_track_step` links each region on
        # its own when the table is labeled with more than one) -- None for
        # a single-field run.
        "track_summary_by_class": ts.get("by_class"),
    }
    # A session rebuilt from a saved bundle never recomputed what earlier
    # stages recorded (a detect setting, say): keep the bundle's own
    # value rather than overwrite it with None. Counts and versions are
    # always the current session's.
    for key, value in (session.source_params or {}).items():
        if params.get(key) is None and key not in _NEVER_INHERITED:
            params[key] = value
    return params


# Manifest keys `run_track_step` produces -- dropped from a rebuilt
# session's `source_params` as soon as it links again.
_TRACK_KEYS = frozenset({
    "sigma_loc_um", "D_est_um2_s", "n_linked_steps", "rms_step_px", "max_step_px",
    "min_track_length", "exclude_flags", "n_points_dropped_invalid",
    "n_points_dropped_flagged", "point_filters", "track_filters", "n_points_dropped_by_filter", "n_tracks_linked",
    "density_um2", "crowding_ratio", "resolvability_verdict", "resolvability_message",
    "track_summary_by_class",
})


# Manifest keys that describe *this* write, never carried over from the
# bundle a session was rebuilt from (`PipelineSession.source_params`).
_NEVER_INHERITED = frozenset({"n_points", "n_tracks", "spotsolve_version"})


def session_from_bundle(
    image_path: str | Path,
    stack: tuple[np.ndarray, StackMetadata],
    points_df: pl.DataFrame,
    tracks_df: Optional[pl.DataFrame],
    manifest: dict,
    labels: Optional[np.ndarray] = None,
    regions: Optional[Regions] = None,
) -> PipelineSession:
    """A `PipelineSession` picking up where a saved bundle left off, so
    its detections can be linked (or its tracks re-filtered and re-saved)
    without re-running detect.

    Built on the manifest's own `pixel_size_um`/`dt_s`/`exposure_s`, not
    the file's or the UI's: `points.parquet`'s µm and s columns were
    derived from those numbers, and linking against different ones would
    leave the bundle internally inconsistent. Everything the manifest
    records about how the detections were made is put back on the
    `*_used` fields, and the whole `params` dict is kept as
    `source_params` for `session_manifest_extra` to fall back on."""
    params = dict(manifest.get("params", {}) or {})
    image, metadata = stack

    def filters(key: str) -> Optional[FilterSpec]:
        return _filters_from_record(params.get(key))

    detect_kwargs = _detect_kwargs_from_record(params.get("detect_kwargs"))
    frame_range = params.get("frame_range")
    has_tracks = tracks_df is not None and tracks_df.height > 0 and "track_id" in tracks_df.columns
    session = PipelineSession(
        image_path=Path(image_path),
        image=image,
        pixel_size_um=params.get("pixel_size_um") or metadata.pixel_size_um,
        dt_s=params.get("dt_s") or metadata.dt_s,
        channel=params.get("channel", 0),
        z_index=params.get("z_index", 0),
        metadata=metadata,
        exposure_s=params.get("exposure_s", metadata.exposure_s),
        sigma=params.get("sigma_px"),
        points_df=points_df,
        camera_kwargs_used=params.get("camera_kwargs"),
        detect_kwargs_used=detect_kwargs,
        detector_used=params.get("detector"),
        frame_range_used=tuple(frame_range) if frame_range is not None else None,
        labels=labels,
        regions=regions,
        source_params=params,
        point_filters_used=filters("point_filters"),
    )
    if has_tracks:
        session.tracks_df = tracks_df
        session.min_track_length_used = params.get("min_track_length")
        session.exclude_flags_used = parse_flag_names(params.get("exclude_flags"))
        session.max_step_used = params.get("max_step_px")
        session.track_filters_used = filters("track_filters")
        session.track_summary = {key: params.get(key) for key in _TRACK_KEYS}
        session.track_summary["by_class"] = params.get("track_summary_by_class")
    if session.pixel_size_um is None or session.dt_s is None:
        raise ValueError(
            f"{Path(image_path).name}: the saved bundle records no pixel size / frame "
            "interval and the file has none either."
        )
    return session


def _filters_from_record(spec: Optional[dict]) -> Optional[FilterSpec]:
    """A manifest's `{column: [lo, hi]}` back as a `FilterSpec`."""
    return {col: tuple(bounds) for col, bounds in spec.items()} if spec else None


def _detect_kwargs_from_record(detect_kwargs: Optional[dict]) -> Optional[dict]:
    """A manifest's `detect_kwargs` back as `run_detect_step` takes them."""
    if detect_kwargs is None:
        return None
    return {
        key: tuple(value) if key == "slack" and value is not None else value
        for key, value in detect_kwargs.items()
        if key not in _OBSOLETE_DETECT_KWARGS
    }


def detect_track_params_from_manifest(manifest: dict) -> dict:
    """The `DetectTrackParams` keyword arguments a saved bundle was made
    with -- what lets a headless run reuse the settings one movie was
    tuned on in the widget (`gemscape2 detect-track`'s `template`).

    Only settings, never what they measured (`D_est_um2_s`, `n_points`,
    ...), and only the keys the manifest actually records, so a missing
    one falls through to `DetectTrackParams`' default. `frame_range` is
    left out: a frame window is a fact about one movie (or a quick test
    on a few frames), not a setting to carry to the next."""
    params = manifest.get("params", {}) or {}
    record = {
        "sigma": params.get("sigma_px"),
        "min_track_length": params.get("min_track_length"),
        "exclude_flags": parse_flag_names(params.get("exclude_flags")),
        # Absent from a bundle linked by spotsolve 0.1, whose linker took
        # no step limit: the config then has to supply one.
        "max_step": params.get("max_step_px"),
        "detector": params.get("detector"),
        "camera_kwargs": params.get("camera_kwargs"),
        "detect_kwargs": _detect_kwargs_from_record(params.get("detect_kwargs")),
        "point_filters": _filters_from_record(params.get("point_filters")),
        "track_filters": _filters_from_record(params.get("track_filters")),
    }
    return {key: value for key, value in record.items() if value is not None}


def _metadata_provenance(session: PipelineSession) -> dict:
    """`StackMetadata.as_manifest_dict()` for this session's image, with
    each of the two required values marked as coming from the file or
    from an explicit override (the values themselves are recorded
    separately, by `session_manifest_extra`). Empty-ish but present for a
    session built without metadata, so the manifest's shape doesn't
    depend on which path produced it."""
    metadata = session.metadata
    if metadata is None:
        return {
            "pixel_size_um_source": "unrecorded",
            "dt_s_source": "unrecorded",
            "exposure_s_source": "unrecorded" if session.exposure_s is None else "given explicitly",
        }
    provenance = metadata.as_manifest_dict()
    if metadata.pixel_size_um is None or metadata.pixel_size_um != session.pixel_size_um:
        provenance["pixel_size_um_source"] = (
            f"given explicitly (file said: {provenance['pixel_size_um_source']})"
        )
    if metadata.dt_s is None or metadata.dt_s != session.dt_s:
        provenance["dt_s_source"] = f"given explicitly (file said: {provenance['dt_s_source']})"
    if session.exposure_s is not None and metadata.exposure_s != session.exposure_s:
        provenance["exposure_s_source"] = (
            f"given explicitly (file said: {provenance['exposure_s_source']})"
        )
    return provenance


def _jsonable_filters(filters: Optional[FilterSpec]) -> Optional[dict]:
    """A filter spec with its `(lo, hi)` tuples as lists -- same reason as
    `_jsonable_detect_kwargs`: `json.dumps` writes both as arrays, so
    normalize here and keep the round-trip honest. An open side is null,
    however it was given (None, or a config's `-inf`/`inf`) -- JSON has no
    infinity."""
    if not filters:
        return None

    def bound(value):
        return None if value is None or not np.isfinite(value) else value

    return {col: [bound(lo), bound(hi)] for col, (lo, hi) in filters.items()}


def parse_flag_names(names: Optional[str]) -> Optional[int]:
    """`flag_names`' inverse, for a manifest read back: `"EDGE|STALLED"`
    -> the bitmask. None stays None (a bundle linked before flags existed)."""
    if names is None:
        return None
    return sum(
        spotsolve.FitFlag[name] for name in names.split("|") if name and name != "none"
    )


def _jsonable_detect_kwargs(detect_kwargs: Optional[dict]) -> Optional[dict]:
    """`detect_kwargs` with `slack` as a list rather than a tuple --
    `json.dumps` writes both as arrays, but reading a manifest back gives
    lists either way, so normalize here and keep the round-trip honest."""
    if detect_kwargs is None:
        return None
    out = dict(detect_kwargs)
    if out.get("slack") is not None:
        out["slack"] = list(out["slack"])
    return out


def run_detect_track(
    image_path: str | Path,
    params: DetectTrackParams,
    pixel_size_um: Optional[float] = None,
    dt_s: Optional[float] = None,
    channel: int = 0,
    z_index: int = 0,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
    exposure_s: Optional[float] = None,
    labels: Optional[np.ndarray] = None,
    regions: Optional[Regions] = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Run the full detect+track pipeline on one timelapse in one call,
    composing `load_session`/`run_detect_step`/`run_track_step` -- for
    the headless CLI. See this module's docstring for the stepwise
    alternative the widget uses.

    `image_path` can be .tif/.tiff, .nd2, or .ims (see `io_formats.load_stack`).
    `channel`/`z_index` pick which plane to track for files with more than
    one (both default to 0). `pixel_size_um`/`dt_s` fall back to the
    file's own metadata if not given explicitly, and so does `exposure_s`
    (recorded in the manifest for the diffusion analysis; never required).

    `cancel_event`, if given, is forwarded to each stage -- see
    `PipelineCancelled`'s docstring for what "cancelled" actually means per
    stage (cooperative, frame-granular for detect, stage-boundary-only for
    track).

    `labels`/`regions`, if given, are the movie's painted regions (see
    `napari_gemscape2.regions`): only painted pixels are localized, each
    detection is stamped with its region, and each region is linked on
    its own -- what the widget does with "restrict to regions" on.

    Returns (points_df, tracks_df, manifest_extra) -- `manifest_extra` is
    meant to be passed as `results.build_manifest`'s `params`.
    """
    session = load_session(
        image_path,
        pixel_size_um=pixel_size_um,
        dt_s=dt_s,
        channel=channel,
        z_index=z_index,
        exposure_s=exposure_s,
    )
    start, end = _resolve_frame_range(params.frame_range, session.image.shape[0])
    n_frames = end - start
    total_steps = n_frames + 1

    def report(done: int, total: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(done, total, stage)

    session.sigma = params.sigma
    mask = None
    if labels is not None:
        if labels.shape != session.image.shape[-2:]:
            raise ValueError(
                f"its regions mask is {labels.shape}, but a frame is {session.image.shape[-2:]}"
            )
        session.labels, session.regions = labels, regions
        mask = labels > 0

    detect_progress = (
        (lambda done, total, stage: report(done, total_steps, stage))
        if progress_callback is not None
        else None
    )
    run_detect_step(
        session,
        camera_kwargs=params.camera_kwargs,
        detect_kwargs=params.detect_kwargs,
        frame_range=params.frame_range,
        progress_callback=detect_progress,
        cancel_event=cancel_event,
        n_threads=params.n_threads,
        detector=params.detector,
        mask=mask,
    )
    if session.labels is not None:
        session.points_df = region_tools.label_points(session.points_df, session.labels, session.regions)

    def track_progress(done: int, total: int, stage: str) -> None:
        report(n_frames + done, total_steps, stage)

    run_track_step(
        session,
        params.max_step,
        min_track_length=params.min_track_length,
        exclude_flags=params.exclude_flags,
        point_filters=params.point_filters,
        track_filters=params.track_filters,
        progress_callback=track_progress,
        cancel_event=cancel_event,
    )

    return session.points_df, session.tracks_df, session_manifest_extra(session)
