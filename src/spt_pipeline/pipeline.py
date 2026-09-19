"""The shared detect+track pipeline, as three composable stages plus one
convenience wrapper that runs all of them.

Both the headless CLI (`cli.py`) and the interactive napari widget
(`widgets/experiment_list.py`) build on this module -- the logic lives
here exactly once, unlike napari-gemscape where the interactive handlers
and its batch subprocess script each reimplemented the pipeline.

Detection and linking are `spotsolve`'s: `spotsolve.localize`/
`localize_stack` (multi-emitter fits by Bayesian model selection -- in each
small box an emitter exists iff it lowers the box's Poisson deviance by a
fixed number of nats, so "how many are there" and "where are they" are
answered as one problem) and `spotsolve.tracking.link` (frame-to-frame
linking scored by likelihood ratio under a per-track posterior over D).
`spotsolve.loctable` defines the table that passes between them, and this
module emits it verbatim -- `se_y`/`se_x` (per-detection CRLB), `flux`,
`is_aggregate` and the rest -- so nothing downstream has to learn a second
schema.

Two things the previous sfwloc-based pipeline did are gone because
`spotsolve` makes them unnecessary rather than because they were dropped:

  - There is one *default* detector, not a per-acquisition choice of
    three. `spotsolve.localize`/`localize_stack` (multi-emitter, the
    default `DetectTrackParams.detector`) fits each box jointly and
    decides how many emitters it holds by Bayesian model selection --
    the same rule handles a crowded field and a sparse one, so there is
    no dense/sparse/DAOPHOT algorithm to hand-pick per file. `spotsolve`
    now also ships `localize_aguet`/`localize_aguet_stack`
    (`detector="aguet"`): independent single-emitter fits behind a LoG
    screen, with no multi-emitter search and no width-band rejection --
    the spotfitlm-compatible baseline for genuinely sparse fields, kept
    on as an option rather than the default because the joint fit's
    model selection is the more general rule when in doubt.
  - Linking takes no gate. `spotsolve.tracking.fit_link_params` measures
    the step-size distribution, detection continuity and CRLB inflation
    from the movie itself, so the old bootstrap-link -> estimate-D ->
    relink-with-a-derived-gate dance collapses into one call. What used to
    be a tuned `bootstrap_gate_px` is now a fitted `LinkParams`, recorded
    in `track_summary` for the record.

`PipelineSession` + `load_session`/`run_preview_frame`/
`run_calibration_step`/`run_detect_step`/`run_track_step` let each stage
run independently and be re-run after tweaking that stage's own knobs,
without redoing earlier stages -- this is what backs the dock widget's
per-tab "Preview frame" / "Run detect" / "Run tracking" buttons (each
stage builds on whatever the session already has: `run_detect_step` uses
`session.sigma` from a prior calibration if the caller doesn't pass one
explicitly; `run_track_step` needs `session.points_df` from a prior
detect). `run_detect_track` composes them in one call for the headless CLI,
where stepwise control isn't needed.

Two things the interactive path does that the headless one doesn't:

  - **Preview instead of calibrate.** `run_preview_frame` localizes one
    frame with the reporting band off and hands back every fit, so the PSF
    width is chosen by looking at the `fit_sigma` distribution and
    adopting its median -- one visible round of exactly what
    `run_calibration_step` iterates out of sight. An explicit
    `DetectTrackParams.sigma` then skips calibration entirely; `sigma_init`
    and `run_calibration_step` remain for unattended batch runs, where
    there is nobody to look at a histogram.
  - **Filters.** `DetectTrackParams.point_filters`/`track_filters` are
    `{column: (lo, hi)}` cuts (see `filter_mask`) on per-detection columns
    and per-track metrics. Like `is_aggregate`, they apply to what linking
    sees and to which tracks survive -- never to `points_df`, which keeps
    every detection so the rejected population stays auditable -- and they
    are recorded in `session_manifest_extra`.
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
from spotsolve import loctable, tracking

from spt_pipeline import regions as region_tools
from spt_pipeline.regions import Regions
from spt_pipeline.io_formats import StackMetadata, load_stack
from spt_pipeline.tracking_diagnostics import check_resolvability

# The camera's own calibration, forwarded to every `spotsolve.localize`/
# `localize_stack`/`calibrate_sigma` call. `offset` (ADU) is subtracted
# before fitting; everything else about the camera -- gain, read noise --
# is measured from each frame's own noise (`spotsolve` no longer takes
# either as an input), so `offset` is the only knob left here.
DEFAULT_CAMERA_KWARGS = dict(
    offset=0.0,
)

# Forwarded to `spotsolve.localize`/`localize_stack` as **kwargs, mirroring
# that function's own defaults (`spotsolve.native`).
#
# `slack` is the width range a fit is allowed to take and `band` the range
# actually reported as a detection, both as multiples of `sigma` -- a fit
# landing outside `band` is an out-of-band reject (too narrow, too wide, or
# too close to an edge) rather than a detection, which is how out-of-focus
# and non-PSF-shaped junk is kept out of the table. `k_max` caps how many
# emitters one box may be fitted with jointly.
#
# One cut on the LoG z-statistic (sd of the frame's own noise), used both
# for which peaks get a box searched around them and for whether a
# leftover residual peak inside a box gets tried as another emitter --
# `spotsolve` used to split these into a seed cut and a birth cut, but
# measurement showed one number (`PEAK_Z`) does the same job. `None`
# (recommended) uses that default; raise it for fewer false positives and
# speed, lower it toward 2.5 for faint, sparse data.
#
# `selection`/`count_penalty` pick the per-box emitter-count rule:
# "fixed" (default) is the existing greedy 10-nat cost, unaffected by
# `count_penalty=0.0` -- i.e. today's behavior, unchanged. "bic" compares
# background-only and multi-emitter fits with an experimental
# BIC-inspired score instead; see spotsolve's docs/COUNT_SELECTION.md
# before trusting a `count_penalty` value against real data.
DEFAULT_DETECT_KWARGS = dict(
    k_max=spotsolve.K_MAX,
    threshold=None,
    slack=spotsolve.SLACK,
    band=spotsolve.BAND,
    selection="fixed",
    count_penalty=0.0,
)

# Forwarded to `spotsolve.localize_aguet`/`localize_aguet_stack` as **kwargs
# when `DetectTrackParams.detector == "aguet"`, mirroring their own defaults
# (`spotsolve.aguet`). None of `DEFAULT_DETECT_KWARGS`' keys apply here --
# Aguet fits one emitter per LoG-screened candidate independently rather
# than searching a box jointly, so there is no `k_max` (nothing to search
# jointly for), no `slack`/`band` (every screened fit is reported; there is
# no width-based accept/reject) and no `threshold` (the screening cut is
# `significance`, a per-pixel level rather than a z-score). `boxsize` is the
# odd fit-crop size around each candidate and `itermax` its optimizer's
# iteration budget -- both rarely need changing.
DEFAULT_SPARSE_KWARGS = dict(
    significance=0.05,
    boxsize=9,
    itermax=50,
)

# Forwarded to `spotsolve.calibrate_sigma` as **kwargs, mirroring its own
# defaults. It localizes the calibration frame(s) with the reporting band
# off, takes the median fitted width, and repeats at that value until the
# estimate moves by less than `tol` (relative) or `max_rounds` runs out;
# `n_boot`/`seed` drive the bootstrap CI on that median.
DEFAULT_CALIBRATION_KWARGS = dict(
    tol=0.002,
    max_rounds=8,
    n_boot=2000,
    seed=0,
)

# ProgressCallback(done, total, stage) -- called from whatever thread the
# stage function executes on; the interactive widget wraps this in a
# QObject signal to cross back onto the Qt event-loop thread safely.
ProgressCallback = Callable[[int, int, str], None]


class PipelineCancelled(Exception):
    """Raised at a `cancel_event` checkpoint (see `run_detect_step`/
    `run_track_step`/`run_calibration_step`'s `cancel_event` argument).
    Cooperative cancellation only -- takes effect at the next chunk
    (detect, when a `progress_callback` puts it on the chunked
    `localize_stack` path) or stage boundary, not instantly, since
    `calibrate_sigma`, `localize_stack` and `link` are each one opaque
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
    the saved bundle got. Same role `calibration_accepted` plays for the
    reporting band."""
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
    # The in-focus PSF width the search runs at. Set it explicitly (the
    # interactive path does, from a preview frame's measured `fit_sigma` --
    # see `run_preview_frame`) to skip calibration entirely; leave it None
    # to have `run_detect_track` measure it per file with
    # `run_calibration_step`, which is what an unattended batch over
    # many acquisitions wants.
    sigma: Optional[float] = None
    sigma_init: float = 1.3
    calibration_frame_index: int = 0
    min_track_length: int = 2
    # Over-bright cut, as a multiple of each frame's own median detection:
    # detections above it are FLAGGED `is_aggregate` (never deleted -- see
    # `run_detect_step`). None uses spotsolve's own AGG_AMP_RATIO.
    agg_ratio: Optional[float] = None
    # Whether linking sees the flagged aggregates. Default True (drop them):
    # an over-bright blob is not a point emitter, and its position is a
    # flux-weighted compromise between whatever is inside it, so linking it
    # produces a trajectory of something that isn't a particle.
    drop_aggregates: bool = True
    # Whether `spotsolve.tracking.link` also scores candidate links by
    # brightness continuity (each detection's `flux`/`se_flux`), on top of
    # position and CRLB. Off by default -- it's an extra cue for a crowded
    # field where position alone leaves ambiguous assignments, not a
    # correction to a broken default.
    link_with_flux: bool = False
    # Which spotsolve detector `run_detect_step` runs: "multi_emitter"
    # (default -- `spotsolve.localize`/`localize_stack`, joint fit + Bayesian
    # model selection) or "aguet" (`localize_aguet`/`localize_aguet_stack`,
    # the independent-fit sparse baseline). Only the production detect step
    # honors this -- `run_calibration_step`/`run_preview_frame` always
    # measure sigma with the multi-emitter detector regardless, since a PSF
    # width is a physical fact about the optics, not a property of which
    # detector will run on it.
    detector: str = "multi_emitter"
    camera_kwargs: dict = field(default_factory=lambda: dict(DEFAULT_CAMERA_KWARGS))
    # None picks `DEFAULT_DETECT_KWARGS` or `DEFAULT_SPARSE_KWARGS` to match
    # `detector` (see `run_detect_step`) -- left as None rather than always
    # defaulting to the multi-emitter dict, which would silently hand
    # `localize_aguet_stack` keyword arguments (`k_max`, `slack`, `band`) it
    # doesn't accept.
    detect_kwargs: Optional[dict] = None
    calibration_kwargs: dict = field(default_factory=lambda: dict(DEFAULT_CALIBRATION_KWARGS))
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
    # `duration_s`). Like `is_aggregate`, these are applied to what linking
    # sees and to the final track set -- never to `points_df` itself, which
    # keeps every detection so the rejected population stays auditable.
    point_filters: FilterSpec = field(default_factory=dict)
    track_filters: FilterSpec = field(default_factory=dict)


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

    sigma: Optional[float] = None
    calib_summary: Optional[dict] = None
    calib_points_df: Optional[pl.DataFrame] = None
    calibration_kwargs_used: Optional[dict] = None
    calibration_frame_used: Optional[int] = None

    # One frame localized with the reporting band OFF, for the Detect
    # tab's "Preview frame" action -- every fit is a row, including the
    # ones a real run would reject as out-of-band, which is the point:
    # the `fit_sigma` histogram over this table is how `sigma` gets
    # chosen. See `run_preview_frame`. Never written to the bundle.
    preview_points_df: Optional[pl.DataFrame] = None
    preview_summary: Optional[dict] = None
    preview_frame_used: Optional[int] = None

    points_df: Optional[pl.DataFrame] = None
    # One row per frame (`loctable.FRAME_SCHEMA`): detection counts, the
    # out-of-band reject breakdown, and the aggregate flux share. Kept on
    # the session rather than folded into `points_df` because it's a fact
    # about the frame, not about any one detection -- and it's what makes
    # "why did this frame find nothing" answerable after the fact. Its
    # `n_too_narrow`/`n_too_wide`/`n_edge` columns are always 0 when
    # `detector_used == "aguet"`: that detector has no width-band rejection
    # to count (see `run_detect_step`).
    frames_df: Optional[pl.DataFrame] = None
    camera_kwargs_used: Optional[dict] = None
    detect_kwargs_used: Optional[dict] = None
    detector_used: Optional[str] = None
    agg_ratio_used: Optional[float] = None
    frame_range_used: Optional[tuple[int, int]] = None
    # The painted regions image (`(H, W)` uint16, 0 = background) whose
    # `labels > 0` was run_detect_step's `mask`, and the table naming each
    # label (see `spt_pipeline.regions`) -- set by the widget (not
    # pipeline.py itself, which stays napari-agnostic), carried through so
    # write_result can persist them alongside points/tracks.
    labels: Optional[np.ndarray] = None
    regions: Optional[Regions] = None
    # The saved manifest's `params`, when this session was rebuilt from a
    # bundle (`session_from_bundle`) rather than run here -- what
    # `session_manifest_extra` falls back on for anything the session
    # never recomputed (the calibration CI, say), so a re-save of a
    # reopened bundle doesn't erase its own provenance.
    source_params: Optional[dict] = None

    tracks_df: Optional[pl.DataFrame] = None
    track_summary: Optional[dict] = None
    link_params: Optional[tracking.LinkParams] = None
    drop_aggregates_used: Optional[bool] = None
    link_with_flux_used: Optional[bool] = None
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
    """Load a timelapse and start a fresh (un-calibrated, un-detected,
    un-tracked) `PipelineSession`. `stack` is an already-read
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


def run_calibration_step(
    session: PipelineSession,
    sigma_init: float,
    calibration_kwargs: Optional[dict] = None,
    camera_kwargs: Optional[dict] = None,
    frame_index: int = 0,
    mask: Optional[np.ndarray] = None,
    cancel_event: Optional[threading.Event] = None,
) -> PipelineSession:
    """Measure the in-focus PSF sigma against `session.image[frame_index]`
    (default: the first frame), via `spotsolve.calibrate_sigma`. Sets
    `session.sigma`/`session.calib_summary` in place (and returns
    `session`, for chaining).

    Always uses the multi-emitter detector (`calibrate_sigma` is hardcoded
    to it), regardless of `DetectTrackParams.detector` -- a PSF width is a
    physical fact about the optics, not a property of which detector will
    later run the production detect step against it.

    `frame_index` matters when the default frame isn't a good calibration
    reference -- e.g. sparser or better-focused elsewhere in the stack.

    `sigma_init` only needs to be within ~25% of the truth:
    `calibrate_sigma` localizes the frame with the reporting band off,
    takes the median fitted width, and re-runs at that value until it
    stops moving (`calibration_kwargs`' `tol`/`max_rounds`).
    `calib_summary` carries the bootstrap 95% CI on that median and
    whether the loop converged -- a `converged=False` estimate is the one
    to distrust.

    `session.calib_points_df` gets a per-spot table for the same frame,
    re-localized at the measured sigma with the reporting band OFF
    (`band=None`), so every fit is a row -- including the ones a detect run
    would reject as out-of-band. Columns are `loctable.LOCALIZATION_SCHEMA`
    (`y`/`x`/`fit_sigma`/`sigma_ratio`/`se_*`/`flux`/...) plus a derived
    `accepted`, so every calibration spot's fit is inspectable and not
    just the aggregate estimate. `run_preview_frame` returns the same
    table for one frame without the fixed-point loop around it, and that
    is what the napari widget shows now -- see `widgets/params_panel.py`'s
    docstring for why this function is the headless path only.

    `mask`, if given, is forwarded as `roi` to both calls -- calibrating on
    the same region detection will run on.

    `cancel_event`, if given, is only checked before this stage starts --
    `calibrate_sigma` is one opaque Rust call with no interruption point
    of its own, so a cancellation requested mid-fit still runs to
    completion (see `PipelineCancelled`'s docstring).
    """
    _check_cancelled(cancel_event)
    kwargs = dict(calibration_kwargs) if calibration_kwargs is not None else dict(DEFAULT_CALIBRATION_KWARGS)
    camera = dict(camera_kwargs) if camera_kwargs is not None else dict(DEFAULT_CAMERA_KWARGS)
    frame = session.image[frame_index]

    calibration = spotsolve.calibrate_sigma(frame, sigma_init, roi=mask, **camera, **kwargs)

    # Re-localize the same frame at the measured sigma, band off, purely to
    # get an inspectable per-spot table -- `calibrate_sigma` returns the
    # aggregate plus the raw widths, not positions.
    result = spotsolve.localize(frame, calibration.sigma, roi=mask, band=None, images=False, **camera)
    calib_points_df, _frame_row, _aggs = loctable.frame_tables(
        result,
        frame=frame_index,
        t=frame_index * session.dt_s,
        pixel_size=session.pixel_size_um,
    )
    calib_points_df = calib_points_df.with_columns(
        calibration_accepted(calib_points_df).alias("accepted")
    )

    lo, hi = calibration.ci
    session.sigma = calibration.sigma
    session.calib_summary = {
        "sigma_estimate": calibration.sigma,
        "sigma_ci_lo": lo,
        "sigma_ci_hi": hi,
        "n_spots_used": calibration.n_spots,
        "n_spots_total": calib_points_df.height,
        "converged": calibration.converged,
        "rounds": len(calibration.guesses),
    }
    session.calib_points_df = calib_points_df
    session.calibration_kwargs_used = kwargs
    session.calibration_frame_used = frame_index
    session.camera_kwargs_used = camera
    return session


def calibration_accepted(
    calib_points_df: pl.DataFrame, band: tuple[float, float] = spotsolve.BAND
) -> pl.Series:
    """Per-spot boolean: would a detect run at this sigma have reported
    this calibration candidate, or rejected it as out-of-band? Just
    `band[0] <= sigma_ratio <= band[1]` -- `calib_points_df` is built with
    the reporting band off precisely so both kinds of spot are in the
    table, and this is the one place that rule is written rather than
    re-derived wherever a caller (e.g. the calibration-spots preview
    layer) wants to show accepted vs. rejected spots.

    Note this is a different question than which spots the sigma estimate
    was computed from: `calibrate_sigma` takes the median over every
    fitted width, band or no band."""
    lo, hi = band
    return (calib_points_df["sigma_ratio"] >= lo) & (calib_points_df["sigma_ratio"] <= hi)


def run_preview_frame(
    session: PipelineSession,
    sigma: float,
    frame_index: int = 0,
    camera_kwargs: Optional[dict] = None,
    detect_kwargs: Optional[dict] = None,
    agg_ratio: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    cancel_event: Optional[threading.Event] = None,
    detector: str = "multi_emitter",
) -> PipelineSession:
    """Localize ONE frame at `sigma` with `detector` ("multi_emitter", the
    default, or "aguet"), so every fit lands in the table -- including, for
    the multi-emitter detector, the ones a real detect run would bin as
    out-of-band (its reporting band is forced off here). Sets
    `session.preview_points_df` / `session.preview_summary` /
    `session.preview_frame_used` in place (and returns `session`, for
    chaining). Never touches `points_df`: a preview is something to look
    at, not a result to link or save.

    This is what makes `calibrate_sigma` unnecessary interactively. That
    function's loop was: localize the frame with the band off, take the
    median fitted width, re-run at it, repeat to a fixed point -- all of it
    invisible, reported as one number plus a bootstrap CI. Here the same
    loop is the user's: preview, look at the `fit_sigma` histogram, adopt
    its median (`summary["fit_sigma_median"]`, what the Detect tab's "Use"
    button reads), preview again. It converges in the same two or three
    rounds, and the distribution it converged on is on screen the whole
    time -- so a bimodal or ragged `fit_sigma` (two focal planes, junk
    fitted as signal) is visible rather than averaged into a median with a
    tight CI around it. `run_calibration_step` is still there for the
    headless path, where there is nobody to look.

    For `detector="multi_emitter"`, `summary` also carries `n_in_band`
    against the CURRENT `detect_kwargs` band -- how many of these fits a
    real run at this sigma would actually report -- since that, not the
    raw fit count, is what a detect run yields. `accepted` is the same
    question per row (see `calibration_accepted`), meant for the preview
    layer's border color. `detector="aguet"` has no band to gate against
    (every LoG-screened fit it makes is already a reported detection), so
    every row previews as accepted and `n_in_band` is omitted; `n_failed`
    (from the fit's own `info["failures"]`) is reported instead.

    `cancel_event` is only checked before the call: one frame is one
    opaque Rust call (see `PipelineCancelled`).
    """
    _check_cancelled(cancel_event)
    camera = dict(camera_kwargs) if camera_kwargs is not None else dict(DEFAULT_CAMERA_KWARGS)
    t = session.image.shape[0]
    frame_index = max(0, min(frame_index, t - 1))
    frame = session.image[frame_index]

    n_failed = None
    if detector == "aguet":
        detect = dict(detect_kwargs) if detect_kwargs is not None else dict(DEFAULT_SPARSE_KWARGS)
        band = None
        result = spotsolve.localize_aguet(frame, sigma, roi=mask, images=False, **camera, **detect)
        n_failed = len(result.info.get("failures", ()))
    else:
        detect = dict(detect_kwargs) if detect_kwargs is not None else dict(DEFAULT_DETECT_KWARGS)
        # `band=None` regardless of what the Detect tab has set: the whole
        # point of a preview is to show the fits the band would have
        # removed, so the band can be chosen against them. `slack` (the
        # range a fit may TAKE, as opposed to be reported at) is honored --
        # it bounds the optimizer, so overriding it would preview a
        # different fit.
        preview_kwargs = dict(detect)
        band = preview_kwargs.pop("band", None)
        preview_kwargs["band"] = None
        result = spotsolve.localize(frame, sigma, roi=mask, images=False, **camera, **preview_kwargs)

    points_df, frame_row, _aggs = loctable.frame_tables(
        result,
        frame=frame_index,
        t=frame_index * session.dt_s,
        pixel_size=session.pixel_size_um,
        agg_ratio=agg_ratio,
    )
    accepted = (
        calibration_accepted(points_df, band)
        if band is not None
        else pl.Series("accepted", np.ones(points_df.height, dtype=bool))
    )
    points_df = points_df.with_columns(accepted.alias("accepted"))

    fit_sigma = points_df["fit_sigma"].drop_nulls().to_numpy() if points_df.height else np.array([])
    session.preview_points_df = points_df
    session.preview_frame_used = frame_index
    session.preview_summary = {
        "sigma_used": sigma,
        "detector": detector,
        "n_fits": points_df.height,
        "n_in_band": int(accepted.sum()) if points_df.height and band is not None else None,
        "n_failed": n_failed,
        "n_flagged": int(points_df["is_aggregate"].sum()) if points_df.height else 0,
        # The median of THIS frame's fitted widths -- one round of what
        # `calibrate_sigma` iterates. Preview again at it to do the next.
        "fit_sigma_median": float(np.median(fit_sigma)) if fit_sigma.size else None,
        "fit_sigma_mad": float(np.median(np.abs(fit_sigma - np.median(fit_sigma)))) if fit_sigma.size else None,
        "band": list(band) if band is not None else None,
        "median_flux": float(frame_row["median_flux"][0]) if frame_row.height else None,
    }
    session.camera_kwargs_used = camera
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
    camera_kwargs: Optional[dict] = None,
    detect_kwargs: Optional[dict] = None,
    agg_ratio: Optional[float] = None,
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
    "multi_emitter" (`spotsolve.localize_stack`, joint per-box fit with
    Bayesian model selection) or "aguet" (`spotsolve.localize_aguet_stack`,
    independent single-emitter fits behind a LoG screen -- the sparse
    baseline). `detect_kwargs` must match whichever is chosen (`None` picks
    `DEFAULT_DETECT_KWARGS`/`DEFAULT_SPARSE_KWARGS` accordingly) -- the two
    detectors take disjoint keyword arguments, so a dict built for one
    raises a `TypeError` if forwarded to the other. `session.frames_df`'s
    `n_too_narrow`/`n_too_wide`/`n_edge` columns are always 0 for
    `detector="aguet"`: it has no width-band rejection to count.

    Sets `session.points_df` (one row per detection,
    `loctable.LOCALIZATION_SCHEMA`) and `session.frames_df` (one row per
    frame, `loctable.FRAME_SCHEMA`). `loc_id` is unique over the whole
    run, and `frame` holds true stack indices (`start`..`end-1`), not a
    re-based 0..n range, on either path below.

    `sigma`, if given, is the in-focus PSF width used as-is, skipping
    calibration entirely. If omitted, falls back to `session.sigma` from a
    prior `run_calibration_step` call -- raises if neither is available.
    Note this is the width the search runs AT; each emitter still gets its
    own fitted width (`fit_sigma`), and -- for `detector="multi_emitter"`
    only -- how far that may stray before the fit stops being reported is
    `detect_kwargs`' `slack`/`band`.

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

    Over-bright detections are FLAGGED (`is_aggregate`), never dropped
    here: `agg_ratio` (a multiple of each frame's own median detection --
    relative to the frame so that one number survives bleaching and
    illumination drift) sets the cut, and `frames_df` records what share
    of each frame's flux landed in aggregates. Whether linking sees them
    is `run_track_step`'s `drop_aggregates`, so "how much of this movie
    was junk" stays an auditable fact about the run rather than a silent
    deletion.

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
                "No sigma available -- run calibration first, or pass sigma explicitly."
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
        locs, frame_row, _aggs = loctable.frame_tables(
            result,
            frame=i,
            t=i * session.dt_s,
            pixel_size=session.pixel_size_um,
            agg_ratio=agg_ratio,
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
    session.agg_ratio_used = agg_ratio
    session.frame_range_used = (start, end)
    # A fresh detection table supersedes whatever bundle this session was
    # rebuilt from, so none of that bundle's recorded numbers still apply.
    session.source_params = None
    return session


def estimate_D_um2_s(linked_df: pl.DataFrame, dt_s: float, pixel_size_um: float, sigma_loc_um: float):
    """Single-step MSD estimate of D, corrected for localization noise:
    mean(r^2) = 4*D*dt + 4*sigma_loc_um^2.

    Kept even though `spotsolve.tracking.fit_link_params` fits its own
    population distribution over D: this one is a direct moment of the
    finished trajectories in physical units, computed here, so it is an
    independent read on the linker's output rather than a restatement of
    the linker's own prior. When the two disagree badly, that disagreement
    is the signal (see `run_track_step`'s `track_summary`)."""
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
    min_track_length: int = 2,
    drop_aggregates: bool = True,
    link_with_flux: bool = False,
    point_filters: Optional[FilterSpec] = None,
    track_filters: Optional[FilterSpec] = None,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> PipelineSession:
    """Link `session.points_df` into trajectories with
    `spotsolve.tracking`, then drop tracks shorter than
    `min_track_length`. Requires `session.points_df` from a prior
    `run_detect_step` call.

    Two calls, and no gate to tune: `fit_link_params` measures the
    population distribution over D, the per-frame detection continuity
    (`p_cont`) and the CRLB inflation factor (`se_inflate`) from this
    movie's own displacements, and `link` then scores each candidate link
    as a likelihood ratio under that fit, using each detection's own
    `se_y`/`se_x`. A dim spot therefore carries a genuinely wider gate
    than a bright one, which is the distinction that decides assignments
    in a dense field. The fitted `LinkParams` is kept on the session and
    summarized into `track_summary` for the record.

    Frame-to-frame only: a missed detection ENDS a track rather than being
    bridged, so trajectories fragment instead of swapping identity. That
    is why `min_track_length` matters more here than a gap-closing linker
    would need it to.

    `drop_aggregates` (default True) removes detections flagged
    `is_aggregate` by `run_detect_step` before linking, via
    `loctable.filter_aggregates`. `session.points_df` is left whole either
    way -- the filter applies to what the linker sees, not to what was
    saved, so the aggregate share stays auditable in `frames_df`.

    `link_with_flux` (default False) additionally scores each candidate
    link by brightness continuity -- each detection's `flux`/`se_flux` --
    on top of position and CRLB (`spotsolve.tracking.link`'s
    `brightness=True`). An extra cue for a crowded field where position
    alone leaves ambiguous assignments; off by default because it isn't a
    correction to a broken default, just a second signal to opt into.

    `point_filters` is the same idea generalized to any per-detection
    column (`{column: (lo, hi)}` -- see `filter_mask`): the Detect tab's
    histogram cuts, applied on top of the aggregate drop to decide what
    linking sees. `track_filters` applies to the linked result, keeping
    whole tracks by their `track_metrics_df` row (see
    `apply_track_filters`). Both leave `session.points_df` whole, for the
    same reason `drop_aggregates` does, and both are recorded in
    `session_manifest_extra` so what the run kept is readable off the
    bundle rather than only off whoever dragged the handle.

    `min_track_length` defaults to 2 (drop singletons only): a length-1
    "track" is just an unlinked detection with no displacement of its own
    -- it contributes nothing to `estimate_D_um2_s` (needs a
    frame-to-frame pair) or to any diffusionkit fit downstream, so keeping
    it around is pure clutter in the saved bundle and the tracks layer.

    `progress_callback`, if given, is called twice: `(0, 2, "fitting link
    parameters")` before the fit, `(1, 2, "linking")` before the link --
    neither has finer-grained progress of its own.

    `cancel_event`, if given, is checked before each of those two calls --
    each is one opaque Rust call with no interruption point of its own, so
    this stage can only be skipped before one starts, not interrupted
    mid-call (see `PipelineCancelled`'s docstring).
    """
    if session.points_df is None:
        raise ValueError("No points available -- run detect first.")
    _check_cancelled(cancel_event)

    def report(done: int, total: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(done, total, stage)

    points_df = session.points_df
    link_input = loctable.filter_aggregates(points_df, keep_flagged=not drop_aggregates)
    n_after_aggregates = link_input.height
    link_input = apply_filters(link_input, point_filters)
    if link_input.height == 0:
        if not points_df.height:
            raise ValueError("No detections to link -- run detect first.")
        if n_after_aggregates == 0:
            raise ValueError(
                "No linkable detections -- every detection was flagged as an aggregate "
                "(raise agg_ratio, or set drop_aggregates=False to link them anyway)."
            )
        raise ValueError(
            f"No linkable detections -- the point filters "
            f"({', '.join(sorted(point_filters or {}))}) rejected all "
            f"{n_after_aggregates} of them. Widen or clear them."
        )

    class_groups = _region_groups(link_input)
    report(0, 2, "fitting link parameters")
    # Fitted on everything even when linking per region: it is both the
    # fallback for a class too sparse to fit on its own and the pooled
    # numbers the top-level summary (and status line) report.
    link_params = tracking.fit_link_params(link_input)

    _check_cancelled(cancel_event)
    report(1, 2, "linking")
    by_class = None
    if class_groups is None:
        tracks_df = tracking.link(link_input, link_params, brightness=link_with_flux)
    else:
        # LinkParams are fitted per class (every cell's nucleus pooled, say):
        # classes are separated because their D and density differ, and a
        # pooled D prior would gate one class's steps by the other's, while
        # one region alone is usually too sparse to fit. Linking is then
        # done one region at a time, which is what guarantees no track
        # crosses a region boundary (nucleus -> cytoplasm, cell -> cell).
        class_areas = _class_areas_px(session)
        pieces, by_class, next_id = [], {}, 0
        for name, class_rows, instances in class_groups:
            class_params, fell_back = link_params, True
            if class_rows.height >= MIN_POINTS_FOR_CLASS_FIT:
                try:
                    class_params, fell_back = tracking.fit_link_params(class_rows), False
                except Exception:
                    pass
            class_pieces = []
            for rows in instances:
                linked = tracking.link(rows, class_params, brightness=link_with_flux)
                if linked.height:
                    linked = linked.with_columns(
                        (pl.col("track_id").cast(pl.Int64) + next_id).alias("track_id")
                    )
                    next_id = int(linked["track_id"].max()) + 1
                class_pieces.append(linked)
            linked = pl.concat(class_pieces, how="vertical_relaxed")
            kept = linked
            if min_track_length > 1 and kept.height:
                kept = kept.filter(pl.len().over("track_id") >= min_track_length)
            by_class[name] = {
                "n_regions": len(instances),
                "fell_back_to_pooled": fell_back,
                **_link_summary(session, class_rows, kept, class_params, class_areas.get(name)),
                "n_tracks": kept["track_id"].n_unique() if kept.height else 0,
            }
            pieces.append(linked)
        tracks_df = pl.concat(pieces, how="vertical_relaxed") if pieces else link_input.head(0)
    n_tracks_linked = tracks_df["track_id"].n_unique() if tracks_df.height else 0
    if min_track_length > 1:
        tracks_df = tracks_df.filter(pl.len().over("track_id") >= min_track_length)
    tracks_df = apply_track_filters(tracks_df, track_filters, session.pixel_size_um, session.dt_s)
    report(2, 2, "done")

    area_px = None
    if session.labels is not None:
        area_px = int((session.labels > 0).sum()) or None
    summary = _link_summary(session, link_input, tracks_df, link_params, area_px)

    if session.source_params:
        # Linking was just redone, so a reopened bundle's recorded link
        # numbers and cuts no longer describe this session -- only its
        # detect/calibration provenance still does.
        session.source_params = {
            key: value for key, value in session.source_params.items() if key not in _TRACK_KEYS
        }
    session.tracks_df = tracks_df
    session.link_params = link_params
    session.drop_aggregates_used = drop_aggregates
    session.link_with_flux_used = link_with_flux
    session.min_track_length_used = min_track_length
    session.point_filters_used = dict(point_filters) if point_filters else None
    session.track_filters_used = dict(track_filters) if track_filters else None
    session.track_summary = {
        **summary,
        "n_points_linked": link_input.height,
        "n_points_dropped_as_aggregate": points_df.height - n_after_aggregates,
        "n_points_dropped_by_filter": n_after_aggregates - link_input.height,
        "n_tracks_linked": n_tracks_linked,
        "by_class": by_class,
    }
    return session


# Below this many linkable detections a class's own `fit_link_params` is
# not attempted: the mixture fit over nearest-neighbour distances needs a
# population to fit, and a handful of points in a small region would give
# a prior worse than the pooled one it falls back on.
MIN_POINTS_FOR_CLASS_FIT = 50

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
    link_params,
    area_px: Optional[int] = None,
) -> dict:
    """The per-run linking numbers `track_summary` reports -- for the
    whole field, or for one region class's slice of it. `area_px` is the
    area the detections could have come from: the region(s) when there are any,
    else the full frame, so a density (and the crowding verdict built on
    it) isn't diluted by area nothing could be detected in."""
    # The linker's own CRLB, pooled, as one localization precision in
    # physical units -- what `estimate_D_um2_s` subtracts off and what the
    # resolvability check reasons about. `se_inflate` is applied because it
    # is the linker's fitted statement that the reported CRLB understates
    # the real error by that factor (in variance).
    se_y_um = link_input["se_y"].to_numpy() * session.pixel_size_um
    se_x_um = link_input["se_x"].to_numpy() * session.pixel_size_um
    sigma_loc_um = float(
        np.sqrt(link_params.se_inflate)
        * np.nanmedian(np.sqrt((se_y_um**2 + se_x_um**2) / 2.0))
    )
    D_est, n_links = estimate_D_um2_s(tracks_df, session.dt_s, session.pixel_size_um, sigma_loc_um)

    # The linker's own fitted population mean D, converted out of
    # px^2/frame into the same units as `D_est` so the two are comparable.
    px2_per_frame_to_um2_s = session.pixel_size_um**2 / session.dt_s
    D_link = link_params.d_mean * px2_per_frame_to_um2_s

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
        # `spotsolve.tracking.LinkParams`, for the record -- nothing here
        # was chosen, all of it was measured from this movie.
        "D_link_um2_s": D_link,
        "immobile_fraction": link_params.d_immobile,
        "p_cont": link_params.p_cont,
        "lam_birth_per_px2": link_params.lam_birth,
        "se_inflate": link_params.se_inflate,
        "density_um2": density_um2,
        "crowding_ratio": resolvability["ratio"],
        "resolvability_verdict": resolvability["verdict"],
        "resolvability_message": resolvability["message"],
    }


def session_manifest_extra(session: PipelineSession) -> dict:
    """Assemble the `manifest.json` `params` payload from a session that's
    been through all three stages -- meant to be passed as
    `results.build_manifest`'s `params`."""
    ts = session.track_summary or {}
    cs = session.calib_summary or {}
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
        "sigma_ci_px": [cs.get("sigma_ci_lo"), cs.get("sigma_ci_hi")] if cs else None,
        "sigma_converged": cs.get("converged"),
        "sigma_loc_um": ts.get("sigma_loc_um"),
        "D_est_um2_s": ts.get("D_est_um2_s"),
        "D_link_um2_s": ts.get("D_link_um2_s"),
        "immobile_fraction": ts.get("immobile_fraction"),
        "p_cont": ts.get("p_cont"),
        "lam_birth_per_px2": ts.get("lam_birth_per_px2"),
        "se_inflate": ts.get("se_inflate"),
        "n_linked_steps": ts.get("n_linked_steps"),
        "min_track_length": session.min_track_length_used,
        "drop_aggregates": session.drop_aggregates_used,
        "link_with_flux": session.link_with_flux_used,
        "agg_ratio": session.agg_ratio_used,
        "n_points_dropped_as_aggregate": ts.get("n_points_dropped_as_aggregate"),
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
        "calibration_kwargs": session.calibration_kwargs_used,
        "calibration_frame": session.calibration_frame_used,
        "frame_range": list(session.frame_range_used) if session.frame_range_used is not None else None,
        "spotsolve_version": spotsolve.__version__,
        # Per-class linking numbers (`run_track_step` links each region on
        # its own, with LinkParams fitted per class, when the table is
        # labeled with more than one) -- None for a single-field run.
        "track_summary_by_class": ts.get("by_class"),
    }
    # A session rebuilt from a saved bundle never recomputed what earlier
    # stages recorded (the calibration's CI, say): keep the bundle's own
    # value rather than overwrite it with None. Counts and versions are
    # always the current session's.
    for key, value in (session.source_params or {}).items():
        if params.get(key) is None and key not in _NEVER_INHERITED:
            params[key] = value
    return params


# Manifest keys `run_track_step` produces -- dropped from a rebuilt
# session's `source_params` as soon as it links again.
_TRACK_KEYS = frozenset({
    "sigma_loc_um", "D_est_um2_s", "D_link_um2_s", "immobile_fraction", "p_cont",
    "lam_birth_per_px2", "se_inflate", "n_linked_steps", "min_track_length",
    "drop_aggregates", "link_with_flux", "n_points_dropped_as_aggregate",
    "point_filters", "track_filters", "n_points_dropped_by_filter", "n_tracks_linked",
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
        spec = params.get(key)
        return {col: tuple(bounds) for col, bounds in spec.items()} if spec else None

    detect_kwargs = params.get("detect_kwargs")
    if detect_kwargs is not None:
        detect_kwargs = {
            key: tuple(value) if key in ("slack", "band") and value is not None else value
            for key, value in detect_kwargs.items()
        }
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
        calibration_kwargs_used=params.get("calibration_kwargs"),
        calibration_frame_used=params.get("calibration_frame"),
        points_df=points_df,
        camera_kwargs_used=params.get("camera_kwargs"),
        detect_kwargs_used=detect_kwargs,
        detector_used=params.get("detector"),
        agg_ratio_used=params.get("agg_ratio"),
        frame_range_used=tuple(frame_range) if frame_range is not None else None,
        labels=labels,
        regions=regions,
        source_params=params,
        point_filters_used=filters("point_filters"),
    )
    if has_tracks:
        session.tracks_df = tracks_df
        session.min_track_length_used = params.get("min_track_length")
        session.drop_aggregates_used = params.get("drop_aggregates")
        session.link_with_flux_used = params.get("link_with_flux")
        session.track_filters_used = filters("track_filters")
        session.track_summary = {key: params.get(key) for key in _TRACK_KEYS}
        session.track_summary["by_class"] = params.get("track_summary_by_class")
    if session.pixel_size_um is None or session.dt_s is None:
        raise ValueError(
            f"{Path(image_path).name}: the saved bundle records no pixel size / frame "
            "interval and the file has none either."
        )
    return session


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
    normalize here and keep the round-trip honest."""
    if not filters:
        return None
    return {col: [lo, hi] for col, (lo, hi) in filters.items()}


def _jsonable_detect_kwargs(detect_kwargs: Optional[dict]) -> Optional[dict]:
    """`detect_kwargs` with `slack`/`band` as lists rather than tuples --
    `json.dumps` writes both as arrays, but reading a manifest back gives
    lists either way, so normalize here and keep the round-trip honest."""
    if detect_kwargs is None:
        return None
    out = dict(detect_kwargs)
    for key in ("slack", "band"):
        if out.get(key) is not None:
            out[key] = list(out[key])
    return out


def run_detect_track(
    image_path: str | Path,
    pixel_size_um: Optional[float] = None,
    dt_s: Optional[float] = None,
    channel: int = 0,
    z_index: int = 0,
    params: Optional[DetectTrackParams] = None,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
    exposure_s: Optional[float] = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Run the full calibrate+detect+track pipeline on one timelapse in
    one call, composing `load_session`/`run_calibration_step`/
    `run_detect_step`/`run_track_step` -- for the headless CLI. See this
    module's docstring for the stepwise alternative the widget uses.

    `image_path` can be .tif/.tiff, .nd2, or .ims (see `io_formats.load_stack`).
    `channel`/`z_index` pick which plane to track for files with more than
    one (both default to 0). `pixel_size_um`/`dt_s` fall back to the
    file's own metadata if not given explicitly, and so does `exposure_s`
    (recorded in the manifest for the diffusion analysis; never required).

    `cancel_event`, if given, is forwarded to each stage -- see
    `PipelineCancelled`'s docstring for what "cancelled" actually means per
    stage (cooperative, frame-granular for detect, stage-boundary-only for
    calibration/track).

    Returns (points_df, tracks_df, manifest_extra) -- `manifest_extra` is
    meant to be passed as `results.build_manifest`'s `params`.
    """
    params = params or DetectTrackParams()
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
    total_steps = n_frames + 3

    def report(done: int, total: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(done, total, stage)

    # An explicit `params.sigma` means the width was already settled --
    # interactively, off a preview frame's `fit_sigma` histogram (see
    # `run_preview_frame`) -- so there is nothing to measure. Only the
    # unattended case, where nobody is looking at a histogram, falls back
    # to `calibrate_sigma`.
    if params.sigma is not None:
        session.sigma = params.sigma
    else:
        report(0, total_steps, "calibrating sigma")
        run_calibration_step(
            session,
            params.sigma_init,
            params.calibration_kwargs,
            camera_kwargs=params.camera_kwargs,
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
        camera_kwargs=params.camera_kwargs,
        detect_kwargs=params.detect_kwargs,
        agg_ratio=params.agg_ratio,
        frame_range=params.frame_range,
        progress_callback=detect_progress,
        cancel_event=cancel_event,
        n_threads=params.n_threads,
        detector=params.detector,
    )

    def track_progress(done: int, total: int, stage: str) -> None:
        report(n_frames + 1 + done, total_steps, stage)

    run_track_step(
        session,
        params.min_track_length,
        drop_aggregates=params.drop_aggregates,
        link_with_flux=params.link_with_flux,
        point_filters=params.point_filters,
        track_filters=params.track_filters,
        progress_callback=track_progress,
        cancel_event=cancel_event,
    )

    return session.points_df, session.tracks_df, session_manifest_extra(session)
