"""Pipeline parameter form for the interactive Run action, laid out as
two tabbed pages -- **Detect** and **Track** -- each of which is a small
detect -> filter -> finalize flow of its own, the shape Imaris and
TrackMate use for spot detection:

    Detect:  Camera -> PSF width -> Detect -> Filter
             (camera + PSF + detection knobs, [Preview frame] /
             [Run detect], then histogram filters on what it found)
    Track:   Link -> Filter -> Save
             (linking knobs, [Run tracking], histogram filters on the
             tracks it linked, then [Save results])

Each arrow there is a page, not a scroll: within a tab the steps are a
`qtkit.StepPager` -- one step visible at a time, `‹`/`›` to leaf between them,
the title a menu to jump. The dock is narrow and short, only one step is
ever being tuned, and a stacked column put that step's Run button
off-screen as often as not. See `qtkit.StepPager` for why the height actually
shrinks (a stacked widget otherwise reserves its tallest page). Stage
completions leaf for you: `ExperimentListWidget` calls
`show_tab("Detect", "Filter")` when a detect run lands, and
`show_tab("Track", "Filter")` after linking.

The filters are `widgets/feature_filters.FeatureFilterPanel` in both
places, and neither deletes anything: a filter decides what the next stage
sees (linking, for the Detect tab's) and what the saved bundle keeps
(tracks.parquet, for the Track tab's), exactly as `drop_aggregates`
already did for flagged aggregates. `points.parquet` holds every detection
either way and `manifest.json` records the ranges, so what a run kept is
readable off the bundle afterwards.

**There is no Calibration tab.** `spotsolve.calibrate_sigma`'s loop was:
localize a frame with the reporting band off, take the median fitted
width, re-run at it, repeat to a fixed point -- and it reported one number
with a bootstrap CI, with the distribution behind it never shown. The
Detect tab's "Preview frame" button is that same loop with the user in it:
preview at the current sigma, look at the `fit_sigma` histogram, click
"Use" to adopt its median, preview again. It settles in the same two or
three rounds and the distribution is on screen throughout, so a bimodal or
ragged `fit_sigma` -- two focal planes, junk being fitted as signal -- is
something you see rather than something a median averages away.
`pipeline.run_calibration_step` is still there for the headless path,
where there is nobody to look; the "calibrate per file" checkbox next to
`sigma` is what routes a multi-file batch run back to it.

The camera group moved here with it. It was on the Calibration tab because
"what is this camera" is a calibration question, but it is read by
preview, detect and calibrate alike -- and with calibration no longer a
tab of its own, the one tab that always needs it is this one.
`PipelineParamsWidget.get_camera_kwargs` remains the single source of
truth that `ExperimentListWidget` forwards to every stage.

A note on units, since two different ones are in play: `sigma` is in
**pixels**, while `slack` and `band` are **multiples of whatever sigma the
search is running at** -- `spotsolve` reports `sigma_ratio = fit_sigma /
sigma` and rejects a fit when that ratio leaves `band`. They are not
multiples of the *initial* guess, and they are not absolute pixels, so
moving `sigma` moves both windows with it. The expert section therefore
prints the current px window under those two rows, recomputed whenever
`sigma` changes.

Each tab shows only the handful of knobs that matter for day-to-day
tuning; the rest collapse under a per-tab "Expert settings"
(`qtkit.CollapsibleSection`, collapsed by default). Tabs across stages and
pages within one keep the dock panel's height bounded to a single step
rather than to the whole form.

Why this form is so much smaller than the sfwloc-era one it replaces: that
pipeline offered three detectors, each with its own ~20-key solver-kwargs
dict, and a linker with a hand-tuned bootstrap gate. `spotsolve` has one
detector whose decision rule is fixed (an emitter exists iff it lowers the
box's Poisson deviance by a set number of nats) and a linker with no dials
at all, so what's left to expose is genuinely the camera, the PSF width,
and the reporting band -- physical facts about the instrument rather than
solver tuning. A knob that isn't here is not hidden; it doesn't exist.

Each tab owns its stage's "Run" button and a one-line status label, wired
to `PipelineParamsWidget`'s `previewRequested`/`detectRequested`/
`trackRequested`/`saveRequested` signals -- `ExperimentListWidget` connects
these to run that single stage (via `pipeline.py`'s `run_preview_frame`/
`run_detect_step`/`run_track_step`) against whatever the earlier stages
already produced, instead of always re-running the whole pipeline as one
atomic unit. The status labels are read-only; the widget calls
`set_preview_status`/`set_detect_status`/`set_track_status` back with each
stage's result.

The Detect tab also exposes *scope* controls -- which frames and which
pixels get analyzed, not how -- kept in its core section since these are
exactly what make "explore one image incrementally" (this widget's whole
point, vs. blindly running a batch job) practical: a frame-range pair
(`get_frame_range`) and a "restrict to ROI" checkbox
(`get_use_roi_mask`), plus a Shapes-layer dropdown (`get_roi_layer_name`)
and a "Draw ROI…" button (`newRoiRequested`) that just asks for a fresh
Shapes layer to draw on. A "cores" spinbox (`get_n_threads`) sits beside
the frame range -- `localize_stack`'s `n_threads`, defaulted to every
core (see `pipeline.run_detect_step`'s docstring for why this speeds up
even the interactively-watched run, not just a headless batch).

The dropdown is what makes several ROIs on screen at once workable: draw
as many Shapes layers as you like (rename them in napari's layer list --
the name shown here follows, and it's the name the ROI is saved under,
see `spt_pipeline.rois`), then pick which one this run is restricted to.
It replaced "whichever Shapes layer happens to be active", where the
targeted region silently changed whenever the layer selection did -- and
where a second ROI could only be used by clicking the right layer first.

This widget stays viewer-agnostic (no napari `Viewer` reference), so
`ExperimentListWidget` (which owns the viewer) is responsible for adding
that layer (persistent/2D, transparent fill, polygon-lasso tool active --
see `_on_new_roi_requested`), for keeping the dropdown's contents in sync
with the viewer's Shapes layers (`set_roi_choices`), and, once the ROI
checkbox is checked, for turning the named layer into the boolean mask
array `spotsolve`'s `roi` argument expects.
"""

from __future__ import annotations

import os
from typing import Optional

import polars as pl
import spotsolve
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from qtkit import CollapsibleSection, StepPager, double_spinbox, hline, note_label, style_status_label, wrapping_label

from spt_pipeline.pipeline import (
    DEFAULT_CALIBRATION_KWARGS,
    DEFAULT_CAMERA_KWARGS,
    DEFAULT_DETECT_KWARGS,
    TRACK_METRIC_COLUMNS,
    DetectTrackParams,
    FilterSpec,
)
from spt_pipeline.widgets.feature_filters import FeatureFilterPanel


def _ispin(value: int, minimum: int, maximum: int, tooltip: str) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setToolTip(tooltip)
    return box


def _expert_section(form: QFormLayout) -> CollapsibleSection:
    body = QWidget()
    body.setLayout(form)
    return CollapsibleSection("Expert settings", body)


def _run_row(button_text: str) -> tuple[QPushButton, QLabel, QHBoxLayout]:
    """A "Run <stage>" button + a status label sharing one row."""
    button = QPushButton(button_text)
    status = wrapping_label("")
    style_status_label(status)
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(button)
    row.addWidget(status, stretch=1)
    return button, status, row


class _CameraFields(QWidget):
    """The camera's own calibration: `offset` (ADU), `gain` (ADU per
    photoelectron) and `read_noise` (electrons rms). Not tuning knobs --
    these convert the raw frame into the photoelectron counts
    `spotsolve`'s Poisson likelihood is written in, so they are facts
    about the chip that every stage must agree on.

    `gain` has an "estimate per frame" checkbox, checked by default,
    because that's what makes the detector usable on an uncharacterized
    camera. It is a fallback and not a preference: a per-frame estimate
    moves with the sample, which quietly rescales every flux in the movie
    along with it. Uncheck it and enter a measured gain when you have one
    -- the tooltip says so, since nothing in the output makes an estimated
    gain look different from a measured one."""

    def __init__(self, defaults: dict) -> None:
        super().__init__()
        self.offset = double_spinbox(
            defaults["offset"], 0.0, 1e6, 1.0, decimals=2,
            tooltip="Camera offset / baseline, ADU -- subtracted before the\n"
            "photoelectron conversion.",
        )
        self.read_noise = double_spinbox(
            defaults["read_noise"], 0.0, 1e4, 0.1, decimals=3,
            tooltip="Read noise, electrons rms -- the variance floor added to\n"
            "the Poisson term.",
        )
        self.gain = double_spinbox(
            1.0, 1e-6, 1e6, 0.1, decimals=4,
            tooltip="Gain, ADU per photoelectron. Prefer a measured value:\n"
            "a per-frame estimate drifts with the sample and rescales every\n"
            "reported flux with it.",
        )
        self.estimate_gain = QCheckBox("estimate per frame")
        self.estimate_gain.setToolTip(
            "Estimate gain from each frame instead of using a fixed value.\n"
            "Usable default on an uncharacterized camera, but a measured gain\n"
            "is strictly better -- see the gain field's tooltip."
        )
        self.estimate_gain.setChecked(defaults["gain"] is None)
        if defaults["gain"] is not None:
            self.gain.setValue(defaults["gain"])
        self.estimate_gain.toggled.connect(self._on_estimate_gain_toggled)
        self._on_estimate_gain_toggled(self.estimate_gain.isChecked())

        gain_row = QHBoxLayout()
        gain_row.setContentsMargins(0, 0, 0, 0)
        gain_row.addWidget(self.gain)
        gain_row.addWidget(self.estimate_gain)

        form = QFormLayout()
        form.setContentsMargins(4, 4, 4, 4)
        form.addRow("offset (ADU)", self.offset)
        form.addRow("gain (ADU/e-)", gain_row)
        form.addRow("read noise (e- rms)", self.read_noise)
        self.setLayout(form)

    def _on_estimate_gain_toggled(self, checked: bool) -> None:
        self.gain.setEnabled(not checked)

    def get_kwargs(self) -> dict:
        return dict(
            offset=self.offset.value(),
            gain=None if self.estimate_gain.isChecked() else self.gain.value(),
            read_noise=self.read_noise.value(),
        )


class _DetectTab(QWidget):
    """Four pages -- Camera, PSF width, Detect, Filter -- leafed through
    with the `qtkit.StepPager` header: the camera, the PSF width, the
    `spotsolve.localize` knobs plus scope and the two actions that produce
    something to look at (preview one frame / run the range), then a
    filter stack over what they found.

    One consequence of paging worth knowing: the preview loop's two
    halves now sit on neighbouring pages -- "Use" on PSF width, the
    `fit_sigma` histogram on Filter. The loop still closes without
    leafing, because `set_preview_result` puts the median (and its MAD) in
    the status line right under the button and on the button's own label;
    page across when what you want is the *shape* of that distribution --
    a bimodal `fit_sigma` is the thing a median hides, and seeing it is
    the whole reason this is a preview loop rather than
    `calibrate_sigma`.

    Core: `sigma` with its preview loop (see this module's docstring --
    this is what replaced the Calibration tab), `k_max` (most emitters one
    box may be fitted with jointly -- the crowding ceiling), the seed and
    birth thresholds, the aggregate cut, a frame range and a "restrict to ROI"
    checkbox. Expert: `slack` and `band`, the width ranges a fit may take
    and be reported at, both as multiples of `sigma` and echoed in px.

    There is no sparsity weight, iteration budget or refinement schedule
    to set: the search's accept/reject rule is a fixed deviance
    improvement, and it runs to its own convergence."""

    previewRequested = Signal()
    runRequested = Signal()
    cancelRequested = Signal()
    newRoiRequested = Signal()
    roiLayerChanged = Signal()
    filtersChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._running = False
        self._measured_sigma: Optional[float] = None
        self._camera = _CameraFields(DEFAULT_CAMERA_KWARGS)
        d = DEFAULT_DETECT_KWARGS
        slack_lo, slack_hi = d["slack"]
        band_lo, band_hi = d["band"]

        # --- PSF width, and the preview loop that measures it ----------
        self.sigma = double_spinbox(
            1.3, 0.3, 10.0, 0.1, decimals=3,
            tooltip="In-focus PSF sigma in PIXELS -- the width the search runs\n"
            "at. Each emitter still gets its own fitted width (fit_sigma);\n"
            "this sets where the search starts and what slack/band are\n"
            "multiples of.\n\n"
            "Don't agonize: preview a frame, read the median fit_sigma off\n"
            "the histogram, click Use, preview again. Two or three rounds is\n"
            "the whole of what calibrate_sigma used to do out of sight.",
        )
        self.sigma.valueChanged.connect(self._update_band_note)
        self.calibrate_per_file = QCheckBox("calibrate per file")
        self.calibrate_per_file.setChecked(True)
        self.calibrate_per_file.setToolTip(
            "Batch runs only (the experiment list's \"Run selected\"): measure\n"
            "sigma separately for each file with spotsolve.calibrate_sigma,\n"
            "taking the value on the left as that fit's starting guess.\n"
            "On by default -- a folder of acquisitions need not share one\n"
            "focus, and nobody is watching a batch run's histograms.\n\n"
            "Preview and Run detect below always use the value on the left\n"
            "exactly as entered, checked or not."
        )
        sigma_row = QHBoxLayout()
        sigma_row.setContentsMargins(0, 0, 0, 0)
        sigma_row.addWidget(self.sigma)
        sigma_row.addWidget(self.calibrate_per_file)

        self._preview_frame = _ispin(
            0, 0, 1_000_000,
            tooltip="Stack frame (0-based) to preview. Frame 0 isn't always the\n"
            "best reference -- somewhere sparser or better-focused often is.",
        )
        self.preview_button = QPushButton("Preview frame")
        self.preview_button.setToolTip(
            "Localize this one frame at the current sigma with the reporting\n"
            "band OFF, so every fit shows up -- including the ones a real run\n"
            "would bin as out-of-band. Nothing is saved and points from a\n"
            "previous Run detect are left alone."
        )
        self.preview_button.clicked.connect(self.previewRequested.emit)
        self.use_measured_button = QPushButton("Use")
        self.use_measured_button.setEnabled(False)
        self.use_measured_button.setToolTip(
            "Adopt the previewed frame's median fit_sigma as sigma -- one\n"
            "round of calibrate_sigma's fixed-point loop. Preview again to\n"
            "take the next; it stops moving after two or three."
        )
        self.use_measured_button.clicked.connect(self._use_measured_sigma)
        preview_row = QHBoxLayout()
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.addWidget(self.preview_button)
        preview_row.addWidget(self._preview_frame)
        preview_row.addWidget(self.use_measured_button)
        preview_row.addStretch()

        self.preview_status = wrapping_label("")
        style_status_label(self.preview_status)

        psf_form = QFormLayout()
        psf_form.setContentsMargins(0, 0, 0, 0)
        psf_form.addRow("sigma (px)", sigma_row)

        psf_body = QWidget()
        psf_layout = QVBoxLayout(psf_body)
        psf_layout.setContentsMargins(0, 0, 0, 0)
        psf_layout.setSpacing(2)
        psf_layout.addLayout(psf_form)
        psf_layout.addLayout(preview_row)
        psf_layout.addWidget(self.preview_status)

        # --- detection knobs -------------------------------------------
        self.k_max = _ispin(
            d["k_max"], 1, 64,
            tooltip="Most emitters one box may be fitted with jointly. The\n"
            "crowding ceiling: raise it for very dense fields, at some cost.",
        )
        # Two cuts on the same LoG z-statistic, failing in opposite
        # directions (see `DEFAULT_DETECT_KWARGS`): the seed cut decides
        # what gets searched and is derived per frame by default; the birth
        # cut decides what gets tried inside a box and is a constant.
        self.seed_threshold = double_spinbox(
            4.0, 0.0, 1e9, 0.1, decimals=2,
            tooltip="Which peaks in the frame get a box searched around them, in sd\n"
            "of the LoG filter's noise. Loose costs only time -- a seed still has\n"
            "to pass the 10-nat test to become a detection -- while too strict\n"
            "loses light that is never fitted. Leave 'derive from frame' checked\n"
            "unless you have a reason.",
        )
        self.derive_seed_threshold = QCheckBox("derive from frame")
        self.derive_seed_threshold.setToolTip(
            "Let the detector derive the seed cut from the frame size\n"
            "(spotsolve's seed_threshold=None). Recommended."
        )
        self.derive_seed_threshold.setChecked(d["seed_threshold"] is None)
        if d["seed_threshold"] is not None:
            self.seed_threshold.setValue(d["seed_threshold"])
        self.derive_seed_threshold.toggled.connect(self._on_derive_seed_threshold_toggled)
        self._on_derive_seed_threshold_toggled(self.derive_seed_threshold.isChecked())
        seed_threshold_row = QHBoxLayout()
        seed_threshold_row.setContentsMargins(0, 0, 0, 0)
        seed_threshold_row.addWidget(self.seed_threshold)
        seed_threshold_row.addWidget(self.derive_seed_threshold)

        self.birth_threshold = double_spinbox(
            d["birth_threshold"], 0.0, 1e9, 0.1, decimals=2,
            tooltip="How strong a leftover residual peak inside a box must be before\n"
            "another emitter is tried there, in sd of the LoG filter's noise.\n"
            "Loose costs precision (false neighbours around bright spots) and\n"
            "time. Raise toward 4 for speed; lower toward 2.5 on faint, sparse data.",
        )

        # Over-bright cut, relative to each frame's own median detection --
        # relative so that one number survives bleaching and illumination
        # drift over a long movie. Detections above it are flagged, not
        # deleted (see pipeline.run_detect_step); the Track tab decides
        # whether linking sees them.
        self.agg_ratio = double_spinbox(
            spotsolve.AGG_AMP_RATIO, 1.0, 1e6, 1.0, decimals=2,
            tooltip="Flag a detection as an aggregate when its flux exceeds this\n"
            "multiple of the frame's median detection. Flagged, never deleted --\n"
            "the Track tab decides whether linking sees them.",
        )

        core_form = QFormLayout()
        core_form.setContentsMargins(0, 0, 0, 0)
        core_form.addRow("k_max", self.k_max)
        core_form.addRow("seed threshold", seed_threshold_row)
        core_form.addRow("birth threshold", self.birth_threshold)
        core_form.addRow("aggregate ratio", self.agg_ratio)

        # What gets analyzed, not how -- kept in core (not expert) since
        # these are exactly the knobs that let one image be explored
        # incrementally (a few frames, a drawn ROI) instead of always
        # committing to the whole stack. `set_frame_bounds` is called by
        # `ExperimentListWidget` once an image's frame count is known;
        # 0/0 means "process everything" (the common case, and the only
        # sane default before any image has been selected).
        self._max_frames: Optional[int] = None
        self.frame_start = _ispin(0, 0, 1_000_000, tooltip="First frame to process (0-based).")
        self.frame_end = _ispin(
            0, 0, 1_000_000,
            tooltip="Frame to stop before (exclusive). 0 = through the last frame.",
        )
        self.frame_end.setSpecialValueText("last")
        frame_row = QHBoxLayout()
        frame_row.setContentsMargins(0, 0, 0, 0)
        frame_row.addWidget(QLabel("frames"))
        frame_row.addWidget(self.frame_start)
        frame_row.addWidget(QLabel("to"))
        frame_row.addWidget(self.frame_end)
        frame_row.addStretch()

        # How many native threads `localize_stack` hands frames to (both
        # the batch path and, chunked, the interactively-watched one --
        # see `pipeline.run_detect_step`'s docstring). Capped at the
        # machine's own core count; defaulting to it is what "use every
        # core" means without a spinbox arrow-key marathon to get there.
        cpu_count = os.cpu_count() or 1
        self.n_threads = _ispin(
            cpu_count, 1, cpu_count,
            tooltip="Worker threads spotsolve's localize_stack hands frames to.\n"
            "Defaults to every core on this machine. Lower it to leave some\n"
            "cores free for other work while a long run is going.",
        )
        cores_row = QHBoxLayout()
        cores_row.setContentsMargins(0, 0, 0, 0)
        cores_row.addWidget(QLabel("cores"))
        cores_row.addWidget(self.n_threads)
        cores_row.addStretch()

        self.use_roi_mask = QCheckBox("Restrict to ROI")
        self.use_roi_mask.setToolTip(
            "Only place emitters inside the shape(s) on the Shapes layer picked\n"
            "in the dropdown (spotsolve's roi argument). Draw one with the button\n"
            "next to it, or in napari, first."
        )
        # Which Shapes layer, chosen by name rather than by whatever
        # happens to be selected in napari's layer list -- see this
        # module's docstring. Kept in sync by `set_roi_choices`; the entry
        # text is the live layer name, so renaming a layer in napari
        # renames it here (and that name is what the ROI is saved under).
        self.roi_layer = QComboBox()
        self.roi_layer.setToolTip(
            "Which Shapes layer to use as the ROI. Draw several and rename them\n"
            "in napari's layer list to keep more than one region around; only the\n"
            "one selected here restricts the run."
        )
        self.roi_layer.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.roi_layer.currentTextChanged.connect(self.roiLayerChanged)
        self.new_roi_button = QPushButton("Draw ROI…")
        self.new_roi_button.setToolTip(
            "Add a new Shapes layer (transparent fill, polygon-lasso tool active)\n"
            "for drawing the ROI -- 2D so it stays visible on every frame instead\n"
            "of only the one it was drawn on. Each click adds another, so several\n"
            "regions can be kept side by side and picked between."
        )
        self.new_roi_button.clicked.connect(self.newRoiRequested.emit)
        roi_row = QHBoxLayout()
        roi_row.setContentsMargins(0, 0, 0, 0)
        roi_row.addWidget(self.use_roi_mask)
        roi_row.addWidget(self.roi_layer, 1)
        roi_row.addWidget(self.new_roi_button)
        self._sync_roi_row()
        self.use_roi_mask.toggled.connect(self._sync_roi_row)

        # `slack` is the width range a fit may take; `band` the narrower
        # range actually reported as a detection. Both are multiples of
        # `sigma`, not absolute pixels -- spotsolve compares them against
        # `sigma_ratio = fit_sigma / sigma` -- so `_update_band_note`
        # prints what they currently come to in px. A fit outside `band` is
        # an out-of-band reject (too narrow / too wide / edge), which is how
        # out-of-focus and non-PSF-shaped junk stays out of the table;
        # `frames_df` counts them per frame.
        self.slack_lo = double_spinbox(
            slack_lo, 0.1, 10.0, 0.05, decimals=3,
            tooltip="Narrowest width a fit may take, as a MULTIPLE OF SIGMA.",
        )
        self.slack_hi = double_spinbox(
            slack_hi, 0.1, 20.0, 0.05, decimals=3,
            tooltip="Widest width a fit may take, as a MULTIPLE OF SIGMA.",
        )
        self.band_lo = double_spinbox(
            band_lo, 0.1, 10.0, 0.05, decimals=3,
            tooltip="Narrowest width reported as a detection, as a MULTIPLE OF\n"
            "SIGMA -- spotsolve tests it against sigma_ratio = fit_sigma/sigma.",
        )
        self.band_hi = double_spinbox(
            band_hi, 0.1, 20.0, 0.05, decimals=3,
            tooltip="Widest width reported as a detection, as a MULTIPLE OF\n"
            "SIGMA -- spotsolve tests it against sigma_ratio = fit_sigma/sigma.",
        )
        for box in (self.slack_lo, self.slack_hi, self.band_lo, self.band_hi):
            box.valueChanged.connect(self._update_band_note)
        self.no_band = QCheckBox("report every fit (no band)")
        self.no_band.setToolTip(
            "Report every fit regardless of width (spotsolve's band=None).\n"
            "A diagnostic: it puts out-of-focus and non-PSF-shaped fits into\n"
            "points_df as if they were detections. Leave unchecked for analysis.\n"
            "Preview frame always does this, whatever this box says."
        )
        self.no_band.toggled.connect(self._on_no_band_toggled)

        slack_row = QHBoxLayout()
        slack_row.setContentsMargins(0, 0, 0, 0)
        slack_row.addWidget(self.slack_lo)
        slack_row.addWidget(QLabel("to"))
        slack_row.addWidget(self.slack_hi)
        band_row = QHBoxLayout()
        band_row.setContentsMargins(0, 0, 0, 0)
        band_row.addWidget(self.band_lo)
        band_row.addWidget(QLabel("to"))
        band_row.addWidget(self.band_hi)

        self._band_note = note_label("")
        expert_form = QFormLayout()
        expert_form.setContentsMargins(0, 0, 0, 0)
        expert_form.addRow("slack (x sigma)", slack_row)
        expert_form.addRow("band (x sigma)", band_row)
        expert_form.addRow("", self._band_note)
        expert_form.addRow("", self.no_band)
        self._update_band_note()

        self.run_button, self.status_label, run_row = _run_row("Run detect")
        self.run_button.clicked.connect(self._on_run_button_clicked)

        # --- filter -----------------------------------------------------
        self.filters = FeatureFilterPanel(
            noun="points",
            hint="Preview a frame or run detect first — then filter on what it found.",
        )
        self.filters.filtersChanged.connect(self.filtersChanged)

        detect_body = QWidget()
        detect_layout = QVBoxLayout(detect_body)
        detect_layout.setContentsMargins(0, 0, 0, 0)
        detect_layout.setSpacing(2)
        detect_layout.addLayout(core_form)
        detect_layout.addWidget(hline())
        detect_layout.addLayout(frame_row)
        detect_layout.addLayout(cores_row)
        detect_layout.addLayout(roi_row)
        detect_layout.addWidget(_expert_section(expert_form))
        detect_layout.addWidget(hline())
        detect_layout.addLayout(run_row)

        self.pager = StepPager()
        self.pager.add_page("Camera", self._camera)
        self.pager.add_page("PSF width", psf_body)
        self.pager.add_page("Detect", detect_body)
        self.pager.add_page("Filter", self.filters)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.pager)
        layout.addStretch()
        self.setLayout(layout)

    def show_step(self, title: str) -> None:
        self.pager.show_step(title)

    # -- PSF / preview ----------------------------------------------------

    def _update_band_note(self) -> None:
        """Restate `slack`/`band` in pixels at the current `sigma`. They
        are ratios against the working sigma, not absolute widths, so this
        line is the only place the actual px window a fit is being judged
        against is visible -- and it moves whenever sigma does."""
        sigma = self.sigma.value()
        band = (
            "band off (every fit reported)"
            if self.no_band.isChecked()
            else f"reported {self.band_lo.value() * sigma:.2f}–{self.band_hi.value() * sigma:.2f} px"
        )
        self._band_note.setText(
            f"× sigma, not px: at sigma = {sigma:.3f} px a fit may take "
            f"{self.slack_lo.value() * sigma:.2f}–{self.slack_hi.value() * sigma:.2f} px, {band}."
        )

    def _use_measured_sigma(self) -> None:
        if self._measured_sigma is not None:
            self.sigma.setValue(self._measured_sigma)

    def set_preview_result(self, summary: Optional[dict], level: str = "neutral") -> None:
        """Report a `pipeline.run_preview_frame` summary -- the fit count,
        the median fitted width that "Use" would adopt, and how many of
        those fits the current band would actually have reported."""
        self._measured_sigma = (summary or {}).get("fit_sigma_median")
        self.use_measured_button.setEnabled(self._measured_sigma is not None)
        if not summary:
            self.use_measured_button.setText("Use")
            self.set_preview_status("", level)
            return
        median = summary.get("fit_sigma_median")
        self.use_measured_button.setText(f"Use {median:.3f}" if median is not None else "Use")
        parts = [f"{summary.get('n_fits', 0)} fits"]
        if median is not None:
            mad = summary.get("fit_sigma_mad")
            spread = f" ± {mad:.3f}" if mad is not None else ""
            parts.append(f"median fit_sigma {median:.3f}{spread} px")
        if summary.get("band") is not None:
            parts.append(f"{summary.get('n_in_band', 0)} in band")
        if summary.get("n_flagged"):
            parts.append(f"{summary['n_flagged']} aggregate")
        self.set_preview_status("  ·  ".join(parts), level)

    def set_preview_status(self, text: str, level: str = "neutral") -> None:
        style_status_label(self.preview_status, level)
        self.preview_status.setText(text)

    def get_sigma(self) -> float:
        return self.sigma.value()

    def get_calibrate_per_file(self) -> bool:
        return self.calibrate_per_file.isChecked()

    def get_preview_frame_index(self) -> int:
        return self._preview_frame.value()

    # -- detect -----------------------------------------------------------

    def _on_derive_seed_threshold_toggled(self, checked: bool) -> None:
        self.seed_threshold.setEnabled(not checked)

    def _on_no_band_toggled(self, checked: bool) -> None:
        self.band_lo.setEnabled(not checked)
        self.band_hi.setEnabled(not checked)
        self._update_band_note()

    def _on_run_button_clicked(self) -> None:
        if self._running:
            self.cancelRequested.emit()
        else:
            self.runRequested.emit()

    def set_running(self, running: bool) -> None:
        """Repurposes the run button into a Cancel button for the duration
        of a run, mirroring the batch Run/Cancel toggle on the experiment
        list's own run button -- see `ExperimentListWidget._cancel_active_run`
        for what cancelling actually does (cooperative, not instant). Deliberately
        doesn't embed the running file's name in the button text: a QPushButton
        can't wrap, so an unbounded filename here would force the button --
        and the row/dock around it -- wider, same as the bug fixed on
        `ExperimentListWidget.progress_label`. The experiment list's own
        progress label already shows which file is running."""
        self._running = running
        self.run_button.setText("Run detect" if not running else "Cancel")
        self.run_button.setEnabled(True)
        self.preview_button.setEnabled(not running)
        if running:
            # A previous run may have left this label green/amber/red
            # (set_status's level) -- reset to neutral so in-flight
            # progress text doesn't read as a stale result.
            style_status_label(self.status_label)

    def set_status(self, text: str, level: str = "neutral") -> None:
        style_status_label(self.status_label, level)
        self.status_label.setText(text)

    def set_progress(self, done: int, total: int, stage: str) -> None:
        """Live frame-by-frame progress while a detect run is active --
        shown right next to the Run/Cancel button, since this row stays in
        view even when the dock's own progress bar has scrolled out of
        sight."""
        self.status_label.setText(f"{stage} -- {done}/{total}" if total else stage)

    def set_frame_bounds(self, n_frames: int) -> None:
        """Called once an image's frame count is known -- clamps the
        frame-range and preview-frame spinboxes' maxima without disturbing
        values the user already chose (Qt clamps the current value down
        automatically if it now exceeds the new maximum)."""
        self._max_frames = n_frames
        self.frame_start.setMaximum(max(n_frames - 1, 0))
        self.frame_end.setMaximum(n_frames)
        self._preview_frame.setMaximum(max(n_frames - 1, 0))

    def get_frame_range(self) -> Optional[tuple[int, int]]:
        """`(start, end)`, or `None` for "whole stack" (both spinboxes at
        their default 0). `end` is passed through as entered -- including
        its `0`/"last" sentinel value -- rather than resolved here against
        `self._max_frames`, which may not be set yet (e.g. a queued batch
        item whose image nothing has loaded on this widget). Resolving
        `end <= 0` into the real last frame is `pipeline._resolve_frame_
        range`'s job: it always has the actual stack length in hand."""
        start = self.frame_start.value()
        end = self.frame_end.value()
        if start == 0 and end == 0:
            return None
        return (start, end)

    def get_n_threads(self) -> int:
        return self.n_threads.value()

    def get_use_roi_mask(self) -> bool:
        return self.use_roi_mask.isChecked()

    def get_roi_layer_name(self) -> Optional[str]:
        """Name of the Shapes layer the ROI checkbox targets, or None if
        the viewer has no Shapes layer to target."""
        name = self.roi_layer.currentText()
        return name or None

    def set_roi_choices(self, names: list[str], preferred: Optional[str] = None) -> None:
        """Replace the dropdown's entries with the viewer's current Shapes
        layer names, selecting `preferred` (the caller's own record of
        which *layer* is targeted, which is how a rename keeps its
        selection instead of jumping elsewhere), else the current text if
        it survived, else the last entry -- the newest layer, since
        `ExperimentListWidget` passes them in layer-list order.

        Signals are blocked across the rebuild so this can't be mistaken
        for the user picking something: `roiLayerChanged` is meant to fire
        only when they actually change the target."""
        current = self.roi_layer.currentText()
        blocked = self.roi_layer.blockSignals(True)
        self.roi_layer.clear()
        self.roi_layer.addItems(names)
        for candidate in (preferred, current):
            if candidate in names:
                self.roi_layer.setCurrentText(candidate)
                break
        else:
            if names:
                self.roi_layer.setCurrentIndex(len(names) - 1)
        self.roi_layer.blockSignals(blocked)
        self._sync_roi_row()

    def _sync_roi_row(self) -> None:
        """Grey the dropdown out unless the ROI checkbox is on and there
        is something to pick, so an empty/irrelevant picker doesn't read
        as a setting that is doing something."""
        self.roi_layer.setEnabled(self.use_roi_mask.isChecked() and self.roi_layer.count() > 0)

    def get_agg_ratio(self) -> float:
        return self.agg_ratio.value()

    def get_camera_kwargs(self) -> dict:
        return self._camera.get_kwargs()

    def get_detect_kwargs(self) -> dict:
        return dict(
            k_max=self.k_max.value(),
            seed_threshold=(
                None if self.derive_seed_threshold.isChecked() else self.seed_threshold.value()
            ),
            birth_threshold=self.birth_threshold.value(),
            slack=(self.slack_lo.value(), self.slack_hi.value()),
            band=None if self.no_band.isChecked() else (self.band_lo.value(), self.band_hi.value()),
        )


class _TrackingTab(QWidget):
    """Three pages -- Link, Filter, Save: what little the linker leaves to
    the caller, then a filter stack over the tracks it produced, then the
    save that finalizes the bundle.

    `spotsolve.tracking.fit_link_params` measures the population
    distribution over D, the per-frame detection continuity and the CRLB
    inflation factor from this movie's own displacements, and `link`
    scores each candidate link as a likelihood ratio under that fit using
    each detection's own `se_y`/`se_x`. So there is no gate, no search
    radius and no cost weighting to set here -- only what to feed it and
    what to keep afterwards.

    `min_track_length` filters the final `tracks_df` (see
    `pipeline.run_track_step`) -- default 2 drops bare singletons (a
    length-1 "track" has no displacement of its own, so it's pure clutter
    downstream, not a meaningful trajectory). Linking is frame-to-frame
    only, so a missed detection ends a track rather than being bridged;
    that fragments trajectories instead of swapping identity, which is
    why this filter matters more here than with a gap-closing linker. It
    stays a spinbox rather than folding into the filter stack below
    because it is applied *inside* the link step -- it changes what
    `estimate_D_um2_s` and the resolvability check are computed over,
    which a post-hoc filter cannot.

    The filter stack is over `pipeline.track_metrics_df` -- one row per
    track, so "412 of 1893 tracks pass" means what it says. The same panel
    will take diffusionkit's per-track fit results (`D`, `alpha`) once
    they've been computed onto the track table, which is the point of
    filtering per-track metrics here rather than per-vertex ones."""

    runRequested = Signal()
    saveRequested = Signal()
    filtersChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.min_track_length = _ispin(
            2, 1, 10_000,
            tooltip="Drop tracks shorter than this (frames) from the final result.\n"
            "1 = keep everything, including singletons.",
        )
        self.drop_aggregates = QCheckBox("exclude flagged aggregates from linking")
        self.drop_aggregates.setChecked(True)
        self.drop_aggregates.setToolTip(
            "Hide detections flagged is_aggregate (Detect tab's aggregate ratio)\n"
            "from the linker. On by default: an over-bright blob is not a point\n"
            "emitter, and its position is a flux-weighted compromise between\n"
            "whatever is inside it. points.parquet keeps them either way."
        )
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("min track length", self.min_track_length)

        self.run_button, self.status_label, run_row = _run_row("Run tracking")
        self.run_button.clicked.connect(self.runRequested.emit)

        link_body = QWidget()
        link_layout = QVBoxLayout(link_body)
        link_layout.setContentsMargins(0, 0, 0, 0)
        link_layout.setSpacing(2)
        link_layout.addLayout(form)
        link_layout.addWidget(self.drop_aggregates)
        link_layout.addWidget(
            note_label(
                "No gate to set: linking parameters are measured from the movie. "
                "The Detect tab's filters decide which detections the linker sees."
            )
        )
        link_layout.addWidget(hline())
        link_layout.addLayout(run_row)

        self.filters = FeatureFilterPanel(
            noun="tracks",
            hint="Run tracking first — then filter on track length, duration or mean step.",
        )
        self.filters.filtersChanged.connect(self.filtersChanged)

        self.save_button = QPushButton("Save results")
        self.save_button.setEnabled(False)
        self.save_button.setToolTip(
            "Write the bundle: points.parquet (every detection, unfiltered),\n"
            "tracks.parquet (the tracks that pass the filters above),\n"
            "manifest.json (what every stage ran with, including both filter\n"
            "specs) and rois.json if an ROI was used.\n\n"
            "Nothing is written until you press this -- adjust the filters and\n"
            "press it again to rewrite the same bundle."
        )
        self.save_button.clicked.connect(self.saveRequested.emit)
        self.save_status = wrapping_label("")
        style_status_label(self.save_status)
        save_row = QHBoxLayout()
        save_row.setContentsMargins(0, 0, 0, 0)
        save_row.addWidget(self.save_button)
        save_row.addWidget(self.save_status, stretch=1)

        save_body = QWidget()
        save_layout = QVBoxLayout(save_body)
        save_layout.setContentsMargins(0, 0, 0, 0)
        save_layout.setSpacing(2)
        save_layout.addLayout(save_row)

        self.pager = StepPager()
        self.pager.add_page("Link", link_body)
        self.pager.add_page("Filter", self.filters)
        self.pager.add_page("Save", save_body)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.pager)
        layout.addStretch()
        self.setLayout(layout)

    def show_step(self, title: str) -> None:
        self.pager.show_step(title)

    def set_status(self, text: str, level: str = "neutral") -> None:
        style_status_label(self.status_label, level)
        self.status_label.setText(text)

    def set_save_status(self, text: str, level: str = "neutral") -> None:
        style_status_label(self.save_status, level)
        self.save_status.setText(text)

    def set_save_enabled(self, enabled: bool) -> None:
        self.save_button.setEnabled(enabled)

    def get_min_track_length(self) -> int:
        return self.min_track_length.value()

    def get_drop_aggregates(self) -> bool:
        return self.drop_aggregates.isChecked()


class PipelineParamsWidget(QWidget):
    """Tabbed `DetectTrackParams` form -- Detect / Track, each a
    detect -> filter -> finalize flow (see this module's docstring).
    `get_params()` reads both tabs' current state back into a fresh
    `DetectTrackParams`, for the batch/multi-select "Run" action.

    For the stepwise per-tab buttons: `previewRequested`/`detectRequested`/
    `trackRequested`/`saveRequested` fire on click; `set_preview_result`/
    `set_detect_status`/`set_track_status`/`set_save_status` report each
    step's result back once `ExperimentListWidget` has run it.
    `pointFiltersChanged`/`trackFiltersChanged` fire whenever a histogram
    handle moves, so the viewer overlay can follow the cut live."""

    previewRequested = Signal()
    detectRequested = Signal()
    detectCancelRequested = Signal()
    trackRequested = Signal()
    saveRequested = Signal()
    newRoiRequested = Signal()
    roiLayerChanged = Signal()
    pointFiltersChanged = Signal()
    trackFiltersChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._detect = _DetectTab()
        self._tracking = _TrackingTab()

        self._detect.previewRequested.connect(self.previewRequested)
        self._detect.runRequested.connect(self.detectRequested)
        self._detect.cancelRequested.connect(self.detectCancelRequested)
        self._detect.newRoiRequested.connect(self.newRoiRequested)
        self._detect.roiLayerChanged.connect(self.roiLayerChanged)
        self._detect.filtersChanged.connect(self.pointFiltersChanged)
        self._tracking.runRequested.connect(self.trackRequested)
        self._tracking.saveRequested.connect(self.saveRequested)
        self._tracking.filtersChanged.connect(self.trackFiltersChanged)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._detect, "Detect")
        self._tabs.addTab(self._tracking, "Track")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tabs)
        self.setLayout(layout)

    def get_params(self) -> DetectTrackParams:
        """Batch (`run_detect_track`) params. `sigma` is left None when
        "calibrate per file" is checked, which is what routes a multi-file
        run back through `run_calibration_step` with the tab's sigma as
        each fit's starting guess -- see `_DetectTab.calibrate_per_file`."""
        per_file = self._detect.get_calibrate_per_file()
        return DetectTrackParams(
            sigma=None if per_file else self._detect.get_sigma(),
            sigma_init=self._detect.get_sigma(),
            calibration_frame_index=self._detect.get_preview_frame_index(),
            min_track_length=self._tracking.get_min_track_length(),
            agg_ratio=self._detect.get_agg_ratio(),
            drop_aggregates=self._tracking.get_drop_aggregates(),
            camera_kwargs=self._detect.get_camera_kwargs(),
            detect_kwargs=self._detect.get_detect_kwargs(),
            calibration_kwargs=dict(DEFAULT_CALIBRATION_KWARGS),
            frame_range=self._detect.get_frame_range(),
            n_threads=self._detect.get_n_threads(),
            point_filters=self.get_point_filters(),
            track_filters=self.get_track_filters(),
        )

    def show_tab(self, name: str, step: Optional[str] = None) -> None:
        """Bring one stage's tab -- and optionally one step page within it
        -- to the front. Called when a stage finishes, so the filters over
        what it just produced are the thing in view rather than something
        to go leafing for: the pages don't scroll past each other now, so
        landing on the right one matters more than it did."""
        for index in range(self._tabs.count()):
            if self._tabs.tabText(index).lower() == name.lower():
                self._tabs.setCurrentIndex(index)
                if step is not None:
                    self._tabs.widget(index).show_step(step)
                return

    # -- detect stage -----------------------------------------------------

    def get_sigma(self) -> float:
        return self._detect.get_sigma()

    def get_preview_frame_index(self) -> int:
        return self._detect.get_preview_frame_index()

    def get_camera_kwargs(self) -> dict:
        """The one source of truth for `offset`/`gain`/`read_noise`, shared
        by preview, detect and (headless) calibration."""
        return self._detect.get_camera_kwargs()

    def get_detect_kwargs(self) -> dict:
        return self._detect.get_detect_kwargs()

    def get_agg_ratio(self) -> float:
        return self._detect.get_agg_ratio()

    def get_frame_range(self) -> Optional[tuple[int, int]]:
        return self._detect.get_frame_range()

    def get_n_threads(self) -> int:
        return self._detect.get_n_threads()

    def get_use_roi_mask(self) -> bool:
        return self._detect.get_use_roi_mask()

    def get_roi_layer_name(self) -> Optional[str]:
        return self._detect.get_roi_layer_name()

    def set_roi_choices(self, names: list[str], preferred: Optional[str] = None) -> None:
        self._detect.set_roi_choices(names, preferred)

    def set_preview_result(self, summary: Optional[dict], level: str = "neutral") -> None:
        self._detect.set_preview_result(summary, level)

    def set_preview_status(self, text: str, level: str = "neutral") -> None:
        self._detect.set_preview_status(text, level)

    def set_detect_status(self, text: str, level: str = "neutral") -> None:
        self._detect.set_status(text, level)

    def set_detect_progress(self, done: int, total: int, stage: str) -> None:
        self._detect.set_progress(done, total, stage)

    def set_detect_running(self, running: bool) -> None:
        self._detect.set_running(running)

    # -- track stage ------------------------------------------------------

    def get_min_track_length(self) -> int:
        return self._tracking.get_min_track_length()

    def get_drop_aggregates(self) -> bool:
        return self._tracking.get_drop_aggregates()

    def set_track_status(self, text: str, level: str = "neutral") -> None:
        self._tracking.set_status(text, level)

    def set_save_status(self, text: str, level: str = "neutral") -> None:
        self._tracking.set_save_status(text, level)

    def set_save_enabled(self, enabled: bool) -> None:
        self._tracking.set_save_enabled(enabled)

    # -- filters ----------------------------------------------------------

    def set_point_filter_source(self, df: Optional[pl.DataFrame]) -> None:
        """Give the Detect tab's filter panel a detections table to draw
        histograms from -- a preview frame's or a finished run's."""
        self._detect.filters.set_source(df)

    def set_track_filter_source(self, df: Optional[pl.DataFrame]) -> None:
        """Give the Track tab's filter panel a per-track metrics table
        (`pipeline.track_metrics_df`), restricted to the metric columns so
        the picker doesn't also offer `y`/`x` centroids."""
        columns = [c for c in TRACK_METRIC_COLUMNS if df is not None and c in df.columns]
        self._tracking.filters.set_source(df, columns)

    def get_point_filters(self) -> FilterSpec:
        return self._detect.filters.filters()

    def get_track_filters(self) -> FilterSpec:
        return self._tracking.filters.filters()

    def set_point_filters(self, spec: Optional[FilterSpec]) -> None:
        self._detect.filters.set_filters(spec)

    def set_track_filters(self, spec: Optional[FilterSpec]) -> None:
        self._tracking.filters.set_filters(spec)

    def clear_filters(self) -> None:
        """Drop every cut on both tabs -- for a selection change, where the
        ranges belong to a table that is no longer loaded."""
        self._detect.filters.set_filters(None)
        self._tracking.filters.set_filters(None)

    def set_frame_bounds(self, n_frames: int) -> None:
        self._detect.set_frame_bounds(n_frames)
