"""Pipeline parameter form for the interactive Run action, laid out as
tabbed pages that mirror the pipeline's own stages (see `pipeline.py`'s
module docstring: calibrate sigma -> find_spots -> bootstrap/final link):

- **Calibration** -- `sfwloc.report.calibrate_sigma_df` (PSF sigma
  estimation from a sparse-regime reference frame).
- **Detect** -- `sfwloc`'s `find_spots` solver knobs.
- **Track** -- the one pre-run tracking knob (`bootstrap_gate_px`); the
  final link gate is auto-derived, not user-set (see
  `sfwloc.report.recommended_gate_px`).

Each tab shows only the handful of knobs that matter for day-to-day
tuning; the rest collapse under a per-tab "Expert settings"
(`superqt.QCollapsible`, collapsed by default) -- see each tab class's
docstring for the split and why. Tabs (rather than one long stacked form)
keep the dock panel's height bounded to whichever single tab is tallest,
which matters now that there are three stages' worth of knobs to fit next
to the folder list.

Each tab also owns a "Run <stage>" button and a one-line status label,
wired to `PipelineParamsWidget`'s `calibrateRequested`/`detectRequested`/
`trackRequested` signals -- `ExperimentListWidget` connects these to run
that single stage (via `pipeline.py`'s `run_calibration_step`/
`run_detect_step`/`run_track_step`) against whatever the earlier stages
already produced, instead of always re-running calibrate->detect->track
as one atomic unit. The status labels are read-only; the widget calls
`set_calibration_status`/`set_detect_status`/`set_track_status` back with
each stage's result.

The Detect tab also exposes *scope* controls -- which frames and which
pixels get analyzed, not how -- kept in its core section since these are
exactly what make "explore one image incrementally" (this widget's whole
point, vs. blindly running a batch job) practical: a frame-range pair
(`get_frame_range`) and a "restrict to ROI" checkbox
(`get_use_roi_mask`). This widget stays viewer-agnostic (no napari
`Viewer` reference) -- it only reports *whether* the ROI checkbox is
checked; `ExperimentListWidget` (which owns the viewer) is responsible
for finding the active Shapes layer and turning it into the boolean mask
array `find_spots`'s `mask` argument expects.
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible

from spt_pipeline.pipeline import (
    DEFAULT_CALIBRATION_KWARGS,
    DEFAULT_SOLVER_KWARGS,
    DetectTrackParams,
)


def _dspin(value: float, minimum: float, maximum: float, step: float, decimals: int, tooltip: str) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setToolTip(tooltip)
    return box


def _ispin(value: int, minimum: int, maximum: int, tooltip: str) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setToolTip(tooltip)
    return box


def _expert_section(form: QFormLayout) -> QCollapsible:
    body = QWidget()
    body.setLayout(form)
    section = QCollapsible("Expert settings")
    section.addWidget(body)
    section.collapse(animate=False)
    return section


def _run_row(button_text: str) -> tuple[QPushButton, QLabel, QHBoxLayout]:
    """A "Run <stage>" button + a status label sharing one row."""
    button = QPushButton(button_text)
    status = QLabel("")
    status.setStyleSheet("color: gray; font-size: 11px;")
    status.setWordWrap(True)
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(button)
    row.addWidget(status, stretch=1)
    return button, status, row


class _CalibrationTab(QWidget):
    """PSF-sigma calibration (`calibrate_sigma_df`), run once per timelapse
    against a single reference frame (frame 0 by default, but pickable --
    see `frame_index`). Core: the initial sigma guess, the fit box size,
    the peak-detection threshold, and the reference frame itself -- the
    knobs that most directly affect whether calibration finds/fits the
    right spots on a given dataset. Expert: LM solver internals (iteration
    budget, damping schedule, convergence tolerances) and secondary
    peak-finder/fit-bound knobs that rarely need touching."""

    runRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        d = DEFAULT_CALIBRATION_KWARGS

        self.sigma_init = _dspin(
            1.3, 0.3, 10.0, 0.1, decimals=2,
            tooltip="Initial PSF sigma (px) guess the LM fit starts from.",
        )
        self.box_size = _ispin(
            d["box_size"], 3, 99,
            tooltip="Square fit-window side length (px, odd) cropped around each peak.",
        )
        self.peak_threshold_rel = _dspin(
            d["peak_threshold_rel"], 0.0, 1.0, 0.01, decimals=3,
            tooltip="Peak-detector threshold, relative to (max - median) matched-filter\n"
            "response. Lower finds more (dimmer/noisier) candidate peaks.",
        )
        # Which frame gets calibrated against -- kept in core alongside the
        # other day-to-day knobs since frame 0 isn't always the best
        # reference (e.g. sparser or better-focused elsewhere in the
        # stack). Clamped in `set_frame_bounds` once an image is loaded,
        # same pattern as the Detect tab's frame range.
        self._max_frames: Optional[int] = None
        self.frame_index = _ispin(0, 0, 1_000_000, tooltip="Stack frame (0-based) to calibrate against.")

        core_form = QFormLayout()
        core_form.setContentsMargins(0, 0, 0, 0)
        core_form.addRow("sigma init (px)", self.sigma_init)
        core_form.addRow("box size (px)", self.box_size)
        core_form.addRow("peak threshold", self.peak_threshold_rel)
        core_form.addRow("calibration frame", self.frame_index)

        self.sigma_lo = _dspin(d["sigma_lo"], 0.01, 20.0, 0.1, decimals=2, tooltip="Lower bound on fitted sigma (px).")
        self.sigma_hi = _dspin(d["sigma_hi"], 0.01, 50.0, 0.1, decimals=2, tooltip="Upper bound on fitted sigma (px).")
        self.delta_bound = _dspin(
            d["delta_bound"], 0.1, 20.0, 0.1, decimals=2,
            tooltip="Bound (px) on fitted peak-center offset from the integer peak location.",
        )
        self.amp_upper = _dspin(d["amp_upper"], 1.0, 1e9, 100.0, decimals=1, tooltip="Upper bound on fitted amplitude.")
        self.bg_upper = _dspin(d["bg_upper"], 0.0, 1e7, 10.0, decimals=1, tooltip="Upper bound on fitted background.")
        self.peak_footprint = _ispin(
            d["peak_footprint"], 1, 51, tooltip="Local-maximum window size (px, odd) for the peak finder."
        )
        self.peak_border = _ispin(
            d["peak_border"], 0, 100, tooltip="Pixels excluded from peak detection at the image border."
        )
        self.peak_top_k = _ispin(
            0, 0, 100000, tooltip="Cap on candidate peaks considered, ranked by response strength (0 = no cap)."
        )
        self.n_iter = _ispin(d["n_iter"], 1, 500, tooltip="Maximum Levenberg-Marquardt iterations per spot fit.")
        self.mu0 = _dspin(d["mu0"], 1e-6, 1e6, 0.1, decimals=4, tooltip="Initial LM damping factor.")
        self.mu_decrease = _dspin(
            d["mu_decrease"], 1e-3, 1.0, 0.05, decimals=4, tooltip="LM damping multiplier on an accepted step."
        )
        self.mu_increase = _dspin(
            d["mu_increase"], 1.0, 1000.0, 0.5, decimals=4, tooltip="LM damping multiplier on a rejected step."
        )
        self.step_tol = _dspin(
            d["step_tol"], 0.0, 1.0, 1e-6, decimals=8, tooltip="LM convergence: minimum step-size norm."
        )
        self.loss_tol = _dspin(
            d["loss_tol"], 0.0, 1.0, 1e-8, decimals=9, tooltip="LM convergence: minimum relative loss decrease."
        )
        self.gtol = _dspin(
            d["gtol"], 0.0, 1.0, 1e-6, decimals=8, tooltip="LM convergence: minimum gradient norm."
        )

        expert_form = QFormLayout()
        expert_form.setContentsMargins(0, 0, 0, 0)
        expert_form.addRow("sigma lo (px)", self.sigma_lo)
        expert_form.addRow("sigma hi (px)", self.sigma_hi)
        expert_form.addRow("delta bound (px)", self.delta_bound)
        expert_form.addRow("amp upper", self.amp_upper)
        expert_form.addRow("bg upper", self.bg_upper)
        expert_form.addRow("peak footprint", self.peak_footprint)
        expert_form.addRow("peak border", self.peak_border)
        expert_form.addRow("peak top-k", self.peak_top_k)
        expert_form.addRow("LM n_iter", self.n_iter)
        expert_form.addRow("LM mu0", self.mu0)
        expert_form.addRow("LM mu decrease", self.mu_decrease)
        expert_form.addRow("LM mu increase", self.mu_increase)
        expert_form.addRow("LM step_tol", self.step_tol)
        expert_form.addRow("LM loss_tol", self.loss_tol)
        expert_form.addRow("LM gtol", self.gtol)

        self.run_button, self.status_label, run_row = _run_row("Run calibration")
        self.run_button.clicked.connect(self.runRequested.emit)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(core_form)
        layout.addWidget(_expert_section(expert_form))
        layout.addLayout(run_row)
        layout.addStretch()
        self.setLayout(layout)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_frame_bounds(self, n_frames: int) -> None:
        """Called once an image's frame count is known -- clamps the
        calibration-frame spinbox's maximum without disturbing a value the
        user already chose."""
        self._max_frames = n_frames
        self.frame_index.setMaximum(max(n_frames - 1, 0))

    def get_sigma_init(self) -> float:
        return self.sigma_init.value()

    def get_frame_index(self) -> int:
        return self.frame_index.value()

    def get_kwargs(self) -> dict:
        kwargs = dict(DEFAULT_CALIBRATION_KWARGS)
        kwargs.update(
            box_size=self.box_size.value(),
            sigma_lo=self.sigma_lo.value(),
            sigma_hi=self.sigma_hi.value(),
            delta_bound=self.delta_bound.value(),
            amp_upper=self.amp_upper.value(),
            bg_upper=self.bg_upper.value(),
            peak_threshold_rel=self.peak_threshold_rel.value(),
            peak_footprint=self.peak_footprint.value(),
            peak_border=self.peak_border.value(),
            peak_top_k=self.peak_top_k.value() or None,
            n_iter=self.n_iter.value(),
            mu0=self.mu0.value(),
            mu_decrease=self.mu_decrease.value(),
            mu_increase=self.mu_increase.value(),
            step_tol=self.step_tol.value(),
            loss_tol=self.loss_tol.value(),
            gtol=self.gtol.value(),
        )
        return kwargs


class _DetectTab(QWidget):
    """`find_spots` solver knobs, plus scope controls for *what* gets
    analyzed. Core: `lam` (sparsity weight), `n_iter` (iteration budget),
    the birth/split crowding-correction toggles, a frame range, and a
    "restrict to ROI" checkbox -- the knobs that matter for whether
    detection finds the right spots, on the right frames/pixels, at a
    given density. Expert: refinement-pass internals rarely worth
    touching per-dataset."""

    runRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        d = DEFAULT_SOLVER_KWARGS

        self.lam = _dspin(
            d["lam"], 1e-4, 100.0, 0.01, decimals=4,
            tooltip="Sparsity (BLASSO) weight. Higher = fewer, stronger spots;\n"
            "lower = more, weaker/noisier spots.",
        )
        self.n_iter = _ispin(
            d["n_iter"], 1, 2000,
            tooltip="Maximum outer SFW iterations (upper bound on spots found).",
        )

        core_form = QFormLayout()
        core_form.setContentsMargins(0, 0, 0, 0)
        core_form.addRow("lam", self.lam)
        core_form.addRow("n_iter", self.n_iter)

        self.birth_test = QCheckBox("birth test")
        self.birth_test.setChecked(d["birth_test"])
        self.birth_test.setToolTip(
            "Local GLRT re-check before committing each new spike -- rejects\n"
            "spurious births near an already-placed spot (crowding)."
        )
        self.split_test = QCheckBox("split test")
        self.split_test.setChecked(d["split_test"])
        self.split_test.setToolTip(
            "Local GLRT re-check after refining each spike -- splits it into\n"
            "two if that fits the data significantly better (crowding)."
        )
        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        toggle_row.addWidget(self.birth_test)
        toggle_row.addWidget(self.split_test)
        toggle_row.addStretch()

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

        self.use_roi_mask = QCheckBox("Restrict to ROI (active Shapes layer)")
        self.use_roi_mask.setToolTip(
            "Only place new spikes inside the shape(s) on the viewer's currently\n"
            "active Shapes layer (find_spots's mask argument). Background/likelihood\n"
            "fitting still uses the whole frame. Draw a Shapes layer in napari first."
        )

        self.refine_lam = _dspin(
            d["refine_lam"], 0.0, 100.0, 0.01, decimals=4,
            tooltip="L1 weight inside the joint position+amplitude refiner\n"
            "(varpro_refine), separate from lam. 0 = unbiased amplitude refit.",
        )
        self.fista_iter = _ispin(
            d["fista_iter"], 1, 1000, tooltip="FISTA iterations for each amplitude-only sub-problem."
        )
        self.n_refine = _ispin(
            d["n_refine"], 0, 50, tooltip="Local refinement sweeps per outer iteration."
        )
        self.refine_iter = _ispin(
            d["refine_iter"], 1, 200, tooltip="Max outer L-BFGS-B iterations for varpro_refine."
        )
        self.varpro_fista_iter = _ispin(
            d["varpro_fista_iter"], 1, 1000,
            tooltip="FISTA iterations for each inner amplitude solve inside varpro_refine.",
        )
        self.prune_tol = _dspin(
            d["prune_tol"], 0.0, 1.0, 1e-5, decimals=6,
            tooltip="Amplitude pruning threshold applied after each outer iteration.",
        )
        self.delta_dev_tol = _dspin(
            d["delta_dev_tol"], 0.0, 10.0, 1e-3, decimals=5,
            tooltip="Minimum mean-deviance improvement required per outer iteration.",
        )
        self.delta_dev_patience = _ispin(
            d["delta_dev_patience"], 1, 50,
            tooltip="Consecutive non-improving iterations tolerated before stopping.",
        )
        self.delta_dev_min_iter = _ispin(
            d["delta_dev_min_iter"], 0, 100,
            tooltip="Minimum outer iterations before the deviance-plateau stop can fire.",
        )

        expert_form = QFormLayout()
        expert_form.setContentsMargins(0, 0, 0, 0)
        expert_form.addRow("refine_lam", self.refine_lam)
        expert_form.addRow("fista_iter", self.fista_iter)
        expert_form.addRow("n_refine", self.n_refine)
        expert_form.addRow("refine_iter", self.refine_iter)
        expert_form.addRow("varpro_fista_iter", self.varpro_fista_iter)
        expert_form.addRow("prune_tol", self.prune_tol)
        expert_form.addRow("delta_dev_tol", self.delta_dev_tol)
        expert_form.addRow("delta_dev_patience", self.delta_dev_patience)
        expert_form.addRow("delta_dev_min_iter", self.delta_dev_min_iter)

        self.run_button, self.status_label, run_row = _run_row("Run detect")
        self.run_button.clicked.connect(self.runRequested.emit)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(core_form)
        layout.addLayout(toggle_row)
        layout.addLayout(frame_row)
        layout.addWidget(self.use_roi_mask)
        layout.addWidget(_expert_section(expert_form))
        layout.addLayout(run_row)
        layout.addStretch()
        self.setLayout(layout)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_frame_bounds(self, n_frames: int) -> None:
        """Called once an image's frame count is known -- clamps the
        frame-range spinboxes' maxima without disturbing values the user
        already chose (Qt clamps the current value down automatically if
        it now exceeds the new maximum)."""
        self._max_frames = n_frames
        self.frame_start.setMaximum(max(n_frames - 1, 0))
        self.frame_end.setMaximum(n_frames)

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

    def get_use_roi_mask(self) -> bool:
        return self.use_roi_mask.isChecked()

    def get_solver_kwargs(self) -> dict:
        kwargs = dict(DEFAULT_SOLVER_KWARGS)
        kwargs.update(
            lam=self.lam.value(),
            refine_lam=self.refine_lam.value(),
            n_iter=self.n_iter.value(),
            fista_iter=self.fista_iter.value(),
            n_refine=self.n_refine.value(),
            refine_iter=self.refine_iter.value(),
            varpro_fista_iter=self.varpro_fista_iter.value(),
            prune_tol=self.prune_tol.value(),
            delta_dev_tol=self.delta_dev_tol.value(),
            delta_dev_patience=self.delta_dev_patience.value(),
            delta_dev_min_iter=self.delta_dev_min_iter.value(),
            birth_test=self.birth_test.isChecked(),
            split_test=self.split_test.isChecked(),
        )
        return kwargs


class _TrackingTab(QWidget):
    """The one pre-run tracking knob. The bootstrap pass exists only to
    get a rough D estimate for auto-deriving the *final* link gate
    (`sfwloc.report.recommended_gate_px`) -- that final gate is not
    user-set."""

    runRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.bootstrap_gate_px = _dspin(
            3.0, 0.5, 50.0, 0.5, decimals=2,
            tooltip="Fixed, generous linking gate (px) for the bootstrap track pass used\n"
            "only to estimate D before the final auto-derived gate is computed.",
        )
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("bootstrap gate (px)", self.bootstrap_gate_px)

        self.run_button, self.status_label, run_row = _run_row("Run tracking")
        self.run_button.clicked.connect(self.runRequested.emit)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(form)
        layout.addLayout(run_row)
        layout.addStretch()
        self.setLayout(layout)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def get_bootstrap_gate_px(self) -> float:
        return self.bootstrap_gate_px.value()


class PipelineParamsWidget(QWidget):
    """Tabbed `DetectTrackParams` form -- Calibration / Detect / Track.
    `get_params()` reads all three tabs' current state back into a fresh
    `DetectTrackParams`, for the batch/multi-select "Run" action.

    For the stepwise per-tab "Run <stage>" buttons: `calibrateRequested`/
    `detectRequested`/`trackRequested` fire on click;
    `set_calibration_status`/`set_detect_status`/`set_track_status` report
    each stage's result back once `ExperimentListWidget` has run it."""

    calibrateRequested = Signal()
    detectRequested = Signal()
    trackRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._calibration = _CalibrationTab()
        self._detect = _DetectTab()
        self._tracking = _TrackingTab()

        self._calibration.runRequested.connect(self.calibrateRequested)
        self._detect.runRequested.connect(self.detectRequested)
        self._tracking.runRequested.connect(self.trackRequested)

        tabs = QTabWidget()
        tabs.addTab(self._calibration, "Calibration")
        tabs.addTab(self._detect, "Detect")
        tabs.addTab(self._tracking, "Track")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tabs)
        self.setLayout(layout)

    def get_params(self) -> DetectTrackParams:
        return DetectTrackParams(
            sigma_init=self._calibration.get_sigma_init(),
            calibration_frame_index=self._calibration.get_frame_index(),
            bootstrap_gate_px=self._tracking.get_bootstrap_gate_px(),
            solver_kwargs=self._detect.get_solver_kwargs(),
            calibration_kwargs=self._calibration.get_kwargs(),
            frame_range=self._detect.get_frame_range(),
        )

    def get_sigma_init(self) -> float:
        return self._calibration.get_sigma_init()

    def get_calibration_frame_index(self) -> int:
        return self._calibration.get_frame_index()

    def get_calibration_kwargs(self) -> dict:
        return self._calibration.get_kwargs()

    def get_solver_kwargs(self) -> dict:
        return self._detect.get_solver_kwargs()

    def get_bootstrap_gate_px(self) -> float:
        return self._tracking.get_bootstrap_gate_px()

    def get_frame_range(self) -> Optional[tuple[int, int]]:
        return self._detect.get_frame_range()

    def get_use_roi_mask(self) -> bool:
        return self._detect.get_use_roi_mask()

    def set_frame_bounds(self, n_frames: int) -> None:
        self._calibration.set_frame_bounds(n_frames)
        self._detect.set_frame_bounds(n_frames)

    def set_calibration_status(self, text: str) -> None:
        self._calibration.set_status(text)

    def set_detect_status(self, text: str) -> None:
        self._detect.set_status(text)

    def set_track_status(self, text: str) -> None:
        self._tracking.set_status(text)
