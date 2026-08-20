"""Pipeline parameter form for the interactive Run action, laid out as
tabbed pages that mirror the pipeline's own stages (see `pipeline.py`'s
module docstring: calibrate sigma -> find_spots -> bootstrap/final link):

- **Calibration** -- `sfwloc.report.fit_spots_sparse_df` +
  `sigma_from_spots` (PSF sigma estimation from a sparse-regime reference
  frame, via Aguet's point-source detector).
- **Detect** -- localizer knobs for either of two algorithms, picked via
  an "algorithm" dropdown: `sfwloc`'s dense SFW `find_spots` solver
  (default, handles overlapping/crowded fields), or the sparse
  `find_spots_sparse_df` per-spot free-sigma LM fit (lighter/faster, only
  valid for genuinely well-separated fields -- reuses `_SparseFitFields`,
  the same knob set the Calibration tab's fit uses).
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
(`get_use_roi_mask`), plus a "Draw ROI…" button (`newRoiRequested`) that
just asks for a fresh Shapes layer to draw on -- this widget stays
viewer-agnostic (no napari `Viewer` reference), so `ExperimentListWidget`
(which owns the viewer) is responsible both for adding that layer
(persistent/2D, transparent fill, polygon-lasso tool active -- see
`_on_new_roi_requested`) and, once the ROI checkbox is checked, for
finding the active Shapes layer and turning it into the boolean mask
array `find_spots`'s `mask` argument expects.
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible

from spt_pipeline.pipeline import (
    DEFAULT_CALIBRATION_KWARGS,
    DEFAULT_SOLVER_KWARGS,
    DEFAULT_SPARSE_KWARGS,
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


class _SparseFitFields(QWidget):
    """Widget knobs for `sfwloc_py.find_spots_sparse`'s shared parameter
    set -- Aguet's point-source detector (a per-pixel significance test
    ANDed with Laplacian-of-Gaussian local maxima) feeding a free-sigma,
    free-background bounded LM Gaussian fit per candidate spot -- both the
    Calibration tab (fit against one reference frame) and the Detect tab's
    sparse algorithm (that same fit run as the localizer, over every
    processed frame) forward as `**kwargs`. Each tab owns its own
    instance/defaults (`defaults`) rather than sharing widget state --
    calibration-frame tuning and whole-stack detect tuning don't have to
    match. `sigma_init`/reference-frame knobs stay with the owning tab
    (not part of this shared fieldset) since only Calibration has a
    reference frame and Detect's sparse mode reads its sigma_init from the
    Calibration tab regardless (see `PipelineParamsWidget.get_sigma_init`).

    Core: fit box size and the detector's significance level -- the knobs
    that most directly affect whether the fit finds/fits the right spots.
    Expert: the detector's kernel-window knob, LM solver internals
    (iteration budget, damping schedule, convergence tolerances), and
    fit-bound knobs that rarely need touching."""

    def __init__(self, defaults: dict) -> None:
        super().__init__()
        d = defaults

        self.box_size = _ispin(
            d["box_size"], 3, 99,
            tooltip="Square fit-window side length (px, odd) cropped around each peak.",
        )
        self.alpha = _dspin(
            d["alpha"], 1e-6, 1.0, 0.01, decimals=4,
            tooltip="Significance level for Aguet's per-pixel point-source detection\n"
            "test. Lower finds fewer, more significant candidate peaks.",
        )

        core_form = QFormLayout()
        core_form.setContentsMargins(0, 0, 0, 0)
        core_form.addRow("box size (px)", self.box_size)
        core_form.addRow("alpha", self.alpha)

        self.sigma_lo = _dspin(d["sigma_lo"], 0.01, 20.0, 0.1, decimals=2, tooltip="Lower bound on fitted sigma (px).")
        self.sigma_hi = _dspin(d["sigma_hi"], 0.01, 50.0, 0.1, decimals=2, tooltip="Upper bound on fitted sigma (px).")
        self.delta_bound = _dspin(
            d["delta_bound"], 0.1, 20.0, 0.1, decimals=2,
            tooltip="Bound (px) on fitted peak-center offset from the integer peak location.",
        )
        self.amp_upper = _dspin(d["amp_upper"], 1.0, 1e9, 100.0, decimals=1, tooltip="Upper bound on fitted amplitude.")
        self.bg_upper = _dspin(d["bg_upper"], 0.0, 1e7, 10.0, decimals=1, tooltip="Upper bound on fitted background.")
        self.truncate = _dspin(
            d["truncate"], 1.0, 20.0, 0.5, decimals=2,
            tooltip="Aguet detector kernel half-width, in units of sigma_init.",
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
        expert_form.addRow("truncate (sigma)", self.truncate)
        expert_form.addRow("LM n_iter", self.n_iter)
        expert_form.addRow("LM mu0", self.mu0)
        expert_form.addRow("LM mu decrease", self.mu_decrease)
        expert_form.addRow("LM mu increase", self.mu_increase)
        expert_form.addRow("LM step_tol", self.step_tol)
        expert_form.addRow("LM loss_tol", self.loss_tol)
        expert_form.addRow("LM gtol", self.gtol)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(core_form)
        layout.addWidget(_expert_section(expert_form))
        self.setLayout(layout)

    def get_kwargs(self) -> dict:
        return dict(
            box_size=self.box_size.value(),
            sigma_lo=self.sigma_lo.value(),
            sigma_hi=self.sigma_hi.value(),
            delta_bound=self.delta_bound.value(),
            amp_upper=self.amp_upper.value(),
            bg_upper=self.bg_upper.value(),
            truncate=self.truncate.value(),
            alpha=self.alpha.value(),
            n_iter=self.n_iter.value(),
            mu0=self.mu0.value(),
            mu_decrease=self.mu_decrease.value(),
            mu_increase=self.mu_increase.value(),
            step_tol=self.step_tol.value(),
            loss_tol=self.loss_tol.value(),
            gtol=self.gtol.value(),
        )


class _CalibrationTab(QWidget):
    """PSF-sigma calibration (`fit_spots_sparse_df` + `sigma_from_spots`),
    run once per timelapse
    against a single reference frame (frame 0 by default, but pickable --
    see `frame_index`). The fit itself (`_SparseFitFields`) is the same
    knob set the Detect tab's sparse algorithm reuses -- see that class's
    docstring. Core (this tab's own, alongside the fieldset's core): the
    initial sigma guess and the reference frame."""

    runRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.sigma_init = _dspin(
            1.3, 0.3, 10.0, 0.1, decimals=2,
            tooltip="Initial PSF sigma (px) guess the LM fit starts from.",
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
        core_form.addRow("calibration frame", self.frame_index)

        self._fit_fields = _SparseFitFields(DEFAULT_CALIBRATION_KWARGS)

        self.run_button, self.status_label, run_row = _run_row("Run calibration")
        self.run_button.clicked.connect(self.runRequested.emit)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(core_form)
        layout.addWidget(self._fit_fields)
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
        return self._fit_fields.get_kwargs()


class _DetectTab(QWidget):
    """`find_spots` solver knobs, plus scope controls for *what* gets
    analyzed. Core: `lam` (sparsity weight), `n_iter` (iteration budget),
    the birth/split crowding-correction toggles, a frame range, and a
    "restrict to ROI" checkbox -- the knobs that matter for whether
    detection finds the right spots, on the right frames/pixels, at a
    given density. Expert: refinement-pass internals rarely worth
    touching per-dataset."""

    runRequested = Signal()
    cancelRequested = Signal()
    newRoiRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._running = False
        d = DEFAULT_SOLVER_KWARGS

        # Which localizer runs -- "dense" (find_spots/find_spots_stack's
        # SFW solver, below) or "sparse" (find_spots_sparse_df's per-spot
        # free-sigma LM fit, `_SparseFitFields` -- the same fit the
        # Calibration tab uses, run per-frame as the actual localizer
        # instead of just a one-off sigma estimate). Only one knob group is
        # shown at a time (`_algorithm_stack`), matching whichever's
        # actually forwarded to `run_detect_step`.
        self.algorithm = QComboBox()
        self.algorithm.addItem("Dense (SFW)", userData="dense")
        self.algorithm.addItem("Sparse (well-separated)", userData="sparse")
        self.algorithm.setToolTip(
            "Dense: sfwloc's SFW solver -- handles overlapping/crowded fields.\n"
            "Sparse: a lighter free-sigma LM fit per spot -- faster, but only\n"
            "valid when spots are genuinely well-separated."
        )
        self.algorithm.currentIndexChanged.connect(self._on_algorithm_changed)
        algorithm_row = QHBoxLayout()
        algorithm_row.setContentsMargins(0, 0, 0, 0)
        algorithm_row.addWidget(QLabel("algorithm"))
        algorithm_row.addWidget(self.algorithm, stretch=1)

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
        self.new_roi_button = QPushButton("Draw ROI…")
        self.new_roi_button.setToolTip(
            "Add a new Shapes layer (transparent fill, polygon-lasso tool active)\n"
            "for drawing the ROI -- 2D so it stays visible on every frame instead\n"
            "of only the one it was drawn on."
        )
        roi_row = QHBoxLayout()
        roi_row.setContentsMargins(0, 0, 0, 0)
        roi_row.addWidget(self.use_roi_mask)
        roi_row.addWidget(self.new_roi_button)
        roi_row.addStretch()

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
        self.source_refine_steps = _ispin(
            d["source_refine_steps"], 0, 200,
            tooltip="Gradient-descent steps for continuous source selection (warm start\n"
            "before the birth test's own local fit takes over).",
        )
        self.source_refine_step = _dspin(
            d["source_refine_step"], 1e-4, 100.0, 0.1, decimals=4,
            tooltip="Initial step size for the source-selection search.",
        )
        self.pos_bound = _dspin(
            d["pos_bound"], 0.01, 50.0, 0.1, decimals=3,
            tooltip="Position box half-width (px) for the joint amplitude+position refiner.",
        )
        self.amp_upper = _dspin(
            d["amp_upper"], 1.0, 1e9, 100.0, decimals=1,
            tooltip="Amplitude upper bound used by the joint refiner.",
        )
        self.split_init_sep = _dspin(
            d["split_init_sep"], 0.01, 10.0, 0.05, decimals=3,
            tooltip="Initial per-side child separation for the K=2 split hypothesis,\n"
            "in units of sigma.",
        )
        self.split_min_sep = _dspin(
            d["split_min_sep"], 0.01, 10.0, 0.05, decimals=3,
            tooltip="Minimum per-side child separation accepted by the K=2 split\n"
            "hypothesis, in units of sigma -- below this the split is rejected\n"
            "even if the GLRT itself passes.",
        )
        self.split_free_sigma = QCheckBox()
        self.split_free_sigma.setChecked(d["split_free_sigma"])
        self.split_free_sigma.setToolTip(
            "Let the K=2 split hypothesis's two children fit sigma freely,\n"
            "instead of holding it fixed at the frame's calibrated sigma."
        )
        self.glrt_footprint_sigma = _dspin(
            d["glrt_footprint_sigma"], 0.5, 50.0, 0.5, decimals=2,
            tooltip="Local patch half-width for the birth/split GLRTs, in units of sigma.",
        )
        self.glrt_alpha = _dspin(
            d["glrt_alpha"], 1e-9, 1.0, 1e-4, decimals=9,
            tooltip="Nominal significance level (chi-squared df=3) for the birth/split\n"
            "GLRTs.",
        )
        self.glrt_lm_iter = _ispin(
            d["glrt_lm_iter"], 1, 500,
            tooltip="LM iterations for each local birth/split GLRT fit.",
        )
        self.glrt_local_bg = QCheckBox()
        self.glrt_local_bg.setChecked(d["glrt_local_bg"])
        self.glrt_local_bg.setToolTip(
            "Fold local background into the birth/split GLRT as a free nuisance\n"
            "parameter, instead of holding it fixed at the frame's global bg."
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
        expert_form.addRow("source_refine_steps", self.source_refine_steps)
        expert_form.addRow("source_refine_step", self.source_refine_step)
        expert_form.addRow("pos_bound (px)", self.pos_bound)
        expert_form.addRow("amp_upper", self.amp_upper)
        expert_form.addRow("split_init_sep", self.split_init_sep)
        expert_form.addRow("split_min_sep", self.split_min_sep)
        expert_form.addRow("split_free_sigma", self.split_free_sigma)
        expert_form.addRow("glrt_footprint_sigma", self.glrt_footprint_sigma)
        expert_form.addRow("glrt_alpha", self.glrt_alpha)
        expert_form.addRow("glrt_lm_iter", self.glrt_lm_iter)
        expert_form.addRow("glrt_local_bg", self.glrt_local_bg)

        dense_group = QWidget()
        dense_layout = QVBoxLayout(dense_group)
        dense_layout.setContentsMargins(0, 0, 0, 0)
        dense_layout.setSpacing(4)
        dense_layout.addLayout(core_form)
        dense_layout.addLayout(toggle_row)
        dense_layout.addWidget(_expert_section(expert_form))

        self._sparse_fields = _SparseFitFields(DEFAULT_SPARSE_KWARGS)

        self._algorithm_stack = QStackedWidget()
        self._algorithm_stack.addWidget(dense_group)
        self._algorithm_stack.addWidget(self._sparse_fields)

        self.run_button, self.status_label, run_row = _run_row("Run detect")
        self.run_button.clicked.connect(self._on_run_button_clicked)
        self.new_roi_button.clicked.connect(self.newRoiRequested)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(algorithm_row)
        layout.addWidget(self._algorithm_stack)
        layout.addLayout(frame_row)
        layout.addLayout(roi_row)
        layout.addLayout(run_row)
        layout.addStretch()
        self.setLayout(layout)

    def _on_algorithm_changed(self, index: int) -> None:
        self._algorithm_stack.setCurrentIndex(index)

    def _on_run_button_clicked(self) -> None:
        if self._running:
            self.cancelRequested.emit()
        else:
            self.runRequested.emit()

    def set_running(self, running: bool, label: str = "") -> None:
        """Repurposes the run button into a Cancel button for the duration
        of a run, mirroring the batch Run/Cancel toggle on the experiment
        list's own run button -- see `ExperimentListWidget._cancel_active_run`
        for what cancelling actually does (cooperative, not instant)."""
        self._running = running
        if not running:
            self.run_button.setText("Run detect")
        elif label:
            self.run_button.setText(f"Cancel (running: {label})")
        else:
            self.run_button.setText("Cancel")
        self.run_button.setEnabled(True)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_progress(self, done: int, total: int, stage: str) -> None:
        """Live frame-by-frame progress while a detect run is active --
        shown right next to the Run/Cancel button, since the SFW solver
        itself can't be sped up and this row stays in view even when the
        dock's own progress bar has scrolled out of sight."""
        self.status_label.setText(f"{stage} -- {done}/{total}" if total else stage)

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

    def get_algorithm(self) -> str:
        return self.algorithm.currentData()

    def get_sparse_kwargs(self) -> dict:
        return self._sparse_fields.get_kwargs()

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
            source_refine_steps=self.source_refine_steps.value(),
            source_refine_step=self.source_refine_step.value(),
            pos_bound=self.pos_bound.value(),
            amp_upper=self.amp_upper.value(),
            birth_test=self.birth_test.isChecked(),
            split_test=self.split_test.isChecked(),
            split_init_sep=self.split_init_sep.value(),
            split_min_sep=self.split_min_sep.value(),
            split_free_sigma=self.split_free_sigma.isChecked(),
            glrt_footprint_sigma=self.glrt_footprint_sigma.value(),
            glrt_alpha=self.glrt_alpha.value(),
            glrt_lm_iter=self.glrt_lm_iter.value(),
            glrt_local_bg=self.glrt_local_bg.isChecked(),
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
    detectCancelRequested = Signal()
    trackRequested = Signal()
    newRoiRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._calibration = _CalibrationTab()
        self._detect = _DetectTab()
        self._tracking = _TrackingTab()

        self._calibration.runRequested.connect(self.calibrateRequested)
        self._detect.runRequested.connect(self.detectRequested)
        self._detect.cancelRequested.connect(self.detectCancelRequested)
        self._detect.newRoiRequested.connect(self.newRoiRequested)
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
            algorithm=self._detect.get_algorithm(),
            solver_kwargs=self._detect.get_solver_kwargs(),
            sparse_kwargs=self._detect.get_sparse_kwargs(),
            calibration_kwargs=self._calibration.get_kwargs(),
            frame_range=self._detect.get_frame_range(),
        )

    def get_sigma_init(self) -> float:
        return self._calibration.get_sigma_init()

    def get_calibration_frame_index(self) -> int:
        return self._calibration.get_frame_index()

    def get_calibration_kwargs(self) -> dict:
        return self._calibration.get_kwargs()

    def get_algorithm(self) -> str:
        return self._detect.get_algorithm()

    def get_solver_kwargs(self) -> dict:
        return self._detect.get_solver_kwargs()

    def get_sparse_kwargs(self) -> dict:
        return self._detect.get_sparse_kwargs()

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

    def set_detect_progress(self, done: int, total: int, stage: str) -> None:
        self._detect.set_progress(done, total, stage)

    def set_detect_running(self, running: bool, label: str = "") -> None:
        self._detect.set_running(running, label)

    def set_track_status(self, text: str) -> None:
        self._tracking.set_status(text)
