"""Pipeline parameter form for the interactive Run action, laid out as
two tabbed pages -- **Detect** and **Track** -- each of which is a small
detect -> filter -> finalize flow of its own, the shape Imaris and
TrackMate use for spot detection:

    Detect:  Detect -> Filter -> Save
             (offset + PSF width + detection knobs, [Run detect], then
             histogram filters on what it found)
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

**There is no calibration or preview step.** The PSF width (`sigma`) is a
setting on the Detect page: run detect on a few frames (the frame range),
read the `fit_sigma` histogram on the Filter page, adjust sigma, run
again. The distribution is on screen throughout, so a bimodal or ragged
width -- two focal planes, junk fitted as signal -- is seen rather than
averaged into one number. One caveat: a real run reports only fits inside
the band (0.8-2.0 x sigma by default), so a badly overestimated sigma
shows as a wall at the histogram's low edge rather than a peak; the
expert "report every fit (no band)" box lifts that for a run.

`offset` is the one camera fact spotsolve takes (noise is measured from
each frame), and `PipelineParamsWidget.get_camera_kwargs` is the single
source of truth `ExperimentListWidget` forwards to every stage.

Above both tabs sits `_ImageInfoPanel`: a one-line readout of the
*image's own* metadata -- frame count, pixel size, frame interval -- over
a folded section holding where each value came from and two spinboxes
that override them. It is outside the tabs because those two numbers are
not a parameter of either stage: they are read off the file and
multiplied through everything both stages produce, and until this panel
existed they were applied without ever being shown. A missing one turns
the line red and unfolds the section, since `pipeline.load_session` will
refuse to run without it and the box to fix it is right there.

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
dict, and a linker with a hand-tuned bootstrap gate. `spotsolve`'s default
detector has a fixed decision rule (an emitter exists iff it lowers the
box's Poisson deviance by a set number of nats) and the linker has no dials
at all, so what's left to expose there is genuinely the camera, the PSF
width, and the reporting band -- physical facts about the instrument rather
than solver tuning. A knob that isn't here is not hidden; it doesn't exist.

`spotsolve` now also ships a second detector, Aguet -- LoG-screened
candidates fitted one at a time, with no multi-emitter search and no
width-band rejection (the spotfitlm-compatible sparse baseline). The
"detector" dropdown at the top of the Detect tab's core section picks
between them; the knobs beneath it swap to match (`k_max`/`threshold`/
`slack`/`band` for the default, `significance`/`boxsize`/`itermax` for
Aguet) rather than showing both detectors' settings at once.

Each tab owns its stage's "Run" button and a one-line status label, wired
to `PipelineParamsWidget`'s `detectRequested`/`trackRequested`/
`saveRequested` signals -- `ExperimentListWidget` connects these to run
that single stage (via `pipeline.py`'s `run_detect_step`/`run_track_step`) against whatever the earlier stages
already produced, instead of always re-running the whole pipeline as one
atomic unit. The status labels are read-only; the widget calls
`set_detect_status`/`set_track_status` back with each
stage's result.

The Detect tab also exposes *scope* controls -- which frames and which
pixels get analyzed, not how -- kept in its core section since these are
exactly what make "explore one image incrementally" (this widget's whole
point, vs. blindly running a batch job) practical: a frame-range pair
(`get_frame_range`) and the regions controls (`regions_panel`, a
`widgets.regions_panel.RegionsPanel`): a "restrict to regions" checkbox,
the Labels layer the regions are painted on, and the table naming each
label's class and cell (see `spt_pipeline.regions`). The "cores" spinbox
(`get_n_threads`, under Expert) is `localize_stack`'s `n_threads`,
defaulted to every core (see `pipeline.run_detect_step`'s
docstring for why this speeds up even the interactively-watched run, not
just a headless batch).

This widget stays viewer-agnostic (no napari `Viewer` reference), so
`ExperimentListWidget` (which owns the viewer) is responsible for adding
a regions layer when asked, keeping the layer picker in step with the
viewer's Labels layers, and turning the chosen layer into the boolean
mask `spotsolve`'s `roi` argument expects and the labels each detection
is stamped with.
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
from qtkit import CollapsibleSection, StepPager, double_spinbox, hline, note_label, style_status_label, wrapping_label

from spt_pipeline import units
from spt_pipeline.io_formats import StackMetadata
from spt_pipeline.pipeline import (
    DEFAULT_CAMERA_KWARGS,
    DEFAULT_DETECT_KWARGS,
    DEFAULT_SPARSE_KWARGS,
    TRACK_METRIC_COLUMNS,
    FilterSpec,
)
from spt_pipeline.widgets.feature_filters import FeatureFilterPanel
from spt_pipeline.widgets.regions_panel import RegionsPanel


# A dock this narrow has no room for a numeric field to claim more width
# than its digits need -- QFormLayout's default AllNonFixedFieldsGrow
# policy stretches every field to fill the row regardless, which is where
# most of the wasted width actually comes from. `_compact_form` turns that
# off; `_SPIN_WIDTH` caps each field at what its digits need so the label
# column isn't squeezed into wrapping.
_SPIN_WIDTH = 72
# Decimal spinboxes need more: napari's theme gives every spinbox wide +/-
# buttons, and at 72 px a value like "100.00" was clipped to "100.(".
_DSPIN_WIDTH = 104
_UNIT_SPIN_WIDTH = 140  # wide enough for a spinbox that also carries a unit suffix


def _compact_form(form: QFormLayout) -> QFormLayout:
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
    form.setHorizontalSpacing(6)
    form.setVerticalSpacing(2)
    return form


def _ispin(value: int, minimum: int, maximum: int, tooltip: str, width: int = _SPIN_WIDTH) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setToolTip(tooltip)
    box.setMaximumWidth(width)
    return box


def _dspin(
    value: float,
    minimum: float,
    maximum: float,
    step: float,
    decimals: int,
    tooltip: str = "",
    suffix: str = "",
    width: Optional[int] = None,
) -> QDoubleSpinBox:
    box = double_spinbox(value, minimum, maximum, step, decimals, tooltip=tooltip, suffix=suffix)
    box.setMaximumWidth(width if width is not None else (_UNIT_SPIN_WIDTH if suffix else _DSPIN_WIDTH))
    return box


def _allow_wrapped_height(label: QLabel) -> QLabel:
    """Let a `qtkit.wrapping_label` claim the height its wrapped text
    actually needs.

    `wrapping_label` gives the label an `Ignored` horizontal policy so a
    long line can never widen the dock -- but a word-wrapped `QLabel`
    still reports its *unwrapped* single-line height as its size hint, so
    in a tight column the second and third wrapped lines are simply
    clipped. Turning on `heightForWidth` makes the layout ask how tall the
    text is at the width it was given, which is the whole point of
    wrapping it. Worth it only for the labels that genuinely run to
    several lines -- a file's metadata provenance, here."""
    policy = label.sizePolicy()
    policy.setHeightForWidth(True)
    label.setSizePolicy(policy)
    return label


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


def _save_page(button_text: str, tooltip: str) -> tuple[QPushButton, QLabel, QWidget]:
    """A stage's "Save" page: its save button (off until there is
    something to save) and status label, as one page body."""
    button, status, row = _run_row(button_text)
    button.setEnabled(False)
    button.setToolTip(tooltip)
    body = QWidget()
    layout = QVBoxLayout(body)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    layout.addLayout(row)
    return button, status, body


class _DetectTab(QWidget):
    """Three pages -- Detect, Filter, Save -- leafed through with the
    `qtkit.StepPager` header: the PSF width, the detector choice and its
    knobs plus scope and the Run button, then a filter stack over what it
    found (where `fit_sigma` is read to settle the PSF width), then Save.

    Core: `sigma`, `offset` (the one camera fact `spotsolve` still takes --
    gain and read noise are measured from each frame now), the `detector`
    dropdown, that detector's own knobs, the aggregate cut, a frame range
    and the regions controls.

    Two detectors, two knob sets, never both on screen at once -- the
    dropdown's choice sets which rows `_on_detector_changed` shows:

      - `multi_emitter` (default, `spotsolve.localize`/`localize_stack`):
        `k_max` (most emitters one box may be fitted with jointly -- the
        crowding ceiling) and the LoG `threshold` (one cut, used both for
        seeding a box and for trying another emitter inside one) in core;
        `slack`/`band` (expert) -- the width ranges a fit may take and be
        reported at, both as multiples of `sigma` and echoed in px. There is
        no sparsity weight, iteration budget or refinement schedule to set:
        the search's accept/reject rule is a fixed deviance improvement, and
        it runs to its own convergence.
      - `aguet` (`spotsolve.localize_aguet`/`localize_aguet_stack`, the
        spotfitlm-compatible sparse baseline): `significance` (the LoG
        screening cut, a per-pixel level rather than a z-score) in core;
        `boxsize`/`itermax` (expert) -- the fit-crop size and this
        detector's own iteration budget. No `k_max` (fits are independent,
        never joint) and no `slack`/`band` (every screened fit is reported;
        there is no width-based reject to gate on)."""

    runRequested = Signal()
    cancelRequested = Signal()
    filtersChanged = Signal()
    saveRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._running = False
        self.offset = _dspin(
            DEFAULT_CAMERA_KWARGS["offset"], 0.0, 1e6, 1.0, decimals=2,
            tooltip="Camera offset / baseline, ADU -- subtracted before fitting.\n"
            "The only camera fact spotsolve needs: noise is measured from\n"
            "each frame, not from a gain/read-noise calibration.",
        )
        d = DEFAULT_DETECT_KWARGS
        s = DEFAULT_SPARSE_KWARGS
        slack_lo, slack_hi = d["slack"]
        band_lo, band_hi = d["band"]

        # --- detector choice --------------------------------------------
        # Which spotsolve function `get_detect_kwargs`/pipeline.run_detect_
        # step actually calls. Switching it swaps the rows below between
        # this detector's own knobs (`_on_detector_changed`) -- the two take
        # disjoint keyword arguments, so showing both at once would just
        # invite setting one that the current choice ignores.
        self.detector = QComboBox()
        self.detector.addItem("Multi-emitter", "multi_emitter")
        self.detector.addItem("Sparse (Aguet)", "aguet")
        self.detector.setToolTip(
            "Multi-emitter (default): fits each box jointly, deciding how many\n"
            "emitters it holds by Bayesian model selection -- one rule for both\n"
            "crowded and sparse fields.\n\n"
            "Sparse (Aguet): the spotfitlm-compatible baseline. LoG-screens\n"
            "candidates, then fits each one independently -- no multi-emitter\n"
            "search, no width-band rejection (every screened fit is reported,\n"
            "so frames_df's too_narrow/too_wide/edge counts stay 0).\n"
            "For genuinely sparse fields where the joint search is unneeded."
        )
        self.detector.currentIndexChanged.connect(self._on_detector_changed)

        # --- PSF width ----------------------------------------------------
        self.sigma = _dspin(
            1.3, 0.3, 10.0, 0.1, decimals=3,
            tooltip="In-focus PSF sigma in PIXELS -- the width the search runs\n"
            "at. Each emitter still gets its own fitted width (fit_sigma);\n"
            "this sets where the search starts and what slack/band are\n"
            "multiples of.\n\n"
            "To settle it: run detect on a few frames, read fit_sigma's peak\n"
            "on the Filter page, set it here, run again. Only fits within the\n"
            "band (0.8-2.0 x sigma by default) are reported, so a pile-up at\n"
            "the histogram's low edge means sigma is set too high.",
        )
        self.sigma.valueChanged.connect(self._update_band_note)

        # --- detection knobs -------------------------------------------
        self.k_max = _ispin(
            d["k_max"], 1, 64,
            tooltip="Most emitters one box may be fitted with jointly. The\n"
            "crowding ceiling: raise it for very dense fields, at some cost.",
        )
        # One cut on the LoG z-statistic, used both for which peaks get a
        # box searched around them and for whether a leftover residual
        # peak inside a box gets tried as another emitter (see
        # `DEFAULT_DETECT_KWARGS`) -- spotsolve used to split these into a
        # seed cut and a birth cut, but one number does the same job.
        self.threshold = _dspin(
            0.0, 0.0, 1e9, 0.1, decimals=2,
            tooltip="The LoG cut, in sd of the frame's own noise, for both getting a\n"
            "box searched and for trying another emitter inside one. 'auto'\n"
            f"(recommended) uses spotsolve's default, {spotsolve.PEAK_Z}. Raise it\n"
            "for fewer false positives and speed; lower it toward 2.5 on faint,\n"
            "sparse data.",
        )
        # The minimum is not a cut anyone would want, so it stands for
        # "no explicit cut" (threshold=None): spotsolve's own PEAK_Z.
        self.threshold.setSpecialValueText("auto")
        if d["threshold"] is not None:
            self.threshold.setValue(d["threshold"])

        # spotsolve's per-emitter-count rule: "fixed" keeps the existing
        # greedy 10-nat cost (count_penalty=0 leaves that behavior exactly
        # as it was); "bic" compares background-only and multi-emitter fits
        # with an experimental BIC-inspired score instead (spotsolve's
        # docs/COUNT_SELECTION.md). Multi-emitter only -- Aguet fits one
        # screened candidate at a time, so there is no count to select.
        self.selection = QComboBox()
        self.selection.addItem("Fixed (default)", "fixed")
        self.selection.addItem("BIC (experimental)", "bic")
        self.selection.setToolTip(
            "How the detector decides how many emitters one box holds.\n\n"
            "Fixed: the existing greedy 10-nat rule.\n"
            "BIC (experimental): compares background-only and multi-emitter\n"
            "fits by an information-criterion-style score instead of a fixed\n"
            "cost. Not calibrated evidence or a false-positive rate -- slower,\n"
            "and still being evaluated on faint/blurred data."
        )
        self.selection.currentIndexChanged.connect(self._on_selection_changed)
        self.count_penalty = _dspin(
            0.0, 0.0, 1e6, 0.5, decimals=2,
            tooltip="Extra cost per emitter added to the count rule above --\n"
            "must be finite and non-negative. Higher values favor fewer\n"
            "emitters. Adds to the 10-nat cost in Fixed mode too (0 leaves\n"
            "Fixed at its original behavior); the value that meets a given\n"
            "false-positive budget in BIC mode is data-dependent -- see\n"
            "spotsolve's docs/COUNT_SELECTION.md before trusting one number.",
        )
        self._selection_note = note_label("")
        self._on_selection_changed()

        # Aguet's one core tuning knob: the per-pixel LoG screening level
        # (not a frame-wide false discovery rate -- see spotsolve.aguet).
        # Plays the same "main cut" role `threshold` plays for the
        # multi-emitter detector, so it sits in the same row position.
        self.significance = _dspin(
            s["significance"], 1e-6, 0.5, 0.01, decimals=4,
            tooltip="Per-pixel screening significance for the Aguet baseline --\n"
            "lower is stricter (fewer candidates screened in). This is NOT a\n"
            "frame-wide false discovery rate. Raise it toward 0.1-0.2 on faint,\n"
            "sparse data; lower it for fewer false positives.",
        )

        # Over-bright cut, relative to each frame's own median detection --
        # relative so that one number survives bleaching and illumination
        # drift over a long movie. Detections above it are flagged, not
        # deleted (see pipeline.run_detect_step); the Track tab decides
        # whether linking sees them. Shared by both detectors.
        self.agg_ratio = _dspin(
            spotsolve.AGG_AMP_RATIO, 1.0, 1e6, 1.0, decimals=2,
            tooltip="Flag a detection as an aggregate when its flux exceeds this\n"
            "multiple of the frame's median detection. Flagged, never deleted --\n"
            "the Track tab decides whether linking sees them.",
        )

        self.core_form = core_form = _compact_form(QFormLayout())
        core_form.setContentsMargins(0, 0, 0, 0)
        core_form.addRow("sigma (px)", self.sigma)
        core_form.addRow("detector", self.detector)
        core_form.addRow("offset (ADU)", self.offset)
        core_form.addRow("k_max", self.k_max)
        # Row labels carry the unit the same way "offset (ADU)" and
        # "sigma (px)" do -- a z-score, a p-value and a multiple of the
        # frame's median flux are exactly the kind of thing a bare number
        # invites getting wrong -- kept short so the column stays narrow.
        core_form.addRow("threshold (sd)", self.threshold)
        core_form.addRow("significance (p)", self.significance)
        core_form.addRow("aggregate (× med)", self.agg_ratio)

        # What gets analyzed, not how -- kept in core (not expert) since
        # these are exactly the knobs that let one image be explored
        # incrementally (a few frames, painted regions) instead of always
        # committing to the whole stack. `set_frame_bounds` is called by
        # `ExperimentListWidget` once an image's frame count is known;
        # 0/0 means "process everything" (the common case, and the only
        # sane default before any image has been selected).
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

        # How many native threads the chosen detector's stack function
        # (`localize_stack` or `localize_aguet_stack`) hands frames to (both
        # the headless CLI and, chunked, the interactively-watched run --
        # see `pipeline.run_detect_step`'s docstring). Capped at the
        # machine's own core count; defaulting to it is what "use every
        # core" means without a spinbox arrow-key marathon to get there.
        cpu_count = os.cpu_count() or 1
        self.n_threads = _ispin(
            cpu_count, 1, cpu_count,
            tooltip="Worker threads the detector's stack function hands frames to.\n"
            "Defaults to every core on this machine. Lower it to leave some\n"
            "cores free for other work while a long run is going.",
        )
        frame_row.addStretch()

        self.regions_panel = RegionsPanel()

        # `slack` is the width range a fit may take; `band` the narrower
        # range actually reported as a detection. Both are multiples of
        # `sigma`, not absolute pixels -- spotsolve compares them against
        # `sigma_ratio = fit_sigma / sigma` -- so `_update_band_note`
        # prints what they currently come to in px. A fit outside `band` is
        # an out-of-band reject (too narrow / too wide / edge), which is how
        # out-of-focus and non-PSF-shaped junk stays out of the table;
        # `frames_df` counts them per frame.
        self.slack_lo = _dspin(
            slack_lo, 0.1, 10.0, 0.05, decimals=3,
            tooltip="Narrowest width a fit may take, as a MULTIPLE OF SIGMA.",
        )
        self.slack_hi = _dspin(
            slack_hi, 0.1, 20.0, 0.05, decimals=3,
            tooltip="Widest width a fit may take, as a MULTIPLE OF SIGMA.",
        )
        self.band_lo = _dspin(
            band_lo, 0.1, 10.0, 0.05, decimals=3,
            tooltip="Narrowest width reported as a detection, as a MULTIPLE OF\n"
            "SIGMA -- spotsolve tests it against sigma_ratio = fit_sigma/sigma.",
        )
        self.band_hi = _dspin(
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
            "points_df as if they were detections. Use it for one few-frame run\n"
            "when the fit_sigma histogram piles up at an edge; leave it\n"
            "unchecked for analysis."
        )
        self.no_band.toggled.connect(self._on_no_band_toggled)

        self._slack_row = slack_row = QHBoxLayout()
        slack_row.setContentsMargins(0, 0, 0, 0)
        slack_row.addWidget(self.slack_lo)
        slack_row.addWidget(QLabel("to"))
        slack_row.addWidget(self.slack_hi)
        self._band_row = band_row = QHBoxLayout()
        band_row.setContentsMargins(0, 0, 0, 0)
        band_row.addWidget(self.band_lo)
        band_row.addWidget(QLabel("to"))
        band_row.addWidget(self.band_hi)

        # Aguet's own expert knobs: the odd fit-crop size and this
        # detector's optimizer iteration budget. Both rarely need changing
        # -- there's no per-emitter search to bound the way slack/band
        # bound the multi-emitter fit.
        self.boxsize = _ispin(
            s["boxsize"], 3, 99,
            tooltip="Odd fit-crop size, px, around each screened candidate.\n"
            "Oversized boxes yield no fits. Leave at the default unless\n"
            "spots sit close enough to overlap the crop.",
        )
        self.itermax = _ispin(
            s["itermax"], 1, 10_000,
            tooltip="Max optimizer iterations per candidate fit. Rarely needs\n"
            "changing.",
        )

        self._band_note = note_label("")
        self.expert_form = expert_form = _compact_form(QFormLayout())
        expert_form.setContentsMargins(0, 0, 0, 0)
        expert_form.addRow("slack (× sigma)", slack_row)
        expert_form.addRow("band (× sigma)", band_row)
        expert_form.addRow("", self._band_note)
        expert_form.addRow("", self.no_band)
        expert_form.addRow("count rule", self.selection)
        expert_form.addRow("penalty", self.count_penalty)
        expert_form.addRow("", self._selection_note)
        expert_form.addRow("cores", self.n_threads)
        expert_form.addRow("boxsize (px)", self.boxsize)
        expert_form.addRow("max iterations", self.itermax)
        self._update_band_note()
        self._on_detector_changed()

        self.run_button, self.status_label, run_row = _run_row("Run detect")
        self.run_button.clicked.connect(self._on_run_button_clicked)

        # --- filter -----------------------------------------------------
        self.filters = FeatureFilterPanel(
            noun="points",
            hint="Run detect (a few frames is enough) — then filter on what it found. "
            "fit_sigma's peak is the PSF width to set on the Detect page.",
        )
        self.filters.filtersChanged.connect(self.filtersChanged)

        # --- save (detections alone, independent of tracking) -----------
        self.save_button, self.save_status, save_body = _save_page(
            "Save detections",
            "Write points.parquet (every detection, unfiltered) and\n"
            "manifest.json alone -- usable as soon as detect has run, before\n"
            "tracking. Removes any tracks.parquet this bundle already had,\n"
            "since it was linked from whatever points.parquet said before.\n\n"
            "Nothing is written until you press this.",
        )
        self.save_button.clicked.connect(self.saveRequested.emit)

        detect_body = QWidget()
        detect_layout = QVBoxLayout(detect_body)
        detect_layout.setContentsMargins(0, 0, 0, 0)
        detect_layout.setSpacing(2)
        detect_layout.addLayout(core_form)
        detect_layout.addWidget(hline())
        detect_layout.addLayout(frame_row)
        detect_layout.addWidget(self.regions_panel)
        detect_layout.addWidget(_expert_section(expert_form))
        detect_layout.addWidget(hline())
        detect_layout.addLayout(run_row)

        self.pager = StepPager()
        self.pager.add_page("Detect", detect_body)
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

    # -- PSF width ---------------------------------------------------------

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

    def get_sigma(self) -> float:
        return self.sigma.value()

    # -- detect -----------------------------------------------------------

    def get_detector(self) -> str:
        return self.detector.currentData()

    def _on_detector_changed(self) -> None:
        """Swap the core/expert rows to match the chosen detector -- see
        this class's docstring for which knobs belong to which. Both
        detectors' widgets exist the whole time (their values persist
        across a switch); only visibility changes, via `QFormLayout.
        setRowVisible` on whichever field object the row was built with."""
        sparse = self.get_detector() == "aguet"
        self.core_form.setRowVisible(self.k_max, not sparse)
        self.core_form.setRowVisible(self.threshold, not sparse)
        self.core_form.setRowVisible(self.significance, sparse)
        self.expert_form.setRowVisible(self._slack_row, not sparse)
        self.expert_form.setRowVisible(self._band_row, not sparse)
        self.expert_form.setRowVisible(self._band_note, not sparse)
        self.expert_form.setRowVisible(self.no_band, not sparse)
        self.expert_form.setRowVisible(self.selection, not sparse)
        self.expert_form.setRowVisible(self.count_penalty, not sparse)
        self.expert_form.setRowVisible(self.boxsize, sparse)
        self.expert_form.setRowVisible(self.itermax, sparse)
        # The count-rule note gets a row only while it has something to
        # say: an empty label still takes a row's height and spacing.
        self._sync_selection_note()

    def _on_no_band_toggled(self, checked: bool) -> None:
        self.band_lo.setEnabled(not checked)
        self.band_hi.setEnabled(not checked)
        self._update_band_note()

    def _on_selection_changed(self) -> None:
        """BIC is still experimental (spotsolve's docs/COUNT_SELECTION.md)
        -- say so right under the control, not only in its tooltip."""
        self._selection_note.setText(
            "Experimental: not calibrated evidence or a false-positive rate."
            if self.selection.currentData() == "bic"
            else ""
        )
        self._sync_selection_note()

    def _sync_selection_note(self) -> None:
        # Called from the selection combo during construction too, before
        # the expert form (and the detector choice's rows) exist.
        if hasattr(self, "expert_form"):
            show = self.get_detector() != "aguet" and bool(self._selection_note.text())
            self.expert_form.setRowVisible(self._selection_note, show)

    def _on_run_button_clicked(self) -> None:
        if self._running:
            self.cancelRequested.emit()
        else:
            self.runRequested.emit()

    def set_running(self, running: bool) -> None:
        """Repurposes the run button into a Cancel button for the duration
        of a run -- see `ExperimentListWidget._cancel_active_run` for what
        cancelling actually does (cooperative, not instant). Deliberately
        doesn't embed the running file's name in the button text: a QPushButton
        can't wrap, so an unbounded filename here would force the button --
        and the row/dock around it -- wider, same as the bug fixed on
        `ExperimentListWidget.progress_label`. The experiment list's own
        progress label already shows which file is running."""
        self._running = running
        self.run_button.setText("Run detect" if not running else "Cancel")
        self.run_button.setEnabled(True)
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

    def set_save_status(self, text: str, level: str = "neutral") -> None:
        style_status_label(self.save_status, level)
        self.save_status.setText(text)

    def set_save_enabled(self, enabled: bool) -> None:
        self.save_button.setEnabled(enabled)

    def set_frame_bounds(self, n_frames: int) -> None:
        """Called once an image's frame count is known -- clamps the
        frame-range spinboxes' maxima without disturbing
        values the user already chose (Qt clamps the current value down
        automatically if it now exceeds the new maximum)."""
        self.frame_start.setMaximum(max(n_frames - 1, 0))
        self.frame_end.setMaximum(n_frames)

    def get_frame_range(self) -> Optional[tuple[int, int]]:
        """`(start, end)`, or `None` for "whole stack" (both spinboxes at
        their default 0). `end` is passed through as entered -- including
        its `0`/"last" sentinel value -- rather than resolved here, where
        the frame count may not be known yet (a run started before the
        row's image finished loading on this widget). Resolving
        `end <= 0` into the real last frame is `pipeline._resolve_frame_
        range`'s job: it always has the actual stack length in hand."""
        start = self.frame_start.value()
        end = self.frame_end.value()
        if start == 0 and end == 0:
            return None
        return (start, end)

    def get_n_threads(self) -> int:
        return self.n_threads.value()

    def get_agg_ratio(self) -> float:
        return self.agg_ratio.value()

    def get_camera_kwargs(self) -> dict:
        return dict(offset=self.offset.value())

    def get_detect_kwargs(self) -> dict:
        if self.get_detector() == "aguet":
            return dict(
                significance=self.significance.value(),
                boxsize=self.boxsize.value(),
                itermax=self.itermax.value(),
            )
        return dict(
            k_max=self.k_max.value(),
            threshold=None if self.threshold.value() == self.threshold.minimum() else self.threshold.value(),
            slack=(self.slack_lo.value(), self.slack_hi.value()),
            band=None if self.no_band.isChecked() else (self.band_lo.value(), self.band_hi.value()),
            selection=self.selection.currentData(),
            count_penalty=self.count_penalty.value(),
        )


class _ImageInfoPanel(QWidget):
    """The image's own physical metadata: what the file says, and what to
    use instead when the file is wrong or silent.

    Two rows above the stage tabs, because these are not a parameter of
    either stage -- they are the scale everything both stages produce is
    expressed in:

      - a one-line summary that is always visible (frame count, pixel
        size, frame interval, each with its unit), coloured neutral /
        amber / red so a file with no calibration is obvious before a
        button is pressed rather than as a `ValueError` after one;
      - a folded "Image metadata" section holding the provenance (which
        metadata field each value was read from, plus whatever the reader
        flagged) and the two override spinboxes.

    Folded by default, because the overwhelmingly common case is a file
    that records both correctly and a user who never needs to think about
    it. It unfolds **itself** exactly when it is needed: when the file is
    missing one of the two, which is the one moment the panel is the next
    thing to interact with.

    The override is deliberately sticky across images: a folder is usually
    one acquisition session, so a pixel size typed for the first file is
    almost always right for its neighbours, and re-typing it per file
    would be its own source of error. It can never be silent about that,
    though -- while it is on, the summary line says `overridden` in amber
    and names what the file itself said, and `manifest.json` records the
    value as `given explicitly (file said: ...)`.

    The camera exposure sits beside them, with two differences. No stage
    here needs it, so a file that doesn't record it (every Andor Fusion
    .ims, for one) is amber, not red, and the section doesn't unfold for
    it; the diffusion analysis does need it -- its blur model -- and reads
    it off the layers (`viewer.layer_units_metadata`). And it is not
    behind the override checkbox: that switch replaces the pixel size and
    frame interval with the spinboxes' rounded values, which is the wrong
    price for supplying a number the file simply lacks. Typing an exposure
    is its own override, sticky across images like the other one, and
    undone by "Reset to file". "Not set" is its own state below 0 rather
    than 0 itself, since 0 is a real claim (an instantaneous exposure)
    that would bias D and z if it were only a default.
    """

    changed = Signal()

    # The exposure spinbox's "not set" value, shown as text: one step
    # below the smallest real exposure (0).
    _EXPOSURE_UNSET = -0.0001

    def __init__(self) -> None:
        super().__init__()
        self._metadata: Optional[StackMetadata] = None
        self._file_pixel_size_um: Optional[float] = None
        self._file_dt_s: Optional[float] = None
        self._file_exposure_s: Optional[float] = None
        # Whether the exposure box holds a value typed here (sticky, like
        # the override) rather than the file's own.
        self._exposure_typed = False
        # Where the values being displayed came from ("file", or "bundle"
        # for a saved result's recorded ones), kept so every line this
        # panel writes names the same thing the values actually are.
        self._source = "file"

        self._summary = _allow_wrapped_height(wrapping_label(""))
        style_status_label(self._summary)

        self._detail = _allow_wrapped_height(note_label(""))

        self._override = QCheckBox("use these values instead of the file's")
        self._override.setToolTip(
            "Analyze with the pixel size and frame interval below rather than\n"
            "with what the file records -- for an acquisition whose metadata is\n"
            "missing, or known to be wrong.\n\n"
            "Stays on when you move to another image, since a folder is usually\n"
            "one session; the line above says so while it is in force, and the\n"
            "saved manifest records the value as given explicitly."
        )
        self._override.toggled.connect(self._on_override_toggled)

        # Ranges wide enough for any light microscope: ~1 nm/px (a
        # simulated or upsampled image) to 100 um/px, and 1 us to an hour
        # per frame. Decimals are set for the small end, where the real
        # values live -- a 108 nm pixel is 0.108, and rounding it to two
        # decimals would be a 10% error in every physical column.
        self._pixel_size = _dspin(
            0.1, 0.0001, 100.0, 0.001, decimals=4, suffix=f" {units.UM}/px",
            tooltip="Pixel size at the sample. Everything physical is this number\n"
            "multiplied through: x_um, mean_step_um, and D as its square.",
        )
        self._dt = _dspin(
            0.03, 0.0001, 3600.0, 0.001, decimals=4, suffix=" s/frame",
            tooltip="Time between consecutive frames. Sets the MSD lag times, so D\n"
            "is inversely proportional to it.",
        )
        self._exposure = _dspin(
            self._EXPOSURE_UNSET, self._EXPOSURE_UNSET, 3600.0, 0.001, decimals=4, suffix=" s",
            tooltip="Camera exposure per frame -- how long the shutter is open, not\n"
            "the frame interval (which also counts readout/dead time).\n\n"
            "Only the diffusion analysis uses it: its MLE models the motion\n"
            "blur of a continuous exposure. Leaving it at 0 when the camera\n"
            "was really exposing for 20 ms biases D by about -25% and shifts\n"
            "the non-Brownian score z by +0.3 to +0.7 -- so 'not set' is kept\n"
            "distinct from 0, and the Diffusion panel asks for it.",
        )
        self._exposure.setSpecialValueText("not set")
        self._exposure.valueChanged.connect(self._on_exposure_changed)
        for box in (self._pixel_size, self._dt):
            box.valueChanged.connect(self._on_value_changed)

        self._reset = QPushButton("Reset to file")
        self._reset.setToolTip("Put the boxes back to what this file's own metadata says.")
        self._reset.clicked.connect(self._reset_to_file)

        form = _compact_form(QFormLayout())
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("pixel size", self._pixel_size)
        form.addRow("frame interval", self._dt)

        exposure_form = _compact_form(QFormLayout())
        exposure_form.setContentsMargins(0, 0, 0, 0)
        exposure_form.addRow("exposure", self._exposure)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(2)
        body_layout.addWidget(self._detail)
        body_layout.addWidget(hline())
        reset_row = QHBoxLayout()
        reset_row.setContentsMargins(0, 0, 0, 0)
        reset_row.addWidget(self._reset)
        reset_row.addStretch()

        body_layout.addLayout(exposure_form)
        body_layout.addWidget(self._override)
        body_layout.addLayout(form)
        body_layout.addLayout(reset_row)

        self._section = CollapsibleSection("Image metadata", body, expanded=False)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._summary)
        layout.addWidget(self._section)
        self.setLayout(layout)
        self._sync_enabled()
        self.set_metadata(None)

    # -- what the file says ----------------------------------------------

    def set_metadata(
        self,
        metadata: Optional[StackMetadata],
        source: str = "file",
        pixel_size_um: Optional[float] = None,
        dt_s: Optional[float] = None,
        exposure_s: Optional[float] = None,
    ) -> None:
        """Show the values for the image now being worked on.

        `metadata` is a freshly-read file's (`io_formats.StackMetadata`).
        `source="bundle"` with explicit `pixel_size_um`/`dt_s` instead
        shows a saved bundle's recorded numbers -- which is what its
        tables are actually in, whether they came from the file or from an
        override at the time.
        """
        self._metadata = metadata
        self._source = source
        if metadata is not None:
            self._file_pixel_size_um = metadata.pixel_size_um
            self._file_dt_s = metadata.dt_s
            self._file_exposure_s = metadata.exposure_s
        else:
            self._file_pixel_size_um = pixel_size_um
            self._file_dt_s = dt_s
            self._file_exposure_s = exposure_s
        if not self._override.isChecked():
            self._load_file_values()
        self._refresh()

    def _load_file_values(self) -> None:
        """Park the spinboxes on the file's own values, so turning the
        override on starts from what the file said rather than from
        whatever was last typed for a different image."""
        for box, value in ((self._pixel_size, self._file_pixel_size_um), (self._dt, self._file_dt_s)):
            if value is None:
                continue
            blocked = box.blockSignals(True)
            box.setValue(float(value))
            box.blockSignals(blocked)
        # A typed exposure stays (see the class docstring); otherwise the
        # box follows the file, and one the file doesn't record goes back
        # to "not set" rather than to the last image's value.
        if self._exposure_typed:
            return
        blocked = self._exposure.blockSignals(True)
        self._exposure.setValue(
            self._file_exposure_s if self._file_exposure_s is not None else self._EXPOSURE_UNSET
        )
        self._exposure.blockSignals(blocked)

    # -- the override -----------------------------------------------------

    def pixel_size_um(self) -> Optional[float]:
        """The pixel size to analyze with, or None to use the file's."""
        return self._pixel_size.value() if self._override.isChecked() else None

    def dt_s(self) -> Optional[float]:
        return self._dt.value() if self._override.isChecked() else None

    def exposure_s(self) -> Optional[float]:
        """The exposure typed here, or None to use the file's."""
        return self._exposure.value() if self._exposure_typed else None

    def _on_override_toggled(self, checked: bool) -> None:
        if not checked:
            self._load_file_values()
        self._sync_enabled()
        self._refresh()
        self.changed.emit()

    def _on_value_changed(self, _value: float) -> None:
        if self._override.isChecked():
            self._refresh()
            self.changed.emit()

    def _on_exposure_changed(self, value: float) -> None:
        # Back to "not set" is not a typed value -- it means "use the
        # file's", which is what the box then shows again.
        self._exposure_typed = value >= 0
        if not self._exposure_typed:
            self._load_file_values()
        self._sync_enabled()
        self._refresh()
        self.changed.emit()

    def _reset_to_file(self) -> None:
        was_typed = self._exposure_typed
        self._exposure_typed = False
        self._load_file_values()
        self._sync_enabled()
        self._refresh()
        if self._override.isChecked() or was_typed:
            self.changed.emit()

    def _sync_enabled(self) -> None:
        overriding = self._override.isChecked()
        self._pixel_size.setEnabled(overriding)
        self._dt.setEnabled(overriding)
        self._reset.setEnabled((overriding and self._has_file_values()) or self._exposure_typed)

    def _has_file_values(self) -> bool:
        return (
            self._file_pixel_size_um is not None
            or self._file_dt_s is not None
            or self._file_exposure_s is not None
        )

    # -- display ----------------------------------------------------------

    def effective(self) -> tuple[Optional[float], Optional[float]]:
        """`(pixel_size_um, dt_s)` that a run would actually use: the
        override where it is in force, the file's otherwise, None where
        neither has one."""
        pixel = self.pixel_size_um()
        dt = self.dt_s()
        return (
            pixel if pixel is not None else self._file_pixel_size_um,
            dt if dt is not None else self._file_dt_s,
        )

    def effective_exposure_s(self) -> Optional[float]:
        """The exposure a run would record: the override's when it is in
        force and set, the file's otherwise, None where neither has one."""
        exposure = self.exposure_s()
        return exposure if exposure is not None else self._file_exposure_s

    def _refresh(self) -> None:
        if self._metadata is None and not self._has_file_values() and not self._override.isChecked():
            self._summary.setText("no image loaded")
            self._summary.setToolTip("")
            self._detail.setText("")
            style_status_label(self._summary)
            self._section.set_title("Image metadata")
            return

        pixel, dt = self.effective()
        exposure = self.effective_exposure_s()
        frames = f"{self._metadata.n_frames} {units.FRAMES} · " if self._metadata is not None else ""
        shown = " · ".join(
            (
                units.fmt_unit(pixel, units.UM + "/px") if pixel is not None else "pixel size ?",
                units.fmt_unit(dt, "s/frame") if dt is not None else "frame interval ?",
                f"exposure {units.fmt_unit(exposure, units.SECONDS)}"
                + (" (entered)" if self._exposure_typed else "")
                if exposure is not None
                else "exposure ?",
            )
        )
        missing = [
            name for name, value in (("pixel size", pixel), ("frame interval", dt)) if value is None
        ]
        notes = self._metadata.notes if self._metadata is not None else ()

        if self._override.isChecked():
            prefix = "overridden: "
            said = self._file_summary()
            if said:
                suffix = f" — {self._source} says {said}"
            elif self._metadata is not None:
                suffix = f" — {self._source} records neither"
            else:
                # Between images (the row's load is debounced) there is no
                # file to compare against yet; claiming it records nothing
                # would be a flash of something untrue.
                suffix = ""
            level = "caution"
        else:
            prefix = "from file: " if self._source == "file" else f"from saved {self._source}: "
            suffix = ""
            level = "neutral"
        if missing:
            suffix = f" — no {' or '.join(missing)}: set them below to run"
            level = "error"
        elif exposure is not None and dt is not None and exposure > dt:
            suffix = " — exposure is longer than the frame interval; check both"
            level = "error"
        elif exposure is None:
            # Not an error: detect and track don't need it. But the
            # diffusion analysis will ask, so say so while it is cheap.
            suffix = " — no exposure time: set it below before diffusion analysis"
            level = "caution"
        elif (
            self._exposure_typed
            and self._file_exposure_s is not None
            and self._file_exposure_s != exposure
        ):
            suffix = (
                f" — {self._source} says exposure "
                f"{units.fmt_unit(self._file_exposure_s, units.SECONDS)}"
            )
            level = "caution"
        elif notes and not self._override.isChecked():
            suffix = " — see the details below"
            level = "caution"

        style_status_label(self._summary, level)
        self._summary.setText(f"{prefix}{frames}{shown}{suffix}")
        if self._metadata is not None:
            self._summary.setToolTip(self._metadata.detail())
            self._detail.setText(self._metadata.provenance())
        else:
            self._summary.setToolTip("")
            self._detail.setText(
                f"Recorded in this {self._source}: {self._file_summary() or 'nothing'}."
            )
        self._section.set_title(
            "Image metadata — overridden" if self._override.isChecked() else "Image metadata"
        )
        # Unfold itself only to ask for something it needs: a file missing
        # a value cannot be analyzed until one is typed here, so the panel
        # opens on that and on nothing else. It is never folded back
        # automatically -- that would fight whoever opened it.
        if missing and not self._section.is_expanded():
            self._section.set_expanded(True)

    def _file_summary(self) -> str:
        parts = [
            units.fmt_unit(value, unit)
            for value, unit in (
                (self._file_pixel_size_um, units.UM + "/px"),
                (self._file_dt_s, "s/frame"),
            )
            if value is not None
        ]
        if self._file_exposure_s is not None:
            parts.append(f"exposure {units.fmt_unit(self._file_exposure_s, units.SECONDS)}")
        return " · ".join(parts)


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

    `link_with_flux` is the one optional extra: it adds each detection's
    `flux`/`se_flux` as a second scoring cue (brightness continuity),
    alongside position and CRLB. Off by default -- position/CRLB alone is
    the well-tested path; flux helps most in a crowded field where two
    candidates sit at nearly the same distance but different brightness.

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
            tooltip="Drop tracks with fewer localizations than this from the final\n"
            "result. Counted in POINTS, not seconds: linking is frame-to-frame,\n"
            "so an n-point track spans n-1 intervals and its duration_s is\n"
            "(n-1) x dt. 1 = keep everything, including singletons.",
        )
        self.drop_aggregates = QCheckBox("exclude flagged aggregates from linking")
        self.drop_aggregates.setChecked(True)
        self.drop_aggregates.setToolTip(
            "Hide detections flagged is_aggregate (Detect tab's aggregate ratio)\n"
            "from the linker. On by default: an over-bright blob is not a point\n"
            "emitter, and its position is a flux-weighted compromise between\n"
            "whatever is inside it. points.parquet keeps them either way."
        )
        self.link_with_flux = QCheckBox("use flux as a link cue")
        self.link_with_flux.setToolTip(
            "Also score candidate links by brightness continuity (each\n"
            "detection's flux/se_flux), alongside position and CRLB. Off by\n"
            "default -- an extra cue for a crowded field where two candidates\n"
            "sit at nearly the same distance but different brightness."
        )
        form = _compact_form(QFormLayout())
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("min track length (points)", self.min_track_length)

        self.run_button, self.status_label, run_row = _run_row("Run tracking")
        self.run_button.clicked.connect(self.runRequested.emit)

        link_body = QWidget()
        link_layout = QVBoxLayout(link_body)
        link_layout.setContentsMargins(0, 0, 0, 0)
        link_layout.setSpacing(2)
        link_layout.addLayout(form)
        link_layout.addWidget(self.drop_aggregates)
        link_layout.addWidget(self.link_with_flux)
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

        self.save_button, self.save_status, save_body = _save_page(
            "Save results",
            "Write the bundle: points.parquet (every detection, unfiltered),\n"
            "tracks.parquet (the tracks that pass the filters above),\n"
            "manifest.json (what every stage ran with, including both filter\n"
            "specs) and labels.tif + regions.json if regions were used.\n\n"
            "Nothing is written until you press this -- adjust the filters and\n"
            "press it again to rewrite the same bundle.",
        )
        self.save_button.clicked.connect(self.saveRequested.emit)

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

    def get_link_with_flux(self) -> bool:
        return self.link_with_flux.isChecked()


class PipelineParamsWidget(QWidget):
    """Tabbed `DetectTrackParams` form -- Detect / Track, each a
    detect -> filter -> finalize flow (see this module's docstring).

    For the stepwise per-tab buttons: `detectRequested`/`trackRequested`/
    `saveRequested` fire on click; `set_detect_status`/`set_track_status`/`set_save_status` report each
    step's result back once `ExperimentListWidget` has run it.
    `pointFiltersChanged`/`trackFiltersChanged` fire whenever a histogram
    handle moves, so the viewer overlay can follow the cut live.

    `saveRequested` is the Track tab's "Save results" (points + tracks
    together); `saveDetectionsRequested` is the Detect tab's own "Save
    detections" (points alone, no tracking required) -- two files, two
    independent saves, since a linked track table only ever makes sense
    once there are detections to have linked, not the other way round."""

    detectRequested = Signal()
    detectCancelRequested = Signal()
    trackRequested = Signal()
    saveRequested = Signal()
    saveDetectionsRequested = Signal()
    pointFiltersChanged = Signal()
    trackFiltersChanged = Signal()
    # The pixel size / frame interval override was turned on, off or
    # retyped. Everything physical already computed for the current image
    # was computed at the old scale, so the host has work to do -- see
    # `ExperimentListWidget._on_image_scale_changed`.
    imageScaleChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._detect = _DetectTab()
        self._tracking = _TrackingTab()

        self._detect.runRequested.connect(self.detectRequested)
        self._detect.cancelRequested.connect(self.detectCancelRequested)
        self._detect.filtersChanged.connect(self.pointFiltersChanged)
        self._detect.saveRequested.connect(self.saveDetectionsRequested)
        self._tracking.runRequested.connect(self.trackRequested)
        self._tracking.saveRequested.connect(self.saveRequested)
        self._tracking.filtersChanged.connect(self.trackFiltersChanged)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._detect, "Detect")
        self._tabs.addTab(self._tracking, "Track")

        # Above the tabs, not inside either one: the pixel size and frame
        # interval are not a detection or a linking setting, they are the
        # two facts every physical number on both tabs (and every `_um`
        # column in the saved bundle) is derived from. They were previously
        # read out of the file and applied without ever being shown, so a
        # wrong or absent calibration only surfaced as an odd D. See
        # `_ImageInfoPanel`, which also owns the override for a file whose
        # metadata is missing or wrong.
        self._image_info = _ImageInfoPanel()
        self._image_info.changed.connect(self.imageScaleChanged)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._image_info)
        layout.addWidget(self._tabs)
        self.setLayout(layout)

    def set_image_metadata(
        self,
        metadata: Optional[StackMetadata],
        source: str = "file",
        pixel_size_um: Optional[float] = None,
        dt_s: Optional[float] = None,
        exposure_s: Optional[float] = None,
    ) -> None:
        """Show what the image now being worked on says about itself --
        see `_ImageInfoPanel.set_metadata`."""
        self._image_info.set_metadata(metadata, source, pixel_size_um, dt_s, exposure_s)

    def get_pixel_size_um(self) -> Optional[float]:
        """The pixel size a run should use, or None to take the file's.

        `ExperimentListWidget` passes both this and `get_dt_s` to
        `pipeline.load_session`, whose own arguments
        have always accepted an explicit value -- until now only the
        headless config could supply one."""
        return self._image_info.pixel_size_um()

    def get_dt_s(self) -> Optional[float]:
        return self._image_info.dt_s()

    def get_exposure_s(self) -> Optional[float]:
        """The exposure to record, or None to take the file's -- passed to
        `load_session` beside `get_dt_s`."""
        return self._image_info.exposure_s()

    def get_effective_exposure_s(self) -> Optional[float]:
        return self._image_info.effective_exposure_s()

    def get_effective_image_scale(self) -> tuple[Optional[float], Optional[float]]:
        """`(pixel_size_um, dt_s)` a run would actually use: the override
        where it is in force, otherwise what the current image's file
        recorded. Either is None when neither has one."""
        return self._image_info.effective()

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

    def get_camera_kwargs(self) -> dict:
        """The one source of truth for `offset`."""
        return self._detect.get_camera_kwargs()

    def get_detect_kwargs(self) -> dict:
        return self._detect.get_detect_kwargs()

    def get_detector(self) -> str:
        return self._detect.get_detector()

    def get_agg_ratio(self) -> float:
        return self._detect.get_agg_ratio()

    def get_frame_range(self) -> Optional[tuple[int, int]]:
        return self._detect.get_frame_range()

    def get_n_threads(self) -> int:
        return self._detect.get_n_threads()

    @property
    def regions_panel(self) -> RegionsPanel:
        return self._detect.regions_panel

    def set_detect_status(self, text: str, level: str = "neutral") -> None:
        self._detect.set_status(text, level)

    def set_detect_progress(self, done: int, total: int, stage: str) -> None:
        self._detect.set_progress(done, total, stage)

    def set_detect_running(self, running: bool) -> None:
        self._detect.set_running(running)

    def set_detect_save_status(self, text: str, level: str = "neutral") -> None:
        self._detect.set_save_status(text, level)

    def set_detect_save_enabled(self, enabled: bool) -> None:
        self._detect.set_save_enabled(enabled)

    # -- track stage ------------------------------------------------------

    def get_min_track_length(self) -> int:
        return self._tracking.get_min_track_length()

    def get_drop_aggregates(self) -> bool:
        return self._tracking.get_drop_aggregates()

    def get_link_with_flux(self) -> bool:
        return self._tracking.get_link_with_flux()

    def set_track_status(self, text: str, level: str = "neutral") -> None:
        self._tracking.set_status(text, level)

    def set_save_status(self, text: str, level: str = "neutral") -> None:
        self._tracking.set_save_status(text, level)

    def set_save_enabled(self, enabled: bool) -> None:
        self._tracking.set_save_enabled(enabled)

    # -- filters ----------------------------------------------------------

    def set_point_filter_source(self, df: Optional[pl.DataFrame]) -> None:
        """Give the Detect tab's filter panel a detections table to draw
        histograms from -- a finished run's."""
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
