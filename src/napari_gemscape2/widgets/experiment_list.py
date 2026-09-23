"""Experiment-list napari dock widget: browse a folder of raw timelapses,
see per-file processing status at a glance, load a bundle's layers on
selection, and run detect+track on the current selection in the background.

Rewritten from scratch rather than ported from napari-gemscape's
`InputFileList`/`FilepathItem` (see that repo's `widgets/file_input_widgets.py`):
status is painted as a colored dot via a real `QStyledItemDelegate` instead
of a text-glyph prefix, keybindings are explicit and shown in the header
instead of a few ad hoc (partly broken -- its Up/Down handlers were
no-ops) `keyPressEvent` branches, and there's exactly one execution path
(threaded, in-process) instead of gemscape's separate interactive-handler
and subprocess-batch-script implementations of the same pipeline.

Deliberately excluded (see the project plan): an in-app code-exec tab and
the diffusion-analysis step -- this widget's job is browse/load/run-
detect-track, nothing else. Pipeline parameter tuning (including the
Detect tab's PSF width and frame-range/regions scope controls)
lives in
`widgets/params_panel.py::PipelineParamsWidget`, which stays viewer-
agnostic; this module is what actually resolves the regions controls
into a boolean mask and a labels image, by reading the Labels layer
picked in that panel (`_build_regions`). "New layer" adds one and
`_on_region_layers_changed` keeps the picker in step with the viewer
(renames included), so which regions a run covers is a deliberate pick
rather than a side-effect of which layer was last clicked. Each label is
one region: detections are stamped with it (`regions.label_points`) and
tracking links each region on its own.

A row with a saved bundle is also a live session as soon as it's shown
(`_adopt_bundle_session`): its detections can be linked, and its tracks
re-filtered and re-saved, without re-running detect.

This widget always runs `run_detect_step` with a `progress_callback`
(for the live frame-count/cancel UI), which is what actually puts it on
`run_detect_step`'s chunked path -- not the mask itself, which
`find_spots_stack_df` accepts directly (see `pipeline.run_detect_step`'s
docstring). The Detect tab's "cores" spinbox (`get_n_threads`) is the
chunk's own `localize_stack` call's `n_threads`, so this path is still
multi-core; only progress/cancel granularity, not parallelism, is traded
away here.

Drag-and-drop accepts a dropped folder anywhere on this dock widget (not
just precisely on the list rows) -- both `_ExperimentListView` and the
outer `ExperimentListWidget` implement it, since the list no longer fills
the whole panel now that the params form sits below it.

The pipeline runs one image at a time, on the current row, from the
params-panel tabs' own buttons. Detect and track run
independently, in the background through one `self._worker` slot, against
a `pipeline.PipelineSession` held in `self._session`, so changing one
stage's knobs and re-running it doesn't force redoing the earlier stages.
The session resets on selection change (see `_on_selection_changed`) --
it's scoped to "the image currently being worked on", not persisted across
items. Detect can be cancelled from its own button (`_cancel_active_run`)
-- cooperatively, see `pipeline.PipelineCancelled`'s docstring for what
that actually guarantees per stage.

Multi-file runs start from a finished movie: "Batch from this movie…"
(`widgets/batch_dialog.py`) takes the current row's *saved* bundle as the
template and runs its settings -- detect+track from its manifest, the
diffusion analysis from its saved `diffusion_summary.json` -- over the
other movies ticked in the dialog. That keeps the filter histograms'
role: the cuts are chosen by looking at one movie, then reused, never
set blind. The batch goes through the same `batch.detect_track_bundle`
and `diffusion_batch.analyze_bundle` as the `gemscape2` CLI, and writes
the CLI config that re-runs it. Pooling results across experiments is a
script's job, over the saved bundles.

Runs write **nothing** until "Save results" is pressed
(`_save_result`). The filter histograms on both tabs are the reason:
the cuts they set are chosen by looking at a finished stage's output, so
committing the bundle the instant linking returned would mean saving
before the decision that shapes it had been made. `has_unsaved_session`
on the list row is what marks the gap in between.

Filters also drive the viewer live: `_update_points_layer` and
`_update_tracks_layer` redraw the "points (preview)"/"tracks (preview)"
layers through the current cuts on every handle move, so a spot that
fails a cut leaves the image as the cut is made. That immediacy is the
point of keeping the histogram next to the viewer instead of in a
report. Saving then hands those same layers their final names
(`_promote_preview_layers`) without touching the image layer or the regions
layers, so the view doesn't reset out from under the user at the moment
the work is committed. Every layer here is built by
`viewer.py`'s `add_image_layer`/`add_points_layer`/`add_tracks_layer`,
which is what makes preview and final look identical.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import polars as pl
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from natsort import natsorted
from qtkit.napari import live_layer
from qtpy.QtCore import QModelIndex, QObject, QRect, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QFontMetrics, QPainter, QPen
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QLabel,
    QListWidget,
    QFrame,
    QHBoxLayout,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from napari_gemscape2 import units
from napari_gemscape2.batch import write_batch_config
from napari_gemscape2.results import (
    build_manifest,
    result_dir_for,
    has_result,
    repo_shas,
    load_manifest,
    write_detection_result,
    write_result,
)
from napari_gemscape2.io_formats import SUPPORTED_SUFFIXES as SUPPORTED_FORMATS
from napari_gemscape2.pipeline import (
    PipelineCancelled,
    PipelineSession,
    apply_filters,
    apply_track_filters,
    filter_mask,
    load_session,
    run_detect_step,
    run_track_step,
    session_from_bundle,
    session_manifest_extra,
    track_features_df,
    track_metrics_df,
)
from napari_gemscape2.regions import Regions, label_points
from napari_gemscape2.viewer import (
    ImageDisplay,
    ResultDisplay,
    add_image_layer,
    REGIONS_LAYER_NAME,
    add_regions_layer,
    region_classes,
    add_points_layer,
    add_tracks_layer,
    layer_units_metadata,
    load_image_display,
    load_result_display,
    set_points_layer_data,
    set_tracks_layer_data,
    show_image,
    show_result,
)
from napari_gemscape2.widgets.batch_dialog import (
    BatchDialog,
    BatchEmitter,
    BatchPlan,
    batch_config_path,
    run_batch_worker,
)
from napari_gemscape2.widgets.params_panel import PipelineParamsWidget


def _repo_shas() -> dict:
    """Provenance for a bundle's manifest: the checkouts that produced it."""
    import spotsolve
    import napari_gemscape2

    return repo_shas(spotsolve, napari_gemscape2)


def _dropped_folder(event) -> Optional[Path]:
    """First local directory among an event's dropped URLs, if any."""
    for url in event.mimeData().urls():
        path = Path(url.toLocalFile())
        if path.is_dir():
            return path
    return None


class Status(str, Enum):
    UNTOUCHED = "untouched"
    COMPLETE = "complete"
    SKIP = "skip"


STATUS_COLORS = {
    Status.UNTOUCHED: QColor("#9a9a9a"),
    Status.COMPLETE: QColor("#22c55e"),
    Status.SKIP: QColor("#5a5a5a"),
}


@dataclass
class ExperimentEntry:
    image_path: Path
    result_dir: Path
    status: Status = Status.UNTOUCHED
    n_tracks: Optional[int] = None
    # True while this item's session holds results (a detect or link run,
    # or filters moved since the last save) that aren't in its bundle --
    # painted as an amber ring by ExperimentItemDelegate. Leaving the item
    # while it is set asks first (`_confirm_leave_session`). Cleared by a
    # save, or by choosing to discard.
    has_unsaved_session: bool = False


class ExperimentItem(QListWidgetItem):
    def __init__(self, entry: ExperimentEntry):
        super().__init__()
        self.entry = entry
        self.setToolTip(str(entry.image_path))

    def set_status(self, status: Status, **extra) -> None:
        self.entry.status = status
        for key, value in extra.items():
            setattr(self.entry, key, value)


class ExperimentItemDelegate(QStyledItemDelegate):
    """Paints a status dot + filename (+ track count once known)."""

    DOT_DIAMETER = 8
    PADDING = 8
    ROW_HEIGHT = 26

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        list_widget = self.parent()
        item = list_widget.item(index.row())
        entry = item.entry

        painter.save()
        selected = option.state & QStyle.StateFlag.State_Selected
        if selected:
            painter.fillRect(option.rect, option.palette.highlight())
            text_color = option.palette.highlightedText().color()
        else:
            text_color = option.palette.text().color()

        dot_rect = QRect(
            option.rect.left() + self.PADDING,
            option.rect.center().y() - self.DOT_DIAMETER // 2,
            self.DOT_DIAMETER,
            self.DOT_DIAMETER,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(STATUS_COLORS[entry.status])
        painter.drawEllipse(dot_rect)

        if entry.has_unsaved_session:
            painter.setPen(QPen(QColor("#f59e0b"), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(dot_rect.adjusted(-3, -3, 3, 3))

        text_rect = option.rect.adjusted(self.PADDING * 2 + self.DOT_DIAMETER, 0, -self.PADDING, 0)
        label = entry.image_path.name
        if entry.status is Status.COMPLETE and entry.n_tracks is not None:
            label += f"   ({entry.n_tracks} tracks)"
        painter.setPen(text_color)
        # Plain drawText into a rect this narrow just hard-clips a long
        # filename mid-character with no visual cue there's more --
        # elide it instead (full path is still available via the
        # item's tooltip, set in `ExperimentItem.__init__`).
        metrics = QFontMetrics(painter.font())
        elided = metrics.elidedText(label, Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), elided)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return QSize(option.rect.width(), self.ROW_HEIGHT)


class _ExperimentListView(QListWidget):
    """The list itself: folder scanning + keybindings. Composed inside
    `ExperimentListWidget`, which owns the run/progress machinery."""

    KEYBINDINGS = "Enter: load · X: skip · U: unmark · F5: rescan"

    def __init__(self) -> None:
        super().__init__()
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        # QAbstractItemView delivers drag/drop events to its internal
        # viewport() widget, not to `self` -- setAcceptDrops(True) above
        # only sets the Qt::WA_AcceptDrops flag on `self`, so without this
        # the OS/Qt platform layer never delivers a real (non-synthetic)
        # drop over the visible list rows, only over `self`'s own margins.
        self.viewport().setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setItemDelegate(ExperimentItemDelegate(self))
        self.folder_path: Optional[Path] = None
        self.results_root: Optional[Path] = None
        # Set by the owning ExperimentListWidget so a folder reload can be
        # refused while a run is in flight (see load_folder) -- a fresh
        # QListWidgetItem per row would otherwise silently detach the
        # in-flight run's own item from the reloaded list (Finding 4).
        self.is_busy: Callable[[], bool] = lambda: False
        self.on_busy_blocked: Callable[[], None] = lambda: None
        # Asked before a reload replaces every row (and with them the
        # in-memory session of whichever row was being worked on); False
        # keeps the current list.
        self.confirm_reload: Callable[[], bool] = lambda: True

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key.Key_X, Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            for item in self.selectedItems():
                item.set_status(Status.SKIP)
            self.viewport().update()
            return
        if key == Qt.Key.Key_U:
            for item in self.selectedItems():
                item.set_status(Status.UNTOUCHED)
            self.viewport().update()
            return
        if key == Qt.Key.Key_F5:
            if self.folder_path is not None:
                self.load_folder(self.folder_path, self.results_root)
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        # QAbstractItemView's own default dragMoveEvent re-validates every
        # move tick against the model's canDropMimeData -- QListModel only
        # understands its own internal "application/x-qabstractitemmodeldatalist"
        # mime type, not an OS file drag's "text/uri-list", so without this
        # override the drop indicator shows "forbidden" for the whole drag
        # even though dragEnterEvent above already accepted it.
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        folder = _dropped_folder(event)
        if folder is not None:
            self.load_folder(folder, self.results_root)

    def load_folder(self, folder_path: Path, results_root: Optional[Path] = None) -> None:
        if self.is_busy():
            self.on_busy_blocked()
            return
        if not self.confirm_reload():
            return
        self.clear()
        self.folder_path = Path(folder_path)
        self.results_root = results_root or (self.folder_path / "results")

        for file_path in natsorted(self.folder_path.iterdir()):
            if file_path.name.startswith("."):
                continue
            if file_path.suffix.lower() not in SUPPORTED_FORMATS:
                continue

            result_dir = result_dir_for(self.results_root, file_path)
            entry = ExperimentEntry(image_path=file_path, result_dir=result_dir)
            item = ExperimentItem(entry)

            if has_result(result_dir):
                n_tracks = load_manifest(result_dir)["params"]["n_tracks"]
                item.set_status(Status.COMPLETE, n_tracks=n_tracks)

            self.addItem(item)

    def items(self) -> list[ExperimentItem]:
        return [self.item(i) for i in range(self.count())]


class _ProgressEmitter(QObject):
    updated = Signal(int, int, str)


class ExperimentListWidget(QWidget):
    """Dock widget: folder-scan list + params tabs + progress bar."""

    def __init__(self, napari_viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        # The one Detect/Track worker, while it runs against
        # `self._session_item`. Also what guards `_on_selection_changed`,
        # so clicking a different row mid-run can't rug the viewer layers
        # out from under it (Finding: napari's layer list going blank on a
        # mid-detect selection change).
        self._worker = None
        self._cancel_event: Optional[threading.Event] = None
        self._session: Optional[PipelineSession] = None
        self._session_item: Optional[ExperimentItem] = None
        # Re-entrancy guard for the `setCurrentItem` call `_on_selection_changed`
        # makes to revert a blocked switch -- without it, that call's own
        # `currentItemChanged` re-entry would run the "leaving this row"
        # cleanup against the row we're refusing to leave.
        self._reverting_selection = False
        # The Labels layer the regions picker has chosen, held as the layer
        # itself rather than its name, so a rename doesn't lose it (see
        # `_on_region_layers_changed`).
        self._regions_layer: Optional[Labels] = None

        # The session's own detections/tracks layers, held as layers rather
        # than looked up by name: a filter drag redraws them in place, Save
        # renames them, and the user may rename them too. Checked for still
        # being in the viewer before each use (`_live`), since anything can
        # remove a layer.
        self._points_layer = None
        self._tracks_layer = None
        # Per-track metrics and per-vertex features for the session's
        # current `tracks_df`, computed once per link run rather than on
        # every handle move: `(tracks_df, metrics, features)`.
        self._track_cache: Optional[tuple] = None
        # Filter drags redraw the viewer through these rather than on every
        # mouse-move event: a Tracks layer rebuild is far slower than the
        # rate a handle emits, and queuing one per event made dragging lag.
        self._points_redraw = _debounce_timer(self, self._update_points_layer)
        self._tracks_redraw = _debounce_timer(self, self._update_tracks_layer)
        # A scale spinbox emits per arrow-click and per keystroke, and
        # applying one re-aggregates every track metric and redraws the
        # tracks layer -- so it lands once the typing stops, on a longer
        # fuse than a filter drag (which is a gesture, not an edit).
        self._scale_change = _debounce_timer(self, self._apply_image_scale_change, msec=300)

        # Showing a row's image (or bundle) runs in a worker, started after
        # a short pause in row changes so that arrowing down the list does
        # not queue a full stack read per row passed over. The generation
        # counter is how a load that finishes after the user moved on (or
        # started working on the row) knows to discard its result.
        self._load_generation = 0
        self._pending_load_item: Optional[ExperimentItem] = None
        self._load_timer = QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.setInterval(150)
        self._load_timer.timeout.connect(self._start_item_load)
        self._loading_text = ""
        # The last row load that landed, kept so the session for that row
        # reuses the stack already in memory instead of reading it again.
        self._loaded: Optional[tuple[ExperimentItem, ImageDisplay]] = None
        self.setAcceptDrops(True)

        header = QLabel(_ExperimentListView.KEYBINDINGS)
        header.setStyleSheet("color: gray; font-size: 11px;")

        self.list_view = _ExperimentListView()
        self.list_view.currentItemChanged.connect(self._on_selection_changed)
        self.list_view.is_busy = lambda: self._worker is not None
        self.list_view.on_busy_blocked = lambda: self.progress_label.setText(
            "a run is in progress — finishing before loading a new folder"
        )
        self.list_view.confirm_reload = self._confirm_reload

        self.open_button = QPushButton("Open folder…")
        self.open_button.clicked.connect(self._open_folder_dialog)
        self.batch_button = QPushButton("Batch from this movie…")
        self.batch_button.setToolTip(
            "Run the selected movie's saved settings (detect+track, and its\n"
            "saved diffusion analysis) over other movies in this folder."
        )
        self.batch_button.clicked.connect(self._open_batch_dialog)
        button_row = QHBoxLayout()
        button_row.addWidget(self.open_button)
        button_row.addWidget(self.batch_button)
        # The running batch: its plan, cancel flag and one line per movie
        # (shown when it ends). `self._worker` holds the worker itself, so
        # everything that waits on a step run also waits on a batch.
        self._batch_plan: Optional[BatchPlan] = None
        self._batch_log: list[str] = []
        self._batch_current = ""

        self.params_panel = PipelineParamsWidget()
        self.params_panel.detectRequested.connect(self._run_detect_step)
        self.params_panel.detectCancelRequested.connect(self._cancel_active_run)
        self.params_panel.trackRequested.connect(self._run_track_step)
        self.params_panel.saveRequested.connect(self._save_result)
        self.params_panel.saveDetectionsRequested.connect(self._save_detection_result)
        regions_panel = self.params_panel.regions_panel
        regions_panel.newLayerRequested.connect(self._on_new_regions_layer_requested)
        regions_panel.layerChosen.connect(self._on_region_layer_chosen)
        # Keep the regions picker showing the viewer's Labels layers.
        # Adding/removing/reordering layers is caught on the layer list
        # itself; a *rename* is an event on the layer, so
        # `_on_region_layers_changed` (re)connects to each Labels layer as
        # it goes -- napari's emitters ignore a duplicate connect, so this
        # can run as often as it likes.
        self.viewer.layers.events.inserted.connect(self._on_region_layers_changed)
        self.viewer.layers.events.removed.connect(self._on_region_layers_changed)
        self.viewer.layers.events.reordered.connect(self._on_region_layers_changed)
        self._on_region_layers_changed()
        self.params_panel.pointFiltersChanged.connect(self._on_point_filters_changed)
        self.params_panel.trackFiltersChanged.connect(self._on_track_filters_changed)
        self.params_panel.imageScaleChanged.connect(self._on_image_scale_changed)

        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        # Without this, an unwrapped/long status string (e.g. a full file
        # name or error message) sets its sizeHint as the label's minimum
        # width, which propagates up through `bottom_layout` and the
        # splitter to the dock widget itself -- making the dock refuse to
        # shrink narrower than whatever the longest message so far was.
        # Ignored lets the label shrink freely; word wrap keeps the text
        # readable instead of clipping it.
        self.progress_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        # Show actual counts, not just a bare percentage -- with the SFW
        # solver itself not speedable, seeing "137/500" (via `_on_progress`
        # driving the bar's range/value directly, frame-granular during
        # detect) is the only progress signal available for a long run.
        self.progress_bar.setFormat("%p%  (%v / %m)")

        # The params panel's tabs (esp. with an "Expert settings" section
        # expanded) want more vertical space than most dock heights
        # comfortably offer -- nested plain QWidget/QVBoxLayouts propagate a
        # child's full sizeHint up as their own *minimum* size, so without
        # this, that requirement would climb all the way to the dock
        # widget itself, capping how small it can ever be dragged (Finding:
        # "can't really resize the widget's vertical size") and pushing the
        # progress bar below the visible viewport on a shorter laptop
        # screen with no way to bring it back. A QScrollArea breaks that
        # chain -- it reports a small minimum size regardless of its
        # content and scrolls internally instead, so only the tabs scroll
        # when space is tight; the progress label/bar sit outside it (after
        # it in `bottom_layout`) so they always get their own space rather
        # than competing with the tabs for room.
        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setFrameShape(QFrame.Shape.NoFrame)
        params_scroll.setWidget(self.params_panel)

        # The list and the params/progress area share a drag-resizable
        # divider instead of a plain stacked QVBoxLayout -- the list already
        # scrolls internally when it overflows its own height
        # (_ExperimentListView is a plain QListWidget), so giving it less
        # room costs nothing but visible rows. A fixed split would just
        # trade one hidden thing for another depending on the dock's
        # height; a splitter lets it be tuned per-session instead, and the
        # bottom pane is pinned non-collapsible so the progress bar can
        # never be dragged out of view entirely.
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(params_scroll, 1)
        bottom_layout.addWidget(self.progress_label)
        bottom_layout.addWidget(self.progress_bar)
        self.batch_cancel_button = QPushButton("Cancel batch")
        self.batch_cancel_button.setVisible(False)
        self.batch_cancel_button.clicked.connect(self._cancel_active_run)
        bottom_layout.addWidget(self.batch_cancel_button)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.list_view)
        splitter.addWidget(bottom)
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([220, 360])

        layout = QVBoxLayout()
        layout.addWidget(header)
        layout.addLayout(button_row)
        layout.addWidget(splitter)
        self.setLayout(layout)

    # -- Batch from the current row's saved bundle --

    def _open_batch_dialog(self) -> None:
        item = self.list_view.currentItem()
        if self._worker is not None:
            self.progress_label.setText("a run is in progress -- wait for it to finish")
            return
        if item is None or not has_result(item.entry.result_dir):
            QMessageBox.information(
                self,
                "Batch from this movie",
                "Select a movie with saved results first: its saved settings are what the "
                "batch runs on the others.",
            )
            return
        template = item.entry.result_dir
        entries = [
            (other.entry.image_path, other.entry.result_dir, other.entry.status is Status.SKIP)
            for other in self.list_view.items()
            if other is not item
        ]
        try:
            dialog = BatchDialog(template, self.list_view.results_root, entries, parent=self)
        except Exception as exc:
            QMessageBox.warning(self, "Batch from this movie", f"Can't read {template.name}: {exc}")
            return
        if dialog.exec() != BatchDialog.DialogCode.Accepted:
            return
        self._start_batch(dialog.plan())

    def _start_batch(self, plan: BatchPlan) -> None:
        if not plan.jobs:
            return
        self._batch_log = []
        if plan.write_config:
            config_path = batch_config_path(plan.results_root, plan.template)
            try:
                plan.results_root.mkdir(parents=True, exist_ok=True)
                write_batch_config(
                    config_path,
                    results_root=plan.results_root,
                    template=plan.template,
                    inputs=[(job.image_path, job.result_dir.name) for job in plan.jobs],
                    diffusion=plan.diffusion,
                )
                self._batch_log.append(f"config: {config_path}")
            except Exception as exc:
                self._batch_log.append(f"! config not written: {exc}")

        self._batch_plan = plan
        self._cancel_event = threading.Event()
        emitter = BatchEmitter(self)
        emitter.job_started.connect(self._on_batch_job_started)
        emitter.progress.connect(self._on_batch_progress)
        emitter.bundle_written.connect(self._on_batch_bundle_written)
        emitter.job_finished.connect(self._on_batch_job_finished)
        worker = run_batch_worker(plan, self._cancel_event, emitter)
        # Kept alive until the worker is done with it.
        worker.finished.connect(emitter.deleteLater)
        self._set_batch_running(True)
        self._start_step_worker(worker, self._on_batch_done, on_error=self._on_batch_error)

    def _set_batch_running(self, running: bool) -> None:
        # The list and the params stay put while movies are being written:
        # showing a row mid-write, or starting a step run, would race it.
        self.list_view.setEnabled(not running)
        self.open_button.setEnabled(not running)
        self.batch_button.setEnabled(not running)
        self.params_panel.setEnabled(not running)
        self.batch_cancel_button.setVisible(running)
        self.batch_cancel_button.setEnabled(running)

    def _batch_job_prefix(self, index: int) -> str:
        return f"[{index + 1}/{len(self._batch_plan.jobs)}]"

    def _on_batch_job_started(self, index: int, name: str) -> None:
        self._batch_current = f"{self._batch_job_prefix(index)} {name}"
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText(self._batch_current)

    def _on_batch_progress(self, done: int, total: int, stage: str) -> None:
        if not total:
            self.progress_bar.setRange(0, 0)
        self._on_progress(done, total, stage)
        self.progress_label.setText(f"{self._batch_current}\n{self.progress_label.text()}")

    def _batch_item(self, index: int) -> Optional[ExperimentItem]:
        job = self._batch_plan.jobs[index]
        for item in self.list_view.items():
            if item.entry.result_dir == job.result_dir:
                return item
        return None

    def _on_batch_bundle_written(self, index: int, n_tracks: int) -> None:
        item = self._batch_item(index)
        if item is not None:
            item.set_status(Status.COMPLETE, n_tracks=n_tracks)
            self.list_view.viewport().update()

    def _on_batch_job_finished(self, index: int, ok: bool, message: str) -> None:
        name = self._batch_plan.jobs[index].image_path.name
        mark = "✓" if ok else "✗"
        self._batch_log.append(f"{mark} {name}: {message}")

    def _on_batch_done(self, completed: bool) -> None:
        plan = self._batch_plan
        n_ok = sum(line.startswith("✓") for line in self._batch_log)
        n_failed = sum(line.startswith("✗") for line in self._batch_log)
        self._end_batch()
        headline = (
            f"Batch finished: {n_ok} of {len(plan.jobs)} movies done"
            if completed
            else f"Batch cancelled: {n_ok} of {len(plan.jobs)} movies done"
        )
        if n_failed:
            headline += f", {n_failed} failed"
        self.progress_label.setText(headline)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning if n_failed else QMessageBox.Icon.Information)
        box.setWindowTitle("Batch from this movie")
        box.setText(headline)
        box.setDetailedText("\n".join(self._batch_log))
        box.exec()

    def _on_batch_error(self, exc: Exception) -> None:
        self._end_batch()
        self.progress_label.setText(f"batch error: {exc}")

    def _end_batch(self) -> None:
        self._finish_step_worker()
        self._set_batch_running(False)
        self._batch_plan = None

    def _open_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder of timelapses")
        if folder:
            self.list_view.load_folder(Path(folder))

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        folder = _dropped_folder(event)
        if folder is not None:
            self.list_view.load_folder(folder)

    def _on_selection_changed(self, current: Optional[ExperimentItem], _previous) -> None:
        if current is None or self._reverting_selection:
            return
        if self._worker is not None and self._session_item is not None and current is not self._session_item:
            # A detect/track worker is still running against
            # `self._session_item` -- switching away would otherwise wipe
            # its viewer layers and orphan the worker's eventual result
            # (see `self._worker`'s comment). Snap the selection back
            # rather than let that happen.
            self._reverting_selection = True
            self.list_view.setCurrentItem(self._session_item)
            self._reverting_selection = False
            self.progress_label.setText(
                f"a run is in progress for {self._session_item.entry.image_path.name} "
                "-- finishing before switching items"
            )
            return
        if self._session_item is not None and current is not self._session_item:
            if not self._confirm_leave_session():
                self._reverting_selection = True
                self.list_view.setCurrentItem(self._session_item)
                self._reverting_selection = False
                return
        self._drop_session()
        self.params_panel.set_detect_status("")
        self.params_panel.set_track_status("")
        self.params_panel.set_save_status("")
        self.params_panel.set_save_enabled(False)
        self.params_panel.set_detect_save_status("")
        self.params_panel.set_detect_save_enabled(False)
        # The histogram ranges belong to a table that's about to be
        # unloaded -- carrying them onto the next image would apply cuts
        # chosen against a different population, which is exactly the
        # mistake the histograms exist to prevent.
        self.params_panel.clear_filters()
        self.params_panel.set_point_filter_source(None)
        self.params_panel.set_track_filter_source(None)
        # Blanked rather than left showing the row being left -- the load
        # that will fill it in is debounced, and a stale pixel size on
        # screen is worse than none.
        self.params_panel.set_image_metadata(None)
        self.list_view.viewport().update()

        # Blank the viewer now rather than when the load lands: until then
        # the previous row's layers would sit there looking like this one's.
        self.viewer.layers.clear()
        self._request_item_load(current)

    def _drop_session(self) -> None:
        """Forget the in-memory session and everything derived from it."""
        if self._session_item is not None:
            self._session_item.entry.has_unsaved_session = False
        self._session = None
        self._session_item = None
        self._track_cache = None
        self._points_layer = None
        self._tracks_layer = None
        self._points_redraw.stop()
        self._tracks_redraw.stop()
        self._scale_change.stop()

    def _confirm_leave_session(self) -> bool:
        """Whether it is fine to drop the current session: True straight
        away if nothing in it is unsaved, otherwise the user's answer to a
        Save / Discard / Stay prompt. Leaving used to discard silently --
        one arrow-key press in the list was enough to lose a finished
        detect-and-link."""
        item, session = self._session_item, self._session
        if item is None or session is None or not item.entry.has_unsaved_session:
            return True
        # Tracks, if any, always take Save down the full points+tracks path
        # (`_save_result`); with points alone (tracking hasn't run, or has
        # but wasn't kept), Save writes just the detections
        # (`_save_detection_result`) -- either way there is something to
        # save as soon as detect has run, which `has_unsaved_session` above
        # already implies.
        has_tracks = session.tracks_df is not None and session.tracks_df.height > 0
        savable = has_tracks or session.points_df is not None
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved results")
        box.setText(f"{item.entry.image_path.name} has results that haven't been saved.")
        box.setInformativeText(
            "Save writes its results bundle first; Discard drops them."
            if savable
            else "Nothing has been detected yet, so there is nothing to save — leaving drops it."
        )
        save = box.addButton("Save", QMessageBox.ButtonRole.AcceptRole) if savable else None
        discard = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        stay = box.addButton("Stay", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(stay)
        box.setEscapeButton(stay)
        box.exec()
        clicked = box.clickedButton()
        if clicked is discard:
            return True
        if save is not None and clicked is save:
            return self._save_result(item) if has_tracks else self._save_detection_result(item)
        return False

    def _confirm_reload(self) -> bool:
        """`_ExperimentListView.confirm_reload`: a reload replaces every
        row, so it leaves the session's row just as a selection change
        does."""
        if not self._confirm_leave_session():
            return False
        self._drop_session()
        self._loaded = None
        return True

    # -- Showing a row (off the GUI thread) --

    def _request_item_load(self, item: ExperimentItem) -> None:
        """Show `item`'s bundle, or its raw image if it has none, once the
        row has been sat on briefly (`_load_timer`)."""
        self._load_generation += 1
        self._pending_load_item = item
        self._loaded = None
        self._load_timer.start()

    def _start_item_load(self) -> None:
        item = self._pending_load_item
        if item is None or self.list_view.currentItem() is not item:
            return
        generation = self._load_generation
        entry = item.entry
        self._loading_text = f"loading {entry.image_path.name}…"
        self.progress_label.setText(self._loading_text)
        worker = _load_item_worker(entry.image_path, entry.result_dir)
        worker.returned.connect(
            lambda loaded, item=item, g=generation: self._on_item_loaded(item, g, loaded)
        )
        worker.errored.connect(
            lambda exc, item=item, g=generation: self._on_item_load_error(item, g, exc)
        )
        worker.start()

    def _load_is_current(self, item: ExperimentItem, generation: int) -> bool:
        return generation == self._load_generation and self.list_view.currentItem() is item

    def _clear_loading_text(self) -> None:
        # Only our own message: a step run shares this label.
        if self.progress_label.text() == self._loading_text:
            self.progress_label.setText("")

    def _on_item_loaded(self, item: ExperimentItem, generation: int, loaded) -> None:
        if not self._load_is_current(item, generation):
            return
        self._clear_loading_text()
        image = loaded.image if isinstance(loaded, ResultDisplay) else loaded
        self._loaded = (item, image)
        self.params_panel.set_frame_bounds(image.image.shape[0])
        # What the file says about itself, shown on selection rather than
        # only discovered when a run fails on it -- see
        # `PipelineParamsWidget.set_image_metadata`.
        self.params_panel.set_image_metadata(image.metadata)
        if isinstance(loaded, ResultDisplay):
            show_result(self.viewer, loaded)
            self._restore_filters_from_bundle(loaded)
            self._adopt_bundle_session(item, loaded)
        else:
            show_image(self.viewer, loaded)

    def _adopt_bundle_session(self, item: ExperimentItem, loaded: ResultDisplay) -> None:
        """Make a saved bundle on screen a live session, not just a
        picture of one: its detections become `session.points_df`, so
        Track links them straight away, and its tracks (if any) can be
        re-filtered and re-saved -- without re-running detect, which is
        what the Track button used to demand of a reopened bundle.

        Runs after `_restore_filters_from_bundle`, whose filter updates
        would otherwise read as edits to this session. The layers
        `show_result` just added are adopted as the session's own, so
        filter drags redraw them in place instead of adding a second copy,
        and the bundle's saved regions become the picked layer again."""
        image = loaded.image
        try:
            session = session_from_bundle(
                image.path,
                (image.image, image.metadata),
                loaded.points_df,
                loaded.tracks_df,
                loaded.manifest,
                loaded.labels,
                loaded.regions,
            )
        except Exception as exc:
            self.params_panel.set_detect_status(f"could not resume the saved bundle: {exc}", level="error")
            return
        self._session = session
        self._session_item = item
        item.entry.has_unsaved_session = False
        self._track_cache = None
        for attr, name in (("_points_layer", "points"), ("_tracks_layer", "tracks")):
            layer = self.viewer.layers[name] if name in self.viewer.layers else None
            setattr(self, attr, layer)
        if loaded.labels is not None:
            self._pick_regions_layer(REGIONS_LAYER_NAME)

        n_points = session.points_df.height
        self.params_panel.set_detect_status(
            f"{n_points} points from the saved bundle — Track links them as they are"
            + self._region_count_text(session),
            level="ok" if n_points else "error",
        )
        self.params_panel.set_detect_save_enabled(False)
        self._update_points_layer()
        if session.tracks_df is not None:
            self.params_panel.set_save_enabled(True)
            n_tracks = session.tracks_df["track_id"].n_unique()
            self.params_panel.set_track_status(
                f"{n_tracks} tracks from the saved bundle" + self._region_track_text(session)
            )
        else:
            self.params_panel.set_track_status("not linked yet — press Track to link the saved points")

    def _on_item_load_error(self, item: ExperimentItem, generation: int, exc: Exception) -> None:
        if not self._load_is_current(item, generation):
            return
        self.progress_label.setText(f"could not load {item.entry.image_path.name}: {exc}")

    def _cancel_active_run(self) -> None:
        """Requests cancellation of the running Detect step -- cooperative
        only, see `PipelineCancelled`'s docstring for how promptly this
        actually takes effect per stage."""
        if self._cancel_event is not None:
            self._cancel_event.set()

    def _on_progress(self, done: int, total: int, stage: str) -> None:
        if total:
            # Drive the bar's range/value with the real counts rather than
            # a pre-computed percentage -- lets progress_bar's "%v / %m"
            # format (set in __init__) show e.g. "137 / 500", the only
            # concrete sense of how much is left on a long detect run that
            # can't be sped up.
            if self.progress_bar.maximum() != total:
                self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(done)
            self.progress_label.setText(f"{stage} -- {done}/{total}")
        else:
            self.progress_label.setText(stage)

    # -- Stepwise Detect / Track (current selection only) --

    def _ensure_session(self, item: ExperimentItem) -> PipelineSession:
        """The session for `item`, loading its image fresh if `item` isn't
        already the one `self._session` was built for (selection changes
        already reset `self._session` to None, but this also covers the
        first click for a freshly-selected item)."""
        if self._session is not None and self._session_item is item:
            return self._session
        # Starting work on the row outranks a load of it still in flight:
        # that load would clear the viewer out from under this session.
        self._load_generation += 1
        self._load_timer.stop()
        self._clear_loading_text()
        loaded = self._loaded[1] if self._loaded is not None and self._loaded[0] is item else None
        session = load_session(
            item.entry.image_path,
            pixel_size_um=self.params_panel.get_pixel_size_um(),
            dt_s=self.params_panel.get_dt_s(),
            exposure_s=self.params_panel.get_exposure_s(),
            stack=(loaded.image, loaded.metadata) if loaded is not None else None,
        )
        if loaded is None:
            # The row's load never landed, so its image isn't on screen.
            # Added under whatever is there (regions painted meanwhile, say)
            # rather than clearing it away.
            layer = add_image_layer(self.viewer, session.image, item.entry.image_path.stem)
            self.viewer.layers.move(self.viewer.layers.index(layer), 0)
        self._session = session
        self._session_item = item
        # The session is what the stages actually run with, so the banner
        # follows it: for a row whose bundle was on screen a moment ago,
        # that swaps the manifest's recorded numbers back to the file's.
        self.params_panel.set_image_metadata(session.metadata)
        return session

    def _region_layers(self) -> list[Labels]:
        """Every Labels layer in the viewer, top of napari's layer list
        first -- the candidates for the regions picker."""
        return [layer for layer in reversed(self.viewer.layers) if isinstance(layer, Labels)]

    def _on_region_layer_chosen(self, name: str) -> None:
        layer = self.viewer.layers[name] if name in self.viewer.layers else None
        self._regions_layer = layer if isinstance(layer, Labels) else None
        self.params_panel.regions_panel.set_layer(self._regions_layer)

    def _on_region_layers_changed(self, event=None) -> None:
        """Push the current Labels layer names into the regions picker,
        and make sure a rename of any of them lands here too (each layer's
        own `events.name`, since the layer list only reports add/remove/
        reorder). The picker follows the *layer* (`self._regions_layer`)
        across a rename; with none chosen, or the chosen one removed, it
        falls to the top Labels layer."""
        layers = self._region_layers()
        for layer in layers:
            layer.events.name.connect(self._on_region_layers_changed)
        if self._regions_layer is not None and not any(self._regions_layer is l for l in layers):
            self._regions_layer = None
        if self._regions_layer is None and layers:
            self._regions_layer = layers[0]
        panel = self.params_panel.regions_panel
        panel.set_layer_choices(
            [layer.name for layer in layers],
            self._regions_layer.name if self._regions_layer is not None else None,
        )
        panel.set_layer(self._regions_layer)

    def _pick_regions_layer(self, name: str) -> None:
        """Make `name` the regions layer (plus switch "restrict to regions"
        on) -- how a reopened bundle's saved regions become the active
        ones again, so re-tracking or re-detecting it covers the same
        regions under the same labels."""
        layer = self.viewer.layers[name] if name in self.viewer.layers else None
        if isinstance(layer, Labels):
            self._regions_layer = layer
            self._on_region_layers_changed()
            self.params_panel.regions_panel.set_use_mask(True)

    def _on_new_regions_layer_requested(self) -> None:
        """Add an empty regions Labels layer the size of one frame, ready
        to paint on: 2D so what is painted shows on every frame, and
        picked straight away, since it was asked for to be used."""
        session = self._session
        image_layers = [layer for layer in self.viewer.layers if isinstance(layer, Image)]
        if session is not None:
            shape = session.image.shape[1:]
        elif image_layers:
            shape = image_layers[0].data.shape[-2:]
        else:
            self.params_panel.set_detect_status("open an image first", level="error")
            return
        existing = {layer.name for layer in self.viewer.layers}
        name, n = REGIONS_LAYER_NAME, 1
        while name in existing:
            n += 1
            name = f"{REGIONS_LAYER_NAME} {n}"
        layer = add_regions_layer(self.viewer, np.zeros(shape, dtype=np.uint16), name=name)
        self.viewer.layers.selection.active = layer
        layer.mode = "paint"
        self._regions_layer = layer
        self._on_region_layers_changed()
        self.params_panel.regions_panel.set_use_mask(True)

    def _build_regions(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, Regions]:
        """`(mask, labels, regions)` for the picked regions layer: a copy
        of its labels image (painting can go on while a run is in flight),
        a copy of its table synced to what is painted, and `labels > 0` --
        the `roi` mask spotsolve is handed. The labels and table go on the
        session so `_save_result` can persist them, and the labels stamp
        each detection with its region (`regions.label_points`).

        Raises if no layer is picked, it is gone, it doesn't match the
        frame, or nothing is painted on it yet."""
        panel = self.params_panel.regions_panel
        layer = self._regions_layer
        if layer is None or not any(layer is l for l in self.viewer.layers):
            raise ValueError('no regions layer — press "New layer" and paint, or pick one')
        labels = np.asarray(layer.data)
        if labels.shape != tuple(shape):
            raise ValueError(
                f"regions layer {layer.name!r} is {labels.shape}, but a frame is {tuple(shape)}"
            )
        regions = panel.regions()
        if not regions.table:
            raise ValueError(f"nothing painted on regions layer {layer.name!r} yet")
        labels = labels.astype(np.uint16, copy=True)
        return labels > 0, labels, copy.deepcopy(regions)

    def _start_step_worker(
        self, worker, on_finished, indeterminate: bool = False, on_error: Optional[Callable] = None
    ) -> None:
        if self._worker is not None:
            return
        self.progress_bar.setRange(0, 0 if indeterminate else 100)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        worker.returned.connect(on_finished)
        worker.errored.connect(on_error or self._on_step_error)
        self._worker = worker
        worker.start()

    def _finish_step_worker(self) -> None:
        self._worker = None
        self._cancel_event = None
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")

    def _on_step_error(self, exc: Exception) -> None:
        self.progress_label.setText(f"error: {exc}")
        self._finish_step_worker()

    def _run_detect_step(self) -> None:
        item = self.list_view.currentItem()
        if item is None:
            return
        try:
            session = self._ensure_session(item)
        except Exception as exc:
            self.params_panel.set_detect_status(f"error: {exc}", level="error")
            return
        self.params_panel.set_frame_bounds(session.image.shape[0])

        mask = None
        session.labels = session.regions = None
        if self.params_panel.regions_panel.get_use_mask():
            try:
                mask, session.labels, session.regions = self._build_regions(session.image.shape[1:])
            except Exception as exc:
                self.params_panel.set_detect_status(f"error: {exc}", level="error")
                return

        # The Detect tab's sigma box, always -- not `session.sigma` from
        # some earlier run, so the number on screen is the one that ran.
        sigma = self.params_panel.get_sigma()
        self.progress_label.setText(f"Finding spots: {item.entry.image_path.name}")
        self._cancel_event = threading.Event()
        self.params_panel.set_detect_running(True)
        emitter = _ProgressEmitter()
        emitter.updated.connect(self._on_progress)
        # Also mirror frame-by-frame progress into the Detect tab's own
        # status line -- that row (next to the Run/Cancel button) is what's
        # actually in view while the user is watching a long detect run;
        # the bottom progress bar can scroll out of sight on a short
        # screen (see the params_scroll wiring in __init__).
        emitter.updated.connect(self.params_panel.set_detect_progress)
        worker = _run_detect_worker(
            session,
            sigma,
            self.params_panel.get_camera_kwargs(),
            self.params_panel.get_detect_kwargs(),
            self.params_panel.get_frame_range(),
            mask,
            self._cancel_event,
            emitter,
            self.params_panel.get_n_threads(),
            self.params_panel.get_detector(),
        )
        self._start_step_worker(
            worker,
            lambda s, item=item: self._on_detect_finished(item, s),
            on_error=lambda exc, item=item: self._on_detect_error(item, exc),
        )

    def _on_detect_finished(self, item: ExperimentItem, session: PipelineSession) -> None:
        self.params_panel.set_detect_running(False)
        if self._session_item is not item:
            self._finish_step_worker()
            return
        self._session = session
        if session.labels is not None and session.points_df is not None:
            # Every detection gets the region it fell in -- the label
            # tracking splits on and the diffusion panel groups by.
            session.points_df = label_points(session.points_df, session.labels, session.regions)
        item.entry.has_unsaved_session = True
        self.list_view.viewport().update()
        n_points = session.points_df.height if session.points_df is not None else 0
        start, end = session.frame_range_used or (0, session.image.shape[0])
        # Shown next to the detection count because a run where most fits
        # carry a FitFlag (usually AT_BOUND: widths pinned against slack)
        # is a sigma or focus problem, and the count alone hides it.
        extra = ""
        frames = session.frames_df
        if frames is not None and frames.height:
            n_flagged = int(frames["n_flagged"].sum())
            if n_flagged:
                extra = f"  ({n_flagged} flagged)"
        self.params_panel.set_detect_status(
            f"{n_points} points across frames {start}-{end - 1}{extra}"
            + self._region_count_text(session),
            level="error" if n_points == 0 else "ok",
        )

        # Hand the run's detections to the filter histograms (where
        # fit_sigma is read to settle the PSF width) and draw the layer
        # through whatever cuts are already set. The Detect tab
        # pages through its steps rather than scrolling, so leaf it to the
        # filter page too -- that histogram is what there is to do next,
        # and it is no longer just below the Run button.
        self.params_panel.set_point_filter_source(session.points_df)
        self.params_panel.show_tab("Detect", "Filter")
        self._update_points_layer(new_data=True)
        self.params_panel.set_save_enabled(False)
        self.params_panel.set_detect_save_enabled(n_points > 0)
        self.params_panel.set_detect_save_status("not saved yet" if n_points > 0 else "", level="caution")
        self._finish_step_worker()

    @staticmethod
    def _region_count_text(session: PipelineSession) -> str:
        """Per-class detection counts for the Detect status line, for a
        run over more than one region, e.g. `nucleus: 412 · cytoplasm:
        1030 (6 cells)`."""
        df = session.points_df
        if session.regions is None or df is None or "region_class" not in df.columns:
            return ""
        if len(session.regions.table) < 2:
            return ""
        counts = dict(df.group_by("region_class").len().iter_rows())
        parts = [f"{name}: {counts.get(name, 0)}" for name in session.regions.class_names()]
        n_cells = len({r.cell for r in session.regions.table.values()})
        return "\n" + " · ".join(parts) + f"  ({n_cells} cell{'s' if n_cells != 1 else ''})"

    @staticmethod
    def _region_track_text(session: PipelineSession) -> str:
        """One short line per region class for the Track status -- each
        class was linked with its own fitted parameters (`run_track_step`),
        so its own D is the number worth reading, not only the pooled one."""
        by_class = (session.track_summary or {}).get("by_class")
        if not by_class:
            return ""
        lines = []
        for name, row in by_class.items():
            line = (
                f"{name}: {row.get('n_tracks', 0)} tracks  "
                f"D ≈ {units.fmt(row.get('D_est_um2_s'), 'D_est_um2_s')}"
            )
            if row.get("fell_back_to_pooled"):
                line += " (too few points to fit alone — pooled link params)"
            lines.append(line)
        return "\n" + "\n".join(lines)

    def _on_detect_error(self, item: ExperimentItem, exc: Exception) -> None:
        self.params_panel.set_detect_running(False)
        if isinstance(exc, PipelineCancelled):
            self.params_panel.set_detect_status("cancelled")
        else:
            self.params_panel.set_detect_status(f"error: {exc}", level="error")
        self._finish_step_worker()

    def _run_track_step(self) -> None:
        item = self.list_view.currentItem()
        if item is None:
            return
        try:
            session = self._ensure_session(item)
        except Exception as exc:
            self.params_panel.set_track_status(f"error: {exc}", level="error")
            return
        if session.points_df is None:
            self.params_panel.set_track_status("error: run detect first", level="error")
            return
        self.progress_label.setText(f"Linking: {item.entry.image_path.name}")
        emitter = _ProgressEmitter()
        emitter.updated.connect(self._on_progress)
        worker = _run_track_worker(
            session,
            self.params_panel.get_min_track_length(),
            self.params_panel.get_exclude_flags(),
            self.params_panel.get_link_with_flux(),
            self.params_panel.get_min_link_margin(),
            # The Detect tab's cuts decide what the linker sees -- applied
            # here rather than to `points_df`, which keeps every detection
            # (see run_track_step's docstring).
            self.params_panel.get_point_filters(),
            emitter,
        )
        self._start_step_worker(worker, lambda s, item=item: self._on_track_finished(item, s))

    def _on_track_finished(self, item: ExperimentItem, session: PipelineSession) -> None:
        if self._session_item is not item:
            self._finish_step_worker()
            return
        self._session = session
        item.entry.has_unsaved_session = True
        summary = session.track_summary or {}
        n_tracks = (
            session.tracks_df["track_id"].n_unique() if session.tracks_df is not None and session.tracks_df.height else 0
        )
        verdict = summary.get("resolvability_verdict", "ok")
        level = {"ok": "ok", "caution": "caution", "unresolvable": "error"}.get(verdict, "neutral")
        message = summary.get("resolvability_message", "")
        # Both D estimates, because they answer the question two ways: the
        # MSD moment of the finished tracks, and the linker's own fitted
        # population mean. Agreement is reassuring; a large gap means the
        # linking is suspect, and neither number alone would show it.
        cuts = [
            f"{n} {what}"
            for n, what in (
                (summary.get("n_points_dropped_invalid"), "points without a usable error"),
                (summary.get("n_points_dropped_flagged"), "flagged points"),
                (summary.get("n_points_dropped_by_filter"), "points cut by filters"),
                (summary.get("n_links_rejected"), "links under the margin"),
            )
            if n
        ]
        filtered = f"  ({', '.join(cuts)})" if cuts else ""
        # Units from `napari_gemscape2.units` rather than spelled out here, so
        # this line, the diffusion panel's fit summaries and every plot
        # axis say µm²/s the same way.
        self.params_panel.set_track_status(
            f"{n_tracks} tracks  D ≈ {units.fmt(summary.get('D_est_um2_s'), 'D_est_um2_s')} "
            f"(linker fit {units.fmt(summary.get('D_link_um2_s'), 'D_link_um2_s')}, "
            f"immobile {summary.get('immobile_fraction', 0.0):.0%}){filtered}"
            f"{self._region_track_text(session)}\n{message}",
            level=level,
        )

        # Feed the Track tab's histograms one row per track, then draw the
        # layer through whatever cuts survive. Nothing is written yet --
        # `_save_result` is the finalize step now, so the filters can
        # be tuned against the linked result before it's committed.
        self._track_cache = None
        self.params_panel.set_track_filter_source(self._track_tables()[0])
        self._update_tracks_layer(new_data=True)
        self.params_panel.set_save_enabled(session.tracks_df is not None and session.tracks_df.height > 0)
        self.params_panel.set_save_status("not saved yet", level="caution")
        self.params_panel.show_tab("Track", "Filter")
        self.list_view.viewport().update()
        self._finish_step_worker()

    # -- Filters -> viewer (live) --

    def _live(self, layer):
        """`layer` if it is still in the viewer, else None -- a held layer
        can be deleted from under us at any time (by the user, or by a
        `layers.clear()`)."""
        return live_layer(self.viewer, layer)

    def _track_tables(self) -> tuple:
        """`(metrics, features)` for the session's tracks: one row per track
        for the Track tab's histograms, and the per-vertex table the Tracks
        layer draws. Cached per `tracks_df`, so a filter drag only filters
        them rather than re-aggregating every track. `(None, None)` with
        nothing linked."""
        session = self._session
        if session is None or session.tracks_df is None or session.tracks_df.height == 0:
            return None, None
        cached = self._track_cache
        if cached is None or cached[0] is not session.tracks_df:
            metrics = track_metrics_df(session.tracks_df, session.pixel_size_um, session.dt_s)
            features = track_features_df(session.tracks_df, session.pixel_size_um, session.dt_s)
            self._track_cache = cached = (session.tracks_df, metrics, features)
        return cached[1], cached[2]

    def _mark_unsaved(self) -> None:
        if self._session_item is not None and not self._session_item.entry.has_unsaved_session:
            self._session_item.entry.has_unsaved_session = True
            self.list_view.viewport().update()

    def _on_point_filters_changed(self) -> None:
        self._points_redraw.start()
        if self._session is not None and self._session.points_df is not None:
            self._mark_unsaved()
        if self._session is not None and self._session.tracks_df is not None:
            # The linked tracks were produced under the previous cuts, so
            # they no longer follow from what's on screen. Say so instead
            # of letting a stale track layer look current.
            self.params_panel.set_track_status(
                "detection filters changed — re-run tracking", level="caution"
            )
            self.params_panel.set_save_enabled(False)

    def _on_image_scale_changed(self) -> None:
        """The override changed -- apply it once the edit settles."""
        self._scale_change.start()

    def _apply_image_scale_change(self) -> None:
        """The pixel size / frame interval / exposure override changed.

        For the next run there is nothing to do -- `_ensure_session`
        reads the panel when it starts. What needs
        handling is a session already holding results, since those were
        computed at the old scale: `points_df`'s `t`/`y_um`/`se_*_um`
        columns were derived by `loctable` at detect time and the track
        summary's D/density/crowding at link time. Linking itself is
        unaffected (it works in pixels), so the tracks are still the same
        tracks -- but the bundle would be internally inconsistent if saved
        now, which is why this re-scales what it cheaply can, says what it
        can't, and blocks the save until a re-run.
        """
        self._scale_change.stop()
        session = self._session
        if session is None:
            return
        pixel_size_um, dt_s = self.params_panel.get_effective_image_scale()
        exposure_s = self.params_panel.get_effective_exposure_s()
        if pixel_size_um is None or dt_s is None:
            return
        scale_changed = (pixel_size_um, dt_s) != (session.pixel_size_um, session.dt_s)
        exposure_changed = exposure_s != session.exposure_s
        if not scale_changed and not exposure_changed:
            return
        session.pixel_size_um = pixel_size_um
        session.dt_s = dt_s
        session.exposure_s = exposure_s

        # Layer metadata first: the diffusion panel reads its conversion
        # factors from there, and it is reading the layers right now.
        metadata = self._session_layer_metadata()
        for layer in (self._live(self._points_layer), self._live(self._tracks_layer)):
            if layer is not None:
                layer.metadata.update(metadata)

        if not scale_changed:
            # Exposure alone: nothing computed here depends on it -- only
            # the manifest records it, for the diffusion analysis -- so
            # the results stand, and only the saved record is behind.
            if session.points_df is not None:
                self._mark_unsaved()
                self.params_panel.set_save_status(
                    "exposure changed — save to record it", level="caution"
                )
            return

        if session.tracks_df is not None:
            # Per-track metrics (`mean_step_um`, `duration_s`) are derived
            # from the scale, so they and the histograms over them are
            # recomputed -- a cut on "mean step < 0.2 um" has to mean the
            # new µm.
            self._track_cache = None
            self.params_panel.set_track_filter_source(self._track_tables()[0])
            self._update_tracks_layer(new_data=True)
        if session.points_df is not None:
            self._mark_unsaved()
            self.params_panel.set_save_enabled(False)
            self.params_panel.set_detect_save_enabled(False)
            self.params_panel.set_detect_status(
                "scale changed — re-run detect so the saved table's µm and s columns match",
                level="caution",
            )
            if session.tracks_df is not None:
                self.params_panel.set_track_status(
                    "scale changed — re-run detect and tracking to update D and the crowding check",
                    level="caution",
                )

    def _on_track_filters_changed(self) -> None:
        self._tracks_redraw.start()
        if self._session is not None and self._session.tracks_df is not None:
            self._mark_unsaved()
            self.params_panel.set_save_status("filters changed — save to apply", level="caution")

    def _session_layer_metadata(self) -> dict:
        """The same `pixel_size_um`/`dt_s`/`result_dir` metadata
        `viewer.show_result` puts on a saved bundle's layers, for the
        in-memory session's layers -- so a widget reading either (e.g.
        widgets/diffusion_panel.py) works the same on a preview as on a
        loaded bundle."""
        session = self._session
        metadata = layer_units_metadata(
            session.pixel_size_um if session is not None else None,
            session.dt_s if session is not None else None,
            self._session_item.entry.result_dir if self._session_item is not None else None,
            session.exposure_s if session is not None else None,
        )
        # What each `region` label on the layers' rows is (see
        # `napari_gemscape2.regions.label_points`).
        metadata["region_classes"] = region_classes(session.regions if session else None)
        return metadata

    def _update_points_layer(self, new_data: bool = False) -> None:
        """Show only the detections that pass the Detect tab's cuts --
        filtered spots vanish from the image as the handle moves, which is
        the whole point of putting the histogram next to the viewer rather
        than in a report.

        A cut only toggles the layer's `shown` mask; the layer is created
        once, and its data replaced in place (`new_data`) only when a
        detect run produced a new table. It used to be deleted and re-added
        on every mouse-move of a handle."""
        self._points_redraw.stop()
        session = self._session
        if session is None or session.points_df is None:
            return
        df = session.points_df
        layer = self._live(self._points_layer)
        if df.height == 0:
            if layer is not None:
                self.viewer.layers.remove(layer)
            self._points_layer = None
            return
        if layer is None:
            layer = add_points_layer(
                self.viewer, df, "points (preview)", self._session_layer_metadata()
            )
            self._points_layer = layer
        elif new_data:
            layer.metadata.update(self._session_layer_metadata())
            set_points_layer_data(layer, df)
            # A re-run after a save: this is a preview again until saved.
            layer.name = "points (preview)"
        filters = self.params_panel.get_point_filters()
        layer.shown = filter_mask(df, filters).to_numpy() if filters else True

    def _update_tracks_layer(self, new_data: bool = False) -> None:
        """Show only the linked tracks that pass the Track tab's cuts, redrawn
        in place (`viewer.set_tracks_layer_data`) so the layer -- and the
        diffusion panel reading it -- survives the drag. A Tracks layer has
        no per-track visibility, so unlike detections this replaces the
        layer's data with the passing tracks; with none passing, the layer
        is removed."""
        self._tracks_redraw.stop()
        session = self._session
        metrics, features = self._track_tables()
        if session is None or features is None:
            return
        filters = self.params_panel.get_track_filters()
        tracks_df = session.tracks_df
        if filters:
            keep = pl.col("track_id").is_in(apply_filters(metrics, filters)["track_id"])
            features = features.filter(keep)
            tracks_df = tracks_df.filter(keep)
        layer = self._live(self._tracks_layer)
        if features.height == 0:
            if layer is not None:
                self.viewer.layers.remove(layer)
            self._tracks_layer = None
            return
        if layer is None:
            self._tracks_layer = add_tracks_layer(
                self.viewer,
                tracks_df,
                session.pixel_size_um,
                session.dt_s,
                "tracks (preview)",
                self._session_layer_metadata(),
            )
            return
        if new_data:
            # Before the data: the diffusion panel re-reads the layer on a
            # data change, and needs this run's region classes when it does.
            layer.metadata.update(self._session_layer_metadata())
        set_tracks_layer_data(layer, features)
        if new_data:
            layer.name = "tracks (preview)"

    def _filtered_tracks(self):
        session = self._session
        if session is None or session.tracks_df is None:
            return None
        return apply_track_filters(
            session.tracks_df,
            self.params_panel.get_track_filters(),
            session.pixel_size_um,
            session.dt_s,
        )

    # -- Finalize (explicit save) --

    def _save_result(self, item: Optional[ExperimentItem] = None) -> bool:
        """Write the bundle for the stepwise session of `item` (default: the
        current row): every detection in points.parquet, the tracks that
        pass the Track tab's cuts in tracks.parquet, and both filter specs
        in manifest.json. Returns whether it was written.

        Explicit rather than automatic on a finished track run, so the
        filters can be tuned against a linked result before it's committed
        -- press it again after moving a handle and the same bundle is
        rewritten. `item` is passed by the leave-this-row prompt, which
        runs while the list's current row is already the one being moved
        to."""
        item = item if item is not None else self.list_view.currentItem()
        session = self._session if (item is not None and self._session_item is item) else None
        if session is None or session.tracks_df is None:
            self.params_panel.set_save_status("nothing to save — run tracking first", level="error")
            return False

        tracks_df = self._filtered_tracks()
        if tracks_df is None or tracks_df.height == 0:
            self.params_panel.set_save_status(
                "nothing to save — the track filters reject every track", level="error"
            )
            return False

        # Record what the save actually used, so `session_manifest_extra`
        # reports the cuts the bundle was written under rather than the
        # ones the link step happened to run with.
        session.track_filters_used = self.params_panel.get_track_filters() or None
        session.point_filters_used = self.params_panel.get_point_filters() or None

        entry = item.entry
        manifest_params = session_manifest_extra(session)
        # n_tracks comes off the session's unfiltered table; the bundle is
        # getting the filtered one, so correct it before it's written.
        manifest_params["n_tracks"] = tracks_df["track_id"].n_unique()
        manifest = build_manifest(
            result_id=entry.result_dir.name,
            source_image_path=entry.image_path,
            params=manifest_params,
            repo_shas=_repo_shas(),
        )
        try:
            write_result(
                entry.result_dir, session.points_df, tracks_df, manifest,
                labels=session.labels, regions=session.regions,
            )
        except Exception as exc:
            self.params_panel.set_save_status(f"error: {exc}", level="error")
            return False

        n_tracks = manifest_params["n_tracks"]
        item.set_status(Status.COMPLETE, n_tracks=n_tracks)
        item.entry.has_unsaved_session = False
        self.list_view.viewport().update()
        self.params_panel.set_save_status(
            f"saved {n_tracks} tracks → {entry.result_dir.name}", level="ok"
        )
        # This bundle's points.parquet is now exactly what a "Save
        # detections" would write anyway -- disabled so pressing it
        # afterwards, unchanged, can't delete the tracks.parquet just
        # written (`write_detection_result` always removes it).
        self.params_panel.set_detect_save_enabled(False)

        if self.list_view.currentItem() is item:
            self._promote_preview_layers()
        return True

    def _save_detection_result(self, item: Optional[ExperimentItem] = None) -> bool:
        """Write points.parquet + manifest.json alone for the stepwise
        session of `item` (default: the current row) -- the Detect tab's
        own save, usable as soon as detect has run, independent of whether
        tracking has too. Returns whether it was written.

        Removes any tracks.parquet this bundle already had
        (`results.write_detection_result`): those tracks were linked from
        whatever points.parquet said before this call, and points.parquet
        just changed under it. Use "Save results" (`_save_result`) once
        tracking has been (re-)run to get a bundle with tracks in it
        again."""
        item = item if item is not None else self.list_view.currentItem()
        session = self._session if (item is not None and self._session_item is item) else None
        if session is None or session.points_df is None:
            self.params_panel.set_detect_save_status(
                "nothing to save — run detect first", level="error"
            )
            return False

        session.point_filters_used = self.params_panel.get_point_filters() or None

        entry = item.entry
        manifest_params = session_manifest_extra(session)
        # This save doesn't touch tracks (and removes any it previously
        # had), so the manifest shouldn't claim a track count either.
        manifest_params["n_tracks"] = None
        manifest = build_manifest(
            result_id=entry.result_dir.name,
            source_image_path=entry.image_path,
            params=manifest_params,
            repo_shas=_repo_shas(),
        )
        try:
            write_detection_result(
                entry.result_dir, session.points_df, manifest,
                labels=session.labels, regions=session.regions,
            )
        except Exception as exc:
            self.params_panel.set_detect_save_status(f"error: {exc}", level="error")
            return False

        n_points = session.points_df.height
        item.set_status(Status.COMPLETE, n_tracks=None)
        # An unsaved link result, if any, is still unsaved -- only the
        # detections just got written.
        item.entry.has_unsaved_session = session.tracks_df is not None and session.tracks_df.height > 0
        self.list_view.viewport().update()
        self.params_panel.set_detect_save_status(
            f"saved {n_points} detections → {entry.result_dir.name}", level="ok"
        )

        if self.list_view.currentItem() is item:
            self._promote_preview_layers(tracks=False)
        return True

    def _promote_preview_layers(self, *, points: bool = True, tracks: bool = True) -> None:
        """Give the session's "(preview)" layers the final "points"/"tracks"
        names the saved bundle loads under.

        A rename rather than a rebuild. The layers already show exactly what
        was written -- every detection, filtered through the same `shown`
        cuts, and the tracks passing the Track tab's filters (the redraw is
        flushed first, in case a drag's debounce is still pending) -- so
        rebuilding them only threw state away: their layer-control
        settings, their place in the layer list, and the diffusion panel's
        fits, which it drops when the Tracks layer it is reading goes away.

        A "points"/"tracks" pair from this row's previously saved bundle is
        superseded by the one just written, so it is removed. The regions layer
        is left alone -- they were this run's input and are already on
        screen.

        `points`/`tracks` pick which pair was actually just written --
        `_save_detection_result` passes `tracks=False` so an unsaved (or
        nonexistent) Tracks layer is left named "(preview)" rather than
        promoted alongside detections it wasn't asked to save."""
        if points:
            self._update_points_layer()
        if tracks:
            self._update_tracks_layer()
        metadata = self._session_layer_metadata()
        pairs = []
        if points:
            pairs.append((self._points_layer, "points"))
        if tracks:
            pairs.append((self._tracks_layer, "tracks"))
        for layer, final_name in pairs:
            layer = self._live(layer)
            if layer is None:
                continue
            if final_name in self.viewer.layers and self.viewer.layers[final_name] is not layer:
                del self.viewer.layers[final_name]
            layer.metadata.update(metadata)
            layer.name = final_name

    def _restore_filters_from_bundle(self, loaded: ResultDisplay) -> None:
        """Put a saved bundle's recorded filter ranges back on the
        histograms when its row is selected -- the manifest is the record
        of which cuts produced it, so re-opening it should show those cuts
        rather than an empty panel. Sources come from the bundle's own
        tables, so the histograms are the distributions those ranges were
        chosen against.

        A restored *track* filter will usually read as inactive, and that
        is correct rather than a failure: tracks.parquet holds only the
        tracks that passed it, so within the saved table the recorded
        range covers everything. The row is still put back at its recorded
        bounds, so what the cut was stays visible. Point filters do survive
        as live cuts, because points.parquet keeps the detections they
        rejected."""
        points_df, tracks_df = loaded.points_df, loaded.tracks_df
        params = loaded.manifest.get("params", {}) or {}
        pixel_size_um = params.get("pixel_size_um") or 1.0
        dt_s = params.get("dt_s") or 1.0
        # The bundle's own recorded conversion factors, which are what its
        # tables are in -- not necessarily what the file would parse as
        # today (a re-saved bundle may have been run with an override, and
        # a reader fix can change what the file yields). Showing the
        # manifest's values while a saved bundle is on screen keeps the
        # banner describing the data actually displayed.
        self.params_panel.set_image_metadata(
            None,
            source="bundle",
            pixel_size_um=params.get("pixel_size_um"),
            dt_s=params.get("dt_s"),
            exposure_s=params.get("exposure_s"),
        )

        self.params_panel.set_point_filter_source(points_df if points_df.height else None)
        metrics = (
            track_metrics_df(tracks_df, pixel_size_um, dt_s)
            if tracks_df.height and "track_id" in tracks_df.columns
            else None
        )
        self.params_panel.set_track_filter_source(metrics)
        self.params_panel.set_point_filters(_filter_spec(params.get("point_filters")))
        self.params_panel.set_track_filters(_filter_spec(params.get("track_filters")))
        n_tracks = params.get("n_tracks")
        self.params_panel.set_save_status(
            f"saved bundle — {n_tracks} tracks" if n_tracks is not None else "saved bundle",
            level="ok",
        )


def _filter_spec(recorded: Optional[dict]) -> Optional[dict]:
    """A manifest's `{column: [lo, hi]}` back as `{column: (lo, hi)}` --
    JSON has no tuples, and `pipeline.FilterSpec` is written in them. A
    null side stays None (unbounded)."""
    if not recorded:
        return None
    def side(value):
        return None if value is None else float(value)

    return {col: (side(lo), side(hi)) for col, (lo, hi) in recorded.items()}


def _debounce_timer(parent: QObject, slot: Callable[[], None], msec: int = 40) -> QTimer:
    """A single-shot timer that calls `slot` once `msec` after the last of
    a burst of `start()` calls -- restarting a running QTimer pushes its
    timeout back."""
    timer = QTimer(parent)
    timer.setSingleShot(True)
    timer.setInterval(msec)
    timer.timeout.connect(slot)
    return timer


@thread_worker(start_thread=False)
def _load_item_worker(image_path: Path, result_dir: Path):
    """A row's saved bundle if it has one, else its raw image -- read off
    the GUI thread, shown by `ExperimentListWidget._on_item_loaded`."""
    if has_result(result_dir):
        return load_result_display(result_dir)
    return load_image_display(image_path)


@thread_worker(start_thread=False)
def _run_detect_worker(
    session: PipelineSession,
    sigma: float,
    camera_kwargs: dict,
    detect_kwargs: dict,
    frame_range: Optional[tuple[int, int]],
    mask: Optional[np.ndarray],
    cancel_event: threading.Event,
    emitter: _ProgressEmitter,
    n_threads: Optional[int] = None,
    detector: str = "multi_emitter",
) -> PipelineSession:
    def progress_cb(done: int, total: int, stage: str) -> None:
        emitter.updated.emit(done, total, stage)

    return run_detect_step(
        session,
        sigma=sigma,
        camera_kwargs=camera_kwargs,
        detect_kwargs=detect_kwargs,
        frame_range=frame_range,
        mask=mask,
        progress_callback=progress_cb,
        cancel_event=cancel_event,
        n_threads=n_threads,
        detector=detector,
    )


@thread_worker(start_thread=False)
def _run_track_worker(
    session: PipelineSession,
    min_track_length: int,
    exclude_flags: int,
    link_with_flux: bool,
    min_link_margin: float,
    point_filters: dict,
    emitter: _ProgressEmitter,
) -> PipelineSession:
    def progress_cb(done: int, total: int, stage: str) -> None:
        emitter.updated.emit(done, total, stage)

    return run_track_step(
        session,
        min_track_length,
        exclude_flags=exclude_flags,
        link_with_flux=link_with_flux,
        min_link_margin=min_link_margin,
        point_filters=point_filters,
        progress_callback=progress_cb,
    )
