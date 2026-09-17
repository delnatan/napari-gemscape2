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
Detect tab's PSF-width preview loop and frame-range/ROI scope controls)
lives in
`widgets/params_panel.py::PipelineParamsWidget`, which stays viewer-
agnostic; this module is what actually resolves the ROI checkbox into a
boolean mask array, by reading the Shapes layer named in that panel's ROI
dropdown (`_build_roi_mask`). Several ROIs can be on screen at once --
"Draw ROI…" adds another Shapes layer each time and `_on_roi_layers_
changed` keeps the dropdown in step with the viewer (renames included),
so which region a run covers is a deliberate pick rather than a
side-effect of which layer was last clicked.

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

Two ways to run the pipeline on the *current* selection, both threaded
through the same `self._worker` slot (only one run -- batch or stepwise
-- active at a time):
- **Batch** ("Run selected" button / `Shift+R` key, possibly multi-select):
  always the full calibrate->detect->track pipeline
  (`pipeline.run_detect_track`). Only `Status.UNTOUCHED`/`Status.ERROR`
  items run by default -- `COMPLETE`/`SKIP` are excluded so this can't
  silently overwrite finished work; press `U` to unmark a `COMPLETE` item
  first if a deliberate re-run is wanted. The same button doubles as
  Cancel while a run is active (`_cancel_active_run`) -- cooperative, see
  `pipeline.PipelineCancelled`'s docstring for what that actually
  guarantees per stage.
- **Stepwise** (the params-panel tabs' own buttons, single current item
  only): preview, detect and track run independently against a
  `pipeline.PipelineSession` held in `self._session`, so changing one
  stage's knobs and re-running it doesn't force redoing the earlier
  stages. The session resets on selection change (see
  `_on_selection_changed`) -- it's scoped to "the image currently being
  worked on", not persisted across items.

  Stepwise runs write **nothing** until "Save results" is pressed
  (`_save_result`). The filter histograms on both tabs are the reason:
  the cuts they set are chosen by looking at a finished stage's output, so
  committing the bundle the instant linking returned would mean saving
  before the decision that shapes it had been made. Batch runs still write
  on completion -- nobody is dragging a handle during one. `has_unsaved_
  session` on the list row is what marks the gap in between.

  Filters also drive the viewer live: `_update_points_layer` and
  `_update_tracks_layer` redraw the "points (preview)"/"tracks (preview)"
  layers through the current cuts on every handle move, so a spot that
  fails a cut leaves the image as the cut is made. That immediacy is the
  point of keeping the histogram next to the viewer instead of in a
  report. Saving then hands those same layers their final names
  (`_promote_preview_layers`) without touching the image layer or the ROI
  layers, so the view doesn't reset out from under the user at the moment
  the work is committed. Every layer here is built by
  `viewer.py`'s `add_image_layer`/`add_points_layer`/`add_tracks_layer`,
  which is what makes preview and final look identical.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import polars as pl
from napari.layers import Shapes
from napari.qt.threading import thread_worker
from natsort import natsorted
from qtkit.napari import live_layer
from qtpy.QtCore import QModelIndex, QObject, QRect, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QFontMetrics, QPainter, QPen
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QFrame,
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

from spt_pipeline.results import (
    build_manifest,
    result_dir_for,
    git_sha,
    has_result,
    load_manifest,
    repo_root_of,
    write_result,
)
from spt_pipeline.io_formats import SUPPORTED_SUFFIXES as SUPPORTED_FORMATS
from spt_pipeline.pipeline import (
    DetectTrackParams,
    PipelineCancelled,
    PipelineSession,
    apply_filters,
    apply_track_filters,
    filter_mask,
    load_session,
    run_detect_step,
    run_detect_track,
    run_preview_frame,
    run_track_step,
    session_manifest_extra,
    track_features_df,
    track_metrics_df,
)
from spt_pipeline.rois import shapes_layer_to_roi
from spt_pipeline.viewer import (
    ImageDisplay,
    ResultDisplay,
    add_image_layer,
    add_points_layer,
    add_tracks_layer,
    load_image_display,
    load_result_display,
    set_points_layer_data,
    set_tracks_layer_data,
    show_image,
    show_result,
)
from spt_pipeline.widgets.params_panel import PipelineParamsWidget


def _dropped_folder(event) -> Optional[Path]:
    """First local directory among an event's dropped URLs, if any."""
    for url in event.mimeData().urls():
        path = Path(url.toLocalFile())
        if path.is_dir():
            return path
    return None


class Status(str, Enum):
    UNTOUCHED = "untouched"
    RUNNING = "running"
    COMPLETE = "complete"
    SKIP = "skip"
    ERROR = "error"


STATUS_COLORS = {
    Status.UNTOUCHED: QColor("#9a9a9a"),
    Status.RUNNING: QColor("#3b82f6"),
    Status.COMPLETE: QColor("#22c55e"),
    Status.SKIP: QColor("#5a5a5a"),
    Status.ERROR: QColor("#ef4444"),
}


@dataclass
class ExperimentEntry:
    image_path: Path
    result_dir: Path
    status: Status = Status.UNTOUCHED
    n_tracks: Optional[int] = None
    error: Optional[str] = None
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
    """Paints a status dot + filename (+ track count/error once known)."""

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
        elif entry.status is Status.ERROR and entry.error:
            label += f"   — {entry.error}"
        painter.setPen(text_color)
        # Plain drawText into a rect this narrow just hard-clips a long
        # filename/error mid-character with no visual cue there's more --
        # elide it instead (full path/error is still available via the
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

    runRequested = Signal(list)  # list[ExperimentItem]

    KEYBINDINGS = "Enter: load · Shift+R: run · X: skip · U: unmark · F5: rescan"

    def __init__(self) -> None:
        super().__init__()
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
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
        if key == Qt.Key.Key_R and (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.runRequested.emit(self.selectedItems())
            return
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
    """Dock widget: folder-scan list + Run button/progress bar."""

    def __init__(self, napari_viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self._worker = None
        self._cancel_event: Optional[threading.Event] = None
        self._session: Optional[PipelineSession] = None
        self._session_item: Optional[ExperimentItem] = None
        # True while a stepwise Calibrate/Detect/Track worker (as opposed to
        # a batch run) is active against `self._session_item` -- guards
        # `_on_selection_changed` so clicking a different row mid-run can't
        # rug the viewer layers out from under it (Finding: napari's layer
        # list going blank on a mid-detect selection change). Batch runs
        # don't set this (they go through `_run_next` directly, not
        # `_start_step_worker`) since they don't hold layers hostage the
        # same way -- `_on_run_finished`/`_on_run_error` already only touch
        # the viewer when the finishing item is still the current one.
        self._step_running = False
        # Re-entrancy guard for the `setCurrentItem` call `_on_selection_changed`
        # makes to revert a blocked switch -- without it, that call's own
        # `currentItemChanged` re-entry would run the "leaving this row"
        # cleanup against the row we're refusing to leave.
        self._reverting_selection = False
        # The Shapes layer the ROI dropdown currently targets, held as the
        # layer itself rather than its name: renaming a layer is the
        # intended way to label one of several ROIs, and a name-keyed
        # record would lose track of the target at exactly that moment
        # (see `_on_roi_layers_changed`).
        self._roi_target: Optional[Shapes] = None

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
        self.list_view.itemSelectionChanged.connect(self._update_run_button_label)
        self.list_view.runRequested.connect(self._run_items)
        self.list_view.is_busy = lambda: self._worker is not None
        self.list_view.on_busy_blocked = lambda: self.progress_label.setText(
            "a run is in progress — finishing before loading a new folder"
        )
        self.list_view.confirm_reload = self._confirm_reload

        open_button = QPushButton("Open folder…")
        open_button.clicked.connect(self._open_folder_dialog)
        self.run_button = QPushButton("Run selected")
        self.run_button.clicked.connect(self._on_run_button_clicked)

        button_row = QHBoxLayout()
        button_row.addWidget(open_button)
        button_row.addWidget(self.run_button)

        self.params_panel = PipelineParamsWidget()
        self.params_panel.previewRequested.connect(self._run_preview_step)
        self.params_panel.detectRequested.connect(self._run_detect_step)
        self.params_panel.detectCancelRequested.connect(self._cancel_active_run)
        self.params_panel.trackRequested.connect(self._run_track_step)
        self.params_panel.saveRequested.connect(self._save_result)
        self.params_panel.newRoiRequested.connect(self._on_new_roi_requested)
        # Keep the Detect tab's ROI dropdown showing the viewer's Shapes
        # layers. Adding/removing/reordering layers is caught on the layer
        # list itself; a *rename* is an event on the layer, so
        # `_on_roi_layers_changed` (re)connects to each Shapes layer as it
        # goes -- napari's emitters ignore a duplicate connect, so this
        # can run as often as it likes.
        self.viewer.layers.events.inserted.connect(self._on_roi_layers_changed)
        self.viewer.layers.events.removed.connect(self._on_roi_layers_changed)
        self.viewer.layers.events.reordered.connect(self._on_roi_layers_changed)
        self.params_panel.roiLayerChanged.connect(self._on_roi_layer_selected)
        self._on_roi_layers_changed()
        self.params_panel.pointFiltersChanged.connect(self._on_point_filters_changed)
        self.params_panel.trackFiltersChanged.connect(self._on_track_filters_changed)

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
        self._update_run_button_label()

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
        if self._step_running and self._session_item is not None and current is not self._session_item:
            # A calibrate/detect/track worker is still running against
            # `self._session_item` -- switching away would otherwise wipe
            # its viewer layers and orphan the worker's eventual result
            # (see `_step_running`'s docstring). Snap the selection back
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
        self.params_panel.set_preview_result(None)
        self.params_panel.set_detect_status("")
        self.params_panel.set_track_status("")
        self.params_panel.set_save_status("")
        self.params_panel.set_save_enabled(False)
        # The histogram ranges belong to a table that's about to be
        # unloaded -- carrying them onto the next image would apply cuts
        # chosen against a different population, which is exactly the
        # mistake the histograms exist to prevent.
        self.params_panel.clear_filters()
        self.params_panel.set_point_filter_source(None)
        self.params_panel.set_track_filter_source(None)
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

    def _confirm_leave_session(self) -> bool:
        """Whether it is fine to drop the current session: True straight
        away if nothing in it is unsaved, otherwise the user's answer to a
        Save / Discard / Stay prompt. Leaving used to discard silently --
        one arrow-key press in the list was enough to lose a finished
        detect-and-link."""
        item, session = self._session_item, self._session
        if item is None or session is None or not item.entry.has_unsaved_session:
            return True
        savable = session.tracks_df is not None and session.tracks_df.height > 0
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved results")
        box.setText(f"{item.entry.image_path.name} has results that haven't been saved.")
        box.setInformativeText(
            "Save writes its results bundle first; Discard drops them."
            if savable
            else "It hasn't been linked yet, so there is nothing to save — leaving drops it."
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
            return self._save_result(item)
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
        # Only our own message: a batch run shares this label.
        if self.progress_label.text() == self._loading_text:
            self.progress_label.setText("")

    def _on_item_loaded(self, item: ExperimentItem, generation: int, loaded) -> None:
        if not self._load_is_current(item, generation):
            return
        self._clear_loading_text()
        image = loaded.image if isinstance(loaded, ResultDisplay) else loaded
        self._loaded = (item, image)
        self.params_panel.set_frame_bounds(image.image.shape[0])
        if isinstance(loaded, ResultDisplay):
            show_result(self.viewer, loaded)
            self._restore_filters_from_bundle(loaded)
        else:
            show_image(self.viewer, loaded)

    def _on_item_load_error(self, item: ExperimentItem, generation: int, exc: Exception) -> None:
        if not self._load_is_current(item, generation):
            return
        self.progress_label.setText(f"could not load {item.entry.image_path.name}: {exc}")

    def _update_run_button_label(self) -> None:
        """Reflects what a click on `run_button` would actually do, given
        the current selection -- kept a no-op while a run is active, since
        `_run_next`/`_cancel_active_run` own the label in that state."""
        if self._worker is not None:
            return
        items = self.list_view.selectedItems()
        runnable = [item for item in items if item.entry.status in (Status.UNTOUCHED, Status.ERROR)]
        n_complete = sum(1 for item in items if item.entry.status is Status.COMPLETE)
        if not items:
            self.run_button.setText("Run selected")
        elif n_complete:
            self.run_button.setText(f"Run selected ({len(runnable)}, {n_complete} complete)")
        else:
            self.run_button.setText(f"Run selected ({len(runnable)})")
        self.run_button.setEnabled(True)

    def _on_run_button_clicked(self) -> None:
        if self._worker is not None:
            self._cancel_active_run()
        else:
            self._run_items(self.list_view.selectedItems())

    def _cancel_active_run(self) -> None:
        """Requests cancellation of whatever's currently running (batch or
        stepwise Detect) -- cooperative only, see `PipelineCancelled`'s
        docstring for how promptly this actually takes effect per stage.
        Also drops any remaining queued batch items so they don't start."""
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._run_queue = []
        self.run_button.setText("Cancelling…")
        self.run_button.setEnabled(False)

    def _run_items(self, items: list[ExperimentItem]) -> None:
        if self._worker is not None:
            return
        runnable = [item for item in items if item.entry.status in (Status.UNTOUCHED, Status.ERROR)]
        n_complete = sum(1 for item in items if item.entry.status is Status.COMPLETE)
        n_skip = sum(1 for item in items if item.entry.status is Status.SKIP)
        if not runnable:
            if n_complete or n_skip:
                self.progress_label.setText(
                    f"nothing to run -- {n_complete} already complete, {n_skip} skipped "
                    "(press U to unmark and re-run)"
                )
            return
        if n_complete or n_skip:
            self.progress_label.setText(
                f"running {len(runnable)}, skipping {n_complete} complete + {n_skip} skipped"
            )
        self._run_queue = runnable
        self._cancel_event = threading.Event()
        self._run_next()

    def _run_next(self) -> None:
        if not self._run_queue:
            self._worker = None
            self._cancel_event = None
            self.progress_bar.setVisible(False)
            self.progress_label.setText("")
            self._update_run_button_label()
            return

        item = self._run_queue.pop(0)
        entry = item.entry
        item.set_status(Status.RUNNING)
        self.list_view.viewport().update()

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Running: {entry.image_path.name}")
        # Filename already shown in `progress_label` above -- a QPushButton
        # can't wrap its text, so embedding an unbounded filename here would
        # force the button (and the row/dock around it) wider for as long
        # as this run's name stays the longest one seen, the same class of
        # bug `progress_label`'s own word-wrap fix addresses.
        self.run_button.setText("Cancel")
        self.run_button.setEnabled(True)

        emitter = _ProgressEmitter()
        emitter.updated.connect(self._on_progress)

        worker = _run_pipeline_worker(
            entry.image_path, self.params_panel.get_params(), self._cancel_event, emitter
        )
        worker.returned.connect(lambda result, item=item: self._on_run_finished(item, result))
        worker.errored.connect(lambda exc, item=item: self._on_run_error(item, exc))
        self._worker = worker
        worker.start()

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

    def _on_run_finished(self, item: ExperimentItem, result) -> None:
        points_df, tracks_df, manifest_extra = result
        entry = item.entry

        import spotsolve
        import spt_pipeline

        repo_shas = {
            "spotsolve": git_sha(repo_root_of(spotsolve)),
            "spt_pipeline": git_sha(repo_root_of(spt_pipeline)),
        }
        manifest = build_manifest(
            result_id=entry.result_dir.name,
            source_image_path=entry.image_path,
            params=manifest_extra,
            repo_shas=repo_shas,
        )
        write_result(entry.result_dir, points_df, tracks_df, manifest)
        item.set_status(Status.COMPLETE, n_tracks=manifest_extra["n_tracks"])
        self.list_view.viewport().update()

        if self.list_view.currentItem() is item and self._session_item is not item:
            self._request_item_load(item)

        self._worker = None
        self._cancel_event = None
        self._run_next()

    def _on_run_error(self, item: ExperimentItem, exc: Exception) -> None:
        cancelled = isinstance(exc, PipelineCancelled)
        if cancelled:
            # Not a real error -- back to untouched so it's safe (and
            # obviously re-runnable) rather than parked in Status.ERROR.
            item.set_status(Status.UNTOUCHED)
        else:
            item.set_status(Status.ERROR, error=str(exc))
        self.list_view.viewport().update()
        self._worker = None
        self._cancel_event = None
        # _run_next()'s empty-queue branch clears progress_label -- set the
        # "cancelled" message after, so it's the one left showing instead of
        # being immediately overwritten by that cleanup.
        self._run_next()
        if cancelled:
            self.progress_label.setText("cancelled")

    # -- Stepwise Calibrate / Detect / Track (current selection only) --

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
        if loaded is None:
            # The row's load never landed, so its image isn't on screen.
            # Added under whatever is there (an ROI drawn meanwhile, say)
            # rather than clearing it away.
            session = load_session(item.entry.image_path)
            layer = add_image_layer(self.viewer, session.image, item.entry.image_path.stem)
            self.viewer.layers.move(self.viewer.layers.index(layer), 0)
        else:
            session = load_session(
                item.entry.image_path, stack=(loaded.image, loaded.pixel_size_um, loaded.dt_s)
            )
        self._session = session
        self._session_item = item
        return session

    def _roi_layers(self) -> list[Shapes]:
        """Every Shapes layer in the viewer, in layer-list order -- the
        candidate ROIs for the Detect tab's dropdown."""
        return [layer for layer in self.viewer.layers if isinstance(layer, Shapes)]

    def _on_roi_layer_selected(self) -> None:
        """Remember which layer the dropdown now names, so a later rename
        of it can be followed (`_on_roi_layers_changed`)."""
        name = self.params_panel.get_roi_layer_name()
        layer = self.viewer.layers[name] if name and name in self.viewer.layers else None
        self._roi_target = layer if isinstance(layer, Shapes) else None

    def _on_roi_layers_changed(self, event=None) -> None:
        """Push the current Shapes layer names into the ROI dropdown, and
        make sure a rename of any of them lands here too (each layer's own
        `events.name`, since the layer list only reports add/remove/
        reorder). Renaming is the intended way to tell several ROIs apart
        -- the name is also what the region is saved under (see
        `spt_pipeline.rois`) -- so the dropdown follows the *layer*
        (`self._roi_target`) across a rename rather than losing it when
        the name it was listed under disappears."""
        layers = self._roi_layers()
        for layer in layers:
            layer.events.name.connect(self._on_roi_layers_changed)
        if not any(layer is self._roi_target for layer in layers):
            self._roi_target = None
        preferred = self._roi_target.name if self._roi_target is not None else None
        self.params_panel.set_roi_choices([layer.name for layer in layers], preferred)
        self._on_roi_layer_selected()

    def _on_new_roi_requested(self) -> None:
        """Add an empty Shapes layer for drawing the ROI, ready to draw on
        immediately: 2D (`ndim=2`, fewer dims than an nD image stack) so a
        shape drawn on it shows on every frame rather than only the one it
        was drawn on -- the same trailing-`(y, x)`-only convention
        `rois.roi_to_shapes_kwargs` uses when an ROI round-trips through
        disk -- transparent fill so it doesn't occlude the image/points
        underneath, and the polygon-lasso tool selected as the active mode
        so the user can start drawing right away.

        Each click adds *another* layer ("roi 1", "roi 2", ...) rather
        than reusing one, so several regions can be kept side by side and
        picked between in the dropdown; the fresh one becomes the
        dropdown's selection (`set_roi_choices` falls back to the last
        entry, and this is the layer that was just appended)."""
        existing = {layer.name for layer in self.viewer.layers}
        n = 1
        while f"roi {n}" in existing:
            n += 1
        layer = self.viewer.add_shapes(
            ndim=2,
            name=f"roi {n}",
            face_color="transparent",
            edge_color="yellow",
        )
        self.viewer.layers.selection.active = layer
        layer.mode = "add_polygon_lasso"
        # The insert event already refreshed the dropdown (keeping
        # whatever was targeted before); point it at the layer the user
        # just asked for instead -- they're about to draw on it.
        self._roi_target = layer
        self._on_roi_layers_changed()

    def _build_roi_mask(self, shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
        """Boolean `(H, W)` mask from the Shapes layer named in the Detect
        tab's ROI dropdown -- the union of every shape drawn on it -- plus
        that layer's polygon ROI record
        (`spt_pipeline.rois.shapes_layer_to_roi`), meant to be stashed on
        `session.roi` so `_save_result` can persist it alongside the
        run's results (see `experiment.write_result`).

        Targeted by name rather than by which layer happens to be selected
        in napari, so clicking around the layer list (or drawing a second
        region for reference) can't quietly change which pixels a run
        covers. Raises if the named layer is gone or has nothing drawn on
        it yet."""
        name = self.params_panel.get_roi_layer_name()
        if name is None:
            raise ValueError('no ROI layer to use — press "Draw ROI…" and draw a region first')
        layer = self.viewer.layers[name] if name in self.viewer.layers else None
        if not isinstance(layer, Shapes):
            raise ValueError(f"ROI layer {name!r} is gone — pick another one or draw a new one")
        if len(layer.data) == 0:
            raise ValueError(f"nothing drawn on ROI layer {name!r} yet")
        mask = np.any(layer.to_masks(shape), axis=0)
        return mask, shapes_layer_to_roi(layer)

    def _start_step_worker(
        self, worker, on_finished, indeterminate: bool = False, on_error: Optional[Callable] = None
    ) -> None:
        if self._worker is not None:
            return
        self._step_running = True
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
        self._step_running = False
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")

    def _on_step_error(self, exc: Exception) -> None:
        self.progress_label.setText(f"error: {exc}")
        self._finish_step_worker()

    def _run_preview_step(self) -> None:
        """Localize ONE frame with the reporting band off and show every
        fit, so `sigma` can be chosen by looking at the `fit_sigma`
        distribution instead of by trusting `calibrate_sigma`'s median --
        see `params_panel`'s module docstring for why this replaced the
        Calibration tab."""
        item = self.list_view.currentItem()
        if item is None:
            return
        try:
            session = self._ensure_session(item)
        except Exception as exc:
            self.params_panel.set_preview_status(f"error: {exc}", level="error")
            return
        self.params_panel.set_frame_bounds(session.image.shape[0])

        mask = None
        if self.params_panel.get_use_roi_mask():
            try:
                mask, _roi = self._build_roi_mask(session.image.shape[1:])
            except Exception as exc:
                self.params_panel.set_preview_status(f"error: {exc}", level="error")
                return

        frame_index = self.params_panel.get_preview_frame_index()
        self.progress_label.setText(f"Previewing frame {frame_index}: {item.entry.image_path.name}")
        self.params_panel.set_preview_status("previewing…")
        worker = _run_preview_worker(
            session,
            self.params_panel.get_sigma(),
            frame_index,
            self.params_panel.get_camera_kwargs(),
            self.params_panel.get_detect_kwargs(),
            self.params_panel.get_agg_ratio(),
            mask,
            self.params_panel.get_detector(),
        )
        self._start_step_worker(
            worker, lambda s, item=item: self._on_preview_finished(item, s), indeterminate=True
        )

    def _on_preview_finished(self, item: ExperimentItem, session: PipelineSession) -> None:
        if self._session_item is not item:
            # Selection changed to a different item while this ran --
            # `_on_selection_changed` already reset `self._session`;
            # don't resurrect a stale one for the item we've left.
            self._finish_step_worker()
            return
        self._session = session
        summary = session.preview_summary or {}
        n_fits = summary.get("n_fits", 0)
        # A preview that found nothing is a real answer (wrong sigma,
        # wrong offset, blank frame) -- say so in red rather than
        # leaving an empty status that reads like it never ran.
        self.params_panel.set_preview_result(summary, level="error" if n_fits == 0 else "ok")
        self._add_preview_layer(session)
        # Preview is also where the filter histograms come from before any
        # full run exists -- tune the cuts on one frame, then run the range.
        # Deliberately only when a real detect hasn't already produced a
        # bigger table: one frame is a worse population to judge against.
        if session.points_df is None:
            self.params_panel.set_point_filter_source(session.preview_points_df)
        self._finish_step_worker()

    def _add_preview_layer(self, session: PipelineSession) -> None:
        """One box-shaped point per previewed fit, `features=` set to the
        full per-spot table (`loctable.LOCALIZATION_SCHEMA`:
        `y`/`x`/`fit_sigma`/`sigma_ratio`/`se_*`/`flux`/...) plus the
        derived `accepted` -- napari shows a hovered/selected point's
        features in the status bar, so every fit is inspectable, not just
        the median in the status label.

        `accepted` (green border) is whether a detect run at this sigma
        would report the spot at all, or reject it as out-of-band (see
        `pipeline.calibration_accepted`); gray-bordered points are the
        out-of-band fits, which the preview includes precisely because the
        multi-emitter detector runs with the band off. That is what makes
        the band choosable: both populations are on the image at once. The
        Aguet detector has no band -- every point previews green.

        Faces are transparent (border color only) so the boxes outline
        each fit without occluding the underlying image."""
        df = session.preview_points_df
        if df is None or df.height == 0:
            if "preview spots" in self.viewer.layers:
                del self.viewer.layers["preview spots"]
            return
        if "preview spots" in self.viewer.layers:
            del self.viewer.layers["preview spots"]

        features = {col: df[col].to_numpy() for col in df.columns}
        # Sized off the sigma previewed at rather than a fit-window
        # parameter -- spotsolve's boxes are an internal of the search, not
        # a knob, so there is no box_size to read back. ~6 sigma is wide
        # enough to frame the spot it marks at a glance.
        sigma = (session.preview_summary or {}).get("sigma_used") or session.sigma or 1.3
        box_size = max(3.0, 6.0 * sigma)

        self.viewer.add_points(
            df.select(["y", "x"]).to_numpy(),
            name="preview spots",
            features=features,
            symbol="square",
            size=box_size,
            face_color="transparent",
            border_color="accepted",
            border_color_cycle=[
                STATUS_COLORS[Status.COMPLETE].name(),
                STATUS_COLORS[Status.UNTOUCHED].name(),
            ],
            border_width=0.15,
        )
        if session.preview_frame_used is not None and self.viewer.dims.ndim:
            # Step the viewer to the frame just previewed -- the boxes are
            # 2D and show on every frame, so without this they'd be drawn
            # over whichever frame happens to be displayed.
            self.viewer.dims.set_current_step(0, session.preview_frame_used)

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
        session.roi = None
        if self.params_panel.get_use_roi_mask():
            try:
                mask, roi = self._build_roi_mask(session.image.shape[1:])
            except Exception as exc:
                self.params_panel.set_detect_status(f"error: {exc}", level="error")
                return
            session.roi = [roi]

        # The Detect tab's sigma box, always -- not `session.sigma` from
        # some earlier run. It IS the setting now that the preview loop
        # feeds it (params_panel's docstring), so honoring a stale session
        # value would mean the number on screen isn't the one that ran.
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
            self.params_panel.get_agg_ratio(),
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
        item.entry.has_unsaved_session = True
        self.list_view.viewport().update()
        n_points = session.points_df.height if session.points_df is not None else 0
        start, end = session.frame_range_used or (0, session.image.shape[0])
        # The reject/aggregate counts are the reason frames_df is kept: a
        # run that found plenty of spots but binned most of them as
        # out-of-band is a focus or sigma problem, and that's only visible
        # if the numbers are shown next to the detection count.
        extra = ""
        frames = session.frames_df
        if frames is not None and frames.height:
            n_rejected = int(
                frames["n_too_narrow"].sum() + frames["n_too_wide"].sum() + frames["n_edge"].sum()
            )
            n_flagged = int(frames["n_locs_flagged"].sum())
            parts = []
            if n_rejected:
                parts.append(f"{n_rejected} out-of-band")
            if n_flagged:
                parts.append(f"{n_flagged} aggregate")
            if parts:
                extra = "  (" + ", ".join(parts) + ")"
        self.params_panel.set_detect_status(
            f"{n_points} points across frames {start}-{end - 1}{extra}",
            level="error" if n_points == 0 else "ok",
        )

        # Hand the real run's detections to the filter histograms (they
        # may have been showing a single preview frame's) and draw the
        # layer through whatever cuts are already set. The Detect tab
        # pages through its steps rather than scrolling, so leaf it to the
        # filter page too -- that histogram is what there is to do next,
        # and it is no longer just below the Run button.
        self.params_panel.set_point_filter_source(session.points_df)
        self.params_panel.show_tab("Detect", "Filter")
        if "preview spots" in self.viewer.layers:
            # The preview's one frame is superseded by the real run; two
            # overlapping spot layers on the same frame is just confusing.
            del self.viewer.layers["preview spots"]
        self._update_points_layer(new_data=True)
        self.params_panel.set_save_enabled(False)
        self._finish_step_worker()

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
        session = self._session if self._session_item is item else None
        if session is None or session.points_df is None:
            self.params_panel.set_track_status("error: run detect first", level="error")
            return
        self.progress_label.setText(f"Linking: {item.entry.image_path.name}")
        emitter = _ProgressEmitter()
        emitter.updated.connect(self._on_progress)
        worker = _run_track_worker(
            session,
            self.params_panel.get_min_track_length(),
            self.params_panel.get_drop_aggregates(),
            self.params_panel.get_link_with_flux(),
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
        dropped = summary.get("n_points_dropped_by_filter") or 0
        filtered = f"  ({dropped} points cut by filters)" if dropped else ""
        self.params_panel.set_track_status(
            f"{n_tracks} tracks  D~{summary.get('D_est_um2_s', 0.0):.4f} um^2/s "
            f"(linker fit {summary.get('D_link_um2_s', 0.0):.4f}, "
            f"immobile {summary.get('immobile_fraction', 0.0):.0%}){filtered}\n{message}",
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
        return {
            "pixel_size_um": session.pixel_size_um if session is not None else 1.0,
            "dt_s": session.dt_s if session is not None else 1.0,
            "result_dir": str(self._session_item.entry.result_dir.resolve())
            if self._session_item is not None
            else None,
        }

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
        import spotsolve
        import spt_pipeline

        repo_shas = {
            "spotsolve": git_sha(repo_root_of(spotsolve)),
            "spt_pipeline": git_sha(repo_root_of(spt_pipeline)),
        }
        manifest_params = session_manifest_extra(session)
        # n_tracks comes off the session's unfiltered table; the bundle is
        # getting the filtered one, so correct it before it's written.
        manifest_params["n_tracks"] = tracks_df["track_id"].n_unique()
        manifest = build_manifest(
            result_id=entry.result_dir.name,
            source_image_path=entry.image_path,
            params=manifest_params,
            repo_shas=repo_shas,
        )
        try:
            write_result(
                entry.result_dir, session.points_df, tracks_df, manifest, rois=session.roi
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

        if self.list_view.currentItem() is item:
            self._promote_preview_layers()
        return True

    def _promote_preview_layers(self) -> None:
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
        superseded by the one just written, so it is removed. The ROI layers
        are left alone -- they were this run's input and are already on
        screen."""
        self._update_points_layer()
        self._update_tracks_layer()
        if "preview spots" in self.viewer.layers:
            del self.viewer.layers["preview spots"]
        metadata = self._session_layer_metadata()
        for layer, final_name in ((self._points_layer, "points"), (self._tracks_layer, "tracks")):
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
def _run_pipeline_worker(
    image_path: Path,
    params: DetectTrackParams,
    cancel_event: threading.Event,
    emitter: _ProgressEmitter,
):
    def progress_cb(done: int, total: int, stage: str) -> None:
        emitter.updated.emit(done, total, stage)

    return run_detect_track(
        image_path, params=params, progress_callback=progress_cb, cancel_event=cancel_event
    )


@thread_worker(start_thread=False)
def _run_preview_worker(
    session: PipelineSession,
    sigma: float,
    frame_index: int,
    camera_kwargs: dict,
    detect_kwargs: dict,
    agg_ratio: float,
    mask: Optional[np.ndarray],
    detector: str = "multi_emitter",
) -> PipelineSession:
    return run_preview_frame(
        session,
        sigma,
        frame_index=frame_index,
        camera_kwargs=camera_kwargs,
        detect_kwargs=detect_kwargs,
        agg_ratio=agg_ratio,
        mask=mask,
        detector=detector,
    )


@thread_worker(start_thread=False)
def _run_detect_worker(
    session: PipelineSession,
    sigma: float,
    camera_kwargs: dict,
    detect_kwargs: dict,
    agg_ratio: float,
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
        agg_ratio=agg_ratio,
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
    drop_aggregates: bool,
    link_with_flux: bool,
    point_filters: dict,
    emitter: _ProgressEmitter,
) -> PipelineSession:
    def progress_cb(done: int, total: int, stage: str) -> None:
        emitter.updated.emit(done, total, stage)

    return run_track_step(
        session,
        min_track_length,
        drop_aggregates=drop_aggregates,
        link_with_flux=link_with_flux,
        point_filters=point_filters,
        progress_callback=progress_cb,
    )
