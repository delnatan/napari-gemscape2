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
Detect tab's frame-range/ROI scope controls) lives in
`widgets/params_panel.py::PipelineParamsWidget`, which stays viewer-
agnostic; this module is what actually resolves the ROI checkbox into a
boolean mask array, by reading the active napari Shapes layer
(`_build_roi_mask`) -- `find_spots` only accepts a mask on its
single-frame path, so a mask forces `run_detect_step` onto the
frame-by-frame loop regardless of whether progress is also being
reported (see `pipeline.run_detect_step`'s docstring).

Drag-and-drop accepts a dropped folder anywhere on this dock widget (not
just precisely on the list rows) -- both `_ExperimentListView` and the
outer `ExperimentListWidget` implement it, since the list no longer fills
the whole panel now that the params form sits below it.

Two ways to run the pipeline on the *current* selection, both threaded
through the same `self._worker` slot (only one run -- batch or stepwise
-- active at a time):
- **Batch** ("Run selected" button / `R` key, possibly multi-select):
  always the full calibrate->detect->track pipeline
  (`pipeline.run_detect_track`), unchanged from before.
- **Stepwise** (each params-panel tab's own "Run <stage>" button, single
  current item only): calibrate, detect, and track run independently
  against a `pipeline.PipelineSession` held in `self._session`, so
  changing one stage's knobs and re-running it doesn't force redoing the
  earlier stages. The session resets on selection change (see
  `_on_selection_changed`) -- it's scoped to "the image currently being
  worked on", not persisted across items.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from napari.layers import Shapes
from napari.qt.threading import thread_worker
from natsort import natsorted
from qtpy.QtCore import QModelIndex, QObject, QRect, QSize, Qt, Signal
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from spt_pipeline.experiment import (
    build_manifest,
    experiment_dir_for,
    git_sha,
    has_experiment,
    load_experiment,
    repo_root_of,
    write_experiment,
)
from spt_pipeline.io_formats import SUPPORTED_SUFFIXES as SUPPORTED_FORMATS
from spt_pipeline.io_formats import load_stack
from spt_pipeline.pipeline import (
    DetectTrackParams,
    PipelineSession,
    load_session,
    run_calibration_step,
    run_detect_step,
    run_detect_track,
    run_track_step,
    session_manifest_extra,
)
from spt_pipeline.viewer import add_experiment_layers
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
    experiment_dir: Path
    status: Status = Status.UNTOUCHED
    n_tracks: Optional[int] = None
    error: Optional[str] = None


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

        text_rect = option.rect.adjusted(self.PADDING * 2 + self.DOT_DIAMETER, 0, -self.PADDING, 0)
        label = entry.image_path.name
        if entry.status is Status.COMPLETE and entry.n_tracks is not None:
            label += f"   ({entry.n_tracks} tracks)"
        elif entry.status is Status.ERROR and entry.error:
            label += f"   — {entry.error}"
        painter.setPen(text_color)
        painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), label)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return QSize(option.rect.width(), self.ROW_HEIGHT)


class _ExperimentListView(QListWidget):
    """The list itself: folder scanning + keybindings. Composed inside
    `ExperimentListWidget`, which owns the run/progress machinery."""

    runRequested = Signal(list)  # list[ExperimentItem]

    KEYBINDINGS = "Enter: load · R: run · X: skip · U: unmark · F5: rescan"

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
        self.experiments_root: Optional[Path] = None

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_R:
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
                self.load_folder(self.folder_path, self.experiments_root)
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        folder = _dropped_folder(event)
        if folder is not None:
            self.load_folder(folder, self.experiments_root)

    def load_folder(self, folder_path: Path, experiments_root: Optional[Path] = None) -> None:
        self.clear()
        self.folder_path = Path(folder_path)
        self.experiments_root = experiments_root or (self.folder_path / "experiments")

        for file_path in natsorted(self.folder_path.iterdir()):
            if file_path.name.startswith("."):
                continue
            if file_path.suffix.lower() not in SUPPORTED_FORMATS:
                continue

            experiment_dir = experiment_dir_for(self.experiments_root, file_path)
            entry = ExperimentEntry(image_path=file_path, experiment_dir=experiment_dir)
            item = ExperimentItem(entry)

            if has_experiment(experiment_dir):
                _, tracks_df, manifest = load_experiment(experiment_dir)
                n_tracks = manifest.get("params", {}).get("n_tracks")
                if n_tracks is None and tracks_df.height:
                    n_tracks = tracks_df["track_id"].n_unique()
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
        self._session: Optional[PipelineSession] = None
        self._session_item: Optional[ExperimentItem] = None
        self.setAcceptDrops(True)

        header = QLabel(_ExperimentListView.KEYBINDINGS)
        header.setStyleSheet("color: gray; font-size: 11px;")

        self.list_view = _ExperimentListView()
        self.list_view.currentItemChanged.connect(self._on_selection_changed)
        self.list_view.runRequested.connect(self._run_items)

        open_button = QPushButton("Open folder…")
        open_button.clicked.connect(self._open_folder_dialog)
        run_button = QPushButton("Run selected")
        run_button.clicked.connect(lambda: self._run_items(self.list_view.selectedItems()))

        button_row = QHBoxLayout()
        button_row.addWidget(open_button)
        button_row.addWidget(run_button)

        self.params_panel = PipelineParamsWidget()
        self.params_panel.calibrateRequested.connect(self._run_calibrate_step)
        self.params_panel.detectRequested.connect(self._run_detect_step)
        self.params_panel.trackRequested.connect(self._run_track_step)

        self.progress_label = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        layout = QVBoxLayout()
        layout.addWidget(header)
        layout.addLayout(button_row)
        layout.addWidget(self.list_view)
        layout.addWidget(self.params_panel)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

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
        if current is None:
            return
        self._session = None
        self._session_item = None
        self.params_panel.set_calibration_status("")
        self.params_panel.set_detect_status("")
        self.params_panel.set_track_status("")

        entry = current.entry
        if has_experiment(entry.experiment_dir):
            add_experiment_layers(self.viewer, entry.experiment_dir)
        else:
            self.viewer.layers.clear()
            image, _, _ = load_stack(entry.image_path)
            self.viewer.add_image(image, name=entry.image_path.stem)

    def _run_items(self, items: list[ExperimentItem]) -> None:
        runnable = [item for item in items if item.entry.status is not Status.RUNNING]
        if not runnable or self._worker is not None:
            return
        self._run_queue = runnable
        self._run_next()

    def _run_next(self) -> None:
        if not self._run_queue:
            self._worker = None
            self.progress_bar.setVisible(False)
            self.progress_label.setText("")
            return

        item = self._run_queue.pop(0)
        entry = item.entry
        item.set_status(Status.RUNNING)
        self.list_view.viewport().update()

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Running: {entry.image_path.name}")

        emitter = _ProgressEmitter()
        emitter.updated.connect(self._on_progress)

        worker = _run_pipeline_worker(entry.image_path, self.params_panel.get_params(), emitter)
        worker.returned.connect(lambda result, item=item: self._on_run_finished(item, result))
        worker.errored.connect(lambda exc, item=item: self._on_run_error(item, exc))
        self._worker = worker
        worker.start()

    def _on_progress(self, done: int, total: int, stage: str) -> None:
        if total:
            self.progress_bar.setValue(int(100 * done / total))
        self.progress_label.setText(stage)

    def _on_run_finished(self, item: ExperimentItem, result) -> None:
        points_df, tracks_df, manifest_extra = result
        entry = item.entry

        import spt_pipeline
        import sfwloc

        repo_shas = {
            "sfwloc": git_sha(repo_root_of(sfwloc)),
            "spt_pipeline": git_sha(repo_root_of(spt_pipeline)),
        }
        manifest = build_manifest(
            experiment_id=entry.experiment_dir.name,
            source_image_path=entry.image_path,
            params=manifest_extra,
            repo_shas=repo_shas,
        )
        write_experiment(entry.experiment_dir, points_df, tracks_df, manifest)
        item.set_status(Status.COMPLETE, n_tracks=manifest_extra["n_tracks"])
        self.list_view.viewport().update()

        if self.list_view.currentItem() is item:
            add_experiment_layers(self.viewer, entry.experiment_dir)

        self._worker = None
        self._run_next()

    def _on_run_error(self, item: ExperimentItem, exc: Exception) -> None:
        item.set_status(Status.ERROR, error=str(exc))
        self.list_view.viewport().update()
        self._worker = None
        self._run_next()

    # -- Stepwise Calibrate / Detect / Track (current selection only) --

    def _ensure_session(self, item: ExperimentItem) -> PipelineSession:
        """The session for `item`, loading its image fresh if `item` isn't
        already the one `self._session` was built for (selection changes
        already reset `self._session` to None, but this also covers the
        first click for a freshly-selected item)."""
        if self._session is not None and self._session_item is item:
            return self._session
        session = load_session(item.entry.image_path)
        self._session = session
        self._session_item = item
        return session

    def _build_roi_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Boolean `(H, W)` mask from the viewer's currently active Shapes
        layer -- the union of every shape drawn on it. Raises if there
        isn't one, or it has nothing drawn on it yet."""
        active = self.viewer.layers.selection.active
        if not isinstance(active, Shapes) or len(active.data) == 0:
            raise ValueError("no active Shapes layer with a shape drawn -- select/draw one as the ROI")
        return np.any(active.to_masks(shape), axis=0)

    def _start_step_worker(self, worker, on_finished, indeterminate: bool = False) -> None:
        if self._worker is not None:
            return
        self.progress_bar.setRange(0, 0 if indeterminate else 100)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        worker.returned.connect(on_finished)
        worker.errored.connect(self._on_step_error)
        self._worker = worker
        worker.start()

    def _finish_step_worker(self) -> None:
        self._worker = None
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")

    def _on_step_error(self, exc: Exception) -> None:
        self.progress_label.setText(f"error: {exc}")
        self._finish_step_worker()

    def _run_calibrate_step(self) -> None:
        item = self.list_view.currentItem()
        if item is None:
            return
        try:
            session = self._ensure_session(item)
        except Exception as exc:
            self.params_panel.set_calibration_status(f"error: {exc}")
            return
        self.params_panel.set_frame_bounds(session.image.shape[0])
        self.progress_label.setText(f"Calibrating: {item.entry.image_path.name}")
        worker = _run_calibration_worker(
            session,
            self.params_panel.get_sigma_init(),
            self.params_panel.get_calibration_kwargs(),
            self.params_panel.get_calibration_frame_index(),
        )
        self._start_step_worker(
            worker, lambda s, item=item: self._on_calibrate_finished(item, s), indeterminate=True
        )

    def _on_calibrate_finished(self, item: ExperimentItem, session: PipelineSession) -> None:
        if self._session_item is not item:
            # Selection changed to a different item while this ran --
            # `_on_selection_changed` already reset `self._session`;
            # don't resurrect a stale one for the item we've left.
            self._finish_step_worker()
            return
        self._session = session
        summary = session.calib_summary or {}
        self.params_panel.set_calibration_status(
            f"sigma = {session.sigma:.3f} px  "
            f"(n={summary.get('n_spots_used', 0)}/{summary.get('n_spots_total', 0)} spots)"
        )
        self._add_calibration_layer(session)
        self._finish_step_worker()

    def _add_calibration_layer(self, session: PipelineSession) -> None:
        """One box-shaped point per calibration spot, `features=` set to
        the full per-spot fit table (`y`/`x`/`sigma`/`se_sigma`/`nll`/
        `converged`/`laplace_ok`/`at_bound`/...) plus a derived
        `accepted` column -- napari shows a hovered/selected point's
        features in the status bar, so every spot's fit quality is
        inspectable, not just the aggregate sigma_estimate in the status
        label. `accepted` (green border) mirrors the same `converged &
        laplace_ok & !at_bound` filter `calibrate_sigma_df`'s own
        `sigma_estimate` aggregate uses -- gray-bordered points didn't
        count towards it. Faces are transparent (border color only) so
        the boxes outline each fit window without occluding the
        underlying image."""
        df = session.calib_points_df
        if df is None or df.height == 0:
            return
        if "calibration spots" in self.viewer.layers:
            del self.viewer.layers["calibration spots"]

        features = {col: df[col].to_numpy() for col in df.columns}
        features["accepted"] = (df["converged"] & df["laplace_ok"] & ~df["at_bound"]).to_numpy()
        box_size = (session.calibration_kwargs_used or {}).get("box_size", 11)

        self.viewer.add_points(
            df.select(["y", "x"]).to_numpy(),
            name="calibration spots",
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

    def _run_detect_step(self) -> None:
        item = self.list_view.currentItem()
        if item is None:
            return
        try:
            session = self._ensure_session(item)
        except Exception as exc:
            self.params_panel.set_detect_status(f"error: {exc}")
            return
        self.params_panel.set_frame_bounds(session.image.shape[0])

        mask = None
        if self.params_panel.get_use_roi_mask():
            try:
                mask = self._build_roi_mask(session.image.shape[1:])
            except Exception as exc:
                self.params_panel.set_detect_status(f"error: {exc}")
                return

        sigma = session.sigma if session.sigma is not None else self.params_panel.get_sigma_init()
        self.progress_label.setText(f"Finding spots: {item.entry.image_path.name}")
        emitter = _ProgressEmitter()
        emitter.updated.connect(self._on_progress)
        worker = _run_detect_worker(
            session,
            sigma,
            self.params_panel.get_solver_kwargs(),
            self.params_panel.get_frame_range(),
            mask,
            emitter,
        )
        self._start_step_worker(worker, lambda s, item=item: self._on_detect_finished(item, s))

    def _on_detect_finished(self, item: ExperimentItem, session: PipelineSession) -> None:
        if self._session_item is not item:
            self._finish_step_worker()
            return
        self._session = session
        n_points = session.points_df.height if session.points_df is not None else 0
        start, end = session.frame_range_used or (0, session.image.shape[0])
        self.params_panel.set_detect_status(f"{n_points} points across frames {start}-{end - 1}")

        if session.points_df is not None:
            if "points (preview)" in self.viewer.layers:
                del self.viewer.layers["points (preview)"]
            self.viewer.add_points(
                session.points_df.select(["frame", "y", "x"]).to_numpy(),
                name="points (preview)",
                size=4,
            )
        self._finish_step_worker()

    def _run_track_step(self) -> None:
        item = self.list_view.currentItem()
        if item is None:
            return
        session = self._session if self._session_item is item else None
        if session is None or session.points_df is None:
            self.params_panel.set_track_status("error: run detect first")
            return
        self.progress_label.setText(f"Linking: {item.entry.image_path.name}")
        emitter = _ProgressEmitter()
        emitter.updated.connect(self._on_progress)
        worker = _run_track_worker(session, self.params_panel.get_bootstrap_gate_px(), emitter)
        self._start_step_worker(worker, lambda s, item=item: self._on_track_finished(item, s))

    def _on_track_finished(self, item: ExperimentItem, session: PipelineSession) -> None:
        if self._session_item is not item:
            self._finish_step_worker()
            return
        self._session = session
        summary = session.track_summary or {}
        n_tracks = (
            session.tracks_df["track_id"].n_unique() if session.tracks_df is not None and session.tracks_df.height else 0
        )
        self.params_panel.set_track_status(
            f"{n_tracks} tracks  D~{summary.get('D_est_um2_s', 0.0):.4f} um^2/s"
        )

        entry = item.entry
        import spt_pipeline
        import sfwloc

        repo_shas = {
            "sfwloc": git_sha(repo_root_of(sfwloc)),
            "spt_pipeline": git_sha(repo_root_of(spt_pipeline)),
        }
        manifest = build_manifest(
            experiment_id=entry.experiment_dir.name,
            source_image_path=entry.image_path,
            params=session_manifest_extra(session),
            repo_shas=repo_shas,
        )
        write_experiment(entry.experiment_dir, session.points_df, session.tracks_df, manifest)
        item.set_status(Status.COMPLETE, n_tracks=n_tracks)
        self.list_view.viewport().update()

        if self.list_view.currentItem() is item:
            if "points (preview)" in self.viewer.layers:
                del self.viewer.layers["points (preview)"]
            add_experiment_layers(self.viewer, entry.experiment_dir)

        self._finish_step_worker()


@thread_worker(start_thread=False)
def _run_pipeline_worker(image_path: Path, params: DetectTrackParams, emitter: _ProgressEmitter):
    def progress_cb(done: int, total: int, stage: str) -> None:
        emitter.updated.emit(done, total, stage)

    return run_detect_track(image_path, params=params, progress_callback=progress_cb)


@thread_worker(start_thread=False)
def _run_calibration_worker(
    session: PipelineSession, sigma_init: float, calibration_kwargs: dict, frame_index: int
) -> PipelineSession:
    return run_calibration_step(session, sigma_init, calibration_kwargs, frame_index=frame_index)


@thread_worker(start_thread=False)
def _run_detect_worker(
    session: PipelineSession,
    sigma: float,
    solver_kwargs: dict,
    frame_range: Optional[tuple[int, int]],
    mask: Optional[np.ndarray],
    emitter: _ProgressEmitter,
) -> PipelineSession:
    def progress_cb(done: int, total: int, stage: str) -> None:
        emitter.updated.emit(done, total, stage)

    return run_detect_step(
        session,
        sigma=sigma,
        solver_kwargs=solver_kwargs,
        frame_range=frame_range,
        mask=mask,
        progress_callback=progress_cb,
    )


@thread_worker(start_thread=False)
def _run_track_worker(session: PipelineSession, bootstrap_gate_px: float, emitter: _ProgressEmitter) -> PipelineSession:
    def progress_cb(done: int, total: int, stage: str) -> None:
        emitter.updated.emit(done, total, stage)

    return run_track_step(session, bootstrap_gate_px, progress_callback=progress_cb)
