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

Deliberately excluded (see the project plan): interactive spot-finding
parameter tuning, an in-app code-exec tab, and the diffusion-analysis step
-- this widget's job is browse/load/run-detect-track, nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

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
    write_experiment,
)
from spt_pipeline.io_formats import SUPPORTED_SUFFIXES as SUPPORTED_FORMATS
from spt_pipeline.io_formats import load_stack
from spt_pipeline.pipeline import DetectTrackParams, run_detect_track
from spt_pipeline.viewer import add_experiment_layers


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
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if path.is_dir():
                self.load_folder(path, self.experiments_root)

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

        self.progress_label = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        layout = QVBoxLayout()
        layout.addWidget(header)
        layout.addLayout(button_row)
        layout.addWidget(self.list_view)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

    def _open_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder of timelapses")
        if folder:
            self.list_view.load_folder(Path(folder))

    def _on_selection_changed(self, current: Optional[ExperimentItem], _previous) -> None:
        if current is None:
            return
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

        worker = _run_pipeline_worker(entry.image_path, DetectTrackParams(), emitter)
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
            "sfwloc": git_sha(Path(sfwloc.__file__).resolve().parents[2]),
            "spt_pipeline": git_sha(Path(spt_pipeline.__file__).resolve().parents[2]),
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


@thread_worker(start_thread=False)
def _run_pipeline_worker(image_path: Path, params: DetectTrackParams, emitter: _ProgressEmitter):
    def progress_cb(done: int, total: int, stage: str) -> None:
        emitter.updated.emit(done, total, stage)

    return run_detect_track(image_path, params=params, progress_callback=progress_cb)
