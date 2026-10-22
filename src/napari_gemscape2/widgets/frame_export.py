"""Screenshot and frame-sequence export: the canvas as PNGs that keep the
colours seen on screen (see `napari_gemscape2.snapshots` for why napari's
own screenshots come out dull on a Mac).

"Screenshot…" saves the current view. "Export frames…" steps the chosen
slider (time, by default) through a range and saves one PNG per step into
a folder, numbered from 0; the ffmpeg command for turning them into a movie
or GIF is written beside them in `ffmpeg.txt`.
"""

from __future__ import annotations

from pathlib import Path

from qtkit import flow_row
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QWidget,
)

from napari_gemscape2.snapshots import COLOR_SPACES, export_frames, ffmpeg_hint, save_png


class FrameExportWidget(QWidget):
    def __init__(self, napari_viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer

        self._axis = QComboBox()
        self._axis.setToolTip("The slider to step through; one frame per step.")
        self._start = QSpinBox()
        self._stop = QSpinBox()
        self._stop.setToolTip("Last step included.")
        self._every = QSpinBox()
        self._every.setRange(1, 1_000_000)
        self._every.setToolTip("Save every n-th step.")
        self._scale = QDoubleSpinBox()
        self._scale.setRange(0.25, 8.0)
        self._scale.setSingleStep(0.5)
        self._scale.setValue(1.0)
        self._scale.setSuffix("×")
        self._scale.setToolTip("Render the canvas at this multiple of its size on screen.")
        self._color = QComboBox()
        self._color.addItems(COLOR_SPACES)
        self._color.setToolTip(
            "The colour profile written into each PNG. Display P3 matches a Mac screen "
            "(napari's canvas is shown without colour matching); sRGB for other screens."
        )

        screenshot = QPushButton("Screenshot…")
        screenshot.setToolTip("Save the current canvas as a PNG.")
        screenshot.clicked.connect(self._save_screenshot)
        export = QPushButton("Export frames…")
        export.setToolTip("Save one PNG per step of the slider into a folder.")
        export.clicked.connect(self._export_frames)
        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        # A form of its own rather than the MSD tab's `labelled` rows: this
        # widget stays importable without the analysis stack behind them.
        layout = QFormLayout(self)
        layout.addRow("along", self._axis)
        layout.addRow("steps", flow_row(self._start, QLabel("to"), self._stop, QLabel("every"), self._every))
        layout.addRow("scale", self._scale)
        layout.addRow("colour", self._color)
        layout.addRow(flow_row(screenshot, export))
        layout.addRow(self._status)

        self._axis.currentIndexChanged.connect(self._on_axis_changed)
        self.viewer.dims.events.range.connect(self._refresh_axes)
        self.viewer.dims.events.axis_labels.connect(self._refresh_axes)
        self._refresh_axes()

    def _refresh_axes(self, event=None) -> None:
        dims = self.viewer.dims
        # Sliders only: the displayed axes are the picture itself.
        sliders = [axis for axis in range(dims.ndim) if axis not in dims.displayed]
        keep = self._axis.currentData()
        self._axis.blockSignals(True)
        self._axis.clear()
        for axis in sliders:
            self._axis.addItem(f"{dims.axis_labels[axis]} ({dims.nsteps[axis]})", axis)
        index = self._axis.findData(keep)
        self._axis.setCurrentIndex(index if index >= 0 else 0)
        self._axis.blockSignals(False)
        self._on_axis_changed()

    def _on_axis_changed(self, index=None) -> None:
        axis = self._axis.currentData()
        last = self.viewer.dims.nsteps[axis] - 1 if axis is not None else 0
        for box in (self._start, self._stop):
            box.setRange(0, last)
        self._start.setValue(0)
        self._stop.setValue(last)

    def _scale_value(self):
        return None if self._scale.value() == 1.0 else self._scale.value()

    def _save_screenshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", "screenshot.png", "PNG (*.png)")
        if not path:
            return
        frame = self.viewer.screenshot(canvas_only=True, scale=self._scale_value(), flash=False)
        path = save_png(frame, Path(path).with_suffix(".png"), self._color.currentText())
        self._status.setText(f"saved {path.name} ({self._color.currentText()})")

    def _export_frames(self) -> None:
        axis = self._axis.currentData()
        if axis is None:
            QMessageBox.information(self, "Export frames", "There is no slider to step through.")
            return
        steps = range(self._start.value(), self._stop.value() + 1, self._every.value())
        if not steps:
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the frames")
        if not folder:
            return
        stem = "frame"
        color = self._color.currentText()

        dialog = QProgressDialog("Saving frames…", "Stop", 0, len(steps), self)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)

        def progress(done: int) -> bool:
            dialog.setValue(done)
            QApplication.processEvents()
            return not dialog.wasCanceled()

        written = export_frames(
            self.viewer, folder, stem, axis, steps, color, self._scale_value(), progress
        )
        dialog.close()
        hint = ffmpeg_hint(stem, color)
        (Path(folder) / "ffmpeg.txt").write_text(hint)
        self._status.setText(f"saved {len(written)} frames to {folder} ({color}); ffmpeg command in ffmpeg.txt")
