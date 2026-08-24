"""Small Qt building blocks shared across the dock widgets -- the
green/amber/red status-label color language and the section-separator
rule, originally `params_panel.py`-private, promoted here once
`diffusion_panel.py` needed the same look. Kept dependency-free of any
other widget module (no napari `Viewer`, no pipeline imports) so every
dock widget can import it without pulling in unrelated state.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QRectF, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QPainter, QPalette, QPen
from qtpy.QtWidgets import QDialog, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

STATUS_LEVEL_COLORS = {
    "neutral": "#9a9a9a",
    "ok": "#22c55e",
    "caution": "#f59e0b",
    "error": "#ef4444",
}


def style_status_label(label: QLabel, level: str = "neutral") -> None:
    color = STATUS_LEVEL_COLORS.get(level, STATUS_LEVEL_COLORS["neutral"])
    label.setStyleSheet(f"color: {color}; font-size: 11px;")


def hline() -> QFrame:
    """A thin horizontal rule for separating a widget's logical sections --
    reads faster than spacing alone once a form has several knob groups."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


class PlotWindow(QDialog):
    """A resizable, non-modal pop-up for one matplotlib Figure, with the
    standard pan/zoom/save toolbar -- a fixed-size canvas embedded in a
    narrow dock panel is unusable for anything but a glance, and dock
    widgets can't be resized independently of the rest of the napari
    window. One instance is meant to be kept around and reused via
    repeated `show_figure` calls (e.g. one window for a population fit's
    ensemble plot, a separate one for a per-track posterior corner plot),
    rather than opening a fresh window per plot."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(700, 550)
        self._canvas: FigureCanvasQTAgg | None = None
        self._toolbar: NavigationToolbar2QT | None = None
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

    def show_figure(self, figure) -> None:
        if self._canvas is not None:
            self._layout.removeWidget(self._toolbar)
            self._toolbar.deleteLater()
            self._layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
        self._canvas = FigureCanvasQTAgg(figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas)
        self._canvas.draw()
        self.show()
        self.raise_()
        self.activateWindow()


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


class DataFrameTableModel(QAbstractTableModel):
    """Read-only Qt table model over a polars DataFrame. Refresh the whole
    table via `setDataFrame`; sortable by clicking a `QTableView`'s header
    (`view.setSortingEnabled(True)`) since `sort()` re-sorts the
    underlying DataFrame and resets the model rather than juggling
    Qt-side row-index mappings."""

    def __init__(self, df: pl.DataFrame | None = None, parent=None) -> None:
        super().__init__(parent)
        self._df = df if df is not None else pl.DataFrame()

    def setDataFrame(self, df: pl.DataFrame) -> None:
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def dataframe(self) -> pl.DataFrame:
        return self._df

    def row_dict(self, row: int) -> dict:
        return self._df.row(row, named=True)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else self._df.height

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df.columns)

    def headerData(self, section: int, orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._df.columns[section]
        return str(section)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        return _format_cell(self._df[index.row(), index.column()])

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        if self._df.height == 0 or not (0 <= column < len(self._df.columns)):
            return
        name = self._df.columns[column]
        self.layoutAboutToBeChanged.emit()
        self._df = self._df.sort(name, descending=(order == Qt.SortOrder.DescendingOrder), nulls_last=True)
        self.layoutChanged.emit()


def _adaptive_spinbox_step(data_min: float, data_max: float) -> tuple[int, float]:
    """(decimals, step) sized to the data's own span -- a fixed 2-decimal
    spinbox is useless on D values (~1e-2) and equally useless (too fussy)
    on amplitude values (~1e2); this picks ~3-4 significant figures'
    worth of precision from the span instead. Ported from pyvistra's
    widgets/histogram.py::compute_spinbox_params."""
    span = data_max - data_min
    if span <= 0:
        span = 1.0
    order = math.floor(math.log10(abs(span)))
    decimals = max(0, min(4 - order, 10))
    step = span * 0.01
    if step > 0:
        step_order = math.floor(math.log10(abs(step)))
        step = round(step, -step_order) if step_order >= 0 else round(step, abs(step_order))
    return decimals, step


def configure_spinbox_for_range(spinbox: QDoubleSpinBox, data_min: float, data_max: float) -> None:
    decimals, step = _adaptive_spinbox_step(data_min, data_max)
    span = data_max - data_min
    margin = span * 0.1 if span > 0 else 1.0
    spinbox.blockSignals(True)
    spinbox.setDecimals(decimals)
    spinbox.setSingleStep(step if step > 0 else 10.0**-decimals)
    spinbox.setRange(data_min - margin, data_max + margin)
    spinbox.blockSignals(False)


class _HistogramCanvas(QWidget):
    """Draggable min/max range handles over a log-scaled histogram --
    ported from pyvistra's widgets/histogram.py (BaseHistogramWidget +
    HistogramWidget), trimmed of that project's image-contrast-specific
    bits (no dtype-based hard bounds -- these are plain numeric columns,
    not image intensities) and palette-aware instead of hardcoded colors,
    so it reads correctly against napari's dark theme without a separate
    color-token module."""

    rangeChanged = Signal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumHeight(48)
        self.setMinimumWidth(120)
        self.hist_counts: np.ndarray | None = None
        self.data_min = 0.0
        self.data_max = 1.0
        self.display_min = 0.0
        self.display_max = 1.0
        self.range_min = 0.0
        self.range_max = 1.0
        self._dragging: str | None = None

    def set_data(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            self.data_min, self.data_max = 0.0, 1.0
            self.hist_counts = np.zeros(60)
        else:
            self.data_min = float(values.min())
            self.data_max = float(values.max())
            span = self.data_max - self.data_min
            min_span = max(abs(self.data_min), abs(self.data_max), 1.0) * 1e-9
            if span < min_span:
                self.data_max = self.data_min + min_span
            counts, _ = np.histogram(values, bins=60, range=(self.data_min, self.data_max))
            self.hist_counts = np.log1p(counts)
        self._update_display_range()
        self.update()

    def set_range(self, vmin: float, vmax: float) -> None:
        self.range_min, self.range_max = vmin, vmax
        self._update_display_range()
        self.update()

    def _update_display_range(self) -> None:
        span = self.data_max - self.data_min
        margin = span * 0.05 if span > 0 else 1.0
        self.display_min = self.range_min - margin if self.range_min < self.data_min else self.data_min
        self.display_max = self.range_max + margin if self.range_max > self.data_max else self.data_max

    def _val_to_x(self, value: float) -> int:
        span = self.display_max - self.display_min
        if span <= 0:
            return 0
        x = int(((value - self.display_min) / span) * self.width())
        return max(-2_147_483_648, min(x, 2_147_483_647))

    def _x_to_val(self, x: int) -> float:
        span = self.display_max - self.display_min
        ratio = x / max(self.width(), 1)
        return max(self.display_min, min(self.display_min + ratio * span, self.display_max))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))

        h, w = self.height(), self.width()
        text_color = palette.color(QPalette.ColorRole.Text)

        if self.hist_counts is not None and self.hist_counts.max() > 0:
            fill = QColor(text_color)
            fill.setAlpha(90)
            painter.setBrush(QBrush(fill))
            painter.setPen(Qt.PenStyle.NoPen)
            n_bins = len(self.hist_counts)
            x_start = self._val_to_x(self.data_min)
            pixel_span = max(self._val_to_x(self.data_max) - x_start, 1)
            max_count = self.hist_counts.max()
            for i, count in enumerate(self.hist_counts):
                bar_h = (count / max_count) * h
                x = x_start + (i / n_bins) * pixel_span
                painter.drawRect(QRectF(x, h - bar_h, pixel_span / n_bins, bar_h))

        x_min, x_max = self._val_to_x(self.range_min), self._val_to_x(self.range_max)
        shade = QColor(0, 0, 0, 120)
        painter.fillRect(0, 0, x_min, h, shade)
        painter.fillRect(x_max, 0, w - x_max, h, shade)

        pen = QPen(text_color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(x_min, 0, x_min, h)
        painter.drawLine(x_max, 0, x_max, h)

    def _event_x(self, event) -> int:
        return int(event.position().x())

    def mousePressEvent(self, event) -> None:
        x = self._event_x(event)
        x_min, x_max = self._val_to_x(self.range_min), self._val_to_x(self.range_max)
        if abs(x - x_min) < 8:
            self._dragging = "min"
        elif abs(x - x_max) < 8:
            self._dragging = "max"
        else:
            self._dragging = None

    def mouseMoveEvent(self, event) -> None:
        x = self._event_x(event)
        x_min, x_max = self._val_to_x(self.range_min), self._val_to_x(self.range_max)
        if abs(x - x_min) < 8 or abs(x - x_max) < 8:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        if self._dragging is None:
            return
        val = self._x_to_val(x)
        if self._dragging == "min":
            self.range_min = min(val, self.range_max - 1e-9)
        else:
            self.range_max = max(val, self.range_min + 1e-9)
        self._update_display_range()
        self.rangeChanged.emit(self.range_min, self.range_max)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = None
        self.setCursor(Qt.CursorShape.ArrowCursor)


class HistogramRangeWidget(QWidget):
    """Min-spinbox / draggable-handle histogram / max-spinbox row for
    setting a numeric range interactively -- the same shape as
    pyvistra's `ChannelRow` contrast control (`widgets/channel_panel.py`
    there), generalized to any 1D numeric column (QC filtering, colormap
    remapping) instead of just image intensity. `rangeChanged` fires on
    every user-driven change (drag or spinbox edit); `set_data`/
    `set_range` are silent (for programmatic setup, e.g. switching which
    column is being edited)."""

    rangeChanged = Signal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._min_spin = QDoubleSpinBox()
        self._min_spin.setKeyboardTracking(False)
        self._min_spin.setFixedWidth(85)
        self._max_spin = QDoubleSpinBox()
        self._max_spin.setKeyboardTracking(False)
        self._max_spin.setFixedWidth(85)
        self._canvas = _HistogramCanvas()

        self._min_spin.valueChanged.connect(self._on_min_spin)
        self._max_spin.valueChanged.connect(self._on_max_spin)
        self._canvas.rangeChanged.connect(self._on_canvas_range)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._min_spin)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(self._max_spin)
        self.setLayout(layout)

    def set_data(self, values: np.ndarray) -> None:
        self._canvas.set_data(values)
        configure_spinbox_for_range(self._min_spin, self._canvas.data_min, self._canvas.data_max)
        configure_spinbox_for_range(self._max_spin, self._canvas.data_min, self._canvas.data_max)

    def set_range(self, vmin: float, vmax: float) -> None:
        self._canvas.set_range(vmin, vmax)
        self._min_spin.blockSignals(True)
        self._max_spin.blockSignals(True)
        self._min_spin.setValue(vmin)
        self._max_spin.setValue(vmax)
        self._min_spin.blockSignals(False)
        self._max_spin.blockSignals(False)

    def data_range(self) -> tuple[float, float]:
        return self._canvas.data_min, self._canvas.data_max

    def range(self) -> tuple[float, float]:
        return self._canvas.range_min, self._canvas.range_max

    def _on_min_spin(self, value: float) -> None:
        if value < self._max_spin.value():
            self.set_range(value, self._max_spin.value())
            self.rangeChanged.emit(*self.range())

    def _on_max_spin(self, value: float) -> None:
        if value > self._min_spin.value():
            self.set_range(self._min_spin.value(), value)
            self.rangeChanged.emit(*self.range())

    def _on_canvas_range(self, vmin: float, vmax: float) -> None:
        self._min_spin.blockSignals(True)
        self._max_spin.blockSignals(True)
        self._min_spin.setValue(vmin)
        self._max_spin.setValue(vmax)
        self._min_spin.blockSignals(False)
        self._max_spin.blockSignals(False)
        self.rangeChanged.emit(vmin, vmax)
