"""Small Qt building blocks shared across the dock widgets -- the
green/amber/red status-label color language and the section-separator
rule, originally `params_panel.py`-private, promoted here once
`diffusion_panel.py` needed the same look. Kept dependency-free of any
other widget module (no napari `Viewer`, no pipeline imports) so every
dock widget can import it without pulling in unrelated state.
"""

from __future__ import annotations

import polars as pl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt
from qtpy.QtWidgets import QDialog, QFrame, QLabel, QVBoxLayout

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
