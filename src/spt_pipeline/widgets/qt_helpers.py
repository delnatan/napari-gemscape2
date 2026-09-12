"""Small Qt building blocks shared across the dock widgets -- the
green/amber/red status-label color language and the section-separator
rule, originally `params_panel.py`-private, promoted here once
`diffusion_panel.py` needed the same look. Kept dependency-free of any
other widget module (no napari `Viewer`, no pipeline imports) so every
dock widget can import it without pulling in unrelated state.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import polars as pl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QPoint, QRect, QRectF, QSize, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QPainter, QPalette, QPen
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

STATUS_LEVEL_COLORS = {
    "neutral": "#9a9a9a",
    "ok": "#22c55e",
    "caution": "#f59e0b",
    "error": "#ef4444",
}


def style_status_label(label: QLabel, level: str = "neutral") -> None:
    color = STATUS_LEVEL_COLORS.get(level, STATUS_LEVEL_COLORS["neutral"])
    label.setStyleSheet(f"color: {color}; font-size: 11px;")


def status_label(text: str = "") -> QLabel:
    """A neutral-styled status/summary label that can never widen its dock.

    A word-wrapped `QLabel` still reports its longest *unwrapped* line as
    its `sizeHint`, and that hint propagates up as the containing dock's
    minimum width -- so a single long error message or file path
    permanently stops the dock from being dragged narrower again
    (`ExperimentListWidget.progress_label` hit this first). `Ignored`
    horizontally lets the label shrink to whatever it is given and wrap
    inside it instead. Every status line in a dock should come from here
    rather than hand-rolling the four calls.
    """
    label = QLabel(text)
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    style_status_label(label)
    return label


# Floor for any internally-scrolling region, in px -- small enough that a
# dock can be dragged genuinely short, tall enough to still show a row or
# two of whatever is inside rather than a bare pair of scrollbars.
_SCROLL_MIN_HEIGHT_PX = 56


def scrolled(widget: QWidget) -> QScrollArea:
    """Wrap `widget` so its height stops dictating its container's minimum
    height -- plain nested layouts propagate a child's full `sizeHint` up
    as the parent's *minimum* size, which is how a tall tab ends up pushing
    a dock's footer buttons off the bottom of a laptop screen with no way
    to reach them. A `QScrollArea` reports a small minimum regardless of
    its content and scrolls internally instead."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(widget)
    # QAbstractScrollArea's own minimumSizeHint is derived from its
    # scrollbars and frame, which still adds up to ~90px per nested area.
    # An explicit floor is what actually lets a stack of these collapse:
    # a few scrollable rows is a usable pane, and it is always reachable
    # by scrolling, unlike content pushed off the bottom of the dock.
    area.setMinimumHeight(_SCROLL_MIN_HEIGHT_PX)
    return area


class FlowLayout(QLayout):
    """A horizontal layout that wraps onto as many lines as it needs.

    A `QHBoxLayout` of small controls has a hard minimum width: the sum of
    all of them. In a napari dock that is the single biggest thing
    stopping the panel from being dragged narrow -- one row of a label, a
    spinbox and two checkboxes was pinning this widget to 350px on its
    own, wider than anything else in it. Reflowing costs a line of height
    at narrow widths and nothing at wide ones, and drops the minimum width
    to that of the widest *single* control.

    This is Qt's documented flow-layout pattern: `heightForWidth` reports
    what the wrap would cost, and `_do_layout` either measures or places
    depending on `test_only`.
    """

    def __init__(self, parent=None, margin: int = 0, spacing: int = 4) -> None:
        super().__init__(parent)
        self._items: list = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    # -- QLayout plumbing --

    def addItem(self, item) -> None:  # noqa: N802 (Qt virtual)
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        # The widest single item, not their sum -- the whole point.
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x, y, line_height = effective.x(), effective.y(), 0
        for item in self._items:
            widget = item.widget()
            space_x = space_y = self.spacing()
            if widget is not None:
                style = widget.style()
                space_x = self.spacing() + style.layoutSpacing(
                    QSizePolicy.ControlType.PushButton,
                    QSizePolicy.ControlType.PushButton,
                    Qt.Orientation.Horizontal,
                )
                space_y = self.spacing() + style.layoutSpacing(
                    QSizePolicy.ControlType.PushButton,
                    QSizePolicy.ControlType.PushButton,
                    Qt.Orientation.Vertical,
                )
            hint = item.sizeHint()
            next_x = x + hint.width() + space_x
            if next_x - space_x > effective.right() and line_height > 0:
                x = effective.x()
                y = y + line_height + space_y
                next_x = x + hint.width() + space_x
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


def flow_row(*widgets: QWidget, spacing: int = 4) -> QWidget:
    """A `FlowLayout` of `widgets` in a container sized to actually use it.

    A widget whose layout has `hasHeightForWidth` only gets the taller
    geometry it asks for when its own size policy advertises the same, so
    the policy is set here rather than left to each caller to remember.
    """
    container = QWidget()
    layout = FlowLayout(container, spacing=spacing)
    for widget in widgets:
        layout.addWidget(widget)
    policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    policy.setHeightForWidth(True)
    container.setSizePolicy(policy)
    return container


class CollapsibleSection(QWidget):
    """A titled section whose body folds away, for controls that are tall
    but only touched occasionally -- a stack of filter histograms, a
    joint-plot picker.

    In a napari dock, vertical space is the scarce resource, and these
    bodies are the widgets that eat it: one `FeatureFilterPanel` row is a
    histogram plus two spinboxes. Folding them is what lets the same dock
    show a useful number of table rows without the user first dragging a
    splitter. The header keeps a disclosure triangle and the title
    visible while collapsed, so what is hidden is never a mystery --
    unlike a splitter pane dragged to zero, which leaves nothing to
    click back."""

    def __init__(self, title: str, content: QWidget, expanded: bool = False) -> None:
        super().__init__()
        self._content = content

        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._toggle.toggled.connect(self._on_toggled)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self._toggle)
        header.addStretch()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(header)
        layout.addWidget(content)
        self.setLayout(layout)
        content.setVisible(expanded)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self._content.setVisible(checked)

    def set_expanded(self, expanded: bool) -> None:
        self._toggle.setChecked(expanded)

    def set_title(self, title: str) -> None:
        self._toggle.setText(title)


def tabify_with_open_widget(napari_viewer, widget: QWidget, sibling_class_name: str) -> None:
    """Land `widget`'s dock as a tab on an already-open dock widget whose
    inner widget's class is named `sibling_class_name`, instead of
    napari's default of stacking a second same-area dock widget below the
    first. Matched by class name (not an imported type) so
    experiment_list.py and diffusion_panel.py don't have to import each
    other. Deferred via `QTimer.singleShot(0, ...)` because this runs from
    `__init__`, before napari wraps `widget` in its `QtViewerDockWidget`
    and docks it (see `_instantiate_dock_widget` in
    `napari._qt.qt_main_window`) -- by the next event-loop tick that
    wrapping is done, whether or not a sibling happens to be open yet."""
    from qtpy.QtCore import QTimer

    def _do_tabify() -> None:
        own_dock = widget.parent()
        if own_dock is None:
            return
        for inner in napari_viewer.window.dock_widgets.values():
            if type(inner).__name__ == sibling_class_name:
                sibling_dock = inner.parent()
                if sibling_dock is not None and sibling_dock is not own_dock:
                    # `_qt_window` (the QMainWindow) is private API, but
                    # it's the same route napari itself uses internally
                    # to tabify dock widgets -- there's no public
                    # equivalent (see `Window._add_viewer_dock_widget`).
                    napari_viewer.window._qt_window.tabifyDockWidget(sibling_dock, own_dock)
                    own_dock.show()
                    own_dock.raise_()
                return

    QTimer.singleShot(0, _do_tabify)


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


class JointPlotControl(QWidget):
    """Reusable x/y column picker + log-scale toggles + "Show plot" button
    for the generic scatter+KDE joint-distribution plot
    (`spt_pipeline.joint_plot.plot_property_joint`) -- one instance per tab
    that wants to let the user compare any two of its own per-track result
    columns, rather than a single hardcoded property pair. The tab owns the
    actual dataframe and plotting call; this widget only tracks the
    picker/checkbox state and tells the tab when to plot."""

    plotRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._x_picker = QComboBox()
        self._y_picker = QComboBox()
        self._log_x = QCheckBox("log x")
        self._log_x.setChecked(True)
        self._log_y = QCheckBox("log y")
        self._button = QPushButton("Show plot")
        self._button.setEnabled(False)
        self._button.clicked.connect(self.plotRequested)

        form = QFormLayout()
        form.addRow("x:", self._x_picker)
        form.addRow("y:", self._y_picker)
        log_row = QHBoxLayout()
        log_row.addWidget(self._log_x)
        log_row.addWidget(self._log_y)
        log_row.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(log_row)
        layout.addWidget(self._button)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def set_columns(
        self,
        columns: list[str],
        prefer_x: Optional[str] = None,
        prefer_y: Optional[str] = None,
    ) -> None:
        for picker, prefer in ((self._x_picker, prefer_x), (self._y_picker, prefer_y)):
            current = prefer or picker.currentText()
            picker.blockSignals(True)
            picker.clear()
            picker.addItems(columns)
            if current in columns:
                picker.setCurrentText(current)
            elif columns:
                picker.setCurrentIndex(0)
            picker.blockSignals(False)
        self._button.setEnabled(bool(columns))

    def clear(self) -> None:
        self._x_picker.clear()
        self._y_picker.clear()
        self._button.setEnabled(False)

    def selection(self) -> tuple[str, str, bool, bool]:
        return (
            self._x_picker.currentText(),
            self._y_picker.currentText(),
            self._log_x.isChecked(),
            self._log_y.isChecked(),
        )


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
