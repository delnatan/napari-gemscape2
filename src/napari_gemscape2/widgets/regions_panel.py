"""The Detect tab's regions controls: which painted Labels layer restricts
and labels a run, and what each of its labels means.

A region is one label value on a napari Labels layer (see
`napari_gemscape2.regions`). Painting is napari's own job -- brush, fill,
polygon and eraser on the layer -- and this panel adds what the image
can't hold by itself: a name per label (its class), edited in a small
table that follows the layer as it is painted (`_on_layer_painted`,
debounced). Every pixel belongs to one label, so a nucleus painted over
its cell is cut out of that cell without anything else to do.

The `Regions` table is kept on the layer itself
(`layer.metadata["regions"]`), so it follows the layer through renames,
switching layers in the picker, and a bundle reload (`viewer.show_result`
puts it back there).

The table is the layer's label list: selecting a row paints with that
label, "+" reserves the next free label and switches to the paint tool,
and "-" (or Delete) erases the selected label's pixels -- through the
layer's undo history -- and drops its row. A label painted any other way
(napari's own label spinbox, say) turns up as `region <label>`, to be
renamed in its row. Names may repeat: each label is still its own
region, and a name pools them as one class.

Like `params_panel`, this stays viewer-agnostic: `ExperimentListWidget`
fills the layer picker (`set_layer_choices`), creates new layers on
`newLayerRequested`, and hands the chosen layer back with `set_layer`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_gemscape2.regions import Region, Regions, default_class, present_labels, sync_table

METADATA_KEY = "regions"


def layer_regions(layer) -> Regions:
    """The `Regions` table a Labels layer carries, created if missing."""
    regions = layer.metadata.get(METADATA_KEY)
    if not isinstance(regions, Regions):
        regions = Regions()
        layer.metadata[METADATA_KEY] = regions
    return regions


class _RegionsTable(QTableWidget):
    """The label table, with Delete/Backspace asking to remove the
    selected row."""

    removeRequested = Signal()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.removeRequested.emit()
            return
        super().keyPressEvent(event)


class RegionsPanel(QWidget):
    newLayerRequested = Signal()
    layerChosen = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._layer = None
        self._shown_labels: list[int] = []
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self.refresh)

        self.use_mask = QCheckBox("restrict to regions")
        self.use_mask.setToolTip(
            "Only place emitters on painted pixels of the regions layer\n"
            "(spotsolve's roi argument), and label each detection with the\n"
            "region it fell in (region and region_class columns).\n"
            "Tracking links each region on its own, so no track crosses a\n"
            "boundary, with link parameters fitted per class. Check it to show\n"
            "the regions layer picker and table."
        )
        self.layer_picker = QComboBox()
        self.layer_picker.setToolTip("Which Labels layer holds the regions.")
        self.layer_picker.currentTextChanged.connect(self._on_picker_changed)
        self.new_layer_button = QPushButton("New layer")
        self.new_layer_button.setToolTip(
            "Add an empty 2D Labels layer the size of a frame, with the paint\n"
            "tool active -- 2D so the regions show on every frame."
        )
        self.new_layer_button.clicked.connect(self.newLayerRequested.emit)

        self.add_button = QPushButton("+")
        self.add_button.setToolTip("Paint with the next free label, as a new region to name in its row.")
        self.add_button.clicked.connect(self._add_region)
        self.remove_button = QPushButton("\u2212")
        self.remove_button.setToolTip(
            "Erase the selected label's pixels and drop its row (Delete in\n"
            "the table does the same; Ctrl-Z on the layer brings the pixels back)."
        )
        self.remove_button.clicked.connect(self._remove_selected)

        self.table = _RegionsTable(0, 3)
        self.table.setHorizontalHeaderLabels(["", "label", "name"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(8 * self.table.fontMetrics().height() + 24)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        self.table.removeRequested.connect(self._remove_selected)

        # Two short rows rather than one: checkbox + picker + button side
        # by side set the whole Detect page's minimum width.
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.layer_picker, 1)
        top.addWidget(self.new_layer_button)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch()
        # Everything under the checkbox folds away while it is off: a run
        # without regions has no use for the picker or the table.
        self._body = QWidget()
        body = QVBoxLayout(self._body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(2)
        body.addLayout(top)
        body.addLayout(buttons)
        body.addWidget(self.table)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.use_mask)
        layout.addWidget(self._body)
        self.use_mask.toggled.connect(self._sync_enabled)
        self._sync_enabled()

    # -- what the host reads --

    def get_use_mask(self) -> bool:
        return self.use_mask.isChecked()

    def set_use_mask(self, enabled: bool) -> None:
        self.use_mask.setChecked(enabled)

    def layer_name(self) -> Optional[str]:
        return self.layer_picker.currentText() or None

    def layer(self):
        return self._layer

    def regions(self) -> Optional[Regions]:
        """The current layer's table, synced to what is painted."""
        if self._layer is None:
            return None
        return sync_table(np.asarray(self._layer.data), layer_regions(self._layer))

    # -- what the host sets --

    def set_layer_choices(self, names: list[str], current: Optional[str]) -> None:
        """Replace the picker's entries with the viewer's Labels layer
        names, selecting `current` (or the first, if it's gone)."""
        blocked = self.layer_picker.blockSignals(True)
        self.layer_picker.clear()
        self.layer_picker.addItems(names)
        if current in names:
            self.layer_picker.setCurrentText(current)
        self.layer_picker.blockSignals(blocked)
        self._sync_enabled()

    def set_layer(self, layer) -> None:
        """Follow `layer` (a napari Labels layer, or None): its paint and
        data events refresh the table."""
        if layer is self._layer:
            self.refresh()
            return
        if self._layer is not None:
            for emitter in self._layer_emitters(self._layer):
                emitter.disconnect(self._on_layer_painted)
        self._layer = layer
        if layer is not None:
            layer_regions(layer)
            for emitter in self._layer_emitters(layer):
                emitter.connect(self._on_layer_painted)
        self.refresh(force=True)

    @staticmethod
    def _layer_emitters(layer) -> list:
        return [getattr(layer.events, name) for name in ("paint", "data") if hasattr(layer.events, name)]

    # -- table --

    def _on_layer_painted(self, event=None) -> None:
        self._refresh_timer.start()

    def refresh(self, force: bool = False) -> None:
        """Rebuild the table from the layer: one row per painted label,
        plus the label about to be painted if "+" reserved it. Skipped
        when those labels are unchanged, unless `force`."""
        self._sync_enabled()
        layer = self._layer
        if layer is None:
            self._shown_labels = []
            self.table.setRowCount(0)
            return
        data = np.asarray(layer.data)
        regions = layer_regions(layer)
        pending = int(layer.selected_label)
        reserved = regions.table.get(pending)
        sync_table(data, regions)
        if reserved is not None and pending not in regions.table:
            regions.table[pending] = reserved  # reserved, not painted yet
        shown = sorted(set(present_labels(data)) | ({pending} if reserved else set()))
        if shown == self._shown_labels and not force:
            return
        self._shown_labels = shown
        blocked = self.table.blockSignals(True)
        self.table.setRowCount(len(shown))
        for row, label in enumerate(shown):
            region = regions.table[label]
            swatch = QTableWidgetItem()
            swatch.setBackground(self._label_color(label))
            self.table.setItem(row, 0, swatch)
            self.table.setItem(row, 1, QTableWidgetItem(str(label)))
            # Editable, offering the names already in use, so a repeated
            # name is picked rather than retyped.
            combo = QComboBox()
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            combo.addItems(regions.class_names())
            combo.setCurrentText(region.class_)
            combo.currentTextChanged.connect(lambda text, label=label: self._set_class(label, text))
            combo.lineEdit().editingFinished.connect(self._refresh_name_choices)
            self.table.setCellWidget(row, 2, combo)
            if label == pending:
                self.table.selectRow(row)
        self.table.blockSignals(blocked)

    def _label_color(self, label: int) -> QColor:
        try:
            rgba = np.asarray(self._layer.colormap.map(np.array([label]))).reshape(-1)[:4]
            return QColor.fromRgbF(*[float(c) for c in rgba])
        except Exception:
            return QColor(Qt.GlobalColor.transparent)

    def _selected_label(self) -> Optional[int]:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return self._shown_labels[rows[0].row()]

    def _on_row_selected(self) -> None:
        label = self._selected_label()
        if label is not None and self._layer is not None:
            self._layer.selected_label = label

    def _set_class(self, label: int, name: str) -> None:
        name = name.strip()
        if name and self._layer is not None and label in layer_regions(self._layer).table:
            layer_regions(self._layer).table[label].class_ = name

    def _refresh_name_choices(self) -> None:
        """Offer every row the names in use now, after one was edited."""
        if self._layer is None:
            return
        names = layer_regions(self._layer).class_names()
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 2)
            if combo is None:
                continue
            blocked = combo.blockSignals(True)
            current = combo.currentText()
            combo.clear()
            combo.addItems(names)
            combo.setCurrentText(current)
            combo.blockSignals(blocked)

    # -- buttons --

    def _add_region(self) -> None:
        layer = self._layer
        if layer is None:
            return
        regions = layer_regions(layer)
        label = regions.next_label(np.asarray(layer.data))
        regions.table[label] = Region(default_class(label))
        layer.selected_label = label
        layer.mode = "paint"
        self.refresh(force=True)

    def _remove_selected(self) -> None:
        layer = self._layer
        label = self._selected_label()
        if layer is None or label is None:
            return
        row = self._shown_labels.index(label)
        layer_regions(layer).table.pop(label, None)
        # data_setitem, not a plain assignment: it goes on the layer's undo
        # history, and its paint event refreshes whatever else follows it.
        indices = np.nonzero(np.asarray(layer.data) == label)
        if indices[0].size:
            layer.data_setitem(indices, 0)
        self.refresh(force=True)
        # Stay put, so repeated Deletes walk down the list. The rebuild may
        # already have moved the selection there with signals blocked, so
        # the layer is told directly rather than through _on_row_selected.
        if self._shown_labels:
            row = min(row, len(self._shown_labels) - 1)
            self.table.selectRow(row)
            layer.selected_label = self._shown_labels[row]

    def _on_picker_changed(self, name: str) -> None:
        if name:
            self.layerChosen.emit(name)

    def _sync_enabled(self) -> None:
        self._body.setVisible(self.use_mask.isChecked())
        has_layer = self._layer is not None
        self.layer_picker.setEnabled(self.layer_picker.count() > 0)
        for widget in (self.add_button, self.remove_button, self.table):
            widget.setEnabled(has_layer)
