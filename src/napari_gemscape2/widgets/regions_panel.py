"""The Detect tab's regions controls: which painted Labels layer restricts
and labels a run, and what each of its labels means.

A region is one label value on a napari Labels layer (see
`napari_gemscape2.regions`). Painting is napari's own job -- brush, fill,
polygon and eraser on the layer -- and this panel adds what the image
can't hold by itself: a class and a cell id per label, edited in a small
table that follows the layer as it is painted (`_on_layer_painted`,
debounced). Every pixel belongs to one label, so a nucleus painted over
its cell is cut out of that cell without anything else to do.

The `Regions` table is kept on the layer itself
(`layer.metadata["regions"]`), so it follows the layer through renames,
switching layers in the picker, and a bundle reload (`viewer.show_result`
puts it back there).

"New cell" and "Add nucleus" are shortcuts for the usual workflow: pick
the next free label, record its class and cell up front, and switch the
layer to the paint tool. A label painted any other way (napari's own
label spinbox, say) turns up in the table as a new cell of the first
class, to be corrected there.

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
    QInputDialog,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_gemscape2.regions import Region, Regions, present_labels, sync_table

METADATA_KEY = "regions"
_NUCLEUS = "nucleus"


def layer_regions(layer) -> Regions:
    """The `Regions` table a Labels layer carries, created if missing."""
    regions = layer.metadata.get(METADATA_KEY)
    if not isinstance(regions, Regions):
        regions = Regions()
        layer.metadata[METADATA_KEY] = regions
    return regions


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
            "region it fell in (region, region_class and cell columns).\n"
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

        self.new_cell_button = QPushButton("New cell")
        self.new_cell_button.setToolTip(
            "Paint with the next free label, recorded as a new cell of the\n"
            "first class (cytoplasm by default). Paint the whole cell."
        )
        self.new_cell_button.clicked.connect(self._new_cell)
        self.add_nucleus_button = QPushButton("Add nucleus")
        self.add_nucleus_button.setToolTip(
            "Paint with the next free label, recorded as the nucleus of the\n"
            "cell selected in the table. Painting it over the cell cuts it\n"
            "out of the cell's cytoplasm."
        )
        self.add_nucleus_button.clicked.connect(self._add_nucleus)
        self.classes_button = QPushButton("Classes")
        self.classes_button.setToolTip("Edit the list of region classes (comma-separated).")
        self.classes_button.clicked.connect(self._edit_classes)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["", "label", "class", "cell"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMaximumHeight(8 * self.table.fontMetrics().height() + 24)
        self.table.itemSelectionChanged.connect(self._on_row_selected)

        # Two short rows rather than one: checkbox + picker + button side
        # by side set the whole Detect page's minimum width.
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.layer_picker, 1)
        top.addWidget(self.new_layer_button)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(self.new_cell_button)
        buttons.addWidget(self.add_nucleus_button)
        buttons.addStretch()
        buttons.addWidget(self.classes_button)
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
        self._shown_labels = []
        self.refresh()

    @staticmethod
    def _layer_emitters(layer) -> list:
        return [getattr(layer.events, name) for name in ("paint", "data") if hasattr(layer.events, name)]

    # -- table --

    def _on_layer_painted(self, event=None) -> None:
        self._refresh_timer.start()

    def refresh(self) -> None:
        """Rebuild the table from the layer: one row per painted label,
        plus the label about to be painted if the buttons reserved it."""
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
        if shown == self._shown_labels:
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
            combo = QComboBox()
            combo.addItems(regions.classes)
            combo.setCurrentText(region.class_)
            combo.currentTextChanged.connect(lambda text, label=label: self._set_class(label, text))
            self.table.setCellWidget(row, 2, combo)
            spin = QSpinBox()
            spin.setRange(1, 65535)
            spin.setValue(region.cell)
            spin.valueChanged.connect(lambda value, label=label: self._set_cell(label, value))
            self.table.setCellWidget(row, 3, spin)
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
        if self._layer is not None and label in layer_regions(self._layer).table:
            layer_regions(self._layer).table[label].class_ = name

    def _set_cell(self, label: int, cell: int) -> None:
        if self._layer is not None and label in layer_regions(self._layer).table:
            layer_regions(self._layer).table[label].cell = cell

    # -- buttons --

    def _reserve(self, region: Region) -> None:
        layer = self._layer
        regions = layer_regions(layer)
        label = regions.next_label(np.asarray(layer.data))
        regions.table[label] = region
        layer.selected_label = label
        layer.mode = "paint"
        self._shown_labels = []
        self.refresh()

    def _new_cell(self) -> None:
        if self._layer is None:
            return
        regions = layer_regions(self._layer)
        self._reserve(Region(regions.classes[0], regions.next_cell()))

    def _add_nucleus(self) -> None:
        if self._layer is None:
            return
        regions = layer_regions(self._layer)
        label = self._selected_label()
        if label is None or label not in regions.table:
            return
        if _NUCLEUS not in regions.classes:
            regions.classes.append(_NUCLEUS)
        self._reserve(Region(_NUCLEUS, regions.table[label].cell))

    def _edit_classes(self) -> None:
        if self._layer is None:
            return
        regions = layer_regions(self._layer)
        text, ok = QInputDialog.getText(
            self, "Region classes", "Classes (comma-separated):", text=", ".join(regions.classes)
        )
        if not ok:
            return
        classes = [c.strip() for c in text.split(",") if c.strip()]
        # A class still assigned to a region can't disappear from the list.
        classes += [c for c in dict.fromkeys(r.class_ for r in regions.table.values()) if c not in classes]
        if classes:
            regions.classes = classes
            self._shown_labels = []
            self.refresh()

    def _on_picker_changed(self, name: str) -> None:
        if name:
            self.layerChosen.emit(name)

    def _sync_enabled(self) -> None:
        self._body.setVisible(self.use_mask.isChecked())
        has_layer = self._layer is not None
        self.layer_picker.setEnabled(self.layer_picker.count() > 0)
        for widget in (self.new_cell_button, self.add_nucleus_button, self.classes_button, self.table):
            widget.setEnabled(has_layer)
