"""A stack of histogram range filters over one table's numeric columns --
the "filter" half of the detect -> filter -> finalize flow that Imaris and
TrackMate use for spot detection, and the one widget every stage of this
pipeline filters through.

The shape is theirs: one filter is one row -- pick a column, see its
distribution, drag two handles across it -- and filters stack, ANDing
together, so a QC decision is made by looking at the population it applies
to rather than by typing a number into a form. What that buys in a dock
panel specifically is that the *decision* and its *evidence* occupy the
same few hundred pixels: `flux > 1200` means nothing until you can see
that the flux histogram is bimodal with a trough at 1200.

Used three times over, against three different tables, which is why it is
here and not inside any one of them:

  - `params_panel._DetectTab` -- per-detection columns (`flux`,
    `fit_sigma`, `se_pos`, `bg`, ...) from a preview frame or a finished
    detect run. Decides what linking sees.
  - `params_panel._TrackingTab` -- per-track metrics
    (`pipeline.TRACK_METRIC_COLUMNS`). Decides which tracks the bundle
    keeps.
  - `diffusion_panel._TracksPane` -- per-track columns again, post-hoc on
    a loaded bundle: the same per-detection QC fields aggregated to the
    track (`flux_min`, `se_x_max`, ...), plus whatever per-track fit
    results (diffusionkit's `D`, `alpha`) have been computed into the
    track table. One panel over one table, rather than a second one at
    per-detection granularity -- a cut like "every point in this track has
    acceptable flux" is a statement about the track's min, so it belongs
    beside the cut on the track's fitted alpha.

`filters()` returns a `pipeline.FilterSpec` -- plain
`{column: (lo, hi)}` -- which is what `pipeline.filter_mask` applies and
what `manifest.json` records, so a range dragged here, a range read back
off a bundle, and a range typed into a headless TOML are the same object.
A row whose handles still span its column's whole data range is not a
filter and is left out of that dict entirely, rather than stored as a
no-op that would show up in the manifest as a cut that was never made.
"""

from __future__ import annotations

from typing import Optional, Sequence

import polars as pl
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from spt_pipeline.pipeline import FilterSpec, filter_mask
from spt_pipeline.widgets.qt_helpers import HistogramRangeWidget

# Columns no filter should offer: identity and coordinates. Filtering a
# detection on its own `y`/`x` is a crop (that's what the ROI is for), and
# on `frame` a frame range (that's the Detect tab's own control) -- both
# already have a better control elsewhere, and both would silently fight
# with it. `y_px`/`x_px` are the same thing one level up: per-track
# centroids (`diffusion_panel._base_track_table`), where a range on one
# axis alone is a half-crop rather than a region.
SKIP_COLUMNS = frozenset(
    {"loc_id", "track_id", "frame", "t", "y", "x", "y_um", "x_um", "y_px", "x_px", "seconds"}
)


def numeric_filter_columns(df: Optional[pl.DataFrame]) -> list[str]:
    """The columns of `df` a range filter can sensibly be put on: numeric,
    not an identity/coordinate column (`SKIP_COLUMNS`), and not constant
    (a column with one distinct value has no distribution to look at, so a
    histogram of it is a single bar and a filter on it is either a no-op or
    rejects everything)."""
    if df is None or df.height == 0:
        return []
    out = []
    for col, dtype in zip(df.columns, df.dtypes):
        if col in SKIP_COLUMNS or not dtype.is_numeric():
            continue
        if df[col].drop_nulls().n_unique() <= 1:
            continue
        out.append(col)
    return out


class _FilterRow(QWidget):
    """One column's histogram + range handles, with the column itself
    re-pickable in place -- so a row that turns out to be looking at the
    wrong feature is re-pointed rather than removed and re-added."""

    changed = Signal()
    removeRequested = Signal(object)

    def __init__(self, columns: Sequence[str], column: Optional[str] = None) -> None:
        super().__init__()
        self._df: Optional[pl.DataFrame] = None
        # Suppresses `changed` while `set_source`/`set_range` are moving the
        # handles programmatically -- otherwise repointing a row or
        # reloading its data would read as a user-made cut.
        self._loading = False

        self._column_picker = QComboBox()
        self._column_picker.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._column_picker.addItems(list(columns))
        if column is not None and column in columns:
            self._column_picker.setCurrentText(column)
        self._column_picker.currentTextChanged.connect(self._on_column_changed)

        self._remove_button = QPushButton("✕")
        self._remove_button.setFixedWidth(24)
        self._remove_button.setToolTip("Remove this filter")
        self._remove_button.clicked.connect(lambda: self.removeRequested.emit(self))

        self._reset_button = QPushButton("⤢")
        self._reset_button.setFixedWidth(24)
        self._reset_button.setToolTip("Widen back to this column's full range (no cut)")
        self._reset_button.clicked.connect(self.reset_range)

        self._histogram = HistogramRangeWidget()
        self._histogram.rangeChanged.connect(self._on_range_changed)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(2)
        head.addWidget(self._column_picker, 1)
        head.addWidget(self._reset_button)
        head.addWidget(self._remove_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)
        layout.addLayout(head)
        layout.addWidget(self._histogram)
        self.setLayout(layout)

    def column(self) -> str:
        return self._column_picker.currentText()

    def set_columns(self, columns: Sequence[str]) -> None:
        """Re-offer the column list (a new table may have different
        columns), keeping this row pointed at its current column if that
        column survived."""
        current = self.column()
        self._column_picker.blockSignals(True)
        self._column_picker.clear()
        self._column_picker.addItems(list(columns))
        if current in columns:
            self._column_picker.setCurrentText(current)
        self._column_picker.blockSignals(False)

    def set_source(self, df: Optional[pl.DataFrame], keep_range: bool = True) -> None:
        """Point this row at a table. `keep_range` holds the handles where
        they are (the table was re-computed -- a re-run detect, say -- and
        the cut the user chose should survive it); otherwise they widen to
        the new column's full span."""
        self._df = df
        column = self.column()
        if df is None or df.height == 0 or column not in df.columns:
            return
        previous = self._histogram.range()
        self._loading = True
        self._histogram.set_data(df[column].drop_nulls().to_numpy())
        self._histogram.set_range(*(previous if keep_range else self._histogram.data_range()))
        self._loading = False

    def range(self) -> tuple[float, float]:
        return self._histogram.range()

    def set_range(self, lo: float, hi: float) -> None:
        self._loading = True
        self._histogram.set_range(lo, hi)
        self._loading = False

    def reset_range(self) -> None:
        self.set_range(*self._histogram.data_range())
        self.changed.emit()

    def is_active(self) -> bool:
        """False when the handles still span the column's whole data range
        -- a row that cuts nothing, which `FeatureFilterPanel.filters`
        leaves out of the spec rather than record as a no-op cut."""
        data_min, data_max = self._histogram.data_range()
        lo, hi = self._histogram.range()
        return not (lo <= data_min and hi >= data_max)

    def _on_column_changed(self, _column: str) -> None:
        self.set_source(self._df, keep_range=False)
        self.changed.emit()

    def _on_range_changed(self, _lo: float, _hi: float) -> None:
        if not self._loading:
            self.changed.emit()


class FeatureFilterPanel(QWidget):
    """A stack of `_FilterRow`s over one table, plus an "N of M pass"
    readout. Emits `filtersChanged` on every user-driven change (a handle
    dragged, a spinbox typed into, a row added/removed/repointed), never
    on `set_source`/`set_filters`.

    `noun` names what a row of the source table is, for that readout --
    "points" when this is filtering detections, "tracks" when it is
    filtering per-track metrics (one row per track, e.g.
    `pipeline.track_metrics_df`). Passing the per-track table rather than
    the per-vertex one is what makes "412 of 1893 tracks pass" the true
    statement it looks like."""

    filtersChanged = Signal()

    def __init__(self, noun: str = "rows", hint: str = "") -> None:
        super().__init__()
        self._noun = noun
        self._df: Optional[pl.DataFrame] = None
        self._columns: list[str] = []
        self._rows: list[_FilterRow] = []

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)

        self._add_button = QPushButton("+ Add filter")
        self._add_button.setToolTip(
            "Add a range filter on another column. Filters AND together:\n"
            f"a {noun.rstrip('s')} must pass every one of them."
        )
        self._add_button.clicked.connect(self._on_add_clicked)
        self._clear_button = QPushButton("Clear all")
        self._clear_button.clicked.connect(self.clear_filters)

        self._summary = QLabel(hint)
        self._summary.setWordWrap(True)
        # Same reason as ExperimentListWidget.progress_label: an unwrapped
        # summary string would set this label's sizeHint as the panel's
        # minimum width and stop the dock shrinking below it.
        self._summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addWidget(self._add_button)
        button_row.addWidget(self._clear_button)
        button_row.addStretch()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(self._rows_layout)
        layout.addLayout(button_row)
        layout.addWidget(self._summary)
        self.setLayout(layout)
        self._update_enabled()

    def set_source(self, df: Optional[pl.DataFrame], columns: Optional[Sequence[str]] = None) -> None:
        """Point every row at a (re-computed) table, keeping the cuts that
        are already set. Silent -- call `filtersChanged`'s handler yourself
        if the new table changes what downstream should show."""
        self._df = df
        self._columns = list(columns) if columns is not None else numeric_filter_columns(df)
        for row in self._rows:
            row.set_columns(self._columns)
            row.set_source(df, keep_range=True)
        self._update_enabled()
        self._update_summary()

    def filters(self) -> FilterSpec:
        """The active cuts as `{column: (lo, hi)}`. A column filtered by
        more than one row is intersected, so two rows on `flux` narrow it
        rather than the second silently replacing the first."""
        spec: FilterSpec = {}
        for row in self._rows:
            if not row.is_active():
                continue
            lo, hi = row.range()
            if row.column() in spec:
                prev_lo, prev_hi = spec[row.column()]
                lo, hi = max(lo, prev_lo), min(hi, prev_hi)
            spec[row.column()] = (lo, hi)
        return spec

    def set_filters(self, spec: Optional[FilterSpec]) -> None:
        """Replace every row with the cuts in `spec` -- for restoring what
        a bundle's `manifest.json` recorded. Silent."""
        for row in list(self._rows):
            self._remove_row(row, notify=False)
        for column, (lo, hi) in (spec or {}).items():
            if column in self._columns:
                self._add_row(column, notify=False).set_range(lo, hi)
        self._update_summary()

    def clear_filters(self) -> None:
        for row in list(self._rows):
            self._remove_row(row, notify=False)
        self._emit_changed()

    def has_filters(self) -> bool:
        return bool(self.filters())

    def n_passing(self) -> Optional[int]:
        """How many rows of the source table pass every active cut, or
        None with no table loaded."""
        if self._df is None:
            return None
        spec = self.filters()
        if not spec:
            return self._df.height
        return int(filter_mask(self._df, spec).sum())

    def summary_text(self) -> str:
        total = self._df.height if self._df is not None else 0
        passing = self.n_passing()
        if passing is None:
            return f"no {self._noun} loaded yet"
        if not self.filters():
            return f"no filters — all {total} {self._noun} pass"
        return f"{passing} of {total} {self._noun} pass"

    def _next_column(self) -> Optional[str]:
        """The first column no row is already on -- so clicking "+" three
        times gives three different features rather than three copies of
        the first one."""
        taken = {row.column() for row in self._rows}
        for column in self._columns:
            if column not in taken:
                return column
        return self._columns[0] if self._columns else None

    def _add_row(self, column: Optional[str], notify: bool = True) -> _FilterRow:
        row = _FilterRow(self._columns, column)
        row.changed.connect(self._emit_changed)
        row.removeRequested.connect(self._remove_row)
        row.set_source(self._df, keep_range=False)
        self._rows.append(row)
        self._rows_layout.addWidget(row)
        if notify:
            self._emit_changed()
        return row

    def _remove_row(self, row: _FilterRow, notify: bool = True) -> None:
        if row not in self._rows:
            return
        self._rows.remove(row)
        self._rows_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        if notify:
            self._emit_changed()

    def _on_add_clicked(self) -> None:
        column = self._next_column()
        if column is not None:
            self._add_row(column)

    def _emit_changed(self) -> None:
        self._update_summary()
        self.filtersChanged.emit()

    def _update_summary(self) -> None:
        self._summary.setText(self.summary_text())
        self._update_enabled()

    def _update_enabled(self) -> None:
        self._add_button.setEnabled(bool(self._columns))
        self._clear_button.setEnabled(bool(self._rows))
