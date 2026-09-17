"""The histogram filter panel every stage of this pipeline filters through
-- `qtkit.FilterPanel`, fed polars tables and told which of their columns
are worth filtering on.

Used three times, against three tables:

  - `params_panel._DetectTab` -- per-detection columns (`flux`,
    `fit_sigma`, `se_pos`, `bg`, ...) from a preview frame or a finished
    detect run. Decides what linking sees.
  - `params_panel._TrackingTab` -- per-track metrics
    (`pipeline.TRACK_METRIC_COLUMNS`). Decides which tracks the bundle
    keeps.
  - `diffusion_panel._TracksPane` -- per-track columns again, post-hoc: the
    per-detection QC fields aggregated to the track (`flux_min`,
    `se_x_max`, ...) plus whatever fit results (`D`, `alpha`) have been
    joined in. A cut like "every point in this track has acceptable flux"
    is a statement about the track's min, so it belongs beside the cut on
    the track's fitted alpha.

`filters()` returns a `pipeline.FilterSpec`, which `pipeline.filter_mask`
applies and `manifest.json` records -- a range dragged here, one read back
off a bundle, and one typed into a headless TOML are the same object.
"""

from __future__ import annotations

from typing import Optional, Sequence

import polars as pl
from qtkit import FilterPanel, columns_of

from spt_pipeline import units

# Columns no filter should offer: identity and coordinates. Filtering a
# detection on its own `y`/`x` is a crop (that's what the ROI is for), and
# on `frame` a frame range (the Detect tab's own control) -- both already
# have a better control elsewhere, and would silently fight with it.
# `y_px`/`x_px` are per-track centroids, where a range on one axis alone is
# a half-crop rather than a region.
SKIP_COLUMNS = frozenset(
    {"loc_id", "track_id", "frame", "t", "y", "x", "y_um", "x_um", "y_px", "x_px", "seconds"}
)


class FeatureFilterPanel(FilterPanel):
    """`qtkit.FilterPanel` over a polars DataFrame.

    `noun` names what one row of the table is, for the "N of M pass"
    readout: pass the per-track table (one row per track) when filtering
    tracks, so "412 of 1893 tracks pass" means what it says.

    Adds one thing to the base panel: each row's histogram carries a
    tooltip saying what its column is measured in
    (`spt_pipeline.units.tooltip`). A row's column picker shows the bare
    column name -- that is the base widget's, and the name is the one the
    cut is recorded under in `manifest.json`, so it should stay literal --
    but the tracks table offers `se_x_max` (px) and `se_x_um_max` (µm)
    side by side, and dragging a handle to "0.03" means two different
    things depending on which one is in front of you."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Rows are created by the base class (the "+ Add filter" button,
        # `set_filters`), so the tooltips are refreshed from the one
        # signal every one of those paths ends in.
        self.filtersChanged.connect(self._refresh_unit_tooltips)

    def _refresh_unit_tooltips(self) -> None:
        for row in self.rows():
            row.histogram.setToolTip(units.tooltip(row.column()))

    def set_source(self, df: Optional[pl.DataFrame], columns: Optional[Sequence[str]] = None) -> None:
        """Point every row at a (re-computed) table, keeping the cuts already
        set. `columns` limits what may be filtered on; by default every
        numeric, non-constant column not in `SKIP_COLUMNS`. Silent."""
        if df is None or df.height == 0:
            super().set_source(None, columns)
            self._refresh_unit_tooltips()
            return
        if columns is None:
            columns = [
                name
                for name, dtype in zip(df.columns, df.dtypes)
                if name not in SKIP_COLUMNS and dtype.is_numeric()
            ]
            # Only numeric columns are converted; the base class drops the
            # constant ones.
            super().set_source(columns_of(df, columns))
            self._refresh_unit_tooltips()
            return
        columns = [c for c in columns if c in df.columns]
        super().set_source(columns_of(df, columns), columns)
        self._refresh_unit_tooltips()

    def set_filters(self, spec) -> None:
        """`FilterPanel.set_filters` -- silent, and so does not go through
        `filtersChanged`, so the new rows' tooltips are set here."""
        super().set_filters(spec)
        self._refresh_unit_tooltips()
