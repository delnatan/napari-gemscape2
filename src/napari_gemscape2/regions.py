"""Regions: a painted integer labels image plus a table naming each label.

`labels` is an `(H, W)` uint16 image -- napari's Labels layer data. 0 is
background: nothing is localized there. Every other value is one region
*instance*, and each pixel belongs to exactly one of them, so cutting a
nucleus out of its cell is just painting the nucleus over the cell: the
pixels change owner, and the cell keeps the rest. No drawing order or
overlap rule is involved.

The image carries no meaning by itself, so `Regions` records a *class*
name per label value ("cytoplasm", "nucleus", anything). Names may
repeat: two cells painted as labels 1 and 2, both "cytoplasm", stay two
instances -- tracking links each label on its own -- and pool as one
class. Detections are stamped with both (`label_points`: `region`,
`region_class`), so pooling by class is a `group_by("region_class")`.

A bundle stores these as `labels.tif` and `regions.json` (see
`napari_gemscape2.results`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

LABELS_DTYPE = np.uint16


@dataclass
class Region:
    class_: str


def default_class(label: int) -> str:
    """The name a label gets until it is renamed."""
    return f"region {label}"


@dataclass
class Regions:
    """What each label value in a labels image means: `table` maps label
    value -> `Region`."""

    table: dict[int, Region] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "regions": [{"label": label, "class": region.class_} for label, region in sorted(self.table.items())],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Regions":
        # Older bundles also carry a "classes" list and a "cell" per region;
        # neither means anything now.
        return cls(table={int(row["label"]): Region(str(row["class"])) for row in data.get("regions", [])})

    def next_label(self, labels: np.ndarray | None = None) -> int:
        """The smallest label value above everything in the table and in
        `labels` -- what a new region is painted with."""
        top = max(self.table, default=0)
        if labels is not None and labels.size:
            top = max(top, int(labels.max()))
        return top + 1

    def class_names(self) -> list[str]:
        """Distinct class names, in label order."""
        return list(dict.fromkeys(r.class_ for _, r in sorted(self.table.items())))


def present_labels(labels: np.ndarray) -> list[int]:
    """Nonzero label values painted somewhere in `labels`, ascending."""
    counts = np.bincount(labels.ravel())
    return [int(v) for v in np.flatnonzero(counts) if v != 0]


def sync_table(labels: np.ndarray, regions: Regions) -> Regions:
    """`regions` with its table matched to what `labels` actually holds:
    a label painted but never named gets `default_class`; an entry whose
    pixels were all painted over or erased is dropped. Modifies and
    returns `regions`."""
    present = present_labels(labels)
    for label in [k for k in regions.table if k not in set(present)]:
        del regions.table[label]
    for label in present:
        if label not in regions.table:
            regions.table[label] = Region(default_class(label))
    return regions


def region_areas_px(labels: np.ndarray) -> dict[int, int]:
    """`{label: pixel count}` for every nonzero label present."""
    counts = np.bincount(labels.ravel())
    return {int(v): int(counts[v]) for v in np.flatnonzero(counts) if v != 0}


def class_areas_px(labels: np.ndarray, regions: Regions) -> dict[str, int]:
    """Pixel area per class, summed over its regions."""
    areas: dict[str, int] = {}
    for label, n in region_areas_px(labels).items():
        region = regions.table.get(label)
        if region is not None:
            areas[region.class_] = areas.get(region.class_, 0) + n
    return areas


REGION_COLUMNS = ("region", "region_class")


def label_points(points_df: pl.DataFrame, labels: np.ndarray, regions: Regions) -> pl.DataFrame:
    """`points_df` plus `region` (Int32 label value, 0 outside every
    region) and `region_class` (Utf8, null outside or unassigned), read off `labels` at each detection's rounded
    `(y, x)`. Existing region columns are replaced, so re-labeling a
    loaded table is safe."""
    points_df = points_df.drop([c for c in REGION_COLUMNS if c in points_df.columns])
    if points_df.height == 0:
        return points_df.with_columns(
            pl.lit(None, dtype=pl.Int32).alias("region"),
            pl.lit(None, dtype=pl.Utf8).alias("region_class"),
        )
    h, w = labels.shape
    yi = np.clip(np.rint(points_df["y"].to_numpy()).astype(np.int64), 0, h - 1)
    xi = np.clip(np.rint(points_df["x"].to_numpy()).astype(np.int64), 0, w - 1)
    region = labels[yi, xi].astype(np.int32)
    lookup = pl.DataFrame(
        {
            "region": [int(k) for k in regions.table],
            "region_class": [r.class_ for r in regions.table.values()],
        },
        schema={"region": pl.Int32, "region_class": pl.Utf8},
    )
    return points_df.with_columns(pl.Series("region", region, dtype=pl.Int32)).join(
        lookup, on="region", how="left", maintain_order="left"
    )
