"""Regions: a painted integer labels image plus a table naming each label.

`labels` is an `(H, W)` uint16 image -- napari's Labels layer data. 0 is
background: nothing is localized there. Every other value is one region
*instance*, and each pixel belongs to exactly one of them, so cutting a
nucleus out of its cell is just painting the nucleus over the cell: the
pixels change owner, and the cell's label is left as the cytoplasm. No
drawing order or overlap rule is involved.

The image carries no meaning by itself, so `Regions` records, per label
value, its *class* (from a user-editable list, "cytoplasm"/"nucleus" by
default) and the *cell* it belongs to -- a cell's cytoplasm and its
nucleus are two labels sharing one cell id. Detections are stamped with
all three (`label_points`: `region`, `region_class`, `cell`), so pooling
by class is a `group_by("region_class")` and per-cell work stays possible.

A bundle stores these as `labels.tif` and `regions.json` (see
`spt_pipeline.results`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

LABELS_DTYPE = np.uint16
DEFAULT_CLASSES = ("cytoplasm", "nucleus")


@dataclass
class Region:
    class_: str
    cell: int


@dataclass
class Regions:
    """What each label value in a labels image means. `classes` is the
    list offered for assignment (its order is the display order); `table`
    maps label value -> `Region`."""

    classes: list[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))
    table: dict[int, Region] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "classes": list(self.classes),
            "regions": [
                {"label": label, "class": region.class_, "cell": region.cell}
                for label, region in sorted(self.table.items())
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Regions":
        table = {
            int(row["label"]): Region(str(row["class"]), int(row["cell"]))
            for row in data.get("regions", [])
        }
        classes = list(data.get("classes") or DEFAULT_CLASSES)
        # A class named in the table but missing from the list (hand-edited
        # JSON) is still a class.
        classes += [c for c in dict.fromkeys(r.class_ for r in table.values()) if c not in classes]
        return cls(classes=classes, table=table)

    def next_label(self, labels: np.ndarray | None = None) -> int:
        """The smallest label value above everything in the table and in
        `labels` -- what a new region is painted with."""
        top = max(self.table, default=0)
        if labels is not None and labels.size:
            top = max(top, int(labels.max()))
        return top + 1

    def next_cell(self) -> int:
        return max((r.cell for r in self.table.values()), default=0) + 1

    def class_names(self) -> list[str]:
        """Classes that some region actually uses, in `classes` order."""
        used = {r.class_ for r in self.table.values()}
        return [c for c in self.classes if c in used]


def present_labels(labels: np.ndarray) -> list[int]:
    """Nonzero label values painted somewhere in `labels`, ascending."""
    counts = np.bincount(labels.ravel())
    return [int(v) for v in np.flatnonzero(counts) if v != 0]


def sync_table(labels: np.ndarray, regions: Regions) -> Regions:
    """`regions` with its table matched to what `labels` actually holds:
    a label painted but never assigned gets the first class and a cell id
    of its own (a new cell); an entry whose pixels were all painted over
    or erased is dropped. Modifies and returns `regions`."""
    present = present_labels(labels)
    for label in [k for k in regions.table if k not in set(present)]:
        del regions.table[label]
    for label in present:
        if label not in regions.table:
            regions.table[label] = Region(regions.classes[0], regions.next_cell())
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


REGION_COLUMNS = ("region", "region_class", "cell")


def label_points(points_df: pl.DataFrame, labels: np.ndarray, regions: Regions) -> pl.DataFrame:
    """`points_df` plus `region` (Int32 label value, 0 outside every
    region), `region_class` (Utf8, null outside or unassigned) and `cell`
    (Int32, null likewise), read off `labels` at each detection's rounded
    `(y, x)`. Existing region columns are replaced, so re-labeling a
    loaded table is safe."""
    points_df = points_df.drop([c for c in REGION_COLUMNS if c in points_df.columns])
    if points_df.height == 0:
        return points_df.with_columns(
            pl.lit(None, dtype=pl.Int32).alias("region"),
            pl.lit(None, dtype=pl.Utf8).alias("region_class"),
            pl.lit(None, dtype=pl.Int32).alias("cell"),
        )
    h, w = labels.shape
    yi = np.clip(np.rint(points_df["y"].to_numpy()).astype(np.int64), 0, h - 1)
    xi = np.clip(np.rint(points_df["x"].to_numpy()).astype(np.int64), 0, w - 1)
    region = labels[yi, xi].astype(np.int32)
    lookup = pl.DataFrame(
        {
            "region": [int(k) for k in regions.table],
            "region_class": [r.class_ for r in regions.table.values()],
            "cell": [int(r.cell) for r in regions.table.values()],
        },
        schema={"region": pl.Int32, "region_class": pl.Utf8, "cell": pl.Int32},
    )
    return points_df.with_columns(pl.Series("region", region, dtype=pl.Int32)).join(
        lookup, on="region", how="left", maintain_order="left"
    )
