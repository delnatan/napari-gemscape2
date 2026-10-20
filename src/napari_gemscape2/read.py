"""Saved analyses back into tables, for plotting and statistics outside napari.

    from napari_gemscape2 import read_results

    res = read_results("~/data/2026-10-16/results")   # a results folder, a bundle, or a list of either
    res.populations     # populations.csv of every bundle: one row per (movie, group)
    res.tracks          # tracks_summary.csv of every bundle: one row per track
    res.distributions_D, res.distributions_alpha   # the populations' curves, by movie and group
    res.summaries       # {result_id: diffusion_summary.json}

Every table carries `result_id` (the bundle's folder name, the movie), so rows from several movies stack and
are told apart; tables are polars DataFrames (`.to_pandas()` for pandas). A bundle is any folder holding a
`diffusion_summary.json`; folders are searched recursively, so a tree of experiment folders, each with its own
`results/`, reads in one call. Nothing is refitted: these are the numbers the widget or `gemscape2 diffusion`
saved, over the tracks that passed the filters saved with them (`passes_filters` in `tracks`).

The movie, not the track, is the replicate when comparing conditions: tracks of one movie share its cell,
its focus and its calibration. Compare `populations` rows (one per movie and group) across conditions, and
read `tracks` for per-track plots.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Union

import polars as pl

from napari_gemscape2.results import (
    DIFFUSION_SUMMARY_FILENAME,
    DISTRIBUTIONS_ALPHA_FILENAME,
    DISTRIBUTIONS_D_FILENAME,
    POPULATIONS_FILENAME,
    TRACKS_SUMMARY_FILENAME,
)

PathLike = Union[str, Path]


@dataclass
class Results:
    """The saved analyses of some bundles, each table stacked across them (empty when none has it)."""

    bundles: list[Path] = field(default_factory=list)
    populations: pl.DataFrame = field(default_factory=pl.DataFrame)
    tracks: pl.DataFrame = field(default_factory=pl.DataFrame)
    distributions_D: pl.DataFrame = field(default_factory=pl.DataFrame)
    distributions_alpha: pl.DataFrame = field(default_factory=pl.DataFrame)
    summaries: dict[str, dict] = field(default_factory=dict)


def find_bundles(paths: Union[PathLike, Iterable[PathLike]]) -> list[Path]:
    """Every folder at or under `paths` holding a saved analysis, sorted, each once."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    found = set()
    for path in paths:
        path = Path(path).expanduser()
        if (path / DIFFUSION_SUMMARY_FILENAME).exists():
            found.add(path.resolve())
        found.update(p.parent.resolve() for p in path.rglob(DIFFUSION_SUMMARY_FILENAME))
    return sorted(found)


def _read_csv(path: Path, result_id: str) -> pl.DataFrame | None:
    if not path.exists():
        return None
    # Every row read before typing a column: one empty for the first rows (MSD, alpha) would be a string.
    table = pl.read_csv(path, infer_schema_length=None)
    if "result_id" in table.columns:
        return table.with_columns(pl.col("result_id").cast(pl.String))
    return table.select(pl.lit(result_id, dtype=pl.String).alias("result_id"), pl.all())


def _stack(tables: list[pl.DataFrame]) -> pl.DataFrame:
    return pl.concat(tables, how="diagonal_relaxed") if tables else pl.DataFrame()


def read_results(paths: Union[PathLike, Iterable[PathLike]]) -> Results:
    """The saved analyses under `paths` (`find_bundles`), stacked: see this module's docstring."""
    bundles = find_bundles(paths)
    if not bundles:
        raise FileNotFoundError(f"no saved analysis ({DIFFUSION_SUMMARY_FILENAME}) under {paths}")
    parts: dict[str, list[pl.DataFrame]] = {name: [] for name in
                                             ("populations", "tracks", "distributions_D", "distributions_alpha")}
    files = {"populations": POPULATIONS_FILENAME, "tracks": TRACKS_SUMMARY_FILENAME,
             "distributions_D": DISTRIBUTIONS_D_FILENAME, "distributions_alpha": DISTRIBUTIONS_ALPHA_FILENAME}
    summaries = {}
    for bundle in bundles:
        result_id = bundle.name
        summaries[result_id] = json.loads((bundle / DIFFUSION_SUMMARY_FILENAME).read_text())
        for name, filename in files.items():
            table = _read_csv(bundle / filename, result_id)
            if table is not None:
                parts[name].append(table)
    return Results(bundles=bundles, summaries=summaries, **{name: _stack(t) for name, t in parts.items()})
