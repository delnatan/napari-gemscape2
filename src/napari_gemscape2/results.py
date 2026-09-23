"""Reproducible per-image results bundle: points + tracks + provenance manifest.

A results bundle is a directory:
    <result_dir>/points.parquet
    <result_dir>/tracks.parquet  (optional -- only once tracking has been saved)
    <result_dir>/manifest.json
    <result_dir>/labels.tif     (optional -- only if regions were used)
    <result_dir>/regions.json   (alongside labels.tif)

Detection and tracking are two files, saved by two different actions
(`write_result` for both, `write_detection_result` for points alone) -- a
bundle can hold detections with no tracks yet, but never the reverse, since
tracks are always linked from some points table. `load_result` reflects
that: it returns an empty `tracks_df` rather than raising when
`tracks.parquet` isn't there.

The source image is referenced by path in the manifest, not copied.

`labels.tif` is the painted regions image (uint16, 0 = background) and
`regions.json` names each of its labels (`regions.Regions.to_json`) -- see
`napari_gemscape2.regions`. Reloading a bundle (`viewer.show_result`) adds the
image back as a Labels layer, so the same regions can be reused or edited.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import tifffile

from napari_gemscape2.regions import LABELS_DTYPE, Regions

POINTS_FILENAME = "points.parquet"
TRACKS_FILENAME = "tracks.parquet"
MANIFEST_FILENAME = "manifest.json"
LABELS_FILENAME = "labels.tif"
REGIONS_FILENAME = "regions.json"
DIFFUSION_FITS_FILENAME = "diffusion_fits.parquet"
DIFFUSION_SUMMARY_FILENAME = "diffusion_summary.json"
# One row per track: identity, size, position, shape, mean detection QC
# and the classical D -- the table to read an experiment's tracks from and
# to pool across experiments (`diffusion.tracks_summary_table`).
TRACKS_SUMMARY_FILENAME = "tracks_summary.parquet"


def git_sha(repo_path: str | Path) -> str | None:
    """`git rev-parse HEAD` in `repo_path` (any directory inside the
    repo), or None if unavailable (not a git repo, git not installed,
    etc.) -- provenance is best-effort."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def repo_shas(*modules) -> dict[str, str | None]:
    """`{package: git SHA}` for each module's source checkout, asked from
    the package's own directory -- so it holds for a `src/` layout
    (spotsolve, this package) and a flat one (diffusionkit) alike. None
    for a package installed into site-packages, which has no checkout of
    its own (and whose enclosing repo, if the venv sits in one, would be
    the wrong answer)."""
    shas = {}
    for module in modules:
        package_dir = Path(module.__file__).resolve().parent
        shas[module.__name__] = (
            None if "site-packages" in package_dir.parts else git_sha(package_dir)
        )
    return shas


def build_manifest(
    *,
    result_id: str,
    source_image_path: str | Path,
    params: dict,
    repo_shas: dict[str, str | None] | None = None,
) -> dict:
    return {
        "result_id": result_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_image_path": str(Path(source_image_path).resolve()),
        "params": params,
        "repo_shas": repo_shas or {},
    }


def write_result(
    result_dir: str | Path,
    points_df: pl.DataFrame,
    tracks_df: pl.DataFrame,
    manifest: dict,
    labels: np.ndarray | None = None,
    regions: Regions | None = None,
) -> None:
    """`labels`/`regions`, if given, are written to `labels.tif` and
    `regions.json` (see `_write_regions`)."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    points_df.write_parquet(result_dir / POINTS_FILENAME)
    tracks_df.write_parquet(result_dir / TRACKS_FILENAME)
    (result_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
    _write_regions(result_dir, labels, regions)


def write_detection_result(
    result_dir: str | Path,
    points_df: pl.DataFrame,
    manifest: dict,
    labels: np.ndarray | None = None,
    regions: Regions | None = None,
) -> None:
    """The Detect stage's own save: `points.parquet` and `manifest.json`
    (plus the regions) alone, usable before tracking has run at all.

    Removes a stale `tracks.parquet`: whatever tracks it held were linked
    from a `points.parquet` that this call just replaced, so keeping it
    would leave a tracks table on disk that no longer matches the points
    beside it. Re-run tracking and `write_result` (the "Save results"
    button) to get a bundle with tracks in it again."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    points_df.write_parquet(result_dir / POINTS_FILENAME)
    (result_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
    _write_regions(result_dir, labels, regions)
    (result_dir / TRACKS_FILENAME).unlink(missing_ok=True)


def _write_regions(result_dir: Path, labels: np.ndarray | None, regions: Regions | None) -> None:
    """`labels.tif` + `regions.json`, or -- with no labels -- remove any
    stale pair from a previous run of this bundle, so a re-run without
    regions doesn't leave behind ones that no longer apply."""
    labels_path = result_dir / LABELS_FILENAME
    regions_path = result_dir / REGIONS_FILENAME
    if labels is not None and regions is not None:
        tifffile.imwrite(labels_path, np.asarray(labels, dtype=LABELS_DTYPE), compression="zlib")
        regions_path.write_text(json.dumps(regions.to_json(), indent=2))
    else:
        labels_path.unlink(missing_ok=True)
        regions_path.unlink(missing_ok=True)


def load_result(
    result_dir: str | Path,
) -> tuple[pl.DataFrame, pl.DataFrame, dict, np.ndarray | None, Regions | None]:
    """Returns `(points_df, tracks_df, manifest, labels, regions)` --
    `labels`/`regions` are None for a bundle that never used regions;
    `tracks_df` is an empty `DataFrame` (rather than raising) for a
    detections-only bundle written by `write_detection_result`."""
    result_dir = Path(result_dir)
    points_df = pl.read_parquet(result_dir / POINTS_FILENAME)
    tracks_path = result_dir / TRACKS_FILENAME
    tracks_df = pl.read_parquet(tracks_path) if tracks_path.exists() else pl.DataFrame()
    manifest = json.loads((result_dir / MANIFEST_FILENAME).read_text())
    labels = regions = None
    labels_path = result_dir / LABELS_FILENAME
    regions_path = result_dir / REGIONS_FILENAME
    if labels_path.exists() and regions_path.exists():
        labels = tifffile.imread(labels_path).astype(LABELS_DTYPE, copy=False)
        regions = Regions.from_json(json.loads(regions_path.read_text()))
    return points_df, tracks_df, manifest, labels, regions


def write_diffusion_results(
    result_dir: str | Path,
    per_track_df: pl.DataFrame,
    summary: dict,
    tracks_summary_df: pl.DataFrame | None = None,
) -> None:
    """Diffusion-widget results, written like the regions: an optional
    extra on top of the core points/tracks/manifest bundle, not every
    bundle has one. `per_track_df` holds one row per (track_id, method) --
    the classical run's fits and any Bayesian per-track fits the user has
    run, distinguished by a `method` column (see
    `widgets/diffusion_panel.py`) -- so all of them live in one file.
    `summary` holds the run settings and population-level scalars.

    `tracks_summary_df`, when given, is the flat one-row-per-track table
    (`TRACKS_SUMMARY_FILENAME`): the one to read, since `per_track_df`
    mixes every method's rows and columns."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    per_track_df.write_parquet(result_dir / DIFFUSION_FITS_FILENAME)
    (result_dir / DIFFUSION_SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))
    if tracks_summary_df is not None:
        tracks_summary_df.write_parquet(result_dir / TRACKS_SUMMARY_FILENAME)


def load_diffusion_results(
    result_dir: str | Path,
) -> tuple[pl.DataFrame | None, dict | None]:
    """`(per_track_df, summary)`, or `(None, None)` if this bundle has no
    saved diffusion results yet."""
    result_dir = Path(result_dir)
    fits_path = result_dir / DIFFUSION_FITS_FILENAME
    summary_path = result_dir / DIFFUSION_SUMMARY_FILENAME
    if not fits_path.exists():
        return None, None
    per_track_df = pl.read_parquet(fits_path)
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    return per_track_df, summary


def load_manifest(result_dir: str | Path) -> dict:
    """Just the manifest -- what a folder scan needs (`params.n_tracks`)
    without reading either table."""
    return json.loads((Path(result_dir) / MANIFEST_FILENAME).read_text())


def has_result(result_dir: str | Path) -> bool:
    return (Path(result_dir) / MANIFEST_FILENAME).exists()


def result_dir_for(results_root: str | Path, image_path: str | Path) -> Path:
    """Default bundle location for a raw image: `<results_root>/<stem>`."""
    return Path(results_root) / Path(image_path).stem
