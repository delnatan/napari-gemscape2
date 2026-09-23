"""Reproducible per-image results bundle: points + tracks + provenance manifest.

A results bundle is a directory:
    <result_dir>/points.parquet
    <result_dir>/tracks.parquet  (optional -- only once tracking has been saved)
    <result_dir>/manifest.json
    <result_dir>/labels.tif     (optional -- only if regions were used)
    <result_dir>/regions.json   (alongside labels.tif)

and, once the diffusion widget has saved an analysis of it
(`write_diffusion_results`):
    <result_dir>/tracks_summary.csv       one row per track
    <result_dir>/posterior_D.parquet      every track's log posterior over D
    <result_dir>/posterior_alpha.parquet  likewise over alpha (exposure 0 only)
    <result_dir>/distributions_D.csv      the ensemble on the D grid
    <result_dir>/distributions_alpha.csv  likewise on the alpha grid
    <result_dir>/diffusion_summary.json   settings + population numbers

Points and tracks are the atomic data, and stay parquet: everything else
is derived from them. The per-track summary and the distributions are the
tables people open in a spreadsheet or Prism, so they are CSV. The
posteriors share one grid per quantity and run to hundreds of rows per
track, so they are long-format parquet (`track_id`, grid value,
`log_posterior`).

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
from typing import Optional

import numpy as np
import polars as pl
import tifffile

from napari_gemscape2.regions import LABELS_DTYPE, Regions

POINTS_FILENAME = "points.parquet"
TRACKS_FILENAME = "tracks.parquet"
MANIFEST_FILENAME = "manifest.json"
LABELS_FILENAME = "labels.tif"
REGIONS_FILENAME = "regions.json"
DIFFUSION_SUMMARY_FILENAME = "diffusion_summary.json"
# One row per track: identity, size, position, shape, mean detection QC
# and the posterior D (median, low, high) -- the table to read an
# experiment's tracks from and to pool across experiments
# (`diffusion.tracks_summary_table`).
TRACKS_SUMMARY_FILENAME = "tracks_summary.csv"
POSTERIOR_D_FILENAME = "posterior_D.parquet"
POSTERIOR_ALPHA_FILENAME = "posterior_alpha.parquet"
DISTRIBUTIONS_D_FILENAME = "distributions_D.csv"
DISTRIBUTIONS_ALPHA_FILENAME = "distributions_alpha.csv"
# Written by earlier versions of the diffusion widget; removed on the next
# save so a bundle never mixes two analyses.
_STALE_DIFFUSION_FILES = ("diffusion_fits.parquet", "tracks_summary.parquet")


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
    *,
    tracks_summary: pl.DataFrame,
    summary: dict,
    posterior_D: Optional[pl.DataFrame] = None,
    posterior_alpha: Optional[pl.DataFrame] = None,
    distributions_D: Optional[pl.DataFrame] = None,
    distributions_alpha: Optional[pl.DataFrame] = None,
) -> None:
    """The diffusion widget's analysis of this bundle's tracks (see this
    module's docstring for the files). An optional table left as None has
    its file removed rather than kept, so what is on disk is always one
    analysis -- an alpha posterior from an earlier run at exposure 0 never
    sits beside a D posterior from a later one at 20 ms."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    tracks_summary.write_csv(result_dir / TRACKS_SUMMARY_FILENAME)
    (result_dir / DIFFUSION_SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))
    for table, filename in (
        (posterior_D, POSTERIOR_D_FILENAME),
        (posterior_alpha, POSTERIOR_ALPHA_FILENAME),
    ):
        if table is None:
            (result_dir / filename).unlink(missing_ok=True)
        else:
            table.write_parquet(result_dir / filename)
    for table, filename in (
        (distributions_D, DISTRIBUTIONS_D_FILENAME),
        (distributions_alpha, DISTRIBUTIONS_ALPHA_FILENAME),
    ):
        if table is None:
            (result_dir / filename).unlink(missing_ok=True)
        else:
            table.write_csv(result_dir / filename)
    for filename in _STALE_DIFFUSION_FILES:
        (result_dir / filename).unlink(missing_ok=True)


def load_diffusion_summary(result_dir: str | Path) -> Optional[dict]:
    """The saved `diffusion_summary.json`, or None if this bundle has no
    saved analysis yet."""
    path = Path(result_dir) / DIFFUSION_SUMMARY_FILENAME
    return json.loads(path.read_text()) if path.exists() else None


def load_diffusion_results(result_dir: str | Path) -> Optional[dict]:
    """What `write_diffusion_results` wrote, under its keyword names
    (`summary`, `tracks_summary`, `posterior_D`, `posterior_alpha`; an
    absent optional table is None) -- or None when there is no saved
    posterior analysis to read back."""
    result_dir = Path(result_dir)
    summary = load_diffusion_summary(result_dir)
    posterior_d_path = result_dir / POSTERIOR_D_FILENAME
    summary_path = result_dir / TRACKS_SUMMARY_FILENAME
    if summary is None or not posterior_d_path.exists() or not summary_path.exists():
        return None
    alpha_path = result_dir / POSTERIOR_ALPHA_FILENAME
    return {
        "summary": summary,
        # Every row read before typing a column: one that is empty for the
        # first thousand tracks (alpha, NUTS) would otherwise be a string.
        "tracks_summary": pl.read_csv(summary_path, infer_schema_length=None),
        "posterior_D": pl.read_parquet(posterior_d_path),
        "posterior_alpha": pl.read_parquet(alpha_path) if alpha_path.exists() else None,
    }


def load_manifest(result_dir: str | Path) -> dict:
    """Just the manifest -- what a folder scan needs (`params.n_tracks`)
    without reading either table."""
    return json.loads((Path(result_dir) / MANIFEST_FILENAME).read_text())


def has_result(result_dir: str | Path) -> bool:
    return (Path(result_dir) / MANIFEST_FILENAME).exists()


def result_dir_for(results_root: str | Path, image_path: str | Path) -> Path:
    """Default bundle location for a raw image: `<results_root>/<stem>`."""
    return Path(results_root) / Path(image_path).stem
