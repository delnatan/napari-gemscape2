"""Reproducible per-image results bundle: points + tracks + provenance manifest.

A results bundle is a directory:
    <result_dir>/points.parquet
    <result_dir>/tracks.parquet  (optional -- only once tracking has been saved)
    <result_dir>/manifest.json
    <result_dir>/labels.tif     (optional -- only if regions were used)
    <result_dir>/regions.json   (alongside labels.tif)

and, once a mask has been painted for the movie (`save_mask`):
    <result_dir>/mask.json      the movie's mask: its regions table
    <result_dir>/mask.tif       (alongside mask.json, unless it is empty)

`labels.tif`/`regions.json` are the mask the results were made with,
written with them; `mask.tif`/`mask.json` are the movie's mask as it is
painted now, saved on every edit whether or not there are results yet.
Any run of the movie -- the GUI's, a batch's, the CLI's -- is restricted
to the latter (`load_mask`), and a movie without one is analyzed over the
whole field. When the two differ the results are out of date
(`mask_is_stale`): the mask was edited since they were made. An empty
`mask.json` (no `mask.tif`) records a mask erased on purpose, so that
too can make results out of date. A result dir with no `mask.json` --
from before masks were saved on their own -- has its results' mask as
the movie's mask.

and, once the diffusion widget has saved an analysis of it
(`write_diffusion_results`):
    <result_dir>/tracks_summary.csv       one row per track
    <result_dir>/loglik_D.parquet         every track's log-likelihood over D
    <result_dir>/distributions_D.csv      the ensemble on the D grid
    <result_dir>/distributions_D_by_length.csv  the same split by track length
    <result_dir>/diffusion_summary.json   settings + population numbers

Across bundles, `gemscape2 pool` writes one more directory (`write_pooled_results`), by default
`<results_root>/pooled/`: the D population of each sample (and of each bundle where a sample has replicates),
the draws a comparison outside the GUI starts from, and optionally the ensemble-averaged MSD, all read from the
saved analyses above.

Points and tracks are the atomic data, and stay parquet: everything else
is derived from them. The per-track summary and the distributions are the
tables people open in a spreadsheet or Prism, so they are CSV. The
likelihoods share one grid and run to hundreds of rows per track, so they
are long-format parquet (`track_id`, `D_um2_s`, `loglik`).

Detection and tracking are two files, saved by two different actions
(`write_result` for both, `write_detection_result` for points alone) -- a
bundle can hold detections with no tracks yet, but never the reverse, since
tracks are always linked from some points table. `load_result` reflects
that: it returns an empty `tracks_df` rather than raising when
`tracks.parquet` isn't there.

The source image is referenced by path in the manifest, not copied.

`labels.tif`/`mask.tif` are painted regions images (uint16, 0 =
background) and `regions.json`/`mask.json` name each of their labels
(`regions.Regions.to_json`) -- see `napari_gemscape2.regions`. Reloading a
bundle (`viewer.show_result`) adds the movie's mask back as a Labels layer,
so it can be reused or edited.
"""

from __future__ import annotations

import importlib.metadata
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
MASK_LABELS_FILENAME = "mask.tif"
MASK_REGIONS_FILENAME = "mask.json"
DIFFUSION_SUMMARY_FILENAME = "diffusion_summary.json"
# One row per track: identity, size, position, shape, mean detection QC
# and the posterior D (median, low, high) -- the table to read an
# experiment's tracks from and to pool across experiments
# (`diffusion.tracks_summary_table`).
TRACKS_SUMMARY_FILENAME = "tracks_summary.csv"
LOGLIK_D_FILENAME = "loglik_D.parquet"
DISTRIBUTIONS_D_FILENAME = "distributions_D.csv"
DISTRIBUTIONS_D_BY_LENGTH_FILENAME = "distributions_D_by_length.csv"
# Written by earlier versions of the diffusion widget; removed on the next
# save so a bundle never mixes two analyses.
_STALE_DIFFUSION_FILES = (
    "diffusion_fits.parquet", "tracks_summary.parquet", "posterior_alpha.parquet", "distributions_alpha.csv",
    "posterior_D.parquet",
)


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


def _installed_commit(dist: importlib.metadata.Distribution) -> str | None:
    """The commit an installer recorded for a package installed from git
    (PEP 610's `direct_url.json`), or None -- a release wheel or a PyPI
    install has none, and its version alone says what it is."""
    try:
        direct_url = json.loads(dist.read_text("direct_url.json") or "null")
    except json.JSONDecodeError:
        return None
    return ((direct_url or {}).get("vcs_info") or {}).get("commit_id")


def package_provenance(*modules) -> dict[str, dict[str, str | None]]:
    """`{package: {"version", "git_sha"}}` for each module.

    `git_sha` is the module's source checkout's HEAD when it runs from one
    (an editable install -- asked from the package's own directory, so it
    holds for a `src/` layout and a flat one alike), else the commit
    recorded at install time for a package installed from git (diffusionkit
    and qtkit in the standard install, everything in `uv tool install`).
    A site-packages install is never asked of git directly: the repo
    enclosing it, if the venv sits in one, would be the wrong answer. None
    for a release wheel (spotsolve's), where `version` identifies it."""
    packages = {}
    for module in modules:
        package_dir = Path(module.__file__).resolve().parent
        try:
            dist = importlib.metadata.distribution(module.__name__)
        except importlib.metadata.PackageNotFoundError:
            dist = None
        if "site-packages" not in package_dir.parts:
            sha = git_sha(package_dir)
        else:
            sha = _installed_commit(dist) if dist is not None else None
        packages[module.__name__] = {
            "version": dist.version if dist is not None else None,
            "git_sha": sha,
        }
    return packages


def build_manifest(
    *,
    result_id: str,
    source_image_path: str | Path,
    params: dict,
    packages: dict[str, dict[str, str | None]] | None = None,
) -> dict:
    return {
        "result_id": result_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_image_path": str(Path(source_image_path).resolve()),
        "params": params,
        "packages": packages or {},
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


def has_regions(result_dir: str | Path) -> bool:
    """Whether the results here were made with a mask (`labels.tif` +
    `regions.json`)."""
    return (Path(result_dir) / LABELS_FILENAME).exists() and (Path(result_dir) / REGIONS_FILENAME).exists()


def load_regions(result_dir: str | Path) -> tuple[np.ndarray | None, Regions | None]:
    """`(labels, regions)` the results in `result_dir` were made with, or
    `(None, None)`."""
    result_dir = Path(result_dir)
    if not has_regions(result_dir):
        return None, None
    labels = tifffile.imread(result_dir / LABELS_FILENAME).astype(LABELS_DTYPE, copy=False)
    regions = Regions.from_json(json.loads((result_dir / REGIONS_FILENAME).read_text()))
    return labels, regions


def save_mask(result_dir: str | Path, labels: np.ndarray | None, regions: Regions | None) -> None:
    """The movie's mask as painted now: `mask.tif` + `mask.json`, or --
    with no labels or no regions -- an empty `mask.json` alone, recording
    that it has none. Results already here are left as they are."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    tif_path = result_dir / MASK_LABELS_FILENAME
    if labels is None or regions is None or not regions.table:
        tif_path.unlink(missing_ok=True)
        regions = Regions()
    else:
        tifffile.imwrite(tif_path, np.asarray(labels, dtype=LABELS_DTYPE), compression="zlib")
    (result_dir / MASK_REGIONS_FILENAME).write_text(json.dumps(regions.to_json(), indent=2))


def load_mask(result_dir: str | Path) -> tuple[np.ndarray | None, Regions | None]:
    """The movie's mask `(labels, regions)` -- what a run of it is
    restricted to -- or `(None, None)`: `save_mask`'s files, else the
    results' mask (`load_regions`) when no mask was ever saved on its own."""
    result_dir = Path(result_dir)
    json_path = result_dir / MASK_REGIONS_FILENAME
    if not json_path.exists():
        return load_regions(result_dir)
    regions = Regions.from_json(json.loads(json_path.read_text()))
    tif_path = result_dir / MASK_LABELS_FILENAME
    if not regions.table or not tif_path.exists():
        return None, None
    return tifffile.imread(tif_path).astype(LABELS_DTYPE, copy=False), regions


def has_mask(result_dir: str | Path) -> bool:
    """Whether the movie has a mask (`load_mask` isn't `(None, None)`),
    without reading the image."""
    result_dir = Path(result_dir)
    if not (result_dir / MASK_REGIONS_FILENAME).exists():
        return has_regions(result_dir)
    return (result_dir / MASK_LABELS_FILENAME).exists()


def mask_is_stale(result_dir: str | Path) -> bool:
    """Whether results are saved here but the movie's mask has changed
    since they were made: painted, repainted, renamed or erased."""
    result_dir = Path(result_dir)
    if not has_result(result_dir) or not (result_dir / MASK_REGIONS_FILENAME).exists():
        return False
    return not _same_mask(load_mask(result_dir), load_regions(result_dir))


def _same_mask(a: tuple, b: tuple) -> bool:
    (labels_a, regions_a), (labels_b, regions_b) = a, b
    if labels_a is None or labels_b is None:
        return labels_a is None and labels_b is None
    return np.array_equal(labels_a, labels_b) and regions_a.to_json() == regions_b.to_json()


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
    labels, regions = load_regions(result_dir)
    return points_df, tracks_df, manifest, labels, regions


def write_diffusion_results(
    result_dir: str | Path,
    *,
    tracks_summary: pl.DataFrame,
    summary: dict,
    loglik_D: Optional[pl.DataFrame] = None,
    distributions_D: Optional[pl.DataFrame] = None,
    distributions_D_by_length: Optional[pl.DataFrame] = None,
) -> None:
    """The diffusion widget's analysis of this bundle's tracks (see this
    module's docstring for the files). An optional table left as None has
    its file removed rather than kept, so what is on disk is always one
    analysis -- the distributions of an earlier run never sit beside the
    per-track table of a later one that wrote none."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    tracks_summary.write_csv(result_dir / TRACKS_SUMMARY_FILENAME)
    (result_dir / DIFFUSION_SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))
    if loglik_D is None:
        (result_dir / LOGLIK_D_FILENAME).unlink(missing_ok=True)
    else:
        loglik_D.write_parquet(result_dir / LOGLIK_D_FILENAME)
    for table, filename in (
        (distributions_D, DISTRIBUTIONS_D_FILENAME),
        (distributions_D_by_length, DISTRIBUTIONS_D_BY_LENGTH_FILENAME),
    ):
        if table is None:
            (result_dir / filename).unlink(missing_ok=True)
        else:
            table.write_csv(result_dir / filename)
    for filename in _STALE_DIFFUSION_FILES:
        (result_dir / filename).unlink(missing_ok=True)


POOLED_DISTRIBUTIONS_FILENAME = "pooled_distributions_D.csv"
POOLED_POPULATIONS_FILENAME = "pooled_populations_D.csv"
POOLED_DRAWS_FILENAME = "pooled_lognormal_draws_D.csv"
# Written by earlier versions; removed so the folder is always one pooling.
_STALE_POOLED_FILES = ("pooled_distances_D.csv",)
POOLED_ENSEMBLE_MSD_FILENAME = "pooled_ensemble_msd.csv"
POOLED_ENSEMBLE_MSD_FITS_FILENAME = "pooled_ensemble_msd_fits.csv"
POOLED_SUMMARY_FILENAME = "pooled_summary.json"


def write_pooled_results(
    out_dir: str | Path,
    *,
    distributions_D: pl.DataFrame,
    populations_D: pl.DataFrame,
    lognormal_draws_D: pl.DataFrame,
    summary: dict,
    ensemble_msd: Optional[pl.DataFrame] = None,
    ensemble_msd_fits: Optional[pl.DataFrame] = None,
) -> None:
    """What `gemscape2 pool` found across bundles. The ensemble MSD tables left as None have their files removed,
    so the directory is always one pooling."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / POOLED_SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))
    distributions_D.write_csv(out_dir / POOLED_DISTRIBUTIONS_FILENAME)
    populations_D.write_csv(out_dir / POOLED_POPULATIONS_FILENAME)
    lognormal_draws_D.write_csv(out_dir / POOLED_DRAWS_FILENAME)
    for filename in _STALE_POOLED_FILES:
        (out_dir / filename).unlink(missing_ok=True)
    for table, filename in (
        (ensemble_msd, POOLED_ENSEMBLE_MSD_FILENAME),
        (ensemble_msd_fits, POOLED_ENSEMBLE_MSD_FITS_FILENAME),
    ):
        if table is None:
            (out_dir / filename).unlink(missing_ok=True)
        else:
            table.write_csv(out_dir / filename)


def load_diffusion_summary(result_dir: str | Path) -> Optional[dict]:
    """The saved `diffusion_summary.json`, or None if this bundle has no
    saved analysis yet."""
    path = Path(result_dir) / DIFFUSION_SUMMARY_FILENAME
    return json.loads(path.read_text()) if path.exists() else None


def load_diffusion_results(result_dir: str | Path) -> Optional[dict]:
    """What `write_diffusion_results` wrote, under its keyword names
    (`summary`, `tracks_summary`, `loglik_D`) -- or None when there is
    no saved posterior analysis to read back."""
    result_dir = Path(result_dir)
    summary = load_diffusion_summary(result_dir)
    loglik_path = result_dir / LOGLIK_D_FILENAME
    summary_path = result_dir / TRACKS_SUMMARY_FILENAME
    if summary is None or not loglik_path.exists() or not summary_path.exists():
        return None
    return {
        "summary": summary,
        # Every row read before typing a column: one that is empty for the
        # first thousand tracks (MSD, NUTS) would otherwise be a string.
        "tracks_summary": pl.read_csv(summary_path, infer_schema_length=None),
        "loglik_D": pl.read_parquet(loglik_path),
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
