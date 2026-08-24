"""Reproducible per-experiment bundle: points + tracks + provenance manifest.

An experiment bundle is a directory:
    <experiment_dir>/points.parquet
    <experiment_dir>/tracks.parquet
    <experiment_dir>/manifest.json
    <experiment_dir>/rois.json      (optional -- only if an ROI was used)

The source image is referenced by path in the manifest, not copied.

`rois.json`, when present, is a JSON list of `{"name": ..., "polygons":
[[[y, x], ...], ...]}` records -- see `spt_pipeline.rois` for why every ROI
shape is flattened to a polygon regardless of how it was drawn (rectangle/
ellipse/polygon) and for the napari Shapes-layer <-> polygon conversion.
`name` is the napari layer name the ROI was drawn on, so reloading a bundle
(`viewer.add_experiment_layers`) recreates a Shapes layer under that same
name -- both to reproduce the analysis (rebuild the same boolean mask) and
to reproduce the visualization, without storing the image-sized mask array
itself.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

POINTS_FILENAME = "points.parquet"
TRACKS_FILENAME = "tracks.parquet"
MANIFEST_FILENAME = "manifest.json"
ROIS_FILENAME = "rois.json"
DIFFUSION_FITS_FILENAME = "diffusion_fits.parquet"
DIFFUSION_SUMMARY_FILENAME = "diffusion_summary.json"


def repo_root_of(module) -> Path:
    """A package's repo root, from its own `__file__` (`<repo>/src/
    <package>/__init__.py`), for `git_sha` provenance -- works from any
    caller regardless of that caller's own nesting depth, since it walks
    up from the target package's `__file__`, not the caller's."""
    return Path(module.__file__).resolve().parents[2]


def git_sha(repo_path: str | Path) -> str | None:
    """`git rev-parse HEAD` in `repo_path`, or None if unavailable (not a
    git repo, git not installed, etc.) -- provenance is best-effort."""
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


def build_manifest(
    *,
    experiment_id: str,
    source_image_path: str | Path,
    params: dict,
    repo_shas: dict[str, str | None] | None = None,
) -> dict:
    return {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_image_path": str(Path(source_image_path).resolve()),
        "params": params,
        "repo_shas": repo_shas or {},
    }


def write_experiment(
    experiment_dir: str | Path,
    points_df: pl.DataFrame,
    tracks_df: pl.DataFrame,
    manifest: dict,
    rois: list[dict] | None = None,
) -> None:
    """`rois`, if given and non-empty, is written to `rois.json` (see
    `spt_pipeline.rois.shapes_layer_to_roi` for how to build one from a
    napari Shapes layer). A stale `rois.json` from a previous run of this
    same bundle is removed when `rois` is `None`/empty, so a re-run without
    an ROI doesn't leave behind a region that no longer applies."""
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    points_df.write_parquet(experiment_dir / POINTS_FILENAME)
    tracks_df.write_parquet(experiment_dir / TRACKS_FILENAME)
    (experiment_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
    rois_path = experiment_dir / ROIS_FILENAME
    if rois:
        rois_path.write_text(json.dumps(rois, indent=2))
    else:
        rois_path.unlink(missing_ok=True)


def load_experiment(experiment_dir: str | Path) -> tuple[pl.DataFrame, pl.DataFrame, dict, list[dict]]:
    """Returns `(points_df, tracks_df, manifest, rois)` -- `rois` is `[]`
    for a bundle written before ROI persistence, or one that simply never
    used one."""
    experiment_dir = Path(experiment_dir)
    points_df = pl.read_parquet(experiment_dir / POINTS_FILENAME)
    tracks_df = pl.read_parquet(experiment_dir / TRACKS_FILENAME)
    manifest = json.loads((experiment_dir / MANIFEST_FILENAME).read_text())
    rois_path = experiment_dir / ROIS_FILENAME
    rois = json.loads(rois_path.read_text()) if rois_path.exists() else []
    return points_df, tracks_df, manifest, rois


def write_diffusion_results(
    experiment_dir: str | Path, per_track_df: pl.DataFrame, summary: dict
) -> None:
    """Diffusion-widget results, written like `rois.json`: an optional
    extra on top of the core points/tracks/manifest bundle, not every
    bundle has one. `per_track_df` holds one row per (track_id, method) --
    the classic-MSD population fit's per-track table and any Bayesian
    per-track fits the user has run, distinguished by a `method` column
    (see `widgets/diffusion_panel.py`) -- so both live in one file.
    `summary` holds the classic-MSD ensemble-level scalars (D/alpha fits,
    localization offset)."""
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    per_track_df.write_parquet(experiment_dir / DIFFUSION_FITS_FILENAME)
    (experiment_dir / DIFFUSION_SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))


def load_diffusion_results(
    experiment_dir: str | Path,
) -> tuple[pl.DataFrame | None, dict | None]:
    """`(per_track_df, summary)`, or `(None, None)` if this bundle has no
    saved diffusion results yet."""
    experiment_dir = Path(experiment_dir)
    fits_path = experiment_dir / DIFFUSION_FITS_FILENAME
    summary_path = experiment_dir / DIFFUSION_SUMMARY_FILENAME
    if not fits_path.exists():
        return None, None
    per_track_df = pl.read_parquet(fits_path)
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    return per_track_df, summary


def has_experiment(experiment_dir: str | Path) -> bool:
    return (Path(experiment_dir) / MANIFEST_FILENAME).exists()


def experiment_dir_for(experiments_root: str | Path, image_path: str | Path) -> Path:
    """Default bundle location for a raw image: `<experiments_root>/<stem>`."""
    return Path(experiments_root) / Path(image_path).stem
