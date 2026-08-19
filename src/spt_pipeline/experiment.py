"""Reproducible per-experiment bundle: points + tracks + provenance manifest.

An experiment bundle is a directory:
    <experiment_dir>/points.parquet
    <experiment_dir>/tracks.parquet
    <experiment_dir>/manifest.json

The source image is referenced by path in the manifest, not copied.
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
) -> None:
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    points_df.write_parquet(experiment_dir / POINTS_FILENAME)
    tracks_df.write_parquet(experiment_dir / TRACKS_FILENAME)
    (experiment_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))


def load_experiment(experiment_dir: str | Path) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    experiment_dir = Path(experiment_dir)
    points_df = pl.read_parquet(experiment_dir / POINTS_FILENAME)
    tracks_df = pl.read_parquet(experiment_dir / TRACKS_FILENAME)
    manifest = json.loads((experiment_dir / MANIFEST_FILENAME).read_text())
    return points_df, tracks_df, manifest


def has_experiment(experiment_dir: str | Path) -> bool:
    return (Path(experiment_dir) / MANIFEST_FILENAME).exists()


def experiment_dir_for(experiments_root: str | Path, image_path: str | Path) -> Path:
    """Default bundle location for a raw image: `<experiments_root>/<stem>`."""
    return Path(experiments_root) / Path(image_path).stem
