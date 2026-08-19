"""`spt` CLI: headless, config-driven, unattended batch runner.

For scripted/reproducible runs outside napari. Calls the exact same
`pipeline.run_detect_track` the interactive widget uses -- see
`pipeline.py`'s module docstring.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import typer

from spt_pipeline.experiment import build_manifest, git_sha, repo_root_of, write_experiment
from spt_pipeline.pipeline import DetectTrackParams, run_detect_track

app = typer.Typer(no_args_is_help=True)


@app.command("detect-track")
def detect_track(
    config: Path = typer.Argument(..., help="TOML config listing input images + params"),
) -> None:
    """Run detect+track over every image listed in CONFIG.

    Config format:
        experiments_root = "experiments"
        [params]
        sigma_init = 1.3

        [[inputs]]
        path = "data/beads_timelapse_dense.tif"
        experiment_id = "beads_dense"   # optional, defaults to the stem
    """
    cfg = tomllib.loads(config.read_text())
    experiments_root = Path(cfg.get("experiments_root", "experiments"))
    params = DetectTrackParams(**cfg.get("params", {}))

    import sfwloc
    import spt_pipeline

    repo_shas = {
        "sfwloc": git_sha(repo_root_of(sfwloc)),
        "spt_pipeline": git_sha(repo_root_of(spt_pipeline)),
    }

    for entry in cfg["inputs"]:
        image_path = Path(entry["path"])
        experiment_id = entry.get("experiment_id", image_path.stem)

        typer.echo(f"[{experiment_id}] {image_path}")
        points_df, tracks_df, manifest_extra = run_detect_track(
            image_path,
            pixel_size_um=entry.get("pixel_size_um"),
            dt_s=entry.get("dt_s"),
            params=params,
        )
        manifest = build_manifest(
            experiment_id=experiment_id,
            source_image_path=image_path,
            params=manifest_extra,
            repo_shas=repo_shas,
        )
        experiment_dir = experiments_root / experiment_id
        write_experiment(experiment_dir, points_df, tracks_df, manifest)
        typer.echo(
            f"  -> {experiment_dir}  "
            f"({manifest_extra['n_points']} points, {manifest_extra['n_tracks']} tracks)"
        )


@app.command("view")
def view(experiment_dir: Path) -> None:
    """Launch napari and load ONE experiment bundle's layers."""
    from spt_pipeline.viewer import launch_viewer

    launch_viewer(experiment_dir)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
