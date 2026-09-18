"""`spt` CLI: headless, config-driven, unattended batch runner.

For scripted/reproducible runs outside napari. Calls the exact same
`pipeline.run_detect_track` the interactive widget uses -- see
`pipeline.py`'s module docstring.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import typer

from spt_pipeline import units
from spt_pipeline.results import build_manifest, repo_shas, write_result
from spt_pipeline.pipeline import DetectTrackParams, run_detect_track

app = typer.Typer(no_args_is_help=True)


@app.command("detect-track")
def detect_track(
    config: Path = typer.Argument(..., help="TOML config listing input images + params"),
) -> None:
    """Run detect+track over every image listed in CONFIG.

    Config format:
        results_root = "results"
        [params]
        sigma_init = 1.3

        [[inputs]]
        path = "data/beads_timelapse_dense.tif"
        result_id = "beads_dense"   # optional, defaults to the stem
    """
    cfg = tomllib.loads(config.read_text())
    results_root = Path(cfg.get("results_root", "results"))
    params = DetectTrackParams(**cfg.get("params", {}))

    import spotsolve
    import spt_pipeline

    shas = repo_shas(spotsolve, spt_pipeline)

    for entry in cfg["inputs"]:
        image_path = Path(entry["path"])
        result_id = entry.get("result_id", image_path.stem)

        typer.echo(f"[{result_id}] {image_path}")
        points_df, tracks_df, manifest_extra = run_detect_track(
            image_path,
            pixel_size_um=entry.get("pixel_size_um"),
            dt_s=entry.get("dt_s"),
            params=params,
            exposure_s=entry.get("exposure_s"),
        )
        manifest = build_manifest(
            result_id=result_id,
            source_image_path=image_path,
            params=manifest_extra,
            repo_shas=shas,
        )
        result_dir = results_root / result_id
        write_result(result_dir, points_df, tracks_df, manifest)
        # The two numbers every physical column in this bundle was
        # computed with, and where they came from -- the headless
        # counterpart of the metadata line the napari widget shows (see
        # `widgets/params_panel.PipelineParamsWidget.set_image_metadata`).
        # An unattended run is exactly where a wrong pixel size goes
        # unnoticed, so it goes in the log next to the counts.
        typer.echo(
            f"  {units.fmt(manifest_extra['pixel_size_um'], 'pixel_size_um')} "
            f"({manifest_extra.get('pixel_size_um_source')}) · "
            f"{units.fmt(manifest_extra['dt_s'], 'dt_s')} "
            f"({manifest_extra.get('dt_s_source')}) · "
            f"exposure {units.fmt(manifest_extra.get('exposure_s'), 'exposure_s')} "
            f"({manifest_extra.get('exposure_s_source')})"
        )
        for note in manifest_extra.get("metadata_notes") or ():
            typer.echo(f"  ! {note}")
        typer.echo(
            f"  -> {result_dir}  "
            f"({manifest_extra['n_points']} points, {manifest_extra['n_tracks']} tracks)"
        )


@app.command("view")
def view(result_dir: Path) -> None:
    """Launch napari and load ONE results bundle's layers."""
    from spt_pipeline.viewer import launch_viewer

    launch_viewer(result_dir)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
