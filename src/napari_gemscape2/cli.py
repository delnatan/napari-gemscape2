"""`gemscape2` CLI: headless, config-driven, unattended batch runner.

For scripted/reproducible runs outside napari. Calls the exact same
`pipeline.run_detect_track` the interactive widget uses -- see
`pipeline.py`'s module docstring.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Optional

import typer

from napari_gemscape2 import units
from napari_gemscape2.batch import detect_track_bundle
from napari_gemscape2.results import (
    has_regions,
    has_result,
    load_diffusion_summary,
    load_manifest,
    package_provenance,
    write_pooled_results,
)
from napari_gemscape2.pipeline import (
    DetectTrackParams,
    detect_track_params_from_manifest,
    parse_flag_names,
)

app = typer.Typer(no_args_is_help=True)


def _load_config(config: Path) -> tuple[dict, Path]:
    """The parsed config and the folder its relative paths are read
    against -- its own, so a config works from wherever it is run."""
    return tomllib.loads(config.read_text()), config.resolve().parent


def _resolve(base: Path, path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else base / path


def _template_dir(cfg: dict, base: Path) -> Optional[Path]:
    """The bundle named by `template` (its folder, or its manifest.json):
    the movie whose saved settings a batch reuses."""
    if cfg.get("template") is None:
        return None
    path = _resolve(base, cfg["template"])
    if path.name == "manifest.json":
        path = path.parent
    if not has_result(path):
        raise typer.BadParameter(f"`template`: no results bundle at {path}", param_hint="CONFIG")
    return path


def _describe_template(template: Path, inherited: dict, overrides: dict) -> None:
    typer.echo(f"settings from template {template.name}: {', '.join(sorted(inherited)) or 'none'}")
    overridden = sorted(set(overrides) & set(inherited))
    if overridden:
        typer.echo(f"  overridden in this config: {', '.join(overridden)}")


def _result_dir(entry: dict, results_root: Path) -> Path:
    return results_root / entry.get("result_id", Path(entry["path"]).stem)


@app.command("detect-track")
def detect_track(
    config: Path = typer.Argument(..., help="TOML config listing input images + params"),
) -> None:
    """Run detect+track over every image listed in CONFIG.

    Config format (relative paths are read from CONFIG's own folder):
        results_root = "results"
        # Optional: reuse the settings a movie was saved with in the
        # napari widget (its manifest.json) -- sigma, detector, camera,
        # linking, point/track filters. Keys under [params] override it.
        template = "results/beads_dense"

        [params]
        sigma = 1.3      # PSF width, px (required unless the template has it)
        max_step = 5.0   # largest linked step, px (likewise required)
        exclude_flags = ["STALLED"]   # optional: spotsolve FitFlag names

        [[inputs]]
        path = "data/beads_timelapse_dense.tif"
        result_id = "beads_dense"   # optional, defaults to the stem
    """
    cfg, base = _load_config(config)
    results_root = _resolve(base, cfg.get("results_root", "results"))
    param_cfg = dict(cfg.get("params", {}))
    if isinstance(param_cfg.get("exclude_flags"), list):
        # TOML names the flags (`exclude_flags = ["EDGE", "STALLED"]`);
        # the pipeline takes spotsolve's bitmask.
        param_cfg["exclude_flags"] = parse_flag_names("|".join(param_cfg["exclude_flags"]))
    template = _template_dir(cfg, base)
    inherited: dict = {}
    if template is not None:
        inherited = detect_track_params_from_manifest(load_manifest(template))
        _describe_template(template, inherited, param_cfg)
    param_cfg = {**inherited, **param_cfg}
    if param_cfg.get("sigma") is None:
        raise typer.BadParameter(
            "[params] needs `sigma` (PSF width, px). Detect a few frames in the napari "
            "widget and read it off the fit_sigma histogram.",
            param_hint="CONFIG",
        )
    if param_cfg.get("max_step") is None:
        raise typer.BadParameter(
            "[params] needs `max_step` (largest linked step, px) -- about 3x the rms step "
            "of the fastest particles. A template linked before spotsolve 0.2 has none.",
            param_hint="CONFIG",
        )
    params = DetectTrackParams(**param_cfg)

    import spotsolve
    import napari_gemscape2

    packages = package_provenance(spotsolve, napari_gemscape2)

    for entry in cfg["inputs"]:
        image_path = _resolve(base, entry["path"])
        result_id = entry.get("result_id", image_path.stem)

        typer.echo(f"[{result_id}] {image_path}")
        result_dir = results_root / result_id
        if has_regions(result_dir):
            typer.echo("  restricted to its saved regions mask (labels.tif)")
        manifest_extra = detect_track_bundle(
            image_path,
            result_dir,
            params,
            packages=packages,
            pixel_size_um=entry.get("pixel_size_um"),
            dt_s=entry.get("dt_s"),
            exposure_s=entry.get("exposure_s"),
        )
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


@app.command("diffusion")
def diffusion(
    config: Path = typer.Argument(..., help="The same TOML config detect-track ran"),
) -> None:
    """Diffusion analysis of every bundle CONFIG's detect-track wrote --
    the "Diffusion analysis" widget's Run + Save analysis, headless.

    Reads `<results_root>/<result_id>/` for each [[inputs]] entry and
    writes the widget's files there (tracks_summary.csv, posterior_D.parquet,
    distributions_D.csv, distributions_D_by_length.csv,
    diffusion_summary.json), so the result reopens
    in the widget as if saved from it.

    Settings come from the template bundle's diffusion_summary.json when
    `template` is set and it has one, overridden by [diffusion]:
        [diffusion]
        min_frames = 3          # shortest track fitted
        # the posterior grid (diffusionkit's GridPostOptions); D's range is
        # the flat prior's support. Any key left out keeps the template's,
        # else diffusionkit's default (shown).
        grid = { D_min_um2_s = 1e-5, D_max_um2_s = 10.0, n_D = 601 }
        msd_comparison = false
        msd_max_lag = 3         # lags the MSD fits use (>= 3)
        exposure_s = 0.01       # optional: overrides each bundle's recorded one
        min_track_length = 1    # which tracks pass (passes_filters, ensemble)
        filters = { D_median_um2_s = [0.001, inf], flux_mean = [800.0, inf] }
    The ensemble's deconvolution has no settings: its smoothness is chosen
    by the data (diffusionkit's `gridpost.deconvolve`).
    An [[inputs]] entry's own `exposure_s` wins over both.
    """
    from napari_gemscape2.diffusion_batch import DiffusionSettings, analyze_bundle, settings_from_summary

    cfg, base = _load_config(config)
    results_root = _resolve(base, cfg.get("results_root", "results"))
    diff_cfg = dict(cfg.get("diffusion", {}))
    unknown = set(diff_cfg) - DiffusionSettings.names()
    if unknown:
        raise typer.BadParameter(
            f"[diffusion] has unknown keys: {', '.join(sorted(unknown))}", param_hint="CONFIG"
        )
    template = _template_dir(cfg, base)
    inherited: dict = {}
    if template is not None:
        saved = load_diffusion_summary(template)
        if saved is None:
            typer.echo(f"template {template.name} has no saved diffusion analysis -- using [diffusion] and defaults")
        else:
            inherited = settings_from_summary(saved)
            _describe_template(template, inherited, diff_cfg)
    settings_cfg = {**inherited, **diff_cfg}
    if isinstance(diff_cfg.get("grid"), dict):
        settings_cfg["grid"] = {**inherited.get("grid", {}), **diff_cfg["grid"]}
    try:
        DiffusionSettings(**{k: v for k, v in settings_cfg.items() if k != "filters"}).options()
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(f"[diffusion] {exc}", param_hint="CONFIG") from exc
    if settings_cfg.get("filters"):
        settings_cfg["filters"] = {col: tuple(bounds) for col, bounds in settings_cfg["filters"].items()}

    failed = 0
    for entry in cfg["inputs"]:
        result_dir = _result_dir(entry, results_root)
        typer.echo(f"[{result_dir.name}] {result_dir}")
        if not has_result(result_dir):
            typer.echo("  ! no results bundle here -- run detect-track first")
            failed += 1
            continue
        entry_cfg = dict(settings_cfg)
        if entry.get("exposure_s") is not None:
            entry_cfg["exposure_s"] = entry["exposure_s"]
        settings = DiffusionSettings(**entry_cfg)
        try:
            report = analyze_bundle(result_dir, settings)
        except ValueError as exc:
            typer.echo(f"  ! skipped: {exc}")
            failed += 1
            continue
        if not report.units_known:
            typer.echo("  ! the bundle records no pixel size / frame interval: every µm and s is px and frames")
        typer.echo(
            f"  {units.fmt(report.pixel_size_um, 'pixel_size_um')} · {units.fmt(report.dt_s, 'dt_s')} · "
            f"exposure {units.fmt(report.exposure_s, 'exposure_s')}"
        )
        typer.echo(
            f"  -> {report.n_fitted} of {report.n_tracks} tracks fitted, "
            f"{report.n_passing} pass the filters"
        )
    if failed:
        raise typer.Exit(code=1)


@app.command("pool")
def pool(
    config: Path = typer.Argument(..., help="The same TOML config diffusion ran"),
) -> None:
    """Pool the saved diffusion analyses of CONFIG's bundles by sample: the D populations of replicates
    together, the distances between samples and replicates, and optionally the ensemble-averaged MSD.

    Nothing is refitted: each bundle's saved posteriors (`gemscape2 diffusion` first) and the filters it
    was saved with are read back, and only tracks that passed them are pooled. Writes
    pooled_distributions_D.csv, pooled_distances_D.csv, pooled_summary.json (and the ensemble MSD tables)
    to `<results_root>/pooled/`, or to `[pool] output`. Bundles are grouped by their `sample`:
        [[inputs]]
        path = "data/wt_1.tif"
        sample = "wt"           # replicates share it; default: the bundle's own name

        [pool]
        output = "pooled"       # a folder under the config's folder (default: results_root/pooled)
        n_boot = 200            # ensemble MSD: bootstrap resamples of tracks
        # Ensemble-averaged MSD, off by default. Its windows are explicit: each track's MSD is computed
        # to ensemble_max_lag, and the fit uses the averaged curve's first ensemble_n_points lags.
        ensemble_msd = true
        ensemble_max_lag = 8
        ensemble_n_points = 4
        ensemble_offset = "provided"   # or "fit": the intercept of the linear fit (no SDs used)
    A bundle without a saved or current analysis stops the pooling: a missing replicate would bias it.
    """
    from napari_gemscape2.diffusion import grid_record
    from napari_gemscape2.pooling import (
        PoolSettings,
        load_pooled_bundle,
        pooled_batch,
        pooled_ensemble_msd,
        population_tables,
    )

    cfg, base = _load_config(config)
    results_root = _resolve(base, cfg.get("results_root", "results"))
    pool_cfg = dict(cfg.get("pool", {}))
    output = pool_cfg.pop("output", None)
    unknown = set(pool_cfg) - PoolSettings.names()
    if unknown:
        raise typer.BadParameter(f"[pool] has unknown keys: {', '.join(sorted(unknown))}", param_hint="CONFIG")
    try:
        settings = PoolSettings(**pool_cfg)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(f"[pool] {exc}", param_hint="CONFIG") from exc
    out_dir = _resolve(base, output) if output else results_root / "pooled"

    bundles, problems = [], []
    for entry in cfg["inputs"]:
        result_dir = _result_dir(entry, results_root)
        try:
            bundles.append(load_pooled_bundle(result_dir, entry.get("sample")))
        except (ValueError, OSError) as exc:
            problems.append(f"[{result_dir.name}] {exc}")
    if problems:
        for line in problems:
            typer.echo(f"  ! {line}")
        raise typer.Exit(code=1)
    for b in bundles:
        n = len(b.analysis.fitted_ids) if b.passing_ids is None else len(set(b.analysis.fitted_ids.tolist()) & b.passing_ids)
        typer.echo(f"[{b.result_id}] sample {b.sample}: {n} fitted tracks pooled")

    try:
        batch = pooled_batch(bundles)
        distributions, distances = population_tables(batch, bundles[0].analysis.level)
        ens = fits = None
        if settings.ensemble_msd:
            ens, fits = pooled_ensemble_msd(bundles, settings)
    except ValueError as exc:
        typer.echo(f"  ! {exc}")
        raise typer.Exit(code=1) from exc

    import diffusionkit
    import napari_gemscape2

    samples: dict[str, list[str]] = {}
    for b in bundles:
        samples.setdefault(b.sample, []).append(b.result_id)
    summary = {
        "analysis": "diffusionkit.gridpost populations pooled over saved posteriors (flat prior in ln D)",
        "samples": samples,
        "credible_level": bundles[0].analysis.level,
        "grid": grid_record(bundles[0].analysis.options),
        "tracks_pooled": "those that passed each bundle's saved filters",
        "ensemble_msd": (
            {**{k: v for k, v in vars(settings).items() if k != "ensemble_msd"}, "exposure": "treated as 0 (no blur model)",
             "weight": "pairs", "resample": "track"}
            if settings.ensemble_msd
            else None
        ),
        "packages": package_provenance(diffusionkit, napari_gemscape2),
    }
    write_pooled_results(
        out_dir,
        distributions_D=distributions,
        distances_D=distances,
        summary=summary,
        ensemble_msd=None if ens is None else ens.curves,
        ensemble_msd_fits=fits,
    )
    typer.echo(f"  -> {out_dir}  ({len(samples)} samples, {len(bundles)} bundles)")


@app.command("view")
def view(result_dir: Path) -> None:
    """Launch napari and load ONE results bundle's layers."""
    from napari_gemscape2.viewer import launch_viewer

    launch_viewer(result_dir)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
