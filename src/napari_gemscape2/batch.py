"""Batch runs from a template bundle: the headless steps the `gemscape2`
CLI and the experiment list's "Batch from this movie…" dialog share.

A batch reuses one movie's saved settings -- its manifest for
detect+track (`pipeline.detect_track_params_from_manifest`), its
`diffusion_summary.json` for the diffusion analysis
(`diffusion_batch.settings_from_summary`) -- on other movies, so both
entry points write the same bundles. `write_batch_config` records a GUI
batch as the TOML config the CLI reads, so it can be re-run headless.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterable, Optional

from napari_gemscape2.pipeline import DetectTrackParams, ProgressCallback, run_detect_track
from napari_gemscape2.results import build_manifest, load_regions, write_result


def detect_track_bundle(
    image_path: Path,
    result_dir: Path,
    params: DetectTrackParams,
    *,
    repo_shas: dict,
    pixel_size_um: Optional[float] = None,
    dt_s: Optional[float] = None,
    exposure_s: Optional[float] = None,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Detect+track one movie and write its bundle to `result_dir`.
    Returns the manifest's `params` (units, their sources, counts).

    A mask already saved in `result_dir` (`results.save_regions`, painted
    in the widget) restricts the run to its regions and is kept in the
    bundle; without one the whole field is analyzed."""
    labels, regions = load_regions(result_dir)
    points_df, tracks_df, manifest_extra = run_detect_track(
        image_path,
        params,
        pixel_size_um=pixel_size_um,
        dt_s=dt_s,
        exposure_s=exposure_s,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        labels=labels,
        regions=regions,
    )
    manifest = build_manifest(
        result_id=result_dir.name,
        source_image_path=image_path,
        params=manifest_extra,
        repo_shas=repo_shas,
    )
    write_result(result_dir, points_df, tracks_df, manifest, labels=labels, regions=regions)
    return manifest_extra


def _relative(path: Path, base: Path) -> str:
    """`path` relative to `base` when it is inside `base`'s parent -- the
    usual `<data>/results/` layout -- so the config moves with the data;
    absolute otherwise."""
    path, base = Path(path).resolve(), Path(base).resolve()
    if path.is_relative_to(base.parent):
        return os.path.relpath(path, base)
    return str(path)


def write_batch_config(
    config_path: Path,
    *,
    results_root: Path,
    template: Path,
    inputs: Iterable[tuple[Path, str]],
    diffusion: bool,
) -> None:
    """Write the TOML config `gemscape2 detect-track`/`diffusion` read
    for this batch: `template` supplies the settings, `inputs` are
    `(image_path, result_id)` pairs. Paths are relative to the config's
    own folder."""
    base = config_path.parent

    def s(value: str) -> str:
        # A JSON string is a valid TOML basic string.
        return json.dumps(value)

    lines = [
        "# Written by the experiment list's \"Batch from this movie…\".",
        "# Settings come from the template bundle; re-run with",
        "#   gemscape2 detect-track <this file>",
    ]
    if diffusion:
        lines.append("#   gemscape2 diffusion <this file>")
    lines += [
        f"results_root = {s(_relative(results_root, base))}",
        f"template = {s(_relative(template, base))}",
    ]
    for image_path, result_id in inputs:
        lines += [
            "",
            "[[inputs]]",
            f"path = {s(_relative(image_path, base))}",
            f"result_id = {s(result_id)}",
        ]
    config_path.write_text("\n".join(lines) + "\n")
