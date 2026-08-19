# spt-pipeline

Batch orchestration and napari visualization for single-particle tracking, connecting:

- [`sfwloc`](https://github.com/delnatan/sfwloc) — 2D particle localization + Stage 1 LAP tracking.
- [`diffusionkit`](https://github.com/delnatan/diffusionkit) — MSD and Bayesian diffusion analysis.

## Reproducible experiment bundles

Each processed acquisition is written as a directory bundle:

```
experiments/<experiment_id>/
    points.parquet
    tracks.parquet
    manifest.json
```

`manifest.json` records the source image path, detection/tracking parameters, and the
git SHA of `sfwloc` and `spt-pipeline` at run time.

## Usage

Headless batch run (see `pyproject.toml`'s `[project.scripts]` entry `spt`):

```
spt detect-track config.toml
```

Interactive: open napari and use the "Experiment list" dock widget to browse a folder of
raw images, run detect+track on a selected image, and inspect the resulting
image/points/tracks layers.
