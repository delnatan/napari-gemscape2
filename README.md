# spt-pipeline

Batch orchestration and napari visualization for single-particle tracking, connecting:

- [`spotsolve`](https://github.com/delnatan/spotsolve) — multi-emitter 2D localization
  by Bayesian model selection, plus frame-to-frame linking. Runs in Rust.
- [`diffusionkit`](https://github.com/delnatan/diffusionkit) — MSD and Bayesian diffusion analysis.

## Particle tracking in the dense regime

Both halves of the problem are handled by `spotsolve` rather than tuned around:

- **Detection** asks "how many emitters are here, and where?" as one estimation
  problem. In each small box an emitter exists iff it lowers that box's Poisson
  deviance by a fixed number of nats, and the box is fitted jointly with every
  emitter it needs at each one's own width — so overlapping spots are resolved
  instead of being merged into one bright centroid. There is one detector, not a
  sparse/dense choice, and no sparsity weight or iteration budget to tune.
- **Linking** scores each candidate link as a likelihood ratio under a per-track
  posterior over the diffusion coefficient, using each detection's own CRLB
  (`se_y`/`se_x`). A dim spot therefore carries a genuinely wider gate than a
  bright one, which is what decides assignments when the field is crowded. Every
  parameter is measured from the movie by `fit_link_params`, so there is no gate
  or search radius to set.

Linking is frame-to-frame only: a missed detection **ends** a track rather than
being bridged across the gap. Fragmenting a trajectory is a safe failure and
switching its identity is not, so trajectories come out short rather than wrong,
and `min_track_length` is the knob that matters afterwards.

`spt_pipeline.tracking_diagnostics.check_resolvability` reports whether a given
(D, dt, density) is trackable at all — the frame-to-frame step against the mean
nearest-neighbor spacing. Read its verdict as an advisory: the closed forms are
sample physics, but its thresholds were calibrated against the older LAP linker
and are conservative for this one (see that module's docstring).

## Reproducible experiment bundles

Each processed acquisition is written as a directory bundle:

```
experiments/<experiment_id>/
    points.parquet
    tracks.parquet
    manifest.json
    rois.json              (optional — only if an ROI was used)
```

`points.parquet` is `spotsolve.loctable`'s localization table verbatim — one row
per detection, carrying `se_y`/`se_x` (per-detection CRLB), `flux`, `bg`,
`fit_sigma`/`sigma_ratio` and the `is_aggregate` flag. `tracks.parquet` is that
same table with a `track_id` column added, so every detector column survives
linking and stays available for QC downstream.

Nothing a QC decision rejects is deleted. Over-bright detections are **flagged**
(`is_aggregate`), and the histogram filters described below are recorded as
ranges rather than applied to `points.parquet` — so how much of a movie was junk
stays an auditable fact about the run instead of a silent subtraction. What those
judgements change is what the *linker* sees and which tracks the bundle keeps.

`manifest.json` records the source image path, the camera and detection
parameters, both filter specs, what the linker actually measured (PSF sigma, both
diffusion-coefficient estimates, the fitted linking parameters), and the git SHA
of `spotsolve` and `spt-pipeline` at run time.

## Detect, then filter, then finalize

Every QC threshold in the interactive UI is set by dragging two handles across the
distribution it applies to, the way Imaris and TrackMate do it — `flux > 1200`
means nothing until you can see that the flux histogram is bimodal with a trough
at 1200. Filters stack and AND together, and the layer in the viewer redraws as
each handle moves, so a spot that fails a cut leaves the image while the cut is
being made. Nothing is written to disk until *Save experiment*, so the cuts are
chosen against a finished stage's real output rather than guessed at beforehand.

This is also how the PSF width is measured, which is why there is no calibration
step any more. **Preview frame** localizes one frame with the reporting band off
— every fit, including the ones a real run would reject — and reports the median
`fit_sigma`; **Use** adopts it. That is one round of what
`spotsolve.calibrate_sigma` iterates internally, and it converges just as fast (on
a synthetic field: 1.100 → 1.4208 → 1.4197 against a true 1.45), with the
distribution on screen throughout. A bimodal or ragged `fit_sigma` — two focal
planes, junk being fitted as signal — becomes something you see rather than
something a median averages away. `calibrate_sigma` is still what unattended batch
runs use, where nobody is looking at a histogram.

One unit caveat, since two are in play: `sigma` is in **pixels**, while `slack`
and `band` are **multiples of whatever sigma the search is running at**
(`spotsolve` tests them against `sigma_ratio = fit_sigma / sigma`). They are not
multiples of the initial guess and not absolute pixels, so changing `sigma` moves
both windows with it — the Detect tab prints the resulting px window underneath
them for that reason.

## Setup

The project has its own `uv`-managed virtual environment in `.venv`. `spotsolve`
and `diffusionkit` are local path dependencies (sibling checkouts), and
`spotsolve-rs` — the Rust extension `spotsolve` needs at import time — is built
from `../spotsolve/rust/spotsolve-py`:

```
uv sync
```

Re-run it after changing `spotsolve`'s Rust core, so the extension is rebuilt.

## Usage

Headless batch run (see `pyproject.toml`'s `[project.scripts]` entry `spt`):

```
spt detect-track config.toml
```

```toml
experiments_root = "experiments"

[params]
# Give `sigma` (px) to use a measured width directly — e.g. one settled on
# interactively with Preview frame. Omit it and each file is calibrated on its
# own, with `sigma_init` as that fit's starting guess, which is usually what a
# folder of separate acquisitions wants.
sigma_init = 1.3
min_track_length = 2
# Measure these once for your camera -- gain omitted means "estimate per
# frame", which works but drifts with the sample.
camera_kwargs = { offset = 100.0, gain = 2.0, read_noise = 2.0 }
# The same QC cuts the UI's histogram filters produce, as {column = [lo, hi]}.
# Applied to what the linker sees and to which tracks are kept -- never to
# points.parquet, which holds every detection either way.
point_filters = { flux = [1200.0, 40000.0], fit_sigma = [1.0, 2.2] }
track_filters = { track_length = [5.0, 1e9] }

[[inputs]]
path = "data/beads_timelapse_dense.tif"
experiment_id = "beads_dense"   # optional, defaults to the stem
```

Interactive: open napari and use the "Experiment list" dock widget to browse a
folder of raw images and work one image through **Detect** (camera, PSF width via
Preview frame, detection knobs, then filters on what it found) and **Track**
(link, then filters on the tracks), pressing *Save experiment* when the result is
worth keeping. The "Diffusion analysis" widget then reads the tracks layer for
MSD, Bayesian and per-track anisotropy fits, and offers the same histogram
filters over per-track results — so a fitted `D` or `alpha` is filterable by the
same drag as any other feature.
