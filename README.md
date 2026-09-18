# spt-pipeline

Batch orchestration and napari visualization for single-particle tracking, connecting:

- [`spotsolve`](https://github.com/delnatan/spotsolve) — multi-emitter 2D localization
  by Bayesian model selection, plus frame-to-frame linking. Runs in Rust.
- [`diffusionkit`](https://github.com/delnatan/diffusionkit) — classical (Brownian displacement MLE) and Bayesian diffusion analysis.

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

## Reproducible results bundles

Each processed acquisition is written as a directory bundle:

```
results/<result_id>/
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

**Several ROIs** can be checked at once in the Detect tab — each Shapes layer is
one region, named after the layer. Detections are then labeled with the region
they fall in (`roi`/`roi_index` columns; where regions overlap, the one higher in
napari's layer list wins), tracking links each region separately with its own
fitted parameters (so no track crosses a boundary, and the manifest carries a
`track_summary_by_roi`), and the Diffusion panel can filter to one ROI and
reports the classical D/z summary per ROI.

Reopening a row that already has a bundle resumes it: the saved points are the
session's detections, so Track links them without re-running detect, and the
saved ROIs come back checked.

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
being made. Nothing is written to disk until *Save results*, so the cuts are
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

## Units, and where they come from

Two numbers turn this pipeline's pixels and frames into physics: the pixel size
and the frame interval. Everything physical is those two multiplied through —
`x_um`, `duration_s`, `mean_step_um`, every fitted `D` and `K` — so both are read
off the image file's own metadata and **shown, with their provenance, before
anything runs**: the line above the Detect/Track tabs says e.g.
`from file: 500 frames · 0.1083 µm/px · 0.0302 s/frame`, and its tooltip says
which metadata field each came from. A headless run prints the same line
(`spt detect-track`), and `manifest.json` records it (`pixel_size_um_source`,
`dt_s_source`, `metadata_notes`), so a bundle says not just what pixel size it
used but where that came from.

A reader refuses to guess rather than defaulting quietly, because a wrong
calibration produces a table that looks completely normal:

- A TIFF's `XResolution` is "pixels per unit" and the tag itself never says
  microns. The unit comes from ImageJ's own `unit` field or from
  `ResolutionUnit` (inch/cm only), or from the OME header when there is one,
  which is preferred since it states a unit per axis. With no usable unit there
  is **no pixel size** — an inch-calibrated file used to read as microns, i.e.
  every physical column out by 25400×.
- An `.ims` with no recorded extents yields a voxel size of 1/width from
  ImarisReader's defaults — plausible-looking and not a calibration. That now
  reads as missing (`ImarisReader.voxel_size_known`).
- An `.nd2`'s exactly-1×1 µm voxel is that reader's placeholder, so it reads as
  missing too.
- A recorded-but-zero frame interval reads as missing rather than dividing into
  `D` later, and non-square pixels or irregular frame timing (measured from
  per-frame timestamps) are flagged as notes rather than averaged away silently.

`pipeline.load_session` then raises naming what is missing and what the file did
say, instead of failing downstream — and the interactive panel turns that line
red before a button is pressed.

**Overriding it.** The same line folds open into "Image metadata": where each
value came from, and two boxes to supply your own when the file is silent or
wrong. It unfolds itself when a value is missing, since that is the one moment
it is the next thing to use; otherwise it stays out of the way. The override is
sticky across images — a folder is usually one session — and never silent about
it: while in force the line reads `overridden: … — file says …` in amber, and
the bundle records `pixel_size_um_source: "given explicitly (file said: …)"`.
Both the stepwise buttons and "Run selected" honor it, as does a config's
`pixel_size_um`/`dt_s` per `[[inputs]]` entry for headless runs.

Changing the scale after a detect or link run re-derives what it cheaply can
(per-track metrics, the layers' units) and blocks *Save* until you re-run —
`points.parquet`'s `t`/`y_um`/`se_*_um` columns are derived at detect time, so a
bundle saved across a scale change would be internally inconsistent.

**Exposure time.** A third number, the camera exposure, is read the same way
(`.nd2` capture text, an Imaris `Channel` `ExposureTime` with a unit, OME
`Plane ExposureTime`) and recorded in the manifest as `exposure_s` /
`exposure_s_source`. Detection and linking never use it; the diffusion
analysis does, because its MLE models the motion blur of a continuous
exposure. It is **never defaulted to 0**: 0 means "instantaneous", and on
real data that biases `D` by about −25% and the non-Brownian score by +0.3 to
+0.7. A file that doesn't record it (e.g. every Andor Fusion `.ims`) shows
`exposure ?` in amber. You type it into "Image metadata" (the exposure box is
not behind the override switch, so supplying it doesn't replace the file's
pixel size or frame interval). The Diffusion panel's own exposure box is
pre-filled from the layer, and it won't run until it has a value.

In results, `spt_pipeline.units` is the single source of truth for what each
column is measured in: it labels the tracks-pane headers (`D_mle [µm²/s]`,
`se_x_max [px]`, `se_x_um_max [µm]`), the filter rows' tooltips, the spatial
map's color scale, every fit readout, and every plot axis — the last in the same
mathtext style diffusionkit's own figures use, so a joint plot and an MSD plot
label the same quantity the same way. A column whose unit isn't registered is
shown bare rather than guessed at.

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
results_root = "results"

[params]
# Give `sigma` (px) to use a measured width directly — e.g. one settled on
# interactively with Preview frame. Omit it and each file is calibrated on its
# own, with `sigma_init` as that fit's starting guess, which is usually what a
# folder of separate acquisitions wants.
sigma_init = 1.3
min_track_length = 2
# `offset` is the only camera fact spotsolve needs -- noise is measured
# from each frame directly.
camera_kwargs = { offset = 100.0 }
# Score candidate links by brightness continuity (flux/se_flux) as well as
# position and CRLB -- an extra cue for a crowded field. Off by default.
link_with_flux = false
# The same QC cuts the UI's histogram filters produce, as {column = [lo, hi]}.
# Applied to what the linker sees and to which tracks are kept -- never to
# points.parquet, which holds every detection either way.
point_filters = { flux = [1200.0, 40000.0], fit_sigma = [1.0, 2.2] }
track_filters = { track_length = [5.0, 1e9] }

[[inputs]]
path = "data/beads_timelapse_dense.tif"
result_id = "beads_dense"   # optional, defaults to the stem
```

Interactive: open napari and use the "Experiment list" dock widget to browse a
folder of raw images and work one image through **Detect** (PSF width via
Preview frame, detection knobs, then filters on what it found) and **Track**
(link, optionally with flux as a second link cue, then filters on the
tracks), pressing *Save results* when the result is worth keeping. The "Diffusion analysis" widget then reads the tracks layer for
the classical per-track Brownian MLE (`D` plus a calibrated non-Brownian score
`z`, read as a population: log D vs z, mean z ± SE; MSD fits only as a
labelled comparison), Bayesian and per-track anisotropy fits, and offers the same histogram
filters over per-track results — so a fitted `D` or `alpha` is filterable by the
same drag as any other feature.
