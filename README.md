# napari-gemscape2

Batch orchestration and napari visualization for single-particle tracking, connecting:

- [`spotsolve`](https://github.com/delnatan/spotsolve) — multi-emitter 2D localization
  as one joint Poisson model per frame, plus frame-to-frame linking. Runs in Rust.
- [`diffusionkit`](https://github.com/delnatan/diffusionkit) — per-track grid posteriors over D and α, their
  ensemble (summed and deconvolved), classic MSD fits, and per-track NUTS.

## Particle tracking in the dense regime

Both halves of the problem are handled by `spotsolve` rather than tuned around:

- **Detection** asks "how many emitters are here, and where?" as one estimation
  problem. Each frame is fitted as one joint Poisson model, every emitter at its
  own width, and emitters are added or removed by likelihood ratio — so
  overlapping spots are resolved instead of being merged into one bright
  centroid. There is one detector, not a sparse/dense choice, and one knob:
  `fp_per_mpx`, the false spots it admits per 10⁶ pixels of pure noise (default
  16; the Detect tab restates it per frame of the image at hand).
- **Linking** is Crocker–Grier: between consecutive frames it picks the
  assignment with the least summed squared displacement, and no step longer than
  `max_step` (px) is linked. `max_step` is the one setting and is not estimated
  from the movie: about 3× the rms step of the fastest particles of interest.
  The Track tab restates it in µm and as the fastest D it admits, and the status
  line after a run reports the linked steps' rms to check it against.

Linking is frame-to-frame only: a missed detection **ends** a track rather than
being bridged across the gap. Fragmenting a trajectory is a safe failure and
switching its identity is not, so trajectories come out short rather than wrong,
and `min_track_length` is the knob that matters afterwards.

`napari_gemscape2.tracking_diagnostics.check_resolvability` reports whether a given
(D, dt, density) is trackable at all — the frame-to-frame step against the mean
nearest-neighbor spacing. Read its verdict as an advisory: the closed forms are
sample physics, but its thresholds were measured on sfwloc's LAP linker and have
not been re-measured on spotsolve's (see that module's docstring).

## Reproducible results bundles

Each processed acquisition is written as a directory bundle:

```
results/<result_id>/
    points.parquet
    tracks.parquet
    manifest.json
    labels.tif             (optional — only if regions were used)
    regions.json           (alongside labels.tif)
```

`points.parquet` is `spotsolve.loctable`'s localization table verbatim — one row
per detection, carrying `se_y`/`se_x` (per-detection CRLB), `flux`, `bg`,
`fit_sigma`/`sigma_ratio` and `flags` (spotsolve's `FitFlag` diagnostics).
`tracks.parquet` is that same table with `track_id` added, so every detector
column survives linking and stays available for QC downstream.

**Regions** are painted on a napari Labels layer (Detect tab → "New regions
layer"). Each label value is one region, and each pixel belongs to exactly one
label, so painting a nucleus over its cell cuts it out of the cell's cytoplasm.
A table in the Detect tab lists the layer's labels, each with a **name** (its
class: cytoplasm, nucleus, anything): "+" paints with the next free label, "−"
(or Delete) erases the selected one, and a row is renamed in place. Names may
repeat -- two cells both named "cytoplasm" are two regions of one class. Only
painted pixels are localized (label 0 is background). Detections are labeled
with `region` (the label) and `region_class` (its name). Tracking links each
region separately, so no track crosses a boundary, with link parameters fitted
per class (the manifest carries `track_summary_by_class`). The Diffusion panel can filter to one class and
reports the classical D summary (and z, when computed) per class. Bundles from
before this change keep their `rois.json`, but its polygons are no longer read.

A movie's mask is saved to its result folder when you move to another movie
(until it has results, after which the mask is saved with them), and its row in
the list is marked with a mask glyph. So you can paint masks across a whole
folder first, then batch it: masked movies are restricted to their regions, and
the rest are analyzed over the whole field.

Reopening a row that already has a bundle resumes it: the saved points are the
session's detections, so Track links them without re-running detect, and the
saved regions come back as the picked regions layer.

Nothing a QC decision rejects is deleted. spotsolve reports every fit with its
`FitFlag`s; the Track tab chooses which flags keep a fit away from the linker
(none by default), and fits without a finite position error are never linked.
The histogram filters described below are recorded as ranges rather than
applied to `points.parquet` — so how much of a movie was junk
stays an auditable fact about the run instead of a silent subtraction. What those
judgements change is what the *linker* sees and which tracks the bundle keeps.

`manifest.json` records the source image path, the camera and detection
parameters, both filter specs, the linking settings (`max_step_px`) and what the
run measured (PSF sigma, the linked steps' rms, the MSD estimate of D), and the git SHA
of `spotsolve` and `napari-gemscape2` at run time.

## Detect, then filter, then finalize

Every QC threshold in the interactive UI is set by dragging two handles across the
distribution it applies to, the way Imaris and TrackMate do it — `flux > 1200`
means nothing until you can see that the flux histogram is bimodal with a trough
at 1200. Filters stack and AND together, and the layer in the viewer redraws as
each handle moves, so a spot that fails a cut leaves the image while the cut is
being made. Nothing is written to disk until *Save results*, so the cuts are
chosen against a finished stage's real output rather than guessed at beforehand.

This is also how the PSF width is measured; there is no calibration or preview
step. Set `sigma` on the Detect page, run detect on a few frames (the frame
range), and read the `fit_sigma` histogram on the Filter page: its peak is the
width to set. Run again and it should stop moving. A bimodal or ragged
`fit_sigma` (two focal planes, junk being fitted as signal) is something you see
rather than something a median averages away. A sigma set far too high shows as
a pile-up at the histogram's low edge, where fits are pinned against the `slack`
bound and flagged `AT_BOUND`.

One unit caveat, since two are in play: `sigma` is in **pixels**, while `slack`
is a **multiple of whatever sigma the search is running at** (`sigma_ratio =
fit_sigma / sigma`). It is not a multiple of the initial guess and not absolute
pixels, so changing `sigma` moves the window with it — the Detect tab prints the
resulting px window underneath it for that reason.

`max_step` is in **pixels** too, like `sigma` and like spotsolve takes it.

## Units, and where they come from

Two numbers turn this pipeline's pixels and frames into physics: the pixel size
and the frame interval. Everything physical is those two multiplied through —
`x_um`, `duration_s`, `mean_step_um`, every fitted `D` and `K` — so both are read
off the image file's own metadata and **shown, with their provenance, before
anything runs**: the line above the Detect/Track tabs says e.g.
`from file: 500 frames · 0.1083 µm/px · 0.0302 s/frame`, and its tooltip says
which metadata field each came from. A headless run prints the same line
(`gemscape2 detect-track`), and `manifest.json` records it (`pixel_size_um_source`,
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
The widget's buttons honor it, as does a config's
`pixel_size_um`/`dt_s` per `[[inputs]]` entry for headless runs.

Changing the scale after a detect or link run re-derives what it cheaply can
(per-track metrics, the layers' units) and blocks *Save* until you re-run —
`points.parquet`'s `t`/`y_um`/`se_*_um` columns are derived at detect time, so a
bundle saved across a scale change would be internally inconsistent.

**Exposure time.** A third number, the camera exposure, is read the same way
(`.nd2` capture text, an Imaris `Channel` `ExposureTime` with a unit, the
camera's `ExposureTime` in an Andor Fusion `.ims`'s acquisition protocol, OME
`Plane ExposureTime`) and recorded in the manifest as `exposure_s` /
`exposure_s_source`. Detection and linking never use it; the diffusion
analysis does, because its D likelihood models the motion blur of a
continuous exposure. It is **never defaulted to 0**: 0 means "instantaneous",
and on real data that biases `D` low. (Exposure 0 is also the only case where
the α posterior is available: it has no blur model.) A file that doesn't record it shows `exposure ?` in amber. You type it into "Image metadata" (the exposure box is
not behind the override switch, so supplying it doesn't replace the file's
pixel size or frame interval). The Diffusion panel's own exposure box is
pre-filled from the layer, and it won't run until it has a value.

In results, `napari_gemscape2.units` is the single source of truth for what each
column is measured in: it labels the tracks-pane headers (`D_median [µm²/s]`,
`se_x_max [px]`, `se_x_um_max [µm]`), the filter rows' tooltips, the spatial
map's color scale, every fit readout, and every plot axis — the last in the same
mathtext style diffusionkit's own figures use, so a joint plot and an MSD plot
label the same quantity the same way. A column whose unit isn't registered is
shown bare rather than guessed at.

## Setup

Two ways to install, each its own `uv` environment.

**Standard** — on any machine with [uv](https://docs.astral.sh/uv/), no Rust
toolchain and no other checkouts:

```
git clone https://github.com/delnatan/napari-gemscape2.git
cd napari-gemscape2
uv sync                  # add --extra bayes for the NUTS tab (JAX, NumPyro)
uv run napari
```

`spotsolve` comes from its GitHub release wheel for your platform (macOS arm64
and x86-64, Linux x86-64 and aarch64, Windows x86-64; anywhere else its sdist,
which needs Rust), `diffusionkit` and `qtkit` from git. `uv.lock` pins all of
them; `uv lock --upgrade-package diffusionkit` (or `qtkit`) moves to the latest
commit, and a new spotsolve release means editing the version in
`pyproject.toml`'s URLs.

**Development** — `dev/` is a uv workspace over local checkouts of
`spotsolve`, `diffusionkit` and `qtkit` next to this repo, all editable, with
spotsolve's Rust extension built from source (needs [Rust](https://rustup.rs)):

```
./dev/bootstrap.sh       # clones whichever of the three is missing, then syncs
uv run --project dev napari
uv run --project dev --package spotsolve pytest     # a library's own tests
```

It includes the `[bayes]` extra plus maturin, pytest and ruff. Python edits in
any of the four packages take effect on restart; after changing spotsolve's
Rust code, rebuild with `uv sync --project dev --reinstall-package spotsolve`.

## Usage

Batch from napari: tune one movie in the widgets and save it (*Save
results*, then *Save analysis* in the diffusion widget). Then, with that
movie selected in the experiment list, press **Batch from this movie…**. In
the dialog, tick the other movies (unanalyzed ones are pre-ticked) and the
steps: detect + track, diffusion analysis, or both. The batch runs in the
background, one movie at a time, turning each row green as its bundle is
written. It also writes `results/batch_<template>.toml`, the config the CLI
below reads, so the same batch can be re-run headless.

Headless batch run (see `pyproject.toml`'s `[project.scripts]` entry `gemscape2`):

```
gemscape2 detect-track config.toml   # detect + track every input
gemscape2 diffusion config.toml      # then the diffusion analysis of each bundle
```

Both read the same config. Relative paths in it are read from the config's
own folder, so it runs the same from anywhere.

The easy way to write one is to tune a single movie in the napari widgets,
save it (*Save results*, then *Save analysis*), and name that bundle as the
`template`. `detect-track` then reuses the settings in its `manifest.json`
and `diffusion` reuses those in its `diffusion_summary.json`, with nothing
copied by hand. Any key written under `[params]` or `[diffusion]` overrides
the template's. Regions are per movie, not the template's: an input whose result
folder already holds a painted mask (`labels.tif` + `regions.json`, saved when
you leave the movie in the widget) is restricted to it, and one without is
analyzed over the whole field.

```toml
results_root = "results"
template = "results/beads_dense"   # optional

[[inputs]]
path = "data/beads_timelapse_dense_2.tif"
result_id = "beads_dense_2"   # optional, defaults to the stem
exposure_s = 0.01             # optional; also pixel_size_um, dt_s
```

Without a template, or to override it:

```toml
[params]
# PSF width in px -- required. Read it off the fit_sigma histogram of a few
# detected frames in the napari widget.
sigma = 1.3
# Largest step linked between consecutive frames, px -- required too
# (a template linked before spotsolve 0.2 has none). About 3x the rms step
# of the fastest particles of interest.
max_step = 5.0
min_track_length = 2
# `offset` is the only camera fact spotsolve needs -- noise is measured
# from each frame directly.
camera_kwargs = { offset = 100.0 }
# The multi-emitter detector's one knob: expected false spots per 10^6
# pixels of pure noise. Lower is stricter.
detect_kwargs = { fp_per_mpx = 16.0 }
# The same QC cuts the UI's histogram filters produce, as {column = [lo, hi]}.
# Applied to what the linker sees and to which tracks are kept -- never to
# points.parquet, which holds every detection either way. TOML has no null:
# an open side is -inf or inf. An inline table must fit on one line.
point_filters = { flux = [1200.0, 40000.0], fit_sigma = [1.0, 2.2], se_pos = [-inf, 0.4] }
track_filters = { track_length = [5.0, 1e9] }

[diffusion]
min_frames = 3          # shortest track fitted
alpha = false           # the alpha posterior: needs exposure 0, ~30x slower
# The posterior grids -- diffusionkit's GridPostOptions fields (also alpha_min,
# alpha_max, n_alpha, n_K). D's range is the flat prior's support, so it is
# part of the analysis. Keys left out keep the template's (or diffusionkit's
# defaults, shown).
grid = { D_min_um2_s = 1e-4, D_max_um2_s = 10.0, n_D = 501 }
msd_comparison = false
exposure_s = 0.01       # optional: overrides each bundle's recorded exposure
# Which tracks pass (`passes_filters`) and so make up the ensemble -- the
# diffusion widget's tracks-pane cuts, on any per-track column.
min_track_length = 5
filters = { flux_mean = [800.0, inf], D_median_um2_s = [0.001, inf] }
# The ensemble's deconvolution; keys left out keep the template's (or the defaults).
deconvolution = { iters = 500, smooth = 0.5 }
```

`gemscape2 diffusion` writes the same files as the widget's *Save analysis*
(see "Reproducible results bundles"), so a bundle analyzed headless reopens in
the widget like any other.

Interactive: open napari and use the "Experiment list" dock widget to browse a
folder of raw images and work one image through **Detect** (PSF width and
detection knobs, then filters on what it found, where `fit_sigma` settles the
width) and **Track**
(link, optionally with flux as a second link cue, then filters on the
tracks), pressing *Save results* when the result is worth keeping. The "Diffusion analysis" widget then reads the tracks layer for
diffusionkit's grid posteriors: for each track, the posterior over D (flat
prior in ln D, the exposure's blur modelled) summarized as its median and 5%/95%
quantiles, and — with exposure 0, as an opt-in that costs ~30× more — the same
for the fBm exponent α. The run is diffusionkit's own `gridpost.analyze_tracks`
with the `GridPostOptions` the Posterior tab shows: *D min*/*D max* (µm²/s) and
*D points* set the D grid, whose range is the flat prior's support, and the
folded *α / K grid* section the rest. A track whose posterior is cut by an edge
is marked `D_at_grid_edge` (its numbers move with the edge — near-immobile
tracks reach *D min* this way), and the summary counts them. The grids are
saved with the analysis, so `GridPostOptions(**summary["grid"])` in a
diffusionkit script reproduces the widget's numbers exactly. About 1 s for ~500 tracks without α. The **Ensemble**
plot reads them across the tracks the filters pass, per region class: the
per-track medians, the *deconvolved* distribution of D (each track's own
uncertainty removed; peak positions and masses are robust, widths are
resolution-limited), and the *summed* posterior (one D shared by every track).
The **Posteriors** plot shows every track's posterior as one row of a heat map,
sorted by median, beside the *pooled* posterior (the tracks' posteriors
averaged), the histogram of medians and the deconvolved distribution. The
*Deconvolution* section sets its iterations and smoothing (it spreads D over
the run's D grid); the plots and summary follow without a re-run. MSD fits are a labelled opt-in comparison; the **Track** plot shows the
selected track's posterior; the **Map** tab colors each track's centroid by any
result; the **NUTS** tab fits the selected track's full posterior (needs
`--extra bayes`).

Opening a bundle with a saved analysis — saved from the widget, a batch, or
`gemscape2 diffusion` — restores it: plots, per-track columns, filters and
settings, with no fit re-run. It is restored only if the tracks it was fitted on
are still the bundle's, vertex for vertex (`tracks_sha256` in the summary);
after a re-track or a new pixel size the saved numbers show as text and Run
re-fits. Analyses saved before the fingerprint existed need one re-run.

*Save analysis* writes into the bundle:

```
tracks_summary.csv        one row per track (below)
posterior_D.parquet       every fitted track's log posterior: track_id, D_um2_s, log_posterior
posterior_alpha.parquet   likewise over alpha (exposure 0 runs with α only)
distributions_D.csv       the ensemble on the D grid, long by group ("all", then each
                          region class): summed_log_posterior, summed_posterior,
                          pooled_posterior, deconvolved
distributions_alpha.csv   likewise on the α grid (summed only)
diffusion_summary.json    settings (dt, exposure, grid -- GridPostOptions' fields --,
                          level, deconvolution),
                          population numbers per group, the tracks-pane filters,
                          the fitted tracks' fingerprint (tracks_sha256), and repo SHAs
```

Points and tracks stay parquet (the atomic data); the tables people open in a
spreadsheet are CSV. `tracks_summary.csv` — the table to read an experiment's
tracks from and to pool across experiments (*Export CSV…* writes the same table
anywhere) — has every track as a row, filtered-out and excluded ones included:

- identity: `result_id` (the bundle), `track_id`, `region_class`, `cell`, and `passes_filters`
  (the tracks pane's length and histogram cuts, recorded in
  `diffusion_summary.json`);
- size and position: `track_length`, `duration_s`, `mean_step_um`, mean
  `x_um`/`y_um` (and px);
- the posterior: `posterior_status`, `D_median_um2_s`, `D_low_um2_s`,
  `D_high_um2_s` (5% and 95% quantiles), `D_at_grid_edge` (the posterior is
  cut by the D grid's range), and with α: `alpha_status`,
  `alpha_median`, `alpha_low`, `alpha_high`; `D_msd_um2_s`/`alpha_msd` when the
  MSD comparison ran, `*_nuts*` for tracks fitted with NUTS;
- shape: `radius_of_gyration_um`, `net_displacement_um`, `straightness`,
  `gyration_asymmetry`;
- track quality, as the mean over the track's detections of each detector
  column: `flux_mean`, `bg_mean`, `fit_sigma_mean`, `se_x_mean`/`se_y_mean`,
  ... — in the detector's units (`fit_sigma`, `se_x` in px;
  the `_um` variants in µm).

The tracks pane's histogram filters cover per-track results too — so a
posterior `D` or `alpha` is filterable by the same drag as any other feature.

The widgets tune one image at a time, on purpose. To process a whole folder
without looking at each one, use **Batch from this movie…** in the experiment
list, or `gemscape2 detect-track` and `gemscape2 diffusion` with one tuned movie
as the `template`. To compare experiments, read the saved
bundles' `tracks_summary.csv` into a script and pool them there (each row
carries its `result_id`).
