# napari-gemscape2

Batch orchestration and napari visualization for single-particle tracking, connecting:

- [`spotsolve`](https://github.com/delnatan/spotsolve) — 2D localization after u-track's
  detector, with overlapping spots fitted jointly, plus frame-to-frame linking. Runs in Rust.
- [`diffusionkit`](https://github.com/delnatan/diffusionkit) — per-track grid posteriors over D, the
  population they make (a shared D, a log-normal, and a deconvolved distribution, also by track
  length), classic MSD fits, and per-track NUTS.

## Install

All you need is [uv](https://docs.astral.sh/uv/) and git. uv fetches Python
3.13 itself. You don't need Rust or a checkout of spotsolve or diffusionkit.

```sh
# 1. Install uv (once per machine)
curl -LsSf https://astral.sh/uv/install.sh | sh                                    # macOS / Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows

# 2. Get napari-gemscape2 and its environment
git clone https://github.com/delnatan/napari-gemscape2.git
cd napari-gemscape2
uv sync                 # or `uv sync --extra bayes` for the NUTS tab (JAX, NumPyro)

# 3. Run it, from this folder
uv run napari           # widgets: Plugins → GEMscape2
uv run gemscape2 --help # headless batch runs (see Usage)
```

To update, run `git pull && uv sync`.

You don't need to create a virtual environment first. `uv sync` makes
`.venv` inside the clone. uv chooses any installed Python 3.13 or newer, or
downloads one. To use a particular version, run `uv sync --python 3.14`.

`uv sync` installs exactly what `uv.lock` pins, so every machine gets the same
spotsolve, diffusionkit and qtkit. spotsolve arrives as a prebuilt wheel from
its GitHub release for macOS (Apple silicon and Intel), Linux (x86-64 and
aarch64) and Windows (x86-64). On any other platform, uv builds it from source,
which needs [Rust](https://rustup.rs).

Don't `pip install spotsolve` or `pip install diffusionkit`. On PyPI those names
belong to unrelated packages. The commands above get ours.

**Without cloning.** This puts `napari` and `gemscape2` on your PATH, so they
run from any folder:

```sh
uv tool install "napari-gemscape2 @ git+https://github.com/delnatan/napari-gemscape2" \
    --with-executables-from napari
uv tool upgrade napari-gemscape2   # to update
```

For the NUTS tab, write `"napari-gemscape2[bayes] @ git+…"` instead. This route
ignores `uv.lock`: it resolves everything fresh and takes the latest diffusionkit
and qtkit, so two machines installed on different days can differ. For analyses
you'll publish, use the clone.

## Particle tracking in the dense regime

Both halves of the problem are handled by `spotsolve` rather than tuned around:

- **Detection** fits each candidate spot on its own window with a Poisson
  model, every emitter at its own width. With the default "mixtures" detector a
  window takes as many emitters as the likelihood ratio supports, so
  overlapping spots are resolved instead of being merged into one bright
  centroid; "single" fits one per window, faster for well-separated spots. One
  knob sets the threshold: `fp_per_mpx`, the false spots admitted per 10⁶
  pixels of pure noise (default 16; the Detect tab restates it per frame of the
  image at hand).
- **Linking** holds each track to its own diffusion scale: between consecutive
  frames a link is scored by the step density the track predicts from its own
  steps, against the cost of ending the track, which spotsolve reads from the
  movie. So a fast particle passing a slow or immobile one keeps its identity.
  No step longer than `max_step` (px) is linked. `max_step` is the one setting
  and is not estimated from the movie: about 3× the rms step of the fastest
  particles of interest (default 20 px). A large value costs slow particles
  little. The Track tab restates it in µm and as the fastest D the linker
  models, and the status line after a run reports the linked steps' rms.

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
run measured (PSF sigma, the linked steps' rms, the MSD estimate of D). Under
`packages` it records the `version` and `git_sha` of `spotsolve` and
`napari-gemscape2` at run time. The SHA comes from the source checkout, or else
from the commit recorded when the package was installed from git. It is `null`
for a release wheel (spotsolve's), whose version says which release it is.

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
range), and read the `fit_sigma` histogram on the Filter page: `sigma` is the
in-focus width, the narrowest a spot can be, so take it from the narrow,
in-focus end of the main peak, not its middle. Fits can't go below `sigma`
(`width` starts at 1.0), so run again and the peak should sit just above it. A
bimodal or ragged `fit_sigma` (two focal planes, junk being fitted as signal) is
something you see rather than something a median averages away. A sigma set too
high shows as a pile-up at the histogram's low edge.

One unit caveat, since two are in play: `sigma` is in **pixels**, while `width`
(the reported widths, expert settings) is a **multiple of sigma** (`sigma_ratio
= fit_sigma / sigma`), so changing `sigma` moves the window with it — the Detect
tab prints the resulting px window underneath it.

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
and on real data that biases `D` low.
A file that doesn't record it shows `exposure ?` in amber. You type it into "Image metadata" (the exposure box is
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

## Development setup

To use the package, see [Install](#install). To change spotsolve, diffusionkit
or qtkit along with it, use the development environment. `dev/` is a uv
workspace over local checkouts of those three, placed next to this repo:

```
<parent>/napari-gemscape2/   this repo
<parent>/spotsolve/
<parent>/diffusionkit/
<parent>/qtkit/
```

All four packages are editable, and spotsolve's Rust extension is built from
source, which needs [Rust](https://rustup.rs):

```sh
./dev/bootstrap.sh                                   # clones whichever of the three is missing, then syncs
uv run --project dev napari
uv run --project dev pytest                          # this repo's tests
uv run --project dev --package spotsolve pytest      # a library's own tests, same env
```

The environment includes the `[bayes]` extra plus maturin, pytest and ruff.
It lives in `dev/.venv`, separate from the standard `.venv`. Python edits in
any of the four packages take effect when you restart. After changing
spotsolve's Rust code, rebuild it with
`uv sync --project dev --reinstall-package spotsolve`.

**Moving the standard install's pins.** Run
`uv lock --upgrade-package diffusionkit` (or `qtkit`) to move to that repo's
latest `main`. For a new spotsolve release, change the version in every URL
under `[tool.uv.sources]` in `pyproject.toml`, then run `uv lock`. Either way,
commit `uv.lock`. That file is what users' `uv sync` installs.

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
gemscape2 pool config.toml           # then pool replicates by sample
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
# of the fastest particles of interest; each track is linked at its own
# scale, so a large value costs slow particles little.
max_step = 20.0
min_track_length = 2
# `offset` is the only camera fact spotsolve needs -- noise is measured
# from each frame directly.
camera_kwargs = { offset = 100.0 }
# "mixtures" (default: several emitters per fit window) or "single".
detector = "mixtures"
# The detector's one threshold: expected false spots per 10^6 pixels of
# pure noise. Lower is stricter.
detect_kwargs = { fp_per_mpx = 16.0 }
# The same QC cuts the UI's histogram filters produce, as {column = [lo, hi]}.
# Applied to what the linker sees and to which tracks are kept -- never to
# points.parquet, which holds every detection either way. TOML has no null:
# an open side is -inf or inf. An inline table must fit on one line.
point_filters = { flux = [1200.0, 40000.0], fit_sigma = [1.0, 2.2], se_pos = [-inf, 0.4] }
track_filters = { track_length = [5.0, 1e9] }

[diffusion]
min_frames = 3          # shortest track fitted
# The posterior grid -- diffusionkit's GridPostOptions fields. D's range is the
# flat prior's support, so it is part of the analysis. Keys left out keep the
# template's (or diffusionkit's defaults, shown).
grid = { D_min_um2_s = 1e-5, D_max_um2_s = 10.0, n_D = 601 }
# The per-track MSD fits always run (a comparison: exposure treated as 0):
# D over each track's first msd_max_lag lags, alpha (log-log) over the first
# msd_alpha_max_lag -- both at least 3, each capped at the track's length.
msd_max_lag = 3
msd_alpha_max_lag = 10
# or shares of each track's longest lag, replacing both lag counts:
# msd_lag_fraction = 0.3        # D: the usual 25-40% rule
# msd_alpha_lag_fraction = 0.5  # alpha
exposure_s = 0.01       # optional: overrides each bundle's recorded exposure
# Which tracks pass (`passes_filters`) and so make up the ensemble -- the
# diffusion widget's tracks-pane cuts, on any per-track column.
min_track_length = 5
filters = { flux_mean = [800.0, inf], D_info_bits = [1.0, inf] }
```

The ensemble's deconvolution has no settings: its smoothness is chosen by the
data. A config that still sets `deconvolution` is rejected as an unknown key.

`gemscape2 diffusion` writes the same files as the widget's *Save analysis*
(see "Reproducible results bundles"), so a bundle analyzed headless reopens in
the widget like any other.

### Pooling replicates: `gemscape2 pool`

Pooling makes one population of each sample: the tracks of all its movies
in one fit, their log-likelihoods added -- the ensemble a sample makes when
each movie has only a few tracks (small cells such as yeast). It is read
the three ways a single movie's Ensemble is: the log-normal (median D,
spread σ, mean D), the deconvolved distribution as the check on its shape,
and the shared D. One fit per sample assumes its movies share one
population, so where a sample has several movies each is also read on its
own, and its rows show whether they agree. Samples are not tested against
each other: comparing samples is left to analysis outside this package, with the movie (or cell), not the track, as the replicate,
starting from `pooled_lognormal_draws_D.csv`.

Replicates of one sample are pooled from what `diffusion` saved: each
bundle's likelihoods and the filters it was saved with are read back, so
nothing is refitted and only tracks that passed those filters are pooled.
Give the replicates a shared `sample` (default: the bundle's own name), and
optionally a `[pool]` table:

```toml
[[inputs]]
path = "data/wt_1.tif"
sample = "wt"

[[inputs]]
path = "data/wt_2.tif"
sample = "wt"

[[inputs]]
path = "data/mut_1.tif"
sample = "mut"

[pool]
output = "pooled"            # default: <results_root>/pooled
# The ensemble-averaged MSD is off unless asked for, and its windows are
# explicit: each track's MSD is computed to ensemble_max_lag; D is fitted
# over the averaged curve's first ensemble_n_points lags (3 or more), alpha
# (log-log) over its first ensemble_alpha_points (default: all of it).
ensemble_msd = true
ensemble_max_lag = 10
ensemble_n_points = 3
ensemble_alpha_points = 10
ensemble_offset = "provided"  # or "fit": the intercept of the linear MSD fit (no SDs used)
n_boot = 200                  # bootstrap resamples of tracks
```

It writes, to the output folder:

```
pooled_distributions_D.csv   the D population per sample (by = "sample") and, where a
                             sample has replicates, per bundle (by = "experiment") on the
                             D grid: n_tracks, D_um2_s, lognormal and deconvolved, each
                             with its band (_low, _high)
pooled_populations_D.csv     one row per sample (and per bundle): the log-normal's median
                             D, spread σ and mean D with intervals, lognormal_problem,
                             and the shared D
pooled_lognormal_draws_D.csv posterior draws of the log-normal's (mu_ln_D, sigma_ln_D)
                             per sample and per bundle: the input to comparing samples
                             outside the GUI
pooled_ensemble_msd.csv      the ensemble-averaged MSD per sample and lag (pair-weighted)
pooled_ensemble_msd_fits.csv linear/Brownian D (and the localization SD when the offset
                             is fitted) and the log-log power law of the averaged curve,
                             with bootstrap intervals
pooled_summary.json          which bundles make up each sample, the grid, packages
```

A sample's population weighs every track equally, whichever movie it came
from, and its interval knows nothing of how much movies differ: where they
do (the movie rows disagree), a sample's interval is too narrow for comparing
samples, which is why that comparison treats the movie as the replicate. The
ensemble MSD treats the exposure as 0,
like the widget's MSD comparison (the MSD estimators have no blur model), and
its intervals resample tracks: they do not cover shared drift or
miscalibrated localization errors. The bundles must share one D grid, and a
bundle with no current saved analysis stops the pooling rather than being
skipped: a missing replicate would bias it.

Interactive: open napari and use the "Experiment list" dock widget to browse a
folder of raw images and work one image through **Detect** (PSF width and
detection knobs, then filters on what it found, where `fit_sigma` settles the
width) and **Track**
(link, optionally with flux as a second link cue, then filters on the
tracks), pressing *Save results* when the result is worth keeping. The "Diffusion analysis" widget then reads the tracks layer for
diffusionkit's grid posteriors: for each track, the posterior over D (flat
prior in ln D, the exposure's blur modelled) summarized as its median and 5%/95%
quantiles. D is the one per-track quantity reported: no shape metric (α, a ratio
of D at two timescales, radius of gyration, straightness) can be pinned down by
short tracks. α stays in the labelled MSD comparison and the NUTS tab. The run
is diffusionkit's own `gridpost.analyze_tracks` with the `GridPostOptions` the
Posterior tab shows: *D min*/*D max* (µm²/s) and *D points* set the D grid,
whose range is the flat prior's support (by default 10⁻⁵ to 10 µm²/s, so
tracks that can't be told from still fall into a low tail, read as upper
bounds). A track whose posterior is cut by an edge
is marked in `D_grid_edge` with the end that cuts it (its numbers move with the edge — near-immobile
tracks reach *D min* this way), and the summary counts them. The grid is
saved with the analysis, so `GridPostOptions(**summary["grid"])` in a
diffusionkit script reproduces the widget's numbers exactly. Tracks are fitted on
a thread pool: about 20 s for ~10,000 tracks. The **Ensemble**
plot reads them across the tracks the filters pass, per region class, by
adding the tracks' log-likelihoods under a model of the population -- never by
histogramming their medians (a short track's median sits where the prior puts
it) or averaging their posteriors. The *log-normal* population is the
headline: ln D normal across tracks, its median D and spread σ (in ln D) each
with an interval, every track's own measurement noise in the model rather than
in σ. The *deconvolved* distribution, with its 90% band, is the same tracks
under a smooth density of any shape (its smoothness chosen by Laplace evidence,
so there is nothing to tune -- a peak narrower than the tracks resolve comes out
as wide as that resolution, and the band widens below the localization floor);
two modes there mean the log-normal's numbers describe the wrong shape. The
*shared* D is one D for every track: when the tracks differ it lands near their
mean D with an interval far too narrow, which σ tells you. The **By length**
plot splits the tracks' own posteriors (unpooled) and their posteriors under the
deconvolved distribution (partially pooled) by track length (`gridpost.by_track_length`), stacked
so the groups add up to the whole, beside each length's own distribution. Fast
particles leave the focal depth within a few frames, so short tracks come
mostly from fast particles and long ones from slow particles. Its window counts
*each track once* (the default and the usual per-trajectory reading: the stack
is the deconvolved distribution of D across tracks) or *each detection* (a track
once per frame, so long tracks weigh more: the make-up of the spots seen in
focus at a moment). Neither is the share of particles, since a fast particle can
leave the focus and return as another track.
Every D axis — the ensemble,
the posteriors, the track plot, and the joint plot when an axis is D at one
frame interval — shows the **localization floor** `D_floor_um2_s`, the D at
which a track's motion per frame equals its localization noise, ⟨σ²⟩ /
(dt − exposure/3) from its own SDs, as a dotted line at the tracks' median over
their 10–90% band. It is a scale to read D against, not a cut, and it moves
with the square of any error in the SDs.
The **Posteriors** plot shows every track's posterior as one row of a heat map,
sorted by where its likelihood peaks, beside the population reads of the
Ensemble plot. The plots and summary follow the
filters without a re-run; each group's deconvolution takes a second or two, on
a worker thread, so they update shortly after a change. The **Track** plot shows
the selected track's posterior; the **Map** tab colors each track's localizations
by any per-track value (track length from the start, then any result); the **NUTS** tab fits the selected track's full posterior with the
same exposure (needs `--extra bayes`).

The classical MSD analysis has its own **MSD** tab. It needs only tracks, and it
is a labelled comparison, run with the exposure treated as 0 (diffusionkit's
MSD estimators have no blur model). D and α are fitted over separate windows
everywhere: D over the first, best-measured lags, α by log-log over a wider span,
since a log-log slope needs about a decade of τ to mean much.

- **Run MSD fits** fits each track's time-averaged MSD: D over its first lags
  (default 3) and α over its first 10, or over shares of each track's own lags
  (D the usual 25–40%), adding `D_msd`/`K_msd`/`α_msd` columns. A batch
  always runs them alongside the posteriors.
- **Track** draws the selected track's MSD on linear (D) and log-log (α) axes,
  each fit over its window, with a linear fit with a free intercept whose
  localization SD is a check on the SDs. No error bars: one track's lags share
  their displacements.
- **Ensemble** averages the MSDs of the tracks the filters pass per region class,
  each squared displacement counting once, and fits D over the curve's first
  30% and α over all of it (both adjustable), with ±1 SEM over tracks and fit
  intervals from resampling whole tracks. The offset is the SDs' or the linear
  fit's intercept. The late lags rest on the long tracks alone (the α panel
  says how many reach its last lag; the Map tab colors tracks by length). A mix
  of slow and fast tracks averages to one curve, which the deconvolved
  distribution would show as two.

Opening a bundle with a saved analysis — saved from the widget, a batch, or
`gemscape2 diffusion` — restores it: plots, per-track columns, filters and
settings, with no fit re-run. It is restored only if the tracks it was fitted on
are still the bundle's, vertex for vertex (`tracks_sha256` in the summary);
after a re-track or a new pixel size the saved numbers show as text and Run
re-fits. Analyses saved before the fingerprint existed, with the α
posterior of earlier versions, or as `posterior_D.parquet` (before the
likelihoods were saved as `loglik_D.parquet`), need one re-run.

*Save analysis* writes into the bundle:

```
tracks_summary.csv        one row per track (below)
loglik_D.parquet          every fitted track's log-likelihood over D: track_id, D_um2_s,
                          loglik (normalized, so also its flat-prior posterior); what
                          every population read is built from
distributions_D.csv       the population on the D grid, long by group ("all", then each
                          region class): shared_loglik (the tracks' log-likelihoods
                          summed, 0 at the peak: one D for every track -- the summary's
                          shared_D_* interval is read below the grid step), lognormal
                          (the log-normal population at its posterior mode) and
                          deconvolved, each with its pointwise band at the credible
                          level (_low, _high)
distributions_D_by_length.csv
                          the same split by track length, long by group, weight
                          (tracks or detections) and length (length_min, length_max
                          in frames; length_max empty for the open last group):
                          n_tracks, n_detections, unpooled (the tracks' own
                          posteriors) and partially_pooled (under the deconvolved
                          distribution, with _low/_high), each the length's share of
                          all the tracks or detections, so a weight's rows sum to 1
diffusion_summary.json    settings (dt, exposure, grid -- GridPostOptions' fields --,
                          level), population numbers per group (lognormal_D_median_*,
                          lognormal_sigma_ln_D*, lognormal_D_mean_*, shared_D_*,
                          deconvolved_* with its evidence-chosen smoothness,
                          deconvolved_D_lambda),
                          the tracks-pane filters,
                          the fitted tracks' fingerprint (tracks_sha256), and packages
                          (diffusionkit's and napari-gemscape2's version and git SHA)
```

Points and tracks stay parquet (the atomic data); the tables people open in a
spreadsheet are CSV. `tracks_summary.csv` — the table to read an experiment's
tracks from and to pool across experiments (*Export CSV…* writes the same table
anywhere) — has every track as a row, filtered-out and excluded ones included.
Point estimates are posterior **medians**, and `_low`/`_high` are the 5% and
95% quantiles (a 90% credible interval). Empty cells mean "not computed for
this track": it was excluded, too short, or that analysis was off.

| Column | Unit | Meaning |
|---|---|---|
| **Identity** | | |
| `result_id` | | The bundle (movie) the track came from; lets you concatenate bundles |
| `track_id` | | Track number within the bundle |
| `region_class` | | Name of the painted region the track is in (only when regions were used) |
| `passes_filters` | | True if the track passes the tracks-pane cuts (recorded in `diffusion_summary.json`) |
| **Size and position** | | |
| `track_length` | points | Number of localizations in the track |
| `duration_s` | s | Time from first to last frame |
| `mean_step_um` | µm | Mean frame-to-frame step length |
| `x_um`, `y_um` / `x_px`, `y_px` | µm / px | Track centroid |
| **D (one frame interval)** | | |
| `posterior_status` | | `ok`, `excluded` (shorter than `min_frames`), or `invalid_input` |
| `D_mean_um2_s` | µm²/s | The track's one point estimate: its posterior mean E[D], Brownian model, exposure blur included. The summary a grid edge barely moves; under the flat prior it reads high for short tracks (about 1/(n − 2) for n frames), so don't average it — the population numbers are the average |
| `D_low_um2_s`, `D_high_um2_s` | µm²/s | 90% credible interval on D |
| `D_grid_edge` | | Which end of the D grid cuts the posterior: `low` (the data only bound D from above — read the row as upper bounds), `high` (lower bounds), `both` (no information), empty when it isn't cut |
| `D_partially_pooled_um2_s` | µm²/s | The track's E[D] with its population (the log-normal of its region class, or of all passing tracks) as the prior: ignores the grid's edges and averages to the population's mean, but moves with the population, so it is in the saved file only, not the filter panel. Empty for tracks outside the filters |
| `D_info_bits` | bits | How much the track narrowed D from the flat prior. Only comparable on the same grid |
| `D_floor_um2_s` | µm²/s | Localization floor: the D at which motion per frame equals localization noise, ⟨σ²⟩ / (dt − exposure/3) from the track's SDs. A reference scale, not a threshold |
| **Optional fits** | | |
| `D_msd_um2_s`, `K_msd_um2_s_alpha`, `alpha_msd` | µm²/s, µm²/s^α, – | Classical MSD fits (D over D's window, α by log-log over its own), when they ran; no uncertainties |
| `*_nuts*` | | Full-posterior NUTS fit, for tracks fitted in the NUTS tab |
| **Detection quality** (mean over the track's detections, spotsolve's columns) | | |
| `sigma` | px | Detector's reference PSF width (the same for every track) |
| `t_mean` | s | Mean detection time |
| `x_um_mean`, `y_um_mean` | µm | Mean position (same as the centroid) |
| `se_x_mean`, `se_y_mean`, `se_pos_mean` | px | Localization standard error (CRLB) per axis, and their hypot |
| `se_x_um_mean`, `se_y_um_mean` | µm | Same, in µm |
| `flux_mean`, `se_flux_mean` | ADU | Background-free integrated brightness and its standard error |
| `flux_snr_mean` | | flux / se_flux |
| `peak_mean` | ADU | On-centre model pixel value; compare with `bg_mean` |
| `bg_mean` | ADU/px | Fitted local background |
| `fit_sigma_mean`, `sigma_se_mean` | px | Fitted spot width and its standard error |
| `sigma_ratio_mean` | | fit_sigma / sigma; well above 1 flags blur or overlapping spots |
| `z_mean` | | sqrt(2 × likelihood ratio) of each detection: evidence, not precision |

The tracks pane's histogram filters cover per-track results too — so a
posterior `D` is filterable by the same drag as any other feature.

The widgets tune one image at a time, on purpose. To process a whole folder
without looking at each one, use **Batch from this movie…** in the experiment
list, or `gemscape2 detect-track` and `gemscape2 diffusion` with one tuned movie
as the `template`. To compare experiments, read the saved
bundles' `tracks_summary.csv` into a script and pool them there (each row
carries its `result_id`).
