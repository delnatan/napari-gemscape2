# napari-gemscape2

Batch orchestration and napari visualization for single-particle tracking, connecting:

- [`spotsolve`](https://github.com/delnatan/spotsolve) — 2D localization after u-track's
  detector, with overlapping spots fitted jointly, plus frame-to-frame linking. Runs in Rust.
- [`diffusionkit`](https://github.com/delnatan/diffusionkit) — per-track grid posteriors over D, the
  population they make (a log-normal and a deconvolved distribution), per-track likelihoods of α
  and D_ref under fractional Brownian motion and their population, spatial classes of D, and
  classic MSD fits as the comparison.

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
uv sync

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

This route ignores `uv.lock`: it resolves everything fresh and takes the latest diffusionkit
and qtkit, so two machines installed on different days can differ. For analyses
you'll publish, use the clone.

## Particle tracking in the dense regime

Both halves of the problem are handled by `spotsolve` rather than tuned around:

- **Detection** fits each candidate spot on its own window, every emitter at
  its own width, with each pixel's noise from the camera's noise curve (the
  camera's gain, given or estimated from the movie, and its read noise,
  measured from the background). With the default
  "mixtures" detector a window takes as many emitters as the likelihood ratio
  supports, so overlapping spots are resolved instead of being merged into one
  bright centroid; "single" fits one per window, faster for well-separated
  spots. One knob sets the threshold: `p_real`, the least probability that a
  reported spot is real rather than noise, fitted to each movie's detection
  scores (default 0.9). For exposures long enough that moving emitters streak,
  turn on "motion blur" (mixtures only): a close pair is then one elongated
  spot, flagged `ELONGATED`, unless two round ones fit better; off, close pairs
  are resolved.
- **Linking** holds each track to its own diffusion scale: between consecutive
  frames a link is scored by the step density the track predicts from its own
  steps, against the cost of ending the track, which spotsolve reads from the
  movie. So a fast particle passing a slow or immobile one keeps its identity.
  The search radius (px) is read from the movie by default: next-frame
  against own-frame pair distances give the fastest diffusion it holds, and
  the radius is 1.5× three of its rms steps. It is soft, so a larger value
  changes the links little; set it on the Track tab to override ("auto" reads
  it). `switch cost` (default 2.5) is the one trade-off: the log odds a link
  must reach against ending the track. A wrong link swaps two identities and
  costs exp(2.5) ≈ 12 missed links; lower it for more links and more swaps.
  Every link gets `link_p`, the probability that it is right (null where a
  track starts; on simulated movies its mean is the share of links that are
  right to within a point). The status line after a run reports the radius
  used, the linked steps' rms and the share of links expected right, and the
  Filter page can cut tracks on `min_link_p`, their weakest link.

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
    labels.tif             (optional — the mask the results were made with)
    regions.json           (alongside labels.tif)
    mask.tif               (optional — the movie's mask as painted now)
    mask.json              (its region names; empty if the mask was erased)
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

A movie's mask is saved to its result folder as you paint it (`mask.tif` +
`mask.json`, a second after the last edit), whether or not the movie has
results yet, and its row in the list is marked with a mask glyph. So you can
paint masks across a whole folder first, then batch it: masked movies are
restricted to their regions, and the rest are analyzed over the whole field.
Results keep the mask they were made with (`labels.tif` + `regions.json`); when
the mask is edited after that -- repainted, renamed, or erased -- the glyph turns
amber, and the batch dialog ticks that movie to be run again. So you can also
batch first, then mask the movies that need it and batch again.

Reopening a row that already has a bundle resumes it: the saved points are the
session's detections, so Track links them without re-running detect, and the
movie's mask comes back as the picked regions layer.

Nothing a QC decision rejects is deleted. spotsolve reports every fit with its
`FitFlag`s; the Track tab chooses which flags keep a fit away from the linker
(none by default), and fits without a finite position error are never linked.
The histogram filters described below are recorded as ranges rather than
applied to `points.parquet` — so how much of a movie was junk
stays an auditable fact about the run instead of a silent subtraction. What those
judgements change is what the *linker* sees and which tracks the bundle keeps.

`manifest.json` records the source image path, the camera and detection
parameters, both filter specs, the linking settings (`search_radius_px`, where it came
from, and `switch_cost`) and what the
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

The camera's **gain** sits beside the offset on the Detect page: the pixel
variance per ADU of light, which scales every detection test and localization
error, so it should be right to about 10%. Type it from the camera's
calibration, or press *Estimate* to measure it from the current movie's bright
spots (its frame range and regions). The box reads "auto" until it holds a
number; a run then estimates the gain itself, and `manifest.json` records the
value used and whether it was given or estimated.

One unit caveat, since two are in play: `sigma` is in **pixels**, while `width`
(the reported widths, expert settings) is a **multiple of sigma** (`sigma_ratio
= fit_sigma / sigma`), so changing `sigma` moves the window with it — the Detect
tab prints the resulting px window underneath it.

`search_radius` is in **pixels** too, like `sigma` and like spotsolve takes it.

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

The environment includes maturin, pytest and ruff.
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
folder already holds a painted mask (`mask.tif` + `mask.json`, saved as you
paint it in the widget) is restricted to it, and one without is analyzed over
the whole field.

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
# Optional: how far the linker looks between frames, px. Left out, each
# movie reads its own (a template passes on only a radius someone set).
# search_radius = 20.0
# Optional: the log odds a link must reach against ending the track
# (default 2.5). Lower links more and swaps identities more.
switch_cost = 2.5
min_track_length = 2
# The camera: `offset` (ADU) and `gain`, the pixel variance per ADU of
# light (conversion gain times any excess-noise factor; 2 for an EMCCD).
# Leave `gain` out to estimate it from each movie; the read noise is
# measured from the background.
camera_kwargs = { offset = 100.0, gain = 2.7 }
# "mixtures" (default: several emitters per fit window) or "single".
detector = "mixtures"
# The detector's one threshold: the least probability that a reported spot
# is real. Higher is stricter. `motion_blur` (mixtures only) reports a close
# pair as one elongated spot unless two fit better.
detect_kwargs = { p_real = 0.9, motion_blur = false }
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
# Also run the α · D (fBm) fits, D_ref at this lag in seconds. A
# template saved after an α · D run sets it. D_ref is not a Brownian D.
alpha_D_tau_ref_s = 0.1
# and their grids (any key left out keeps the default, shown): α points per
# track, then the population's cells: α rows per cell, D cell width in
# decades, likelihood points per D cell.
alpha_D_grid = { n_alpha = 99, alpha_group = 3, d_log10_D = 0.05, cell_points = 4 }
# Also find spatial classes of D among the tracks that pass (a template saved
# with them sets it). The number of classes is each movie's suggested one
# unless set here; the pattern test shuffles the positions spatial_shuffles times.
spatial = true
# spatial_n_classes = 2
spatial_shuffles = 99
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

# One spatial class of a movie, by the class its tracks' neighbours give them
# (see "Spatial classes"). Classes are numbered per movie, so match them
# yourself; list a movie once per class to pool its classes into different samples.
[[inputs]]
path = "data/wt_1.tif"
sample = "wt inner"
spatial_class = 1

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
diffusionkit's grid likelihoods. Above its tabs sit the two run inputs every
run shares: the camera **exposure** (pre-filled from the layer; the runs
stay off until it has a value) and **min points**, the shortest track fitted.

The **D** tab is the standard quantification (*Run likelihoods*): for each
track, the Brownian likelihood over D (the exposure's blur modelled), which
normalized is its posterior under a flat prior in ln D, summarized as its mean
E[D] and 5%/95% quantiles. The motion's shape (α) is a separate
fit, on the **α · D** tab (below). The run
is diffusionkit's own `gridpost.analyze_tracks` with the `GridPostOptions` the
D tab shows: *D min*/*D max* (µm²/s) and *D points* set the D grid,
whose range is the flat prior's support (by default 10⁻⁵ to 10 µm²/s, so
tracks that can't be told from still fall into a low tail, read as upper
bounds). A track whose likelihood is cut by an edge
is marked in `D_grid_edge` with the end that cuts it (its numbers move with the edge — near-immobile
tracks reach *D min* this way), and the summary counts them. The grid is
saved with the analysis, so `GridPostOptions(**summary["grid"])` in a
diffusionkit script reproduces the widget's numbers exactly. Tracks are fitted on
a thread pool: about 20 s for ~10,000 tracks. The **Ensemble**
plot reads them across the tracks the filters pass, per region class, by
adding the tracks' log-likelihoods under a model of the population -- never by
histogramming their medians (a short track's median sits where the prior puts
it) or averaging their normalized likelihoods. The *log-normal* population is the
headline: ln D normal across tracks, its median D and spread σ (in ln D) each
with an interval, every track's own measurement noise in the model rather than
in σ. The *deconvolved* distribution, with its 90% band, is the same tracks
under a smooth density of any shape (its smoothness chosen by Laplace evidence,
so there is nothing to tune -- a peak narrower than the tracks resolve comes out
as wide as that resolution, and the band widens below the localization floor);
two modes there mean the log-normal's numbers describe the wrong shape. The
*shared* D is one D for every track: when the tracks differ it lands near their
mean D with an interval far too narrow, which σ tells you. The **By length**
plot deconvolves the population within each track-length group (3-4, 5-9,
10-19, 20+ frames; `gridpost.by_track_length`), stacked by the groups' shares
so they add up to a population, beside each length's own population. A group
with few tracks is mostly the deconvolution's smoothness prior. Fast
particles leave the focal depth within a few frames, so short tracks come
mostly from fast particles and long ones from slow particles. Its window shares
the groups by *tracks* (the default and the usual per-trajectory reading) or by
*detections* (a track once per frame, so long tracks weigh more: the make-up of
the spots seen in focus at a moment); within a group each track counts once. Neither is the share of particles, since a fast particle can
leave the focus and return as another track.
Every D axis — the ensemble,
the per-track likelihoods, the track plot, and the joint plot when an axis is D at one
frame interval — shows the **localization floor** `D_floor_um2_s`, the D at
which a track's motion per frame equals its localization noise, ⟨σ²⟩ /
(dt − exposure/3) from its own SDs, as a thin dotted line at the tracks'
median. It is not a resolution limit: information on D accumulates over a
track's steps, and over the tracks a population pools, so D below the floor is
still resolved. It marks accuracy instead: below it most of each step's
variance is localization noise being subtracted, so D leans on the SDs being
calibrated. It moves with the square of any error in them, a bias that more
frames do not average away.
The plots and summary follow the
filters without a re-run; each group's deconvolution takes a second or two, on
a worker thread, so they update shortly after a change. The **Track** plot shows
the selected track's posterior; the **Map** tab colors each track's localizations
by any per-track value (track length from the start, then any result).

The **α · D** tab runs a second per-track fit, diffusionkit's `fbmpost`: the
anomalous exponent α and an amplitude under fractional Brownian motion, with
the same exposure blur and localization SDs (at α = 1 its likelihood is the D
tab's). It has no prior. For each α on a grid over 0.02–1.98 the amplitude is
set to its best value, which gives the track's **profile likelihood** of α;
α̂ is its peak and the interval is every α within 1.92 log-likelihood units
of it (a 95% likelihood-ratio interval). The same profile tests α = 1:
`alpha_p_brownian` is that test's p-value, so a small one says the track is
not Brownian. A track that cannot tell α values apart (motion buried in the
localization noise) has no α̂, and its interval spans the whole grid.

- **Run α · D fits** adds the `alpha_*` and `D_ref_*` columns (about 6 ms per
  track of 10 frames, 40 ms at 40 frames), so the Map tab can color tracks by
  α̂ and the filters can cut on `alpha_p_brownian` or the interval's width.
  The summary counts the tracks whose α is identified, bounded on both sides,
  and not Brownian, and gives the **shared α**: one α for every track the
  filters pass, from their profiles summed, with its interval and its test of
  α = 1, per region class.
- **Grid** (collapsed above Run) sets what the likelihood is evaluated on: the
  α points per track (default 99, steps of 0.02), and the population's cells,
  α points per cell (3), D cell width (0.05 decades of D at the frame
  interval) and likelihood points per D cell (4). A run with the population
  costs mostly the cells, in proportion to points per cell over cell width, so
  coarser cells or fewer points run faster at the cost of blurring narrow
  populations. The grid is saved with the run (`"grid"` in the summary) and a
  batch template reruns it (`alpha_D_grid`).
- **α–D** draws every fitted track's α̂ against its D_ref with likelihood-ratio
  intervals on both, one panel per region class. Tracks whose α interval spans
  the grid are grey; tracks that only bound D_ref from above are grey "<"
  marks at their upper bound.
- **Track** draws the selected track's relative likelihood over (α, D_ref),
  the 95% region outlined, with the profile of α and of D_ref beside it, each
  crossing the threshold where the interval ends.
- **Population** draws the distribution of α and D_ref across the tracks the
  filters pass, one row per region class, with the shared α marked (needs at
  least 20 fitted tracks; ~10 s per thousand tracks). Its numbers (P(α < 1),
  the mean and SD of α, each with an interval) join the tab's summary, and
  *Save analysis* writes them to `diffusion_summary.json` with the joint
  likelihoods in `loglik_alpha_D.parquet` and the profiles in
  `profile_alpha.parquet`.

> **D_ref is not a Brownian D.** It is the fBm amplitude expressed as
> D_ref = MSD₁D(τ_ref) / (2 τ_ref), the true motion's apparent D at one lag
> τ_ref (default 0.1 s, set on the tab). It has the units of D only so the
> scatter has a readable axis. It equals a Brownian D only at α = 1; otherwise
> it changes with τ_ref (at α = 0.3, going from 0.1 s to 1 s divides it by 5),
> so compare D_ref only at the same τ_ref. Below its floor
> (`D_ref_floor_um2_s`) read it as an upper bound; `D_ref_low_um2_s` is 0 when
> the data do not rule out no motion. Never compare it with `D_mean_um2_s` or
> an MSD D, and never pair `D_mean_um2_s` with this tab's α. For how fast
> tracks move, and for populations of D, use the D tab.

**α near 0 is a reading, not a failure.** Motion confined to a region the
particle crosses within a frame looks, frame to frame, like a fixed position
plus noise, and fBm reads that as α at the grid's low end. In simulation,
confined tracks whose relaxation time was 0.1–1 frame gave a median α̂ of
0.02 at 30 and at 100 frames. A prior that ruled out α near 0 would hide
those tracks instead of flagging them, which is why the tab has none.

A track's α interval says what that track alone rules out. Two reads combine
tracks. The **shared α** asks which single α fits them all: it adds their
profiles, which is valid because each profile is a likelihood. The
**population** asks how α and D_ref are spread across tracks: diffusionkit's
2D deconvolution of the tracks' joint (α, D_ref) likelihoods, the way the D
tab's deconvolved distribution comes from the D likelihoods. When the shared
α's interval is narrow but the population is wide, the tracks differ. Below
the D_ref floor the tracks barely tell D_ref values apart, and α there is
mostly the population's smoothness carrying α from above, so the plot and the
summary also give α above the floor and the share of the population below it.

### Spatial classes

The **Spatial** tab asks whether tracks of similar D sit
together in the cell, and if so, where. It reads the D run's likelihoods of the
tracks the filters pass and their centroids (diffusionkit's `spatial`): a few
classes of D, and a prior that neighbouring tracks (each track's 8 nearest)
tend to share a class. Nothing is refitted per track. Classes are numbered 1, 2,
… in increasing order of D and have no names: what a class is, and which class
of another movie it matches, is yours to judge from its populations.

- **Run** fits 1 to 4 classes. *classes* picks how many to use: by default the
  suggested number, the last whose gain clears 0.02 nats per track, counting up.
  **Number of classes** plots the score against the number of classes, with
  that threshold, so the elbow can be read and picked instead.
- The run tests for a spatial pattern: the tracks' positions are shuffled 99
  times and refitted, and the neighbours' gain on the real arrangement is
  compared with theirs (p down to 0.01). Without a pattern the classes are an
  ordinary mixture of D, and **Class map** draws only the tracks, no map.
- **Class map** draws, with a pattern, each class's places: the colour of the
  most probable class at each point, paler where less sure, blank more than two
  track spacings from any track, with each track's centroid in its class's
  colour. The Map tab can also color tracks by `spatial_class`, in the same
  colours.
- **Class populations** draws each class's D population, deconvolved from its
  tracks' likelihoods with its band, all classes on one axis, and its α
  population when there is an α · D run. A track belongs to the class its
  neighbours give it (`spatial_class_neighbours`), its own data left out, so a
  class's α is not selected by the same track's errors in D.

The run adds `spatial_class`, `spatial_p` (its probability),
`spatial_class_neighbours`, `spatial_p_neighbours` and `D_spatial_*` (the
track's D with its class as the prior: it borrows from the class, so read
`D_mean_um2_s` for the track's own) to the tracks pane. *Save analysis* writes
the run to `diffusion_summary.json` (`"spatial"`) and `spatial_classes.parquet`,
and, when there is a pattern, the class map pixel by pixel on the movie's frame
(`spatial_class_map.tif`, `spatial_class_probability.tif`) for figures of your own,
each class's D and α populations to `populations.csv`, `distributions_D.csv` and
`distributions_alpha.csv` (group `spatial class c`), and the bundle reopens with it. `gemscape2 diffusion` runs it with `spatial = true`, and
`gemscape2 pool` pools a class across movies with `spatial_class` on an input.

The classical MSD analysis has its own **MSD** tab. It needs only tracks, and it
is a labelled comparison, run with the exposure treated as 0 (diffusionkit's
MSD estimators have no blur model). D and α are fitted over separate windows
everywhere: D over the first, best-measured lags, α by log-log over a wider span,
since a log-log slope needs about a decade of τ to mean much.

- **Run MSD fits** fits each track's time-averaged MSD: D over its first lags
  (default 3) and α over its first 10, or over shares of each track's own lags
  (D the usual 25–40%), adding `D_msd`/`K_msd`/`α_msd` columns. A batch
  always runs them alongside the D likelihoods.
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
re-fits. The D and α · D runs are restored independently, each by its own
fingerprint. Analyses saved before the fingerprint existed, with the α
posterior of earlier versions (before the profile fits), or as `posterior_D.parquet` (before the
likelihoods were saved as `loglik_D.parquet`), need one re-run.

*Save analysis* writes into the bundle:

```
tracks_summary.csv        one row per track (below)
loglik_D.parquet          every fitted track's log-likelihood over D: track_id, D_um2_s,
                          loglik (normalized, so also its flat-prior posterior); what
                          every population read is built from
loglik_alpha_D.parquet    after an α · D run, every fitted track's likelihood over
                          (α, log10 D_ref) cells: track_id, n_frames, D_ref_floor_um2_s,
                          alpha, log10_D_ref, loglik (0 at each track's maximum); what
                          the α population is built from
profile_alpha.parquet     after an α · D run, every fitted track's profile
                          log-likelihood of α: track_id, n_frames, alpha, loglik
                          (0 at each track's maximum); what the shared α sums
spatial_classes.parquet   after a spatial-classes run, each of its tracks' class
                          probabilities and centroid: track_id, x_um, y_um, q_1..q_C;
                          with "spatial" in the summary, what the run is restored from
spatial_class_map.tif     with a spatial pattern, the class map on the frame's pixels
                          (the grid of labels.tif): uint8 (H, W), each pixel's most
                          probable class 1..C, 0 where no track is within 2 typical
                          spacings -- what the "Class map" figure shades
spatial_class_probability.tif  float32 (C, H, W): each class's probability at each
                          pixel, NaN where the map is blank
populations.csv           one row per group of tracks ("all" -- the tracks the filters
                          pass --, then each region class, then each spatial class):
                          the D population (lognormal_D_median_um2_s, lognormal_sigma_ln_D,
                          lognormal_D_mean_um2_s, each with _low/_high; the deconvolved
                          distribution's mode and median; the median localization floor)
                          and, after an α · D run, the α population (alpha_mean, alpha_sd,
                          p_alpha_below_1, each with _low/_high; the share below the D_ref
                          floor; shared_alpha with its interval and p_brownian). The
                          table to compare movies from (see "Reading results in a script")
distributions_D.csv       the D populations on the D grid, long by group (the groups of
                          populations.csv): lognormal (the log-normal population at its
                          posterior mode) and deconvolved, each with its pointwise band
                          at the credible level (_low, _high); each sums to 1 over the grid
distributions_alpha.csv   after an α · D run, the α populations on the α grid, long by
                          group: deconvolved (α's margin over D_ref, summing to 1) with
                          its band (_low, _high), and deconvolved_above_floor (the
                          population above the D_ref floor alone)
diffusion_summary.json    settings (dt, exposure, grid -- GridPostOptions' fields --,
                          level), population numbers per group (lognormal_D_median_*,
                          lognormal_sigma_ln_D*, lognormal_D_mean_*, shared_D_*,
                          deconvolved_* with its evidence-chosen smoothness,
                          deconvolved_D_lambda),
                          the tracks-pane filters,
                          the fitted tracks' fingerprint (tracks_sha256), and packages
                          (diffusionkit's and napari-gemscape2's version and git SHA);
                          after an α · D run, an "alpha_D" entry: its settings (tau_ref_s,
                          the alpha grid, "grid", lr_level), counts, fingerprint,
                          "shared_alpha" (alpha, low, high, p_brownian) and
                          "population" (P(α < 1), the mean and SD of α, the α-D_ref
                          correlation and the share below the D_ref floor, each
                          [median, low, high]; the same above the floor; overall and
                          per region class); its per-track numbers are columns of
                          tracks_summary.csv
```

Points and tracks stay parquet (the atomic data); the tables people open in a
spreadsheet are CSV. `tracks_summary.csv` — the table to read an experiment's
tracks from and to pool across experiments (*Export CSV…* writes the same table
anywhere) — has every track as a row, filtered-out and excluded ones included.
Point estimates are posterior **means**, and `_low`/`_high` are the 5% and
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
| `D_info_bits` | bits | How much the track narrowed D from the flat prior. Only comparable on the same grid |
| `D_floor_um2_s` | µm²/s | Localization floor: the D at which motion per frame equals localization noise, ⟨σ²⟩ / (dt − exposure/3) from the track's SDs. Not a resolution limit or a threshold: below it, D depends on the SDs being calibrated |
| **α and D_ref (fBm), after an α · D run** | | |
| `fbm_status` | | `ok`, `excluded` (shorter than `min_frames`), or `invalid_input` |
| `alpha_hat`, `alpha_low`, `alpha_high` | | α's maximum-likelihood value (amplitude profiled out; empty when the track does not identify α) and its 95% likelihood-ratio interval on 0.02–1.98 |
| `alpha_edge` | | Which end of the α grid the interval reaches: `low`, `high`, `both` (α not bounded), empty when it reaches neither |
| `alpha_dll_brownian`, `alpha_p_brownian` | nats, – | The likelihood-ratio test of α = 1: the profile's drop from α̂ to 1 and its p-value (χ²₁ of twice the drop). A small p-value says the track is not Brownian |
| `D_ref_hat_um2_s`, `D_ref_low_um2_s`, `D_ref_high_um2_s` | µm²/s | D_ref = MSD₁D(τ_ref) / (2 τ_ref) at the run's τ_ref (in `diffusion_summary.json`): at the joint maximum, with the 95% likelihood-ratio interval over all α. A low of 0 means only an upper bound. **Not a Brownian D** (see above) |
| `log10_D_ref_hat` | | log10 of `D_ref_hat_um2_s` |
| `D_ref_floor_um2_s` | µm²/s | The D_ref at which the motion's MSD at τ_ref equals the localization noise's, ⟨σ²⟩ / τ_ref |
| **Optional fits** | | |
| `D_msd_um2_s`, `K_msd_um2_s_alpha`, `alpha_msd` | µm²/s, µm²/s^α, – | Classical MSD fits (D over D's window, α by log-log over its own), when they ran; no uncertainties |
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
as the `template`. To compare experiments, read the saved bundles in a script
(below).

### Screenshots and frames for movies

napari's own **File → Save Screenshot** looks duller than the window on a Mac.
The canvas is drawn without colour matching, so on a Display P3 screen (any
recent Mac) its colours are P3's, but the PNG napari writes has no colour
profile and every viewer reads it as the narrower sRGB.

**Plugins → GEMscape2 → Screenshots and frames** saves the canvas with the
profile written into the PNG instead (Display P3 by default on macOS, sRGB
elsewhere). **Screenshot…** saves the current view. **Export frames…** steps a
slider (time, by default) through a range and saves `frame_00000.png`,
`frame_00001.png`, … into a folder, together with an `ffmpeg.txt` that holds
the ffmpeg commands for an MP4 and a GIF. The MP4 command tags the movie's
colour space, because ffmpeg ignores the PNGs' profile. A GIF is always shown
as sRGB, so export its frames with the colour set to sRGB.

### Reading results in a script

`read_results` stacks the saved analyses of every bundle under a folder (searched
recursively, so a tree of experiment folders each with its own `results/` reads in
one call) into polars tables, each row carrying its bundle's `result_id`:

```python
from napari_gemscape2 import read_results
import matplotlib.pyplot as plt

res = read_results("~/data/2026-10-16/results")
res.populations          # populations.csv: one row per (movie, group)
res.tracks               # tracks_summary.csv: one row per track
res.distributions_D      # the D populations' curves, by movie and group
res.distributions_alpha  # the α populations' curves
res.summaries            # {result_id: diffusion_summary.json}

# One point per movie: its population's median D and mean α, with their intervals.
pops = res.populations.filter(group="all").to_pandas()
plt.errorbar(pops.lognormal_D_median_um2_s, pops.alpha_mean,
             xerr=[pops.lognormal_D_median_um2_s - pops.lognormal_D_median_low_um2_s,
                   pops.lognormal_D_median_high_um2_s - pops.lognormal_D_median_um2_s],
             yerr=[pops.alpha_mean - pops.alpha_mean_low, pops.alpha_mean_high - pops.alpha_mean], fmt="o")
plt.xscale("log")

# Each movie's deconvolved D population.
for rid, curve in res.distributions_D.filter(group="all").group_by("result_id"):
    plt.semilogx(curve["D_um2_s"], curve["deconvolved"], label=rid[0])
```

Nothing is refitted: these are the numbers the widget or `gemscape2 diffusion`
saved, over the tracks that passed the filters saved with them. Map the movies to
conditions yourself (from `result_id` or a table of your own); when comparing
conditions, the movie, not the track, is the replicate. To combine a sample's
movies into one population first (each with only a few tracks), use `gemscape2
pool`.
