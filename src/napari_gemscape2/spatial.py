"""Spatial classes of one movie's tracks (`diffusionkit.spatial`), read off the D run's likelihoods.

The D posterior run keeps every fitted track's log-likelihood over D; `analyze_spatial` hands those of the
tracks the filters pass, with their centroids in µm, to diffusionkit's hidden Potts model on the track graph.
Nothing is refitted per track. What comes back:

- classes numbered 1..C in increasing order of D, with no names: which class is which across movies is the
  user's call, from each class's populations;
- `selection`, the score against C, which the user picks the elbow from (`spatial_plot.plot_class_selection`);
  the suggested C (the last whose gain clears 0.02 nats per track) is only the default;
- the pattern test, the neighbour gain against shuffled positions: without a pattern the classes are a plain
  mixture and no class map is drawn;
- per-track columns joined into the tracks pane and `tracks_summary.csv` (`SPATIAL_COLUMNS`);
- groups by the class each track's neighbours give it (`class_groups`), the tracks a class's D and alpha
  populations are deconvolved from, as any other group's.

A movie is fitted on its own ("per cell"). Pooling a class across movies is `pooling`'s `spatial_class` on an
input, after the user has matched the class numbers.

Saved with the analysis: the run's record under "spatial" in `diffusion_summary.json` and each track's class
probabilities and position in `spatial_classes.parquet`; `restore_spatial` rebuilds the run from those and the
saved likelihoods.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np
import polars as pl
from diffusionkit import spatial as dk_spatial
from diffusionkit.gridpost import GridLikelihoods
from diffusionkit.spatial import SpatialAnalysis, SpatialOptions

from napari_gemscape2.diffusion import PosteriorAnalysis

SPATIAL_COLUMNS = (
    "spatial_class", "spatial_p", "spatial_class_neighbours", "spatial_p_neighbours",
    "D_spatial_mean_um2_s", "D_spatial_lo_um2_s", "D_spatial_hi_um2_s",
)
MAX_CLASSES = SpatialOptions().max_classes
N_SHUFFLES = SpatialOptions().n_shuffles
# Class colours, by class number in every figure: none of them the region colours (`track_map`) or viridis's
# ends, and apart from each other in lightness as well as hue.
CLASS_COLORS = ("#2a6fdb", "#e0a100", "#14a37f", "#b0399a", "#6b4f2a", "#00a3c4")
# A class map leaves out points whose k-th nearest track is farther than this many typical spacings.
FIELD_REACH = 2.0
GROUP_PREFIX = "spatial class "


def class_color(c: int) -> str:
    """Class `c`'s colour (1-based)."""
    return CLASS_COLORS[(c - 1) % len(CLASS_COLORS)]


def spatial_options(n_classes: Optional[int] = None, n_shuffles: int = N_SHUFFLES,
                    max_classes: int = MAX_CLASSES) -> SpatialOptions:
    """The run's options: `n_classes` None takes the suggested C. Raises ValueError for bad ones, among them a
    number of shuffles too small for p to reach `p_pattern` (0 skips the test)."""
    options = SpatialOptions(max_classes=max(max_classes, n_classes or 1), n_classes=n_classes, n_shuffles=n_shuffles)
    if n_shuffles and 1 / (1 + n_shuffles) > options.p_pattern:
        need = int(np.ceil(1 / options.p_pattern)) - 1
        raise ValueError(f"{n_shuffles} shuffles cannot show a pattern at p <= {options.p_pattern:g}: "
                         f"use at least {need}, or 0 to skip the test")
    return options


def positions_um(base: pl.DataFrame, pixel_size_um: float) -> pl.DataFrame:
    """Each track's centroid in µm (`track_id`, `x_um`, `y_um`) from the widget's per-track table, whose
    centroids are in pixels (`diffusion.base_track_table`)."""
    return base.select(
        pl.col("track_id").cast(pl.Int64),
        (pl.col("x_px") * pixel_size_um).alias("x_um"),
        (pl.col("y_px") * pixel_size_um).alias("y_um"),
    ).drop_nulls()


def likelihoods_of(analysis: PosteriorAnalysis, ids: Optional[set]) -> GridLikelihoods:
    """The D run's kept likelihoods of the tracks `ids` (None: every fitted track)."""
    rows = analysis.rows_for(ids)
    return GridLikelihoods(analysis.fitted_ids[rows], analysis.fitted_frames[rows], analysis.loglik_D[rows])


def analyze_spatial(
    analysis: PosteriorAnalysis,
    base: pl.DataFrame,
    pixel_size_um: float,
    ids: Optional[set],
    options: SpatialOptions = SpatialOptions(),
    *,
    progress: Optional[Callable[[int, int], None]] = None,
    threads: Optional[int] = None,
) -> SpatialAnalysis:
    """Spatial classes of the D run's tracks `ids` (None: all fitted), at their centroids. The shuffles run on
    a pool of `threads` (default one per CPU); `progress(done, total)` counts fits."""
    lik = likelihoods_of(analysis, ids)
    if len(lik.track_ids) <= options.k:
        raise ValueError(f"spatial classes need more than {options.k} fitted tracks, got {len(lik.track_ids)}")
    pos = positions_um(base, pixel_size_um)
    threads = threads or os.cpu_count() or 1
    with ThreadPoolExecutor(threads) as pool:
        return dk_spatial.analyze_likelihoods(lik, analysis.u_D, pos, options, map_fn=pool.map, progress=progress)


def class_groups(s: SpatialAnalysis) -> dict[str, set]:
    """`{"spatial class c": its track ids}`, each track in the class its neighbours give it
    (`SpatialAnalysis.groups`): the groups a class's populations are read from."""
    return {f"{GROUP_PREFIX}{c}": set(ids.tolist()) for c, ids in s.groups().items()}


def spatial_record(s: SpatialAnalysis) -> dict:
    """`diffusion_summary.json`'s "spatial" entry: `SpatialAnalysis.record` with what it is."""
    return {
        "analysis": "diffusionkit.spatial: classes of D on the k-nearest-neighbour graph of the track centroids "
        "(hidden Potts model over the D run's likelihoods), one movie on its own; classes numbered in increasing "
        "order of D. Pattern test: the neighbour gain against shuffled positions. Groups (spatial_class_neighbours) "
        "assign each track by its neighbours alone.",
        **s.record(),
    }


def restore_spatial(analysis: PosteriorAnalysis, record: dict, probabilities: pl.DataFrame) -> SpatialAnalysis:
    """The saved run (`spatial_record`, `SpatialAnalysis.probabilities`) over the restored D run."""
    return dk_spatial.restore(likelihoods_of(analysis, None), analysis.u_D, record, probabilities)


def spatial_columns(s: SpatialAnalysis) -> pl.DataFrame:
    """The per-track columns the run adds to the tracks pane and `tracks_summary.csv`."""
    return s.columns().with_columns(pl.col("track_id").cast(pl.Int64)).select("track_id", *SPATIAL_COLUMNS)


def describe(s: SpatialAnalysis) -> str:
    """One or two lines on the run: the C used and the pattern test."""
    sel = s.selection
    suggested = sel.filter(pl.col("suggested"))["n_classes"].item()
    chosen = f"{s.n_classes} classes" + (" (suggested)" if s.n_classes == suggested else f" (suggested {suggested})")
    if s.null is None:
        test = "no pattern test with one class" if s.n_classes == 1 else "no pattern test (no shuffles)"
    else:
        verdict = "a spatial pattern" if s.patterned else "no spatial pattern"
        test = (f"{verdict}: neighbour gain {s.null.gain:.1f} nats, p = {s.null.p:.2g} "
                f"against {len(s.null.shuffled_gain)} shuffles")
    counts = " · ".join(f"class {r['spatial_class']}: {r['n_tracks']}" for r in s.report().iter_rows(named=True))
    return f"{len(s.track_ids)} tracks · {chosen} · {test}\ntracks by their neighbours' class: {counts}"
