"""Headless diffusion analysis of a saved results bundle -- what the
"Diffusion analysis" widget's Posterior and MSD runs + "Save analysis" do,
without napari.

`analyze_bundle` rebuilds the table the widget reads off the "tracks"
layer (`bundle_track_table`), runs the same posteriors and per-track MSD
fits (always both: the MSD fits take seconds), and writes the same files
through the same `diffusion` functions, so a bundle analyzed by
`gemscape2 diffusion` reopens in the widget exactly like one analyzed
there.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Optional

import polars as pl
from diffusionkit import Acquisition

from napari_gemscape2.diffusion import (
    EXPOSURE_CLAMP_FRACTION,
    GRID_FIELDS,
    MIN_FRAMES,
    analysis_summary,
    analysis_tables,
    analyze_posteriors,
    base_track_table,
    filter_record,
    MSD_ALPHA_LAG_FRACTION,
    MSD_ALPHA_MAX_LAG,
    MSD_MAX_LAG,
    MSDWindow,
    msd_fits_blur_free,
    msd_track_table,
    partially_pooled_table,
    passing_track_ids,
    posterior_options,
    posterior_results_table,
    region_class_groups,
    tracks_summary_table,
    tracks_to_diffusionkit_df,
)
from napari_gemscape2.pipeline import FilterSpec, track_features_df
from napari_gemscape2.regions import Regions
from napari_gemscape2.results import load_result, package_provenance, write_diffusion_results
from napari_gemscape2.viewer import layer_units_metadata, region_classes


@dataclass
class DiffusionSettings:
    """The widget's Posterior- and MSD-tab controls and tracks-pane filters."""

    min_frames: int = MIN_FRAMES
    # The posterior grid (`diffusion.GRID_FIELDS`, `GridPostOptions`'
    # names): D's range is the flat prior's support. Unset keys are
    # diffusionkit's defaults.
    grid: dict = field(default_factory=dict)
    # The per-track MSD fits' windows (`diffusion.MSDWindow`): D over the
    # first `msd_max_lag` lags, alpha over the first `msd_alpha_max_lag`.
    msd_max_lag: int = MSD_MAX_LAG
    msd_alpha_max_lag: int = MSD_ALPHA_MAX_LAG
    # When set, replaces both: shares of each track's own longest lag, D
    # over `msd_lag_fraction` (the 25-40% rule), alpha over the other.
    msd_lag_fraction: Optional[float] = None
    msd_alpha_lag_fraction: float = MSD_ALPHA_LAG_FRACTION
    # Overrides the bundle's recorded exposure. None: use the manifest's.
    exposure_s: Optional[float] = None
    # Which tracks `passes_filters` marks and the ensemble is over. Every
    # track is fitted either way, as the widget does by default.
    min_track_length: int = 1
    filters: FilterSpec = field(default_factory=dict)

    def __post_init__(self):
        self.msd_window  # validates

    @property
    def msd_window(self) -> MSDWindow:
        return MSDWindow(self.msd_max_lag, self.msd_alpha_max_lag, self.msd_lag_fraction, self.msd_alpha_lag_fraction)

    @classmethod
    def names(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    def options(self):
        """The run's `GridPostOptions`; raises ValueError for a bad grid."""
        return posterior_options(self.min_frames, self.grid)


def settings_from_summary(summary: dict) -> dict:
    """`DiffusionSettings` keyword arguments from a saved
    `diffusion_summary.json` -- the settings one movie was analyzed with
    in the widget. Not its exposure, which belongs to that movie."""
    record = summary.get("tracks_summary_filters") or {}
    out = {
        "min_frames": summary.get("min_frames"),
        # Only the D grid: a summary saved with the alpha posterior also
        # records alpha's grid, which no run here has any more.
        "grid": {k: v for k, v in summary["grid"].items() if k in GRID_FIELDS} if summary.get("grid") else None,
        **{k: v for k, v in (summary.get("msd") or {}).items() if k.startswith("msd_")},
        "min_track_length": record.get("min_track_length"),
        "filters": (
            {col: tuple(bounds) for col, bounds in record["ranges"].items()}
            if record.get("ranges")
            else None
        ),
    }
    return {key: value for key, value in out.items() if value is not None}


# The Tracks layer's own `data` columns (see `viewer._tracks_layer_arrays`).
_TRACK_DATA_COLUMNS = ("track_id", "frame", "y", "x")


def bundle_track_table(
    tracks_df: pl.DataFrame, pixel_size_um: float, dt_s: float, regions: Optional[Regions]
) -> pl.DataFrame:
    """The per-vertex table the widget reads off the bundle's "tracks"
    layer (`DiffusionAnalysisWidget._layer_track_table`): the vertices,
    every numeric per-vertex property (`viewer.add_tracks_layer` puts
    `track_features_df`'s columns there), and `region_class` named from
    the bundle's regions."""
    feat = track_features_df(tracks_df, pixel_size_um, dt_s)
    props = [
        col
        for col, dtype in feat.schema.items()
        if col not in _TRACK_DATA_COLUMNS and (dtype.is_numeric() or dtype == pl.Boolean)
    ]
    table = feat.select(
        pl.col("track_id").cast(pl.Int64),
        pl.col("frame").cast(pl.Int64),
        pl.col("y").cast(pl.Float64),
        pl.col("x").cast(pl.Float64),
        *props,
    )
    classes = region_classes(regions)
    if "region" in table.columns and classes:
        lookup = pl.DataFrame(
            {"region": list(classes), "region_class": list(classes.values())},
            schema={"region": table.schema["region"], "region_class": pl.Utf8},
        )
        table = table.join(lookup, on="region", how="left", maintain_order="left")
    return table


def run_exposure(exposure_s: Optional[float], dt_s: float) -> float:
    """The exposure to run with -- the widget's rule
    (`_PosteriorTab._exposure_for_run`): required, and at most the frame
    interval, a hair over being read as equal to it."""
    if exposure_s is None:
        raise ValueError(
            "no camera exposure recorded for this bundle -- set `exposure_s` in "
            "[diffusion] (or on its [[inputs]] entry)"
        )
    if exposure_s > dt_s * (1 + EXPOSURE_CLAMP_FRACTION):
        raise ValueError(f"exposure {exposure_s} s is longer than the frame interval {dt_s} s")
    return min(exposure_s, dt_s)


@dataclass
class BundleReport:
    n_tracks: int
    n_fitted: int
    n_passing: int
    exposure_s: float
    units_known: bool
    pixel_size_um: float
    dt_s: float


def analyze_bundle(
    result_dir: str | Path,
    settings: DiffusionSettings,
    *,
    progress: Optional[Callable[[int, int], None]] = None,
) -> BundleReport:
    """Run the grid posteriors over one bundle's tracks and save the
    analysis into it (`results.write_diffusion_results`)."""
    result_dir = Path(result_dir)
    _points, tracks_df, manifest, _labels, regions = load_result(result_dir)
    if tracks_df.height == 0 or "track_id" not in tracks_df.columns:
        raise ValueError("the bundle has no tracks")
    if "se_x" not in tracks_df.columns or "se_y" not in tracks_df.columns:
        raise ValueError("the tracks have no se_x/se_y (localization error) columns")

    params = manifest.get("params", {})
    units = layer_units_metadata(
        params.get("pixel_size_um"), params.get("dt_s"), result_dir, params.get("exposure_s")
    )
    pixel_size_um, dt_s = units["pixel_size_um"], units["dt_s"]
    exposure_s = run_exposure(
        settings.exposure_s if settings.exposure_s is not None else units["exposure_s"], dt_s
    )

    track_points = bundle_track_table(tracks_df, pixel_size_um, dt_s, regions)
    diffkit_tracks = tracks_to_diffusionkit_df(track_points, pixel_size_um, dt_s)
    base, _hideable = base_track_table(diffkit_tracks, track_points)

    analysis = analyze_posteriors(
        diffkit_tracks, Acquisition(dt_s=dt_s, exposure_s=exposure_s), settings.options(), progress=progress
    )
    msd = msd_track_table(msd_fits_blur_free(diffkit_tracks, dt_s, settings.min_frames, settings.msd_window))
    results = posterior_results_table(analysis, msd)

    joined = base.join(results, on="track_id", how="left")
    ids = passing_track_ids(joined, settings.min_track_length, settings.filters)
    by_class = region_class_groups(base, ids)
    # Each track's E[D] under its population goes in the file, not the filters.
    results = results.join(partially_pooled_table(analysis, ids, by_class), on="track_id", how="left")
    table = tracks_summary_table(
        base, results, result_id=result_dir.name, pixel_size_um=pixel_size_um, passing_ids=ids
    )
    summary = analysis_summary(analysis, ids, by_class, msd=settings.msd_window)
    summary["tracks_summary_filters"] = filter_record(settings.min_track_length, settings.filters)
    import diffusionkit
    import napari_gemscape2

    summary["packages"] = package_provenance(diffusionkit, napari_gemscape2)
    write_diffusion_results(
        result_dir, tracks_summary=table, summary=summary, **analysis_tables(analysis, ids, by_class)
    )
    return BundleReport(
        n_tracks=base.height,
        n_fitted=len(analysis.fitted_ids),
        n_passing=base.height if ids is None else len(ids),
        exposure_s=exposure_s,
        units_known=units["units_known"],
        pixel_size_um=pixel_size_um,
        dt_s=dt_s,
    )
