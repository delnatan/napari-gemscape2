"""Spatial classes of a movie's tracks: the batch run, what it saves, restoring it, its figures and pooling a class."""

import json

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from napari_gemscape2.cli import app
from napari_gemscape2.diffusion import ensemble, restore_analysis, tracks_to_diffusionkit_df
from napari_gemscape2.diffusion_batch import DiffusionSettings, analyze_bundle, bundle_track_table, settings_from_summary
from napari_gemscape2.results import (
    build_manifest, load_diffusion_results, load_result, load_spatial_classes, write_result,
)
from napari_gemscape2.spatial import SPATIAL_COLUMNS, class_groups, restore_spatial
from napari_gemscape2.spatial_plot import plot_class_map, plot_class_populations, plot_class_selection

PX, DT = 0.1, 0.03
SLOW, FAST = 0.02, 0.3


def _cell(root, name, seed, n_tracks=300, n=10):
    """Tracks over a 60 µm field; inside a disk of radius 15 µm at its centre they are slow, outside fast."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tracks):
        start = rng.uniform(0, 600, 2)
        D = SLOW if np.linalg.norm(start - 300) < 150 else FAST
        pos = np.cumsum(rng.normal(size=(n, 2)) * np.sqrt(2 * D * DT) / PX, axis=0) + start
        pos += rng.normal(size=(n, 2)) * 0.2
        rows += [
            dict(track_id=t, frame=f, x=pos[f, 0], y=pos[f, 1], se_x=0.2, se_y=0.2, flux=1000.0, bg=10.0,
                 fit_sigma=1.3, loc_id=t * n + f)
            for f in range(n)
        ]
    tracks = pl.DataFrame(rows)
    manifest = build_manifest(
        result_id=name, source_image_path=root / f"{name}.tif",
        params=dict(pixel_size_um=PX, dt_s=DT, exposure_s=0.01, n_points=tracks.height, n_tracks=n_tracks),
    )
    write_result(root / "results" / name, tracks, tracks, manifest)
    return root / "results" / name


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("spatial")
    for name, seed in (("cell_1", 1), ("cell_2", 2)):
        analyze_bundle(_cell(root, name, seed), DiffusionSettings(spatial=True, spatial_shuffles=19))
    return root


def _restored(result_dir):
    _p, tracks_df, manifest, _l, regions = load_result(result_dir)
    tracks = tracks_to_diffusionkit_df(bundle_track_table(tracks_df, PX, DT, regions), PX, DT)
    saved = load_diffusion_results(result_dir)
    analysis = restore_analysis(tracks, **saved).analysis
    return analysis, restore_spatial(analysis, saved["summary"]["spatial"], load_spatial_classes(result_dir)), saved


def test_a_batch_finds_two_spatially_organized_classes(project):
    summary = json.loads((project / "results" / "cell_1" / "diffusion_summary.json").read_text())
    sp = summary["spatial"]
    assert sp["n_classes"] == 2 and sp["patterned"]
    assert sp["null"]["p"] == pytest.approx(1 / 20)
    assert "share" not in sp["report"][0]
    table = pl.read_csv(project / "results" / "cell_1" / "tracks_summary.csv", infer_schema_length=None)
    assert set(SPATIAL_COLUMNS) <= set(table.columns)
    inside = ((table["x_um"] - 30) ** 2 + (table["y_um"] - 30) ** 2) < 12 ** 2
    assert table.filter(inside)["spatial_class_neighbours"].mean() < 1.2
    assert table.filter(~inside & ((table["x_um"] - 30) ** 2 + (table["y_um"] - 30) ** 2 > 18 ** 2))[
        "spatial_class_neighbours"].mean() > 1.8


def test_the_class_map_is_saved_pixel_by_pixel(project):
    import tifffile

    result_dir = project / "results" / "cell_1"
    label = tifffile.imread(result_dir / "spatial_class_map.tif")
    probability = tifffile.imread(result_dir / "spatial_class_probability.tif")
    # No frame shape recorded in these bundles: the grid reaches the farthest track (60 µm at PX).
    assert label.dtype == np.uint8 and probability.dtype == np.float32
    assert probability.shape == (2, *label.shape)
    assert set(np.unique(label)) <= {0, 1, 2}
    near = label > 0
    assert np.allclose(np.nansum(probability, axis=0)[near], 1, atol=1e-5)
    assert np.isnan(probability[:, ~near]).all()
    assert np.array_equal(label[near], probability[:, near].argmax(axis=0) + 1)
    # The slow disk at the field's centre is class 1, the fast outside class 2.
    centre = (int(30 / PX), int(30 / PX))
    assert label[centre] == 1
    assert label[int(3 / PX), int(3 / PX)] in (0, 2)


def test_each_class_is_a_group_of_the_distributions(project):
    dist = pl.read_csv(project / "results" / "cell_1" / "distributions_D.csv")
    assert {"all", "spatial class 1", "spatial class 2"} <= set(dist["group"].unique().to_list())


def test_the_saved_run_restores(project):
    analysis, s, saved = _restored(project / "results" / "cell_1")
    assert s.n_classes == 2 and s.patterned
    cols = s.columns().join(saved["tracks_summary"].select("track_id", "spatial_class_neighbours"), on="track_id")
    assert (cols["spatial_class_neighbours"] == cols["spatial_class_neighbours_right"]).all()


def test_a_template_reruns_the_spatial_classes_but_not_its_number(project):
    summary = json.loads((project / "results" / "cell_1" / "diffusion_summary.json").read_text())
    kw = settings_from_summary(summary)
    assert kw["spatial"] is True and kw["spatial_shuffles"] == 19
    assert "spatial_n_classes" not in kw


def test_figures_draw(project):
    analysis, s, _ = _restored(project / "results" / "cell_1")
    plot_class_selection(s)
    plot_class_map(s, shape_px=(600, 600), pixel_size_um=PX)
    groups = class_groups(s)
    panels = []
    for c, (name, ids) in enumerate(groups.items(), start=1):
        ens = ensemble(analysis, ids)
        panels.append({"name": name, "class": c, "n_tracks": ens.n_tracks, "deconvolved": ens.deconvolved_D,
                       "band": ens.deconvolved_D_band})
    fig = plot_class_populations(analysis.D_grid_um2_s, panels)
    peaks = [analysis.D_grid_um2_s[np.argmax(p["deconvolved"])] for p in panels]
    assert peaks[0] < 0.1 < peaks[1]
    assert fig.axes


def test_matched_classes_pool_across_movies(project):
    cfg = project / "pool.toml"
    cfg.write_text(
        'results_root = "results"\n'
        '[[inputs]]\npath = "cell_1.tif"\nsample = "inner"\nspatial_class = 1\n'
        '[[inputs]]\npath = "cell_2.tif"\nsample = "inner"\nspatial_class = 1\n'
        '[[inputs]]\npath = "cell_1.tif"\nsample = "outer"\nspatial_class = 2\n'
        '[[inputs]]\npath = "cell_2.tif"\nsample = "outer"\nspatial_class = 2\n'
    )
    result = CliRunner().invoke(app, ["pool", str(cfg)])
    assert result.exit_code == 0, result.output
    pops = pl.read_csv(project / "results" / "pooled" / "pooled_populations_D.csv")
    by_sample = {r["group"]: r for r in pops.filter(pl.col("by") == "sample").iter_rows(named=True)}
    assert by_sample["inner"]["lognormal_D_median_um2_s"] < 0.1 < by_sample["outer"]["lognormal_D_median_um2_s"]
    t1 = pl.read_csv(project / "results" / "cell_1" / "tracks_summary.csv", infer_schema_length=None)
    t2 = pl.read_csv(project / "results" / "cell_2" / "tracks_summary.csv", infer_schema_length=None)
    inner = sum(int((t["spatial_class_neighbours"] == 1).sum()) for t in (t1, t2))
    assert by_sample["inner"]["n_tracks"] + by_sample["inner"]["n_excluded"] == inner


def test_a_class_a_movie_does_not_have_stops_the_pooling(project):
    cfg = project / "bad.toml"
    cfg.write_text('results_root = "results"\n[[inputs]]\npath = "cell_1.tif"\nspatial_class = 5\n')
    result = CliRunner().invoke(app, ["pool", str(cfg)])
    assert result.exit_code == 1 and "not one of this movie's classes" in result.output


def test_too_few_shuffles_to_reach_the_pattern_threshold_are_refused():
    with pytest.raises(ValueError, match="at least 19"):
        DiffusionSettings(spatial=True, spatial_shuffles=9)
    DiffusionSettings(spatial=True, spatial_shuffles=0)
    DiffusionSettings(spatial=True, spatial_shuffles=19)


def test_the_populations_table_has_a_row_per_group_and_reads_back(project):
    from napari_gemscape2 import read_results

    pops = pl.read_csv(project / "results" / "cell_1" / "populations.csv")
    assert pops["group"].to_list() == ["all", "spatial class 1", "spatial class 2"]
    slow, fast = pops.row(1, named=True), pops.row(2, named=True)
    assert slow["lognormal_D_median_um2_s"] < 0.1 < fast["lognormal_D_median_um2_s"]
    assert pops["alpha_mean"].is_null().all()  # no α · D run in this batch
    assert not (project / "results" / "cell_1" / "distributions_alpha.csv").exists()

    res = read_results(project)
    assert [b.name for b in res.bundles] == ["cell_1", "cell_2"]
    assert set(res.populations["result_id"]) == {"cell_1", "cell_2"}
    assert res.tracks.height == 600 and set(res.tracks["result_id"]) == {"cell_1", "cell_2"}
    assert {"result_id", "group", "D_um2_s", "deconvolved"} <= set(res.distributions_D.columns)
    assert res.distributions_alpha.is_empty()
    assert set(res.summaries) == {"cell_1", "cell_2"}
