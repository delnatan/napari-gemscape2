"""`gemscape2 pool`: saved analyses of several bundles pooled by sample, nothing refitted."""

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from napari_gemscape2.cli import app
from napari_gemscape2.diffusion_batch import DiffusionSettings, analyze_bundle
from napari_gemscape2.results import build_manifest, write_result

PX, DT = 0.1, 0.03


def _bundle(root, name, D, seed, n_tracks=50, n=12):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tracks):
        pos = np.cumsum(rng.normal(size=(n, 2)) * np.sqrt(2 * D * DT) / PX, axis=0) + rng.uniform(0, 100, 2)
        pos += rng.normal(size=(n, 2)) * 0.1
        rows += [
            dict(track_id=t, frame=f, x=pos[f, 0], y=pos[f, 1], se_x=0.1, se_y=0.1, flux=1000.0 + t, bg=10.0,
                 fit_sigma=1.3, loc_id=t * n + f)
            for f in range(n)
        ]
    tracks = pl.DataFrame(rows)
    manifest = build_manifest(
        result_id=name, source_image_path=root / "x.tif",
        params=dict(pixel_size_um=PX, dt_s=DT, exposure_s=0.01, n_points=tracks.height, n_tracks=n_tracks),
    )
    write_result(root / "results" / name, tracks, tracks, manifest)
    return root / "results" / name


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("pool")
    for name, D, seed in (("wt_1", 0.05, 1), ("wt_2", 0.05, 2), ("mut_1", 0.5, 3)):
        analyze_bundle(_bundle(root, name, D, seed), DiffusionSettings())
    return root


def _config(root, pool=""):
    path = root / "cfg.toml"
    path.write_text(
        'results_root = "results"\n' + pool +
        '[[inputs]]\npath = "wt_1.tif"\nsample = "wt"\n'
        '[[inputs]]\npath = "wt_2.tif"\nsample = "wt"\n'
        '[[inputs]]\npath = "mut_1.tif"\nsample = "mut"\n'
    )
    return path


def test_each_sample_is_one_population_of_all_its_movies(project):
    result = CliRunner().invoke(app, ["pool", str(_config(project))])
    assert result.exit_code == 0, result.output
    out = project / "results" / "pooled"
    pops = pl.read_csv(out / "pooled_populations_D.csv")
    samples = pops.filter(pl.col("by") == "sample")
    assert dict(zip(samples["group"], samples["n_tracks"])) == {"wt": 100, "mut": 50}
    # The bundles hold instantaneous positions but declare a 10 ms exposure, which the posterior models as
    # blur: it reads D * dt / (dt - exposure / 3).
    blur = DT / (DT - 0.01 / 3)
    for name, D in (("wt", 0.05 * blur), ("mut", 0.5 * blur)):
        row = samples.filter(pl.col("group") == name).row(0, named=True)
        assert row["lognormal_D_median_low_um2_s"] < D < row["lognormal_D_median_high_um2_s"]
        assert row["lognormal_sigma_ln_D_high"] < 0.5  # one D per sample: sigma near 0
    # A sample has several movies, so each movie is also read on its own, with its sample named.
    movies = pops.filter(pl.col("by") == "experiment")
    assert dict(zip(movies["group"], movies["sample"])) == {"wt_1": "wt", "wt_2": "wt", "mut_1": "mut"}
    draws = pl.read_csv(out / "pooled_lognormal_draws_D.csv")
    assert draws.columns == ["by", "group", "draw", "mu_ln_D", "sigma_ln_D"]
    assert set(draws["group"]) == {"wt", "mut", "wt_1", "wt_2", "mut_1"}
    assert not (out / "pooled_distances_D.csv").exists()  # comparing samples is left outside the GUI
    assert not (out / "pooled_ensemble_msd.csv").exists()


def test_ensemble_msd_needs_an_explicit_window(project):
    result = CliRunner().invoke(app, ["pool", str(_config(project, "[pool]\nensemble_msd = true\n"))])
    assert result.exit_code != 0
    assert "ensemble_n_points" in result.output


def test_ensemble_msd_recovers_D(project):
    pool = "[pool]\nensemble_msd = true\nensemble_max_lag = 6\nensemble_n_points = 4\nn_boot = 20\n"
    result = CliRunner().invoke(app, ["pool", str(_config(project, pool))])
    assert result.exit_code == 0, result.output
    fits = pl.read_csv(project / "results" / "pooled" / "pooled_ensemble_msd_fits.csv")
    D = dict(zip(*fits.filter(pl.col("model") == "brownian").select("group", "D_um2_s")))
    assert D["wt"] == pytest.approx(0.05, rel=0.2)
    assert D["mut"] == pytest.approx(0.5, rel=0.2)
    # alpha is fitted over the whole curve unless told otherwise.
    power = fits.filter(pl.col("model") == "power_law")
    assert set(power["n_points"]) == {6}
    assert power["alpha"].to_list() == pytest.approx([1.0, 1.0], abs=0.2)


def test_a_bundle_without_an_analysis_stops_the_pooling(project, tmp_path):
    _bundle(tmp_path, "wt_1", 0.05, 1)  # saved tracks, never analyzed
    path = tmp_path / "cfg.toml"
    path.write_text('results_root = "results"\n[[inputs]]\npath = "wt_1.tif"\n')
    result = CliRunner().invoke(app, ["pool", str(path)])
    assert result.exit_code == 1
    assert "gemscape2 diffusion" in result.output


def test_the_distributions_are_the_log_normal_and_the_deconvolution_with_bands(project):
    result = CliRunner().invoke(app, ["pool", str(_config(project))])
    assert result.exit_code == 0, result.output
    pops = pl.read_csv(project / "results" / "pooled" / "pooled_distributions_D.csv")
    for (_by, _group), rows in pops.group_by("by", "group", maintain_order=True):
        for column in ("lognormal", "deconvolved"):
            assert rows[column].sum() == pytest.approx(1.0)
            assert np.all(rows[f"{column}_low"].to_numpy() <= rows[f"{column}_high"].to_numpy() + 1e-12)


def test_the_pooled_figure_draws(project):
    import matplotlib

    matplotlib.use("Agg")
    from napari_gemscape2.joint_plot import plot_pooled_populations
    from napari_gemscape2.pooling import load_pooled_bundle, run_pooling, PoolSettings

    bundles = [load_pooled_bundle(project / "results" / name, sample)
               for name, sample in (("wt_1", "wt"), ("wt_2", "wt"), ("mut_1", "mut"))]
    result = run_pooling(bundles, PoolSettings())
    fig = plot_pooled_populations(result.distributions, result.populations, result.summary["samples"],
                                  result.summary["credible_level"])
    fig.canvas.draw()
    assert len(fig.axes) == 3


def test_a_config_written_for_a_pooling_reruns_it(project):
    from napari_gemscape2.pooling import PoolSettings, guess_sample, write_pool_config

    names = ("wt_1", "wt_2", "mut_1")
    assert [guess_sample(n) for n in names] == ["wt", "wt", "mut"]
    assert guess_sample("2024") == "2024"
    settings = PoolSettings(
        ensemble_msd=True, ensemble_max_lag=6, ensemble_n_points=3, ensemble_alpha_points=5, n_boot=10
    )
    path = project / "pool.toml"
    write_pool_config(
        path,
        results_root=project / "results",
        inputs=[(project / f"{n}.tif", n, guess_sample(n)) for n in names],
        settings=settings,
        output=project / "results" / "written",
    )
    result = CliRunner().invoke(app, ["pool", str(path)])
    assert result.exit_code == 0, result.output
    out = project / "results" / "written"
    summary = __import__("json").loads((out / "pooled_summary.json").read_text())
    assert summary["samples"] == {"wt": ["wt_1", "wt_2"], "mut": ["mut_1"]}
    assert summary["ensemble_msd"]["ensemble_n_points"] == 3
    assert summary["ensemble_msd"]["ensemble_alpha_points"] == 5
    with pytest.raises(ValueError, match="ensemble_alpha_points"):
        PoolSettings(ensemble_msd=True, ensemble_max_lag=6, ensemble_n_points=3, ensemble_alpha_points=7)
    assert (out / "pooled_ensemble_msd_fits.csv").exists()


def test_a_batch_runs_the_msd_fits_alongside_and_they_reopen(project):
    from napari_gemscape2.diffusion import MSDWindow, restore_analysis, tracks_to_diffusionkit_df
    from napari_gemscape2.diffusion_batch import bundle_track_table
    from napari_gemscape2.results import load_diffusion_results, load_result

    bundle = project / "results" / "wt_1"
    tables = load_diffusion_results(bundle)
    assert MSDWindow.from_settings(tables["summary"]["msd"]) == MSDWindow()
    assert tables["tracks_summary"]["D_msd_um2_s"].drop_nulls().len() == 50
    _points, tracks_df, _manifest, _labels, regions = load_result(bundle)
    tracks = tracks_to_diffusionkit_df(bundle_track_table(tracks_df, PX, DT, regions), PX, DT)
    saved = restore_analysis(tracks, **tables)
    assert set(saved.msd.columns) == {"track_id", "D_msd_um2_s", "K_msd_um2_s_alpha", "alpha_msd"}
