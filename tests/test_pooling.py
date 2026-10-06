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


def test_replicates_are_closer_than_samples(project):
    result = CliRunner().invoke(app, ["pool", str(_config(project))])
    assert result.exit_code == 0, result.output
    out = project / "results" / "pooled"
    dist = pl.read_csv(out / "pooled_distances_D.csv")
    replicate = dist.filter(pl.col("same_sample"))["W1_ln_D_median"].item()
    between = dist.filter(pl.col("by") == "sample")["W1_ln_D_median"].item()
    assert replicate < 0.5 < 2 < between
    pops = pl.read_csv(out / "pooled_distributions_D.csv")
    n = pops.group_by("by", "group").agg(pl.col("n_tracks").first()).filter(pl.col("by") == "sample")
    assert dict(zip(n["group"], n["n_tracks"])) == {"wt": 100, "mut": 50}
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


def test_a_bundle_without_an_analysis_stops_the_pooling(project, tmp_path):
    _bundle(tmp_path, "wt_1", 0.05, 1)  # saved tracks, never analyzed
    path = tmp_path / "cfg.toml"
    path.write_text('results_root = "results"\n[[inputs]]\npath = "wt_1.tif"\n')
    result = CliRunner().invoke(app, ["pool", str(path)])
    assert result.exit_code == 1
    assert "gemscape2 diffusion" in result.output
