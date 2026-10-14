"""The α · D run (`napari_gemscape2.fbm`): diffusionkit's fBm posteriors reshaped for the tracks
pane, saved in the per-track summary and restored from it, and the batch settings that rerun it."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest
from diffusionkit import Acquisition
from diffusionkit.fbmpost import simulate_tracks

from napari_gemscape2 import fbm
from napari_gemscape2.diffusion import StaleAnalysisError, floor_references, tracks_summary_table
from napari_gemscape2.diffusion_batch import DiffusionSettings, settings_from_summary
from napari_gemscape2.joint_plot import plot_alpha_D, plot_alpha_population, plot_fbm_track

ACQ = Acquisition(0.044, 0.02)


def _tracks(n_tracks, alpha, D_ref, seed, first=0):
    t = simulate_tracks(n_tracks, 25, alpha, D_ref, 0.013, ACQ, rng=np.random.default_rng(seed), first_track_id=first)
    return t.with_columns(t_s=pl.col("frame") * ACQ.dt_s)


@pytest.fixture(scope="module")
def tracks():
    short = pl.DataFrame({"track_id": [99, 99], "frame": [0, 1], "x_um": [0.0, 0.01], "y_um": [0.0, 0.0],
                          "sigma_x_um": [0.013] * 2, "sigma_y_um": [0.013] * 2, "t_s": [0.0, 0.044]})
    return pl.concat([_tracks(20, 0.4, 3e-3, 0), _tracks(20, 1.0, 1e-2, 1, first=20), short])


@pytest.fixture(scope="module")
def analysis(tracks):
    return fbm.analyze_fbm(tracks, ACQ, fbm.fbm_options(), threads=1)


def test_one_row_per_track_with_the_panes_columns(analysis):
    fits = analysis.fits
    assert fits.height == 41
    assert set(fbm.FBM_SCHEMA) <= set(fits.columns)
    assert fits.filter(pl.col("track_id") == 99)["fbm_status"].item() == "excluded"
    ok = analysis.ok
    assert ok.height == 40
    assert ((ok["alpha_low"] <= ok["alpha_mean"]) & (ok["alpha_mean"] <= ok["alpha_high"])).all()
    # The two populations separate in alpha on average: subdiffusive vs Brownian.
    low, high = ok.filter(pl.col("track_id") < 20), ok.filter(pl.col("track_id") >= 20)
    assert low["alpha_mean"].median() < 0.75 < high["alpha_mean"].median()


def test_summary_counts_without_population_numbers(analysis):
    s = fbm.summarize_fbm(analysis)
    assert s["n_tracks"] == 41 and s["n_ok"] == 40 and s["n_excluded"] == 1
    assert 0 < s["n_alpha_informative"] <= 40
    assert not any("median_alpha" in k or "alpha_mean" in k for k in s)


def test_saved_columns_restore_the_run(tracks, analysis):
    base = tracks.group_by("track_id").agg(pl.len().alias("track_length"),
                                           pl.col("x_um").mean().alias("x_px"), pl.col("y_um").mean().alias("y_px"))
    table = tracks_summary_table(base, analysis.columns(), result_id="m", pixel_size_um=1.0, passing_ids=None)
    # Through CSV, as `tracks_summary.csv` is saved and read back.
    table = pl.read_csv(table.write_csv().encode(), infer_schema_length=None)
    record = fbm.fbm_record(analysis, None, None)
    restored = fbm.restore_fbm(tracks, record, table)
    assert restored.tau_ref_s == analysis.tau_ref_s
    assert restored.likelihoods is None  # saved without loglik_alpha_D: no population
    a, b = analysis.fits.sort("track_id"), restored.fits.sort("track_id")
    np.testing.assert_allclose(a["alpha_mean"].fill_null(np.nan), b["alpha_mean"].fill_null(np.nan))
    np.testing.assert_allclose(a["alpha_D_ref_corr"].fill_null(np.nan), b["alpha_D_ref_corr"].fill_null(np.nan))
    moved = tracks.with_columns(pl.when(pl.col("track_id") == 3).then(pl.col("x_um") + 1e-3).otherwise(pl.col("x_um")))
    with pytest.raises(StaleAnalysisError):
        fbm.restore_fbm(moved, record, table)


def test_d_ref_columns_are_read_against_their_own_floor(analysis):
    refs = floor_references(analysis.ok, ("D_ref_mean_um2_s", "alpha_mean"))
    assert set(refs) == {"D_ref_mean_um2_s"}
    assert refs["D_ref_mean_um2_s"][1] == pytest.approx(0.013**2 / 0.1)


def test_a_template_with_an_alpha_D_run_reruns_it():
    settings = settings_from_summary({"min_frames": 3, "alpha_D": {"tau_ref_s": 0.25}})
    assert DiffusionSettings(**settings).alpha_D_tau_ref_s == 0.25
    assert "alpha_D_tau_ref_s" not in settings_from_summary({"min_frames": 3})


def test_plots_draw(tracks, analysis):
    panels = fbm.scatter_panels(analysis, {"a": set(range(20)), "b": set(range(20, 40))})
    assert [p["name"] for p in panels] == ["a", "b"]
    assert len(plot_alpha_D(panels).axes) == 2
    panel = fbm.track_panel(analysis, tracks.filter(pl.col("track_id") == 5))
    da = panel["alphas"][1] - panel["alphas"][0]
    dx = panel["log10_D_ref"][1] - panel["log10_D_ref"][0]
    assert panel["density"].sum() * da * dx == pytest.approx(1, abs=1e-3)
    assert (panel["p_alpha"] * panel["alphas"]).sum() * da == pytest.approx(panel["fit"]["alpha_mean"], abs=1e-6)
    plot_fbm_track(panel)
    assert fbm.track_panel(analysis, tracks.filter(pl.col("track_id") == 99)) is None


LOW, HIGH = set(range(20)), set(range(20, 40))


def test_the_run_keeps_each_fitted_tracks_joint_likelihood(analysis):
    lik = analysis.likelihoods
    assert list(lik.track_ids) == list(range(40))  # the 2-frame track was not fitted
    assert lik.loglik.shape[:2] == (40, len(lik.alphas))
    # Cropped to the cells the tracks tell apart, not diffusionkit's 1e-6..1e2 default.
    assert -6 < lik.log10_D_ref[0] and lik.log10_D_ref[-1] < 2


def test_population_separates_the_two_groups(analysis):
    low, high = fbm.alpha_population(analysis, LOW), fbm.alpha_population(analysis, HIGH)
    assert np.median(low.alpha_mean()) < 0.75 < np.median(high.alpha_mean())
    assert fbm.alpha_population(analysis, LOW) is low  # cached
    assert fbm.population_is_cached(analysis, LOW)
    assert fbm.alpha_population(analysis, set(range(10))) is None  # under MIN_POPULATION_TRACKS
    s = fbm.population_summary(high)
    assert s["n_tracks"] == 20
    for key in ("alpha_mean", "alpha_sd", "p_alpha_below_1", "alpha_D_ref_corr", "share_below_floor"):
        median, low_, high_ = s[key]
        assert low_ <= median <= high_


def test_record_carries_the_population_overall_and_by_class(analysis):
    record = fbm.fbm_record(analysis, None, {"a": LOW, "b": HIGH})
    assert record["population"]["n_tracks"] == 40
    assert record["by_region_class"]["a"]["population"]["alpha_mean"][0] < 0.75
    assert record["by_region_class"]["a"]["n_ok"] == 20


def test_saved_likelihoods_restore_the_population(tracks, analysis):
    base = tracks.group_by("track_id").agg(pl.len().alias("track_length"),
                                           pl.col("x_um").mean().alias("x_px"), pl.col("y_um").mean().alias("y_px"))
    table = tracks_summary_table(base, analysis.columns(), result_id="m", pixel_size_um=1.0, passing_ids=None)
    table = pl.read_csv(table.write_csv().encode(), infer_schema_length=None)
    loglik = fbm.loglik_long_table(analysis)
    loglik = pl.read_parquet(_parquet_roundtrip(loglik))
    restored = fbm.restore_fbm(tracks, fbm.fbm_record(analysis, None, None), table, loglik)
    a, b = analysis.likelihoods, restored.likelihoods
    np.testing.assert_array_equal(a.track_ids, b.track_ids)
    np.testing.assert_allclose(a.log10_D_ref, b.log10_D_ref)
    np.testing.assert_allclose(a.loglik, b.loglik, atol=1e-4 * np.abs(a.loglik).max())
    np.testing.assert_allclose(a.D_ref_floor_um2_s, b.D_ref_floor_um2_s)
    assert np.median(fbm.alpha_population(restored, LOW).alpha_mean()) < 0.75
    with pytest.raises(StaleAnalysisError):
        fbm.restore_fbm(tracks, fbm.fbm_record(analysis, None, None), table,
                        loglik.filter(pl.col("track_id") != 3))


def _parquet_roundtrip(df):
    import io

    buffer = io.BytesIO()
    df.write_parquet(buffer)
    buffer.seek(0)
    return buffer


def test_population_plot_draws(analysis):
    panels = fbm.population_panels(analysis, {"a": LOW, "b": HIGH, "few": set(range(5))})
    assert [p["name"] for p in panels] == ["a", "b"]
    assert len(plot_alpha_population(panels).axes) == 4
