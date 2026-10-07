"""The population read of a posterior run: tracks combined by their log-likelihoods, never by their medians."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest
from diffusionkit import Acquisition

from napari_gemscape2 import diffusion as dfn
from napari_gemscape2.joint_plot import plot_d_by_length, plot_d_ensemble, plot_d_posteriors

DT = 0.03
MEDIAN_D, SIGMA = 0.05, 0.6


@pytest.fixture(scope="module")
def analysis():
    rng = np.random.default_rng(0)
    rows = []
    for i, D in enumerate(np.exp(np.log(MEDIAN_D) + SIGMA * rng.standard_normal(300))):
        n = int(rng.integers(5, 30))
        sd = rng.uniform(0.02, 0.03, (n, 2))
        pos = np.cumsum(rng.normal(0, np.sqrt(2 * D * DT), (n, 2)), 0) + sd * rng.standard_normal((n, 2))
        rows.append(pl.DataFrame({"track_id": i, "frame": np.arange(n), "t_s": np.arange(n) * DT,
                                  "x_um": pos[:, 0], "y_um": pos[:, 1],
                                  "sigma_x_um": sd[:, 0], "sigma_y_um": sd[:, 1], "track_length": n}))
    return dfn.analyze_posteriors(pl.concat(rows), Acquisition(DT), dfn.posterior_options(), threads=1)


def test_the_summary_reads_the_population_not_the_medians(analysis):
    s = dfn.summarize(analysis)
    assert not {"median_D_um2_s", "q25_D_um2_s", "q75_D_um2_s"} & set(s)
    assert s["lognormal_D_median_low_um2_s"] < MEDIAN_D < s["lognormal_D_median_high_um2_s"]
    assert s["lognormal_sigma_ln_D_low"] < SIGMA < s["lognormal_sigma_ln_D_high"]
    assert s["lognormal_D_mean_low_um2_s"] < s["lognormal_D_mean_um2_s"] < s["lognormal_D_mean_high_um2_s"]
    assert s["shared_D_low_um2_s"] < s["shared_D_median_um2_s"] < s["shared_D_high_um2_s"]
    assert "lognormal_problem" not in s


def test_the_tables_name_their_models(analysis):
    dist = dfn.distributions_table(analysis)
    assert dist.columns == ["group", "n_tracks", "D_um2_s", "shared_loglik", "lognormal", "lognormal_low",
                            "lognormal_high", "deconvolved", "deconvolved_low", "deconvolved_high"]
    assert dist["lognormal"].sum() == pytest.approx(1.0)
    assert dist["shared_loglik"].max() == 0.0
    by_length = dfn.length_distributions_table(analysis)
    assert {"unpooled", "partially_pooled", "partially_pooled_low", "partially_pooled_high"} <= set(by_length.columns)
    loglik = dfn.loglik_long_table(analysis)
    assert loglik.columns == ["track_id", "D_um2_s", "loglik"]
    per_track = loglik.group_by("track_id").agg(pl.col("loglik").exp().sum())
    assert np.allclose(per_track["loglik"].to_numpy(), 1.0)  # normalized: also the flat-prior posterior


def test_the_heat_map_is_sorted_by_likelihood_peak(analysis):
    (panel,) = dfn.ensemble_panels(analysis, track_posteriors=True)
    peaks = panel["track_posteriors"].argmax(axis=1)
    assert np.all(np.diff(peaks) >= 0)
    assert set(panel) >= {"lognormal", "lognormal_band", "summary", "deconvolved", "shared_interval"}


def test_the_figures_draw(analysis):
    grid = analysis.D_grid_um2_s
    groups = {"early": set(range(150)), "late": set(range(150, 300))}
    for fig in (plot_d_ensemble(grid, dfn.ensemble_panels(analysis, groups)),
                plot_d_posteriors(grid, dfn.ensemble_panels(analysis, groups, track_posteriors=True)),
                plot_d_by_length(dfn.length_panels(analysis, groups))):
        fig.canvas.draw()
        assert len(fig.axes) >= 2
