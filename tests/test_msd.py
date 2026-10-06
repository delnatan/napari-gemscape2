"""The MSD comparison's windows: per track (fixed lags or a share of each track) and the ensemble average."""

import numpy as np
import polars as pl
import pytest
from diffusionkit.classic import window_lags

from napari_gemscape2.diffusion import (
    ensemble_msd_blur_free,
    ensemble_msd_panels,
    group_experiments,
    msd_fits_blur_free,
    msd_track_curve,
)
from napari_gemscape2.diffusion_batch import DiffusionSettings, settings_from_summary

DT, SE = 0.03, 0.03


@pytest.fixture(scope="module")
def tracks():
    """Two populations, D = 0.05 (ids < 150) and 0.4 um^2/s, of mixed lengths."""
    rng = np.random.default_rng(0)
    rows = []
    for t in range(300):
        D = 0.05 if t < 150 else 0.4
        n = int(rng.integers(6, 40))
        pos = np.cumsum(rng.normal(size=(n, 2)) * np.sqrt(2 * D * DT), axis=0) + rng.normal(size=(n, 2)) * SE
        rows += [dict(track_id=t, frame=f, x_um=pos[f, 0], y_um=pos[f, 1], sigma_x_um=SE, sigma_y_um=SE)
                 for f in range(n)]
    return pl.DataFrame(rows)


def test_a_share_of_each_track_fits_more_lags_on_longer_tracks(tracks):
    lengths = dict(tracks.group_by("track_id").len().iter_rows())
    fits = msd_fits_blur_free(tracks, DT, 5, lag_fraction=0.3).filter(pl.col("model") == "brownian")
    for track_id, n_lags in fits.select("track_id", "n_lags").iter_rows():
        assert n_lags == window_lags(lengths[track_id] - 1, 0.3)
    fixed = msd_fits_blur_free(tracks, DT, 5, max_lag=3).filter(pl.col("model") == "brownian")
    assert set(fixed["n_lags"]) == {3}


def test_the_track_plot_reads_the_offset_off_the_curve_too(tracks):
    data = msd_track_curve(tracks, DT, 3, 5, lag_fraction=0.4)
    assert len(data["tau_s"]) >= 3
    assert data["sigma_sds_um"] == pytest.approx(SE)
    assert data["D_linear_um2_s"] is not None


def test_the_ensemble_recovers_each_group(tracks):
    groups = {"slow": set(range(150)), "fast": set(range(150, 300))}
    ens = ensemble_msd_blur_free(group_experiments(tracks, DT, groups), 5, 10, n_boot=50)
    for offset in ("provided", "fit"):
        panels = {p["name"]: p for p in ensemble_msd_panels(ens, 4, offset)}
        assert panels["slow"]["linear"]["D_um2_s"] == pytest.approx(0.05, rel=0.15)
        assert panels["fast"]["linear"]["D_um2_s"] == pytest.approx(0.4, rel=0.15)
        assert panels["slow"]["power_law"]["alpha"] == pytest.approx(1.0, abs=0.15)
        assert panels["slow"]["n_tracks"] == 150
    fitted = {p["name"]: p for p in ensemble_msd_panels(ens, 4, "fit")}
    assert fitted["slow"]["linear"]["localization_sd_um"] == pytest.approx(SE, rel=0.3)
    with pytest.raises(ValueError, match="n_points"):
        ensemble_msd_panels(ens, 11, "provided")


def test_the_window_round_trips_through_a_saved_summary():
    assert settings_from_summary({"msd_comparison": True, "msd_lag_fraction": 0.3})["msd_lag_fraction"] == 0.3
    with pytest.raises(ValueError, match="msd_lag_fraction"):
        DiffusionSettings(msd_lag_fraction=1.5)
    with pytest.raises(ValueError, match="msd_max_lag"):
        DiffusionSettings(msd_max_lag=2)
