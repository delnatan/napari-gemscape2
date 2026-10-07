"""The MSD comparison's windows: per track (fixed lags or a share of each track) and the ensemble average."""

import numpy as np
import polars as pl
import pytest
from diffusionkit.classic import window_lags

from napari_gemscape2.diffusion import (
    MSDWindow,
    ensemble_msd_blur_free,
    ensemble_msd_fits,
    ensemble_msd_panels,
    ensemble_windows,
    group_experiments,
    msd_fits_blur_free,
    msd_track_panel,
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
    fits = msd_fits_blur_free(tracks, DT, 5, MSDWindow(lag_fraction=0.3))
    for track_id, n_lags in fits.filter(pl.col("model") == "brownian").select("track_id", "n_lags").iter_rows():
        assert n_lags == window_lags(lengths[track_id] - 1, 0.3)
    fixed = msd_fits_blur_free(tracks, DT, 5, MSDWindow())
    assert set(fixed.filter(pl.col("model") == "brownian")["n_lags"]) == {3}


def test_alpha_is_fitted_by_log_log_over_its_own_wider_window(tracks):
    lengths = dict(tracks.group_by("track_id").len().iter_rows())
    power = msd_fits_blur_free(tracks, DT, 5, MSDWindow()).filter(pl.col("model") == "power_law")
    assert set(power["method"]) == {"msd_loglog"}
    for track_id, n_lags in power.select("track_id", "n_lags").iter_rows():
        # Up to 10 lags, capped at the track's own; lags with nothing left
        # after the offset cannot be logged and are dropped.
        assert n_lags <= min(10, lengths[track_id] - 1)
    long_ids = [t for t, n in lengths.items() if n >= 30]
    alpha = power.filter(pl.col("track_id").is_in(long_ids))["alpha"]
    assert alpha.median() == pytest.approx(1.0, abs=0.15)


def test_the_track_plot_is_the_ensemble_figure_for_one_track(tracks):
    panel = msd_track_panel(tracks, DT, 3, 5, MSDWindow(lag_fraction=0.4))
    n = tracks.filter(pl.col("track_id") == 3).height
    assert (panel["n_points"], panel["alpha_points"]) == (window_lags(n - 1, 0.4), window_lags(n - 1, 0.5))
    assert len(panel["tau_s"]) == max(panel["n_points"], panel["alpha_points"])
    assert panel["unit"] == "pairs" and np.isnan(panel["se_um2"]).all()
    assert panel["intercept_check"]["sigma_sds_um"] == pytest.approx(SE)
    assert panel["intercept_check"]["D_um2_s"] is not None
    assert panel["linear"]["D_um2_s"] is not None


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



def test_alpha_takes_a_wider_window_than_D(tracks):
    assert ensemble_windows(10, 0.3, 1.0) == (3, 10)
    assert ensemble_windows(4, 0.3, 0.5) == (3, 3)  # never under three lags
    groups = {"slow": set(range(150)), "fast": set(range(150, 300))}
    ens = ensemble_msd_blur_free(group_experiments(tracks, DT, groups), 5, 10, n_boot=30)
    fits = ensemble_msd_fits(ens, 3, "fit", alpha_points=10)
    by_model = {(r["group"], r["model"]): r for r in fits.iter_rows(named=True)}
    assert by_model["slow", "linear"]["n_points"] == 3
    assert by_model["slow", "power_law"]["n_points"] == 10
    assert by_model["slow", "power_law"]["alpha"] == pytest.approx(1.0, abs=0.1)
    # D (and the fitted offset alpha is taken net of) come from D's own window.
    same_D = ensemble_msd_fits(ens, 3, "fit").filter(pl.col("model") == "linear")
    assert fits.filter(pl.col("model") == "linear")["D_um2_s"].to_list() == same_D["D_um2_s"].to_list()
    panel = ensemble_msd_panels(ens, 3, "fit", alpha_points=10)[0]
    assert (panel["n_points"], panel["alpha_points"]) == (3, 10)


def test_a_group_too_short_for_the_window_is_flagged_not_fatal(tracks):
    # Every track of "short" has at most 9 lags; the window asks for 12.
    lengths = dict(tracks.group_by("track_id").len().iter_rows())
    short = {t for t, n in lengths.items() if n <= 10}
    groups = {"short": short, "all": None}
    ens = ensemble_msd_blur_free(group_experiments(tracks, DT, groups), 5, 12, n_boot=0)
    fits = ensemble_msd_fits(ens, 4, "provided", alpha_points=12)
    flagged = fits.filter(pl.col("group") == "short")
    assert set(flagged["status"]) == {"insufficient_data"}
    assert "no track reaches lag 12" in flagged["message"][0]
    assert fits.filter(pl.col("group") == "all")["status"].to_list() == ["ok", "ok"]


def test_the_window_round_trips_through_a_saved_summary():
    for window in (MSDWindow(), MSDWindow(5, 12), MSDWindow(lag_fraction=0.3, alpha_lag_fraction=0.6)):
        record = window.record()
        assert MSDWindow.from_settings(record) == window
        restored = DiffusionSettings(**settings_from_summary({"msd": record})).msd_window
        assert restored.text() == window.text()
    with pytest.raises(ValueError, match="msd_lag_fraction"):
        DiffusionSettings(msd_lag_fraction=1.5)
    with pytest.raises(ValueError, match="msd_max_lag"):
        DiffusionSettings(msd_max_lag=2)
    with pytest.raises(ValueError, match="msd_alpha_max_lag"):
        DiffusionSettings(msd_alpha_max_lag=2)
