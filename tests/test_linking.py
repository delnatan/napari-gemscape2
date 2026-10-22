"""Linking settings: the search radius read from the movie or given, the
switch cost, and `link_p` carried onto the tracks, the summary and the
manifest."""

import numpy as np
import polars as pl
import pytest

from napari_gemscape2.io_formats import StackMetadata
from napari_gemscape2.pipeline import (
    SWITCH_COST,
    detect_track_params_from_manifest,
    load_session,
    run_detect_step,
    run_track_step,
    session_manifest_extra,
    track_features_df,
    track_metrics_df,
)


def _movie(seed=0, T=20, H=48, W=48, n=6):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W]
    pos = rng.uniform(8, H - 8, (n, 2))
    img = np.full((T, H, W), 100.0)
    for t in range(T):
        pos = np.clip(pos + rng.normal(0, 0.7, (n, 2)), 5, H - 5)
        for y, x in pos:
            img[t] += 400 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.3**2))
    return rng.poisson(img).astype(np.uint16)


@pytest.fixture(scope="module")
def detected(tmp_path_factory):
    path = tmp_path_factory.mktemp("m") / "m.tif"
    stack = _movie()
    session = load_session(path, pixel_size_um=0.1, dt_s=0.03, exposure_s=0.01,
                           stack=(stack, StackMetadata(path=path, shape=stack.shape)))
    run_detect_step(session, sigma=1.3, camera_kwargs=dict(offset=100.0, gain=1.0))
    return session


def _manifest(session):
    return {"params": session_manifest_extra(session)}


def test_radius_read_from_the_movie(detected):
    session = run_track_step(detected, min_track_length=3)
    assert session.search_radius_source == "movie"
    assert np.isfinite(session.search_radius_used) and session.search_radius_used > 0
    assert session.switch_cost_used == SWITCH_COST

    tracks = session.tracks_df
    starts = tracks.sort(["track_id", "frame"]).group_by("track_id", maintain_order=True).head(1)
    assert starts["link_p"].null_count() == starts.height
    links = tracks["link_p"].drop_nulls()
    assert links.len() and links.min() >= 0 and links.max() <= 1
    assert session.track_summary["mean_link_p"] == pytest.approx(links.mean())

    params = _manifest(session)["params"]
    assert params["search_radius_source"] == "movie"
    assert params["search_radius_px"] == session.search_radius_used
    assert params["switch_cost"] == SWITCH_COST
    # Each movie reads its own radius: a template passes on only the switch cost.
    template = detect_track_params_from_manifest(_manifest(session))
    assert "search_radius" not in template and template["switch_cost"] == SWITCH_COST


def test_given_radius_and_switch_cost(detected):
    session = run_track_step(detected, search_radius=5.0, switch_cost=1.0, min_track_length=3)
    assert session.search_radius_source == "given" and session.search_radius_used == 5.0
    assert session.switch_cost_used == 1.0
    template = detect_track_params_from_manifest(_manifest(session))
    assert template["search_radius"] == 5.0 and template["switch_cost"] == 1.0


@pytest.mark.parametrize("kwargs", [dict(search_radius=0.0), dict(search_radius=np.inf),
                                    dict(switch_cost=-1.0), dict(switch_cost=np.nan)])
def test_bad_settings_refused(detected, kwargs):
    with pytest.raises(ValueError):
        run_track_step(detected, **kwargs)


def test_min_link_p_and_layer_features(detected):
    session = run_track_step(detected, min_track_length=1)
    tracks = session.tracks_df
    metrics = track_metrics_df(tracks, 0.1, 0.03)
    weakest = tracks.group_by("track_id").agg(pl.col("link_p").min().alias("p"))
    joined = metrics.join(weakest, on="track_id")
    assert joined["min_link_p"].null_count() == 0
    has_link = joined["p"].is_not_null()
    assert (joined.filter(has_link)["min_link_p"] == joined.filter(has_link)["p"]).all()
    assert (joined.filter(~has_link)["min_link_p"] == 1.0).all()
    # The Tracks layer's colormaps need a value on every vertex.
    features = track_features_df(tracks, 0.1, 0.03)
    assert features["link_p"].null_count() == 0 and not features["link_p"].is_nan().any()
