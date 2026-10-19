"""A detect run replaces the detections, so it drops the tracks linked from the old ones."""

import numpy as np

from napari_gemscape2.pipeline import load_session, run_detect_step, run_track_step
from napari_gemscape2.io_formats import StackMetadata


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


def test_detect_clears_the_previous_link_run(tmp_path):
    stack = _movie()
    session = load_session(tmp_path / "m.tif", pixel_size_um=0.1, dt_s=0.03, exposure_s=0.01,
                           stack=(stack, StackMetadata(path=tmp_path / "m.tif", shape=stack.shape)))
    camera = dict(offset=100.0, gain=1.0)
    run_detect_step(session, sigma=1.3, camera_kwargs=camera)
    run_track_step(session, max_step=4.0, min_track_length=3)
    assert session.tracks_df is not None and session.tracks_df.height

    run_detect_step(session, sigma=1.3, camera_kwargs=camera)
    assert session.points_df.height
    assert session.tracks_df is None
    assert session.track_summary is None and session.max_step_used is None
