"""Minimal, in-memory image-stack loading: .tif/.tiff, .nd2, .ims.

Adapted from pyvistra.io (github.com/delnatan/pyvistra, same author) but
deliberately not a dependency on pyvistra itself: pyvistra's lazy 5D
proxy/memmap machinery exists to support its interactive viewer, and this
pipeline always wants the full stack in memory anyway
(`spotsolve.localize_stack` is rayon-parallel over the whole array). Keeping this module self-contained
(numpy/tifffile/h5py + the optional `nd2` package) also means spt-pipeline
doesn't need to drag in pyvistra's viewer/app dependencies, which matters
since this package is meant to eventually stand alone outside the
microscopy workspace.

Every loader here returns one channel/z-plane as a (T, Y, X) float64
array plus (pixel_size_um, dt_s) -- both None if not recoverable from the
file's own metadata, in which case the caller supplies them explicitly or
raises (see pipeline.run_detect_track).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from spt_pipeline.readers.imaris import ImarisReader

SUPPORTED_SUFFIXES = {".tif", ".tiff", ".nd2", ".ims"}


def load_stack(image_path: str | Path, channel: int = 0, z_index: int = 0):
    """Load one channel/z-plane of `image_path` as (stack, pixel_size_um, dt_s).

    `channel`/`z_index` (both default 0, always valid -- every axis is
    size >= 1) pick which plane to use for files with more than one.
    """
    image_path = Path(image_path)
    suffix = image_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        return _load_tiff(image_path)
    if suffix == ".nd2":
        return _load_nd2(image_path, channel, z_index)
    if suffix == ".ims":
        return _load_ims(image_path, channel, z_index)
    raise ValueError(
        f"{image_path}: unsupported format {suffix!r} "
        f"(supported: {sorted(SUPPORTED_SUFFIXES)})"
    )


def _load_tiff(image_path: Path):
    with tifffile.TiffFile(image_path) as tf:
        im = tf.asarray().astype(np.float64)
        ij = tf.imagej_metadata or {}
        pixel_size_um = None
        try:
            xres_num, xres_den = tf.pages[0].tags["XResolution"].value
            pixel_size_um = xres_den / xres_num
        except KeyError:
            pass
        dt_s = ij.get("finterval")
    return im, pixel_size_um, dt_s


def _load_nd2(image_path: Path, channel: int, z_index: int):
    try:
        import nd2
    except ImportError as exc:
        raise ImportError(
            "Reading .nd2 files requires the 'nd2' package (pip install nd2)"
        ) from exc

    with nd2.ND2File(str(image_path)) as f:
        raw = np.asarray(f.asarray())
        axis = {dim.lower(): i for i, dim in enumerate(f.sizes.keys())}

        if "y" not in axis or "x" not in axis:
            raise ValueError(f"{image_path}: no Y/X axes in nd2 sizes {f.sizes}")
        extra = [d for d in axis if d not in ("t", "z", "c", "y", "x")]
        if extra:
            raise ValueError(
                f"{image_path}: unsupported nd2 axes {extra} in {f.sizes} "
                "(only T/Z/C/Y/X handled)"
            )

        # Reorder to canonical (t, z, c, y, x) -- omitting whichever axes
        # this file doesn't have -- then index z/c down to the requested
        # plane, leaving (T, Y, X) or (Y, X) if the file has no T axis.
        present = [d for d in ("t", "z", "c", "y", "x") if d in axis]
        transposed = np.transpose(raw, [axis[d] for d in present])
        selector = tuple(
            z_index if d == "z" else channel if d == "c" else slice(None)
            for d in present
        )
        plane = transposed[selector]
        if "t" not in present:
            plane = plane[np.newaxis, ...]
        stack = np.asarray(plane, dtype=np.float64)

        voxel = f.voxel_size()
        vy = float(getattr(voxel, "y", 1.0) or 1.0)
        vx = float(getattr(voxel, "x", 1.0) or 1.0)
        # nd2's own fallback for "no calibration" is exactly 1.0 um/px.
        pixel_size_um = None if (vy == 1.0 and vx == 1.0) else (vy + vx) / 2.0

        dt_s = _nd2_frame_interval_s(f, stack.shape[0])

    return stack, pixel_size_um, dt_s


def _nd2_frame_interval_s(nd2_file, t_size: int):
    """Median frame interval (s) from nd2's per-frame relative timestamps."""
    loop_indices = getattr(nd2_file, "loop_indices", None) or []
    if not loop_indices or t_size < 2:
        return None

    t_seconds = [None] * t_size
    for seq_idx, index_map in enumerate(loop_indices):
        t_idx = index_map.get("T") if isinstance(index_map, dict) else None
        if t_idx is None or t_idx >= t_size or t_seconds[t_idx] is not None:
            continue
        try:
            frame_meta = nd2_file.frame_metadata(seq_idx)
        except Exception:
            continue
        channels = getattr(frame_meta, "channels", None) or []
        if not channels:
            continue
        time_meta = getattr(channels[0], "time", None)
        rel_ms = getattr(time_meta, "relativeTimeMs", None)
        if rel_ms is None:
            continue
        try:
            t_seconds[t_idx] = float(rel_ms) / 1000.0
        except (TypeError, ValueError):
            continue

    diffs = [
        b - a
        for a, b in zip(t_seconds[:-1], t_seconds[1:])
        if a is not None and b is not None and b > a
    ]
    return float(np.median(diffs)) if diffs else None


def _load_ims(image_path: Path, channel: int, z_index: int):
    reader = ImarisReader(str(image_path))
    try:
        n_t, n_c, n_z, h, w = reader.shape
        if not (0 <= channel < n_c):
            raise ValueError(f"{image_path}: channel={channel} out of range for n_channels={n_c}")
        if not (0 <= z_index < n_z):
            raise ValueError(f"{image_path}: z_index={z_index} out of range for size_z={n_z}")

        stack = np.empty((n_t, h, w), dtype=np.float64)
        for t in range(n_t):
            stack[t] = reader.read(c=channel, t=t, z=z_index)

        _, vy, vx = reader.voxel_size
        pixel_size_um = float((vy + vx) / 2.0)

        timestamps = [ts for ts in reader.timestamps if ts is not None]
        dt_s = None
        if len(timestamps) >= 2:
            diffs = [
                (b - a).total_seconds()
                for a, b in zip(timestamps[:-1], timestamps[1:])
                if (b - a).total_seconds() > 0
            ]
            if diffs:
                dt_s = float(np.median(diffs))
    finally:
        reader.close()

    return stack, pixel_size_um, dt_s
