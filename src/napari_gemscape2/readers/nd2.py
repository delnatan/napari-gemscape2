"""napari reader for Nikon `.nd2` files, so one can be dragged into the viewer.

This only *displays* the file. The analysis path (`io_formats.load_stack`) is separate: it picks one
channel/z-plane and keeps the calibration for the pipeline. Here the whole file is opened lazily (a dask
array, nothing read until a frame is shown), with each channel as its own layer.

No `scale=` is set, to match the rest of the app: layer coordinates are pixels.
"""

from __future__ import annotations

from pathlib import Path


def get_reader(path):
    """napari's hook: a reader for `path` if it is a single `.nd2` file, else None."""
    if isinstance(path, (list, tuple)):
        if len(path) != 1:
            return None
        path = path[0]
    if not str(path).lower().endswith(".nd2"):
        return None
    return read_nd2


def read_nd2(path):
    """`[(data, add_kwargs, "image")]` for one `.nd2` file (napari's LayerData)."""
    import nd2

    if isinstance(path, (list, tuple)):
        path = path[0]
    path = Path(path)

    with nd2.ND2File(str(path)) as f:
        data = f.to_dask()
        dims = [str(dim).upper() for dim in f.sizes]

    kwargs: dict = {"name": path.stem}
    if "C" in dims and data.shape[dims.index("C")] > 1:
        kwargs["channel_axis"] = dims.index("C")
    elif "C" in dims:
        data = data.squeeze(axis=dims.index("C"))
    return [(data, kwargs, "image")]
