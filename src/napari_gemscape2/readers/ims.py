"""napari reader for Imaris `.ims` files, so one can be dragged into the viewer.

Display only, like `readers.nd2`: the analysis path (`io_formats.load_stack`) is separate. The file is
read lazily -- a dask array with one chunk per (time point, channel) volume, each opened on demand -- and
each channel becomes its own layer. Size-1 time and z axes are dropped. No `scale=` is set, to match the
rest of the app: layer coordinates are pixels.
"""

from __future__ import annotations

from pathlib import Path


def get_reader(path):
    """napari's hook: a reader for `path` if it is a single `.ims` file, else None."""
    if isinstance(path, (list, tuple)):
        if len(path) != 1:
            return None
        path = path[0]
    if not str(path).lower().endswith(".ims"):
        return None
    return read_ims


def _read_volume(path: str, t: int, c: int):
    from napari_gemscape2.readers.imaris import ImarisReader

    with ImarisReader(path) as reader:
        return reader.read(c=c, t=t)[None, None]  # (1, 1, Z, Y, X)


def read_ims(path):
    """`[(data, add_kwargs, "image")]` for one `.ims` file (napari's LayerData)."""
    import dask
    import dask.array as da

    from napari_gemscape2.readers.imaris import ImarisReader

    if isinstance(path, (list, tuple)):
        path = path[0]
    path = Path(path)

    with ImarisReader(str(path)) as reader:
        n_t, n_c, n_z, n_y, n_x = reader.shape
        dtype = reader.dtype

    read = dask.delayed(_read_volume, pure=True)
    rows = [
        da.concatenate(
            [da.from_delayed(read(str(path), t, c), shape=(1, 1, n_z, n_y, n_x), dtype=dtype) for c in range(n_c)],
            axis=1,
        )
        for t in range(n_t)
    ]
    data = da.concatenate(rows, axis=0)  # (T, C, Z, Y, X)

    kwargs: dict = {"name": path.stem}
    axes = ["T", "C", "Z", "Y", "X"]
    # Drop size-1 axes from the back so the indices stay valid; keep a real channel axis for napari to split.
    for axis in (2, 0):
        if data.shape[axis] == 1:
            data = data.squeeze(axis=axis)
            del axes[axis]
    if data.shape[axes.index("C")] > 1:
        kwargs["channel_axis"] = axes.index("C")
    else:
        data = data.squeeze(axis=axes.index("C"))
    return [(data, kwargs, "image")]
