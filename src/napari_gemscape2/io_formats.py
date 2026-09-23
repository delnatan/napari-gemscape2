"""Minimal, in-memory image-stack loading: .tif/.tiff, .nd2, .ims.

Adapted from pyvistra.io (github.com/delnatan/pyvistra, same author) but
deliberately not a dependency on pyvistra itself: pyvistra's lazy 5D
proxy/memmap machinery exists to support its interactive viewer, and this
pipeline always wants the full stack in memory anyway
(`spotsolve.localize_stack` is rayon-parallel over the whole array). Keeping this module self-contained
(numpy/tifffile/h5py + the optional `nd2` package) also means napari-gemscape2
doesn't need to drag in pyvistra's viewer/app dependencies, which matters
since this package is meant to eventually stand alone outside the
microscopy workspace.

Every loader here returns one channel/z-plane as a (T, Y, X) float64
array plus a `StackMetadata` -- the two physical facts the pipeline
cannot run without (`pixel_size_um`, `dt_s`), each with a plain-language
record of *where it came from*, and any notes about how far the file's own
metadata can be trusted. Either may be None, in which case the caller
supplies it explicitly or raises (see `pipeline.load_session`).

A third fact, the camera `exposure_s`, is read the same way but is not
required to run: detection and linking never use it. The diffusion
analysis does -- diffusionkit's displacement MLE models the motion blur
of a continuous exposure, and treating a 20 ms exposure as instantaneous
biases D by about -25% and the non-Brownian score by +0.3 to +0.7. So a
missing exposure is None ("ask the user"), never 0 ("instantaneous").

Provenance, not just values, because everything downstream silently
adopts these two numbers: `x_um`, `duration_s`, `D_um2_s` and every
physical column in a saved bundle is this pixel size and this frame
interval multiplied through. A wrong-by-25400x pixel size (a TIFF whose
resolution is in inches) or a placeholder one (an .ims with no recorded
extents, whose voxel size then computes as 1/width) produces a table that
looks entirely normal, so each loader below states which metadata field
it read, refuses to guess a unit it doesn't recognize, and returns None
rather than a number it can't stand behind. `StackMetadata.notes` carries
what the UI shows the user (`widgets/params_panel.set_image_metadata`)
and what `manifest.json` records (`pipeline.session_manifest_extra`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

import numpy as np
import tifffile

from napari_gemscape2 import units
from napari_gemscape2.readers.imaris import ImarisReader

SUPPORTED_SUFFIXES = {".tif", ".tiff", ".nd2", ".ims"}

# Length units, as any of the spellings the formats here use, in µm. A
# unit that isn't in this table is one this module won't convert: a pixel
# size is only meaningful with its unit, so an unrecognized one yields no
# pixel size rather than a number in unknown units (see `_to_um`).
_LENGTH_UNITS_UM = {
    "m": 1e6,
    "meter": 1e6,
    "metre": 1e6,
    "cm": 1e4,
    "centimeter": 1e4,
    "centimetre": 1e4,
    "mm": 1e3,
    "millimeter": 1e3,
    "millimetre": 1e3,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,  # U+03BC, as opposed to the U+00B5 micro sign above
    "micron": 1.0,
    "microns": 1.0,
    "micrometer": 1.0,
    "micrometre": 1.0,
    "nm": 1e-3,
    "nanometer": 1e-3,
    "nanometre": 1e-3,
    "inch": 25400.0,
    "in": 25400.0,
    '"': 25400.0,
}

# A frame interval this far from the median (relatively) makes the
# timing irregular enough to say so: MSD lag times, `t = frame * dt_s`
# and every fit downstream assume one uniform interval.
_DT_IRREGULAR_FRACTION = 0.05

# Relative difference between the Y and X pixel sizes past which
# averaging them into one number is worth a warning rather than silent.
# `spotsolve` and every physical column here assume square pixels.
_ANISOTROPY_FRACTION = 0.01

UNKNOWN = "not recorded in the file"

# Time units an exposure is recorded in, in seconds. As with lengths, a
# value whose unit isn't here -- or that has no unit at all -- is not
# converted: a bare "20" is 20 ms from one acquisition program and 20 s
# from nowhere, and guessing wrong is a 1000x error in the blur model.
_TIME_UNITS_S = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "msec": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "us": 1e-6,
    "µs": 1e-6,
    "μs": 1e-6,
    "microsecond": 1e-6,
    "microseconds": 1e-6,
}

_VALUE_WITH_UNIT = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([^\d\s].*?)?\s*$")


@dataclass(frozen=True)
class StackMetadata:
    """What one loaded stack's file says about itself.

    `pixel_size_um`/`dt_s` are the two the pipeline needs; the `*_source`
    strings say which metadata field each was read from (or `UNKNOWN`),
    and `notes` holds anything the user should see before trusting them --
    non-square pixels, irregular frame timing, a missing calibration.
    Both `*_source` and `notes` are plain language: they are shown in the
    UI verbatim and written into `manifest.json` as-is."""

    path: Path
    shape: tuple[int, int, int]
    pixel_size_um: Optional[float] = None
    dt_s: Optional[float] = None
    pixel_size_source: str = UNKNOWN
    dt_source: str = UNKNOWN
    channel: int = 0
    z_index: int = 0
    n_channels: int = 1
    n_z: int = 1
    # Median absolute deviation of the per-frame intervals, where the file
    # timestamps each frame (.nd2/.ims). None for a format that records
    # one nominal interval (.tif) -- "no spread measured", not "zero".
    dt_spread_s: Optional[float] = None
    # Camera exposure per frame -- not required to detect or link, only
    # by the diffusion analysis's blur model (see this module's
    # docstring). None when the file doesn't say, in a unit this module
    # can convert.
    exposure_s: Optional[float] = None
    exposure_source: str = UNKNOWN
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_frames(self) -> int:
        return int(self.shape[0])

    def missing(self) -> tuple[str, ...]:
        """Which of the two required facts the file didn't provide."""
        absent = []
        if self.pixel_size_um is None:
            absent.append("pixel size")
        if self.dt_s is None:
            absent.append("frame interval")
        return tuple(absent)

    def summary(self) -> str:
        """One line for the UI: what was parsed, in its units, with
        missing values named rather than defaulted."""
        parts = [f"{self.n_frames} {units.FRAMES}"]
        parts.append(
            f"{units.fmt_unit(self.pixel_size_um, units.UM + '/px')}"
            if self.pixel_size_um is not None
            else "pixel size ?"
        )
        parts.append(
            f"{units.fmt_unit(self.dt_s, 's/frame')}"
            if self.dt_s is not None
            else "frame interval ?"
        )
        parts.append(
            f"exposure {units.fmt_unit(self.exposure_s, units.SECONDS)}"
            if self.exposure_s is not None
            else "exposure ?"
        )
        if self.n_channels > 1:
            parts.append(f"channel {self.channel} of {self.n_channels}")
        if self.n_z > 1:
            parts.append(f"z {self.z_index} of {self.n_z}")
        return " · ".join(parts)

    def detail(self) -> str:
        """The provenance, for a tooltip: every field and where it came
        from, one per line, headed by the file it describes."""
        return f"{self.path}\n{self.provenance()}"

    def provenance(self) -> str:
        """`detail` without the file path -- for showing inside a panel
        that is already about one selected file, where an absolute path is
        the longest and least informative line on offer. The tooltip keeps
        the full thing."""
        lines = [
            f"frames: {self.n_frames}   shape (Y, X): {self.shape[1]} × {self.shape[2]}",
            f"pixel size: {units.fmt_unit(self.pixel_size_um, units.UM + '/px')}"
            f"  ({self.pixel_size_source})",
            f"frame interval: {units.fmt_unit(self.dt_s, 's/frame')}  ({self.dt_source})",
            f"exposure: {units.fmt_unit(self.exposure_s, units.SECONDS)}  ({self.exposure_source})",
        ]
        if self.dt_spread_s is not None:
            lines.append(
                f"frame-interval spread (MAD): {units.fmt_unit(self.dt_spread_s, units.SECONDS)}"
            )
        if self.n_channels > 1 or self.n_z > 1:
            lines.append(
                f"channels: {self.n_channels} (using {self.channel})   "
                f"z planes: {self.n_z} (using {self.z_index})"
            )
        lines.extend(f"note: {note}" for note in self.notes)
        return "\n".join(lines)

    def as_manifest_dict(self) -> dict:
        """The provenance fields for `manifest.json`, so a bundle records
        not just which pixel size was used but where it came from."""
        return {
            "pixel_size_um_source": self.pixel_size_source,
            "dt_s_source": self.dt_source,
            "dt_spread_s": self.dt_spread_s,
            "exposure_s_source": self.exposure_source,
            "n_frames_in_file": self.n_frames,
            "n_channels_in_file": self.n_channels,
            "n_z_in_file": self.n_z,
            "metadata_notes": list(self.notes),
        }


def _positive(value) -> Optional[float]:
    """`value` as a float if it is finite and strictly positive, else
    None. The gate every parsed pixel size / interval passes through: a
    0.0 or NaN from a metadata field that exists but was never set is
    absence, and treating it as a value divides by it later
    (`t = frame * dt_s`, `D = px^2/dt`)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _to_um(value: Optional[float], unit: Optional[str]) -> tuple[Optional[float], Optional[str]]:
    """`value` converted from `unit` into µm, plus the canonical unit name
    it was read as -- `(None, None)` if either is missing or the unit
    isn't one `_LENGTH_UNITS_UM` knows."""
    number = _positive(value)
    if number is None or not unit:
        return None, None
    key = str(unit).strip().lower()
    factor = _LENGTH_UNITS_UM.get(key)
    if factor is None:
        return None, key
    return number * factor, key


def _to_seconds(value, unit: Optional[str] = None) -> Optional[float]:
    """A recorded time as seconds: `value` is a number with `unit`
    beside it, or a string carrying both ("20 ms", "0.02s"). None for
    anything without a unit `_TIME_UNITS_S` knows -- including a bare
    number, whose unit is exactly what can't be assumed -- and for a
    negative or non-finite value. Zero is kept: some acquisitions really
    do record an instantaneous (stroboscopic) exposure."""
    if isinstance(value, str) and unit is None:
        match = _VALUE_WITH_UNIT.match(value)
        if match is None:
            return None
        value, unit = match.group(1), match.group(2)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or not unit:
        return None
    factor = _TIME_UNITS_S.get(str(unit).strip().lower())
    return number * factor if factor is not None else None


def _exposure_from_values(
    recorded: list, where: str, notes: list[str]
) -> tuple[Optional[float], str]:
    """`(exposure_s, source)` from every exposure value one file records
    for the plane in use -- a raw string or number each. One distinct
    convertible value is the answer; several different ones (a multi-
    channel acquisition whose channels can't be told apart in the text)
    or an unconvertible one is noted and treated as missing, since the
    blur model has to be told the real exposure, not a plausible one."""
    if not recorded:
        return None, UNKNOWN
    converted = [_to_seconds(value) for value in recorded]
    if any(value is None for value in converted):
        bad = next(raw for raw, value in zip(recorded, converted) if value is None)
        notes.append(
            f"{where} records an exposure of {bad!r}, with no time unit this reader "
            "knows — exposure treated as missing"
        )
        return None, UNKNOWN
    distinct = sorted(set(converted))
    if len(distinct) > 1:
        notes.append(
            f"{where} records different exposures ("
            + ", ".join(units.fmt_unit(v, units.SECONDS) for v in distinct)
            + ") and this reader can't tell which belongs to this channel — "
            "exposure treated as missing"
        )
        return None, UNKNOWN
    return distinct[0], where


def _mean_pixel_size(
    size_y: Optional[float], size_x: Optional[float], notes: list[str]
) -> Optional[float]:
    """One pixel size from a Y and an X one, warning into `notes` when
    they differ: everything downstream (`spotsolve`'s isotropic PSF,
    `y * pixel_size_um`, every step length) assumes square pixels, so an
    anisotropic acquisition is analyzed slightly wrong in one axis and
    the user should know rather than the average hiding it."""
    if size_y is None or size_x is None:
        return size_y if size_x is None else size_x
    mean = (size_y + size_x) / 2.0
    if mean > 0 and abs(size_y - size_x) / mean > _ANISOTROPY_FRACTION:
        notes.append(
            f"non-square pixels: {size_y:.5g} × {size_x:.5g} {units.UM} (Y × X); "
            f"analysis uses their mean, {mean:.5g} {units.UM}"
        )
    return mean


def _interval_stats(
    times_s: list[Optional[float]], notes: list[str], what: str
) -> tuple[Optional[float], Optional[float]]:
    """`(median interval, MAD of the intervals)` from per-frame absolute
    times, warning into `notes` when the timing is too irregular for one
    interval to describe it -- a dropped or re-triggered frame makes
    `t = frame * dt_s` (and every MSD lag built on it) wrong for every
    frame after it, which is worth a line on screen."""
    diffs = np.array(
        [
            b - a
            for a, b in zip(times_s[:-1], times_s[1:])
            if a is not None and b is not None and b > a
        ],
        dtype=float,
    )
    if diffs.size == 0:
        return None, None
    median = float(np.median(diffs))
    spread = float(np.median(np.abs(diffs - median)))
    if median > 0 and spread / median > _DT_IRREGULAR_FRACTION:
        notes.append(
            f"irregular frame timing in {what}: intervals vary by "
            f"{units.fmt_unit(spread, units.SECONDS)} (MAD) around "
            f"{units.fmt_unit(median, units.SECONDS)}; the analysis assumes one uniform interval"
        )
    return _positive(median), spread


def load_stack(image_path: str | Path, channel: int = 0, z_index: int = 0):
    """Load one channel/z-plane of `image_path` as `(stack, metadata)` --
    a (T, Y, X) float64 array and its `StackMetadata`.

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


def _shape_tyx(im: np.ndarray) -> tuple[int, int, int]:
    """`im`'s shape as (T, Y, X), treating a single 2D frame as T=1."""
    if im.ndim == 2:
        return (1, int(im.shape[0]), int(im.shape[1]))
    return (int(im.shape[0]), int(im.shape[-2]), int(im.shape[-1]))


@dataclass
class _Calibration:
    """One metadata source's answer, before it is merged with another's
    (a TIFF can carry both an OME header and ImageJ's resolution tags).

    Notes are kept per fact rather than in one list, so merging can drop
    the fallback's complaints about a fact the preferred source already
    answered: an OME-TIFF also has bare resolution tags with no unit on
    them, and "pixel size treated as missing" is wrong -- and alarming --
    next to a pixel size the OME header gave perfectly well."""

    pixel_size_um: Optional[float] = None
    dt_s: Optional[float] = None
    pixel_size_source: str = UNKNOWN
    dt_source: str = UNKNOWN
    exposure_s: Optional[float] = None
    exposure_source: str = UNKNOWN
    pixel_notes: list[str] = field(default_factory=list)
    dt_notes: list[str] = field(default_factory=list)
    exposure_notes: list[str] = field(default_factory=list)
    general_notes: list[str] = field(default_factory=list)

    @property
    def notes(self) -> list[str]:
        return [*self.general_notes, *self.pixel_notes, *self.dt_notes, *self.exposure_notes]

    def fill_from(self, other: "_Calibration") -> None:
        """Take whichever of the two facts this source is missing from
        `other`, keeping each value's own provenance with it, and keep
        only the notes that still describe an unanswered fact."""
        wanted_pixel = self.pixel_size_um is None
        wanted_dt = self.dt_s is None
        wanted_exposure = self.exposure_s is None
        if wanted_exposure and other.exposure_s is not None:
            self.exposure_s = other.exposure_s
            self.exposure_source = other.exposure_source
        if wanted_exposure:
            self.exposure_notes.extend(other.exposure_notes)
        if wanted_pixel and other.pixel_size_um is not None:
            self.pixel_size_um = other.pixel_size_um
            self.pixel_size_source = other.pixel_size_source
        if wanted_dt and other.dt_s is not None:
            self.dt_s = other.dt_s
            self.dt_source = other.dt_source
        if wanted_pixel:
            self.pixel_notes.extend(other.pixel_notes)
        if wanted_dt:
            self.dt_notes.extend(other.dt_notes)
        self.general_notes.extend(other.general_notes)


def _ome_calibration(tf) -> _Calibration:
    """An OME-TIFF's own XML header.

    Preferred over the bare `XResolution` tag when both are present: OME
    states the unit explicitly per axis (`PhysicalSizeXUnit`), where a
    TIFF resolution tag only has `ResolutionUnit`'s three values, none of
    which is microns -- so an OME file's own header is the one place a
    micron pixel size is unambiguous."""
    calibration = _Calibration()
    if not getattr(tf, "is_ome", False):
        return calibration
    try:
        root = ElementTree.fromstring(tf.ome_metadata)
    except (ElementTree.ParseError, TypeError, ValueError):
        calibration.general_notes.append("OME-TIFF header present but unreadable")
        return calibration
    pixels = next((el for el in root.iter() if el.tag.rpartition("}")[2] == "Pixels"), None)
    if pixels is None:
        return calibration

    sizes: list[Optional[float]] = []
    for axis in ("Y", "X"):
        value, unit_name = _to_um(
            pixels.get(f"PhysicalSize{axis}"), pixels.get(f"PhysicalSize{axis}Unit", units.UM)
        )
        if value is None and unit_name is not None:
            calibration.pixel_notes.append(
                f"OME PhysicalSize{axis}Unit={unit_name!r} is not a length unit this reader knows"
            )
        sizes.append(value)
    calibration.pixel_size_um = _mean_pixel_size(sizes[0], sizes[1], calibration.pixel_notes)
    if calibration.pixel_size_um is not None:
        calibration.pixel_size_source = "OME-TIFF PhysicalSizeX/Y"

    increment = _positive(pixels.get("TimeIncrement"))
    if increment is not None:
        unit_name = (pixels.get("TimeIncrementUnit") or "s").strip().lower()
        if unit_name in ("s", "sec", "second", "seconds"):
            calibration.dt_s = increment
            calibration.dt_source = "OME-TIFF TimeIncrement"
        elif unit_name in ("ms", "millisecond", "milliseconds"):
            calibration.dt_s = increment / 1000.0
            calibration.dt_source = "OME-TIFF TimeIncrement (ms)"
        else:
            calibration.dt_notes.append(
                f"OME TimeIncrementUnit={unit_name!r} not understood — frame interval ignored"
            )

    # Per-plane, with its own unit attribute (OME's default is seconds).
    # A single-channel stack is all this loader reads, so every plane's
    # exposure should agree; `_exposure_from_values` says so if not.
    recorded = [
        f"{plane.get('ExposureTime')} {plane.get('ExposureTimeUnit', 's')}"
        for plane in root.iter()
        if plane.tag.rpartition("}")[2] == "Plane" and plane.get("ExposureTime") is not None
    ]
    calibration.exposure_s, calibration.exposure_source = _exposure_from_values(
        recorded, "OME-TIFF Plane ExposureTime", calibration.exposure_notes
    )
    return calibration


def _tiff_resolution_unit(tf) -> tuple[Optional[str], str]:
    """`(unit name, where it came from)` for what a TIFF's resolution
    tags are *per*.

    ImageJ writes the real unit into its own metadata block (`unit`:
    "micron", "nm", "inch", or "pixel" for an uncalibrated stack). The
    baseline TIFF `ResolutionUnit` tag only distinguishes inch (2) from
    centimetre (3) from none (1), so it can never say microns -- which is
    why a micron pixel size read off `XResolution` alone is an assumption
    rather than a measurement."""
    ij = tf.imagej_metadata or {}
    unit_name = ij.get("unit")
    if unit_name:
        return str(unit_name), "ImageJ 'unit'"
    tag = tf.pages[0].tags.get("ResolutionUnit")
    raw = getattr(tag, "value", None)
    try:
        code = int(getattr(raw, "value", raw) or 0)
    except (TypeError, ValueError):
        code = 0
    return {2: "inch", 3: "cm"}.get(code), f"TIFF ResolutionUnit={code}"


def _pixels_per_unit(tf, tag_name: str) -> Optional[float]:
    """A TIFF resolution tag as a plain "pixels per unit" float -- it is
    stored as a rational."""
    tag = tf.pages[0].tags.get(tag_name)
    if tag is None:
        return None
    try:
        numerator, denominator = tag.value
        return _positive(numerator / denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return _positive(getattr(tag, "value", None))


def _imagej_calibration(tf) -> _Calibration:
    """An ImageJ/plain TIFF's `XResolution`/`YResolution` tags plus the
    unit they are per (`_tiff_resolution_unit`), and ImageJ's own frame
    interval.

    The resolution tags give pixels per unit, so the pixel size is their
    reciprocal -- in that unit. Without a unit this reader can convert,
    no pixel size is returned, rather than the previous behaviour of
    assuming microns: that silently mis-scaled an inch-calibrated file
    (every physical column out by 25400x) and read an uncalibrated
    "pixel"-unit stack as if 1 px were 1 µm."""
    ij = tf.imagej_metadata or {}
    calibration = _Calibration()
    unit_name, unit_source = _tiff_resolution_unit(tf)

    sizes: list[Optional[float]] = []
    for tag_name in ("YResolution", "XResolution"):
        per_unit = _pixels_per_unit(tf, tag_name)
        value, _ = _to_um(1.0 / per_unit if per_unit else None, unit_name)
        if value is None and per_unit is not None and tag_name == "XResolution":
            calibration.pixel_notes.append(
                f"TIFF resolution is {per_unit:.6g} px per {unit_name!r} ({unit_source}), "
                "which this reader cannot convert to µm — pixel size treated as missing"
                if unit_name
                else "TIFF has a resolution tag but records no unit for it "
                f"({unit_source}) — pixel size treated as missing"
            )
        sizes.append(value)
    calibration.pixel_size_um = _mean_pixel_size(sizes[0], sizes[1], calibration.pixel_notes)
    if calibration.pixel_size_um is not None:
        calibration.pixel_size_source = f"TIFF XResolution, {unit_source}={unit_name!r}"

    # ImageJ records either the interval directly or a frame rate.
    interval = _positive(ij.get("finterval"))
    fps = _positive(ij.get("fps"))
    if interval is not None:
        calibration.dt_s, calibration.dt_source = interval, "ImageJ 'finterval'"
    elif fps is not None:
        calibration.dt_s, calibration.dt_source = 1.0 / fps, "ImageJ 'fps'"
    elif "finterval" in ij or "fps" in ij:
        calibration.dt_notes.append(
            "ImageJ frame interval is recorded but zero/unset — treated as missing"
        )
    return calibration


def _load_tiff(image_path: Path):
    with tifffile.TiffFile(image_path) as tf:
        im = tf.asarray().astype(np.float64)
        calibration = _ome_calibration(tf)
        calibration.fill_from(_imagej_calibration(tf))

    metadata = StackMetadata(
        path=image_path,
        shape=_shape_tyx(im),
        pixel_size_um=calibration.pixel_size_um,
        dt_s=calibration.dt_s,
        pixel_size_source=calibration.pixel_size_source,
        dt_source=calibration.dt_source,
        exposure_s=calibration.exposure_s,
        exposure_source=calibration.exposure_source,
        notes=tuple(calibration.notes),
    )
    return im, metadata


def _load_nd2(image_path: Path, channel: int, z_index: int):
    try:
        import nd2
    except ImportError as exc:
        raise ImportError(
            "Reading .nd2 files requires the 'nd2' package (pip install nd2)"
        ) from exc

    notes: list[str] = []
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
        vy = _positive(getattr(voxel, "y", None))
        vx = _positive(getattr(voxel, "x", None))
        # nd2's own fallback for "no calibration" is exactly 1.0 um/px on
        # both axes -- indistinguishable from a real 1 um pixel, but a
        # 1 um pixel is not a single-molecule acquisition, so it is read
        # as "uncalibrated" and said so rather than analyzed.
        if vy == 1.0 and vx == 1.0:
            pixel_size_um = None
            pixel_source = UNKNOWN
            notes.append(
                "nd2 voxel size is exactly 1 × 1 µm, which is the reader's "
                "placeholder for an uncalibrated file — treated as missing"
            )
        else:
            pixel_size_um = _mean_pixel_size(vy, vx, notes)
            pixel_source = "nd2 voxel_size()"

        dt_s, dt_spread_s, dt_source = _nd2_frame_interval_s(f, stack.shape[0], notes)
        exposure_s, exposure_source = _nd2_exposure_s(f, notes)
        n_channels = int(f.sizes.get("C", 1))
        n_z = int(f.sizes.get("Z", 1))

    metadata = StackMetadata(
        path=image_path,
        shape=_shape_tyx(stack),
        pixel_size_um=pixel_size_um,
        dt_s=dt_s,
        pixel_size_source=pixel_source,
        dt_source=dt_source,
        channel=channel,
        z_index=z_index,
        n_channels=n_channels,
        n_z=n_z,
        dt_spread_s=dt_spread_s,
        exposure_s=exposure_s,
        exposure_source=exposure_source,
        notes=tuple(notes),
    )
    return stack, metadata


_ND2_EXPOSURE = re.compile(r"Exposure:\s*([^\r\n,;]+)", re.IGNORECASE)


def _nd2_exposure_s(nd2_file, notes: list[str]) -> tuple[Optional[float], str]:
    """The camera exposure NIS-Elements writes into its capture text
    ("Exposure: 20 ms"). The `capturing` block is read in preference to
    `description`, which repeats it; a multi-channel file lists one line
    per channel, and `_exposure_from_values` refuses to pick among
    differing ones rather than guess which is this channel's."""
    try:
        text_info = nd2_file.text_info or {}
    except Exception:
        return None, UNKNOWN
    for key in ("capturing", "description"):
        recorded = _ND2_EXPOSURE.findall(str(text_info.get(key) or ""))
        if recorded:
            return _exposure_from_values(
                [value.strip() for value in recorded], f"nd2 text_info '{key}' Exposure", notes
            )
    return None, UNKNOWN


def _nd2_frame_interval_s(nd2_file, t_size: int, notes: list[str]):
    """`(median interval, MAD, source)` from nd2's per-frame relative
    timestamps -- the acquisition's real timing rather than its requested
    interval, which is why the spread is worth reporting (see
    `_interval_stats`)."""
    loop_indices = getattr(nd2_file, "loop_indices", None) or []
    if not loop_indices or t_size < 2:
        return None, None, UNKNOWN

    t_seconds: list[Optional[float]] = [None] * t_size
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

    dt_s, spread = _interval_stats(t_seconds, notes, "nd2 frame timestamps")
    return dt_s, spread, "nd2 per-frame timestamps (median)" if dt_s else UNKNOWN


def _load_ims(image_path: Path, channel: int, z_index: int):
    notes: list[str] = []
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

        # An .ims with no recorded extents leaves ImarisReader's defaults
        # in place, from which a voxel size of 1/width computes -- a
        # plausible-looking 0.001 um/px that is not a calibration. The
        # reader reports whether the extents were really there, so that
        # case reads as missing instead.
        _, vy, vx = reader.voxel_size
        pixel_size_um = None
        pixel_source = UNKNOWN
        if getattr(reader, "voxel_size_known", True):
            unit = getattr(reader, "length_unit", None) or units.UM
            size_y, unit_name = _to_um(vy, unit)
            size_x, _ = _to_um(vx, unit)
            if size_y is None and size_x is None:
                notes.append(
                    f"Imaris extents are in {unit_name or unit!r}, not a length unit this "
                    "reader knows — treated as missing"
                )
            else:
                pixel_size_um = _mean_pixel_size(size_y, size_x, notes)
                pixel_source = f"Imaris DataSetInfo/Image extents (unit {unit_name or unit})"
        else:
            notes.append(
                "Imaris file records no image extents, so its voxel size is a "
                "placeholder derived from the frame width — treated as missing"
            )

        # Relative to the first timestamp that exists, not to
        # `timestamps[0]` -- an .ims whose first time point failed to
        # parse has None there, and subtracting it raises. Unparsed
        # entries stay None and their neighbouring intervals are skipped
        # rather than measured across the gap.
        base = next((ts for ts in reader.timestamps if ts is not None), None)
        times = [
            None if (ts is None or base is None) else (ts - base).total_seconds()
            for ts in reader.timestamps
        ]
        dt_s, dt_spread_s = _interval_stats(times, notes, "Imaris timestamps")

        # `ImarisReader` hands back the attribute as a float when it is a
        # bare number and as the raw string when it carries a unit ("20
        # ms"); only the latter can be converted. Imaris's own writer puts
        # it on `Channel <i>`. Andor Fusion leaves it out and records the
        # camera's exposure in its acquisition protocol instead
        # (`ImarisReader.fusion_camera_exposures`), read when the channel
        # attribute is absent.
        info = reader.channels_info[channel] or {}
        recorded = info.get("exposure_time")
        if recorded is not None:
            exposure_s, exposure_source = _exposure_from_values(
                [recorded], f"Imaris Channel {channel} ExposureTime", notes
            )
        else:
            exposure_s, exposure_source = _exposure_from_values(
                reader.fusion_camera_exposures(info.get("name")),
                "Andor Fusion protocol (camera ExposureTime)",
                notes,
            )
    finally:
        reader.close()

    metadata = StackMetadata(
        path=image_path,
        shape=_shape_tyx(stack),
        pixel_size_um=pixel_size_um,
        dt_s=dt_s,
        pixel_size_source=pixel_source,
        dt_source="Imaris TimeInfo timestamps (median)" if dt_s else UNKNOWN,
        channel=channel,
        z_index=z_index,
        n_channels=int(n_c),
        n_z=int(n_z),
        dt_spread_s=dt_spread_s,
        exposure_s=exposure_s,
        exposure_source=exposure_source,
        notes=tuple(notes),
    )
    return stack, metadata
