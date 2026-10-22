"""Screenshots and frame sequences of the napari canvas, saved so they look
as they do on screen.

Why napari's own File > Save Screenshot looks dull on a Mac: the canvas is
OpenGL, which Qt and macOS draw without colour management, so its 8-bit
values go to the display as they are, in the display's own gamut -- Display
P3 on every Mac screen since ~2016, whose primaries are a good deal more
saturated than sRGB's. The PNG napari writes carries no colour profile, and
an untagged image is read as sRGB by Preview, Keynote, browsers and ffmpeg
alike: the same numbers mapped onto the narrower sRGB primaries, i.e. the
picture you saw, desaturated. The pixel values are right; the label is
missing.

So every PNG here is written with an ICC profile: Display P3 by default on
macOS (what the canvas looked like there), sRGB elsewhere. A colour-managed
viewer then shows the file as it was on screen. The profile is Qt's own
(`QColorSpace`), so this needs nothing beyond what napari already brings.

Movies are left to ffmpeg (see `FFMPEG_HINT`) -- it has to be told the frames'
colour space too, or it reads them as sRGB again.
"""

from __future__ import annotations

import sys
from contextlib import closing
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

# The profiles offered, by label; the first is the platform's default.
COLOR_SPACES = ("Display P3", "sRGB") if sys.platform == "darwin" else ("sRGB", "Display P3")

# Frame files are `<stem>_<index>.png`, zero-padded to this many digits, so
# both a sorted listing and ffmpeg's `%05d` pattern read them in order.
FRAME_DIGITS = 5

# For a sequence written with the Display P3 profile. ffmpeg ignores a PNG's
# ICC profile, so the primaries are named again for the output (codes 12/13
# are P3-D65 primaries with the sRGB curve); the scale filter makes the
# RGB -> YUV conversion explicit (BT.709 matrix, video range) to match the
# tags. Without these, players assume other primaries and the movie dulls
# again. A GIF has no colour tags at all and is always shown as sRGB, so it
# only matches a sequence exported as sRGB.
FFMPEG_HINT = """\
# MP4 (QuickTime, Keynote, PowerPoint, browsers):
ffmpeg -framerate 10 -i {stem}_%0{digits}d.png \\
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p" \\
  -c:v libx264 -crf 16 \\
  -color_primaries {primaries} -color_trc iec61966-2-1 -colorspace bt709 -color_range tv \\
  {stem}.mp4

# GIF (always shown as sRGB: export the frames as sRGB for this):
ffmpeg -framerate 10 -i {stem}_%0{digits}d.png \\
  -vf "split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=none" \\
  {stem}.gif
"""
FFMPEG_PRIMARIES = {"Display P3": "smpte432", "sRGB": "bt709"}


def qt_color_space(name: str):
    """The `QColorSpace` for one of `COLOR_SPACES`."""
    from qtpy.QtGui import QColorSpace

    named = {"Display P3": QColorSpace.NamedColorSpace.DisplayP3, "sRGB": QColorSpace.NamedColorSpace.SRgb}
    return QColorSpace(named[name])


def save_png(rgba: np.ndarray, path, color_space: str = COLOR_SPACES[0]) -> Path:
    """Write an (H, W, 3|4) uint8 screenshot as a PNG carrying `color_space`'s
    ICC profile. Alpha is dropped: the canvas is opaque, and an alpha channel
    only invites a viewer to composite it over white."""
    from qtpy.QtGui import QImage

    rgb = np.ascontiguousarray(np.asarray(rgba)[..., :3], dtype=np.uint8)
    height, width = rgb.shape[:2]
    image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
    image.setColorSpace(qt_color_space(color_space))
    path = Path(path)
    if not image.save(str(path), "PNG"):
        raise OSError(f"could not write {path}")
    return path


def frame_path(directory, stem: str, index: int) -> Path:
    return Path(directory) / f"{stem}_{index:0{FRAME_DIGITS}d}.png"


def ffmpeg_hint(stem: str, color_space: str) -> str:
    return FFMPEG_HINT.format(stem=stem, digits=FRAME_DIGITS, primaries=FFMPEG_PRIMARIES[color_space])


def iter_frames(
    viewer,
    axis: int,
    steps,
    scale: Optional[float] = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Step `viewer` along dims `axis` through `steps` and yield
    `(step, canvas screenshot)` for each. The slider is put back where it
    was afterwards, even if the caller stops early."""
    dims = viewer.dims
    start = tuple(dims.current_step)
    try:
        for step in steps:
            dims.set_current_step(axis, int(step))
            yield int(step), viewer.screenshot(canvas_only=True, scale=scale, flash=False)
    finally:
        dims.current_step = start


def export_frames(
    viewer,
    directory,
    stem: str,
    axis: int,
    steps,
    color_space: str = COLOR_SPACES[0],
    scale: Optional[float] = None,
    progress: Optional[Callable[[int], bool]] = None,
) -> list[Path]:
    """Save one tagged PNG per step; files are numbered from 0 in the order
    written, whatever the steps themselves are, so ffmpeg reads them as a
    gapless sequence. `progress(n_done)` returning False stops the run."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    with closing(iter_frames(viewer, axis, steps, scale)) as frames:
        for index, (_, frame) in enumerate(frames):
            written.append(save_png(frame, frame_path(directory, stem, index), color_space))
            if progress is not None and progress(index + 1) is False:
                break
    return written
