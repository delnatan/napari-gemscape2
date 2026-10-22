"""Screenshots and frame sequences carry a colour profile, and stepping
through frames leaves the slider where it was."""

import numpy as np
import pytest

pytest.importorskip("pytestqt")


def _icc_description(path):
    from qtpy.QtGui import QColorSpace, QImage

    return QImage(str(path)).colorSpace().description()


@pytest.mark.parametrize("color_space", ["Display P3", "sRGB"])
def test_png_carries_profile_and_pixels(qtbot, tmp_path, color_space):
    from qtpy.QtGui import QImage

    from napari_gemscape2.snapshots import save_png

    rgba = np.zeros((5, 7, 4), np.uint8)
    rgba[..., 0] = np.arange(7) * 30
    rgba[..., 1] = 200
    rgba[..., 3] = 255
    path = save_png(rgba, tmp_path / "shot.png", color_space)
    assert _icc_description(path) == color_space
    image = QImage(str(path)).convertToFormat(QImage.Format.Format_RGB888)
    back = np.frombuffer(image.constBits().asstring(image.sizeInBytes()), np.uint8)
    back = back.reshape(5, image.bytesPerLine())[:, : 7 * 3].reshape(5, 7, 3)
    np.testing.assert_array_equal(back, rgba[..., :3])


class _Viewer:
    """Just what `export_frames` touches: dims and a screenshot of the
    current step."""

    def __init__(self):
        from napari.components import Dims

        self.dims = Dims(ndim=3, range=((0, 6, 1), (0, 4, 1), (0, 4, 1)))
        self.dims.set_current_step(0, 2)

    def screenshot(self, canvas_only, scale, flash):
        frame = np.zeros((4, 4, 4), np.uint8)
        frame[..., 0] = self.dims.current_step[0]
        frame[..., 3] = 255
        return frame


def test_export_frames_numbers_from_zero_and_restores_slider(qtbot, tmp_path):
    from qtpy.QtGui import QImage

    from napari_gemscape2.snapshots import export_frames

    viewer = _Viewer()
    written = export_frames(viewer, tmp_path, "frame", 0, range(1, 6, 2))
    assert [p.name for p in written] == ["frame_00000.png", "frame_00001.png", "frame_00002.png"]
    assert [QImage(str(p)).pixelColor(0, 0).red() for p in written] == [1, 3, 5]
    assert viewer.dims.current_step[0] == 2

    stopped = export_frames(viewer, tmp_path / "early", "frame", 0, range(6), progress=lambda n: n < 2)
    assert len(stopped) == 2
    assert viewer.dims.current_step[0] == 2
