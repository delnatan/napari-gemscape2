"""The experiment list: each folder loaded keeps its bundles in its own results/."""

import numpy as np
import pytest
import tifffile

pytest.importorskip("pytestqt")


def _folder(root, name):
    folder = root / name
    folder.mkdir()
    tifffile.imwrite(folder / f"{name}_movie.tif", np.zeros((3, 16, 16), np.uint16))
    return folder


class _Drop:
    """What `dropEvent` reads off a drag of one folder from the file browser."""

    def __init__(self, folder):
        from qtpy.QtCore import QMimeData, QUrl

        self._mime = QMimeData()
        self._mime.setUrls([QUrl.fromLocalFile(str(folder))])

    def mimeData(self):
        return self._mime


@pytest.mark.parametrize("onto", ["list", "widget"])
def test_each_folder_keeps_its_own_results(qtbot, tmp_path, onto):
    from napari.components import ViewerModel

    from napari_gemscape2.widgets.experiment_list import ExperimentListWidget

    widget = ExperimentListWidget(ViewerModel())
    qtbot.addWidget(widget)
    view = widget.list_view
    first, second = _folder(tmp_path, "first"), _folder(tmp_path, "second")
    view.load_folder(first)
    for folder in (second, first):
        (view if onto == "list" else widget).dropEvent(_Drop(folder))
        assert view.results_root == folder / "results"
        assert [item.entry.result_dir.parent for item in view.items()] == [folder / "results"]
