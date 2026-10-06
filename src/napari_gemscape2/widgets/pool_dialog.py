"""The experiment list's "Pool analyses…": pool the saved diffusion
analyses of several movies by sample, in the background.

`PoolDialog` picks the movies, names each one's sample and sets the
optional ensemble MSD; `run_pool_worker` reads each movie's saved analysis
back (`pooling.load_pooled_bundle`) and runs `pooling.run_pooling` -- what
`gemscape2 pool` runs -- writing the same tables to the same folder, so a
pooling done here is the one the CLI would do from the config
`pooling.write_pool_config` records.

Nothing is refitted: each movie's saved posteriors and the filters it was
saved with are what is pooled, so a movie only appears here once its
analysis is saved (the Diffusion analysis widget's "Save analysis", a
batch, or `gemscape2 diffusion`). Movies that share a sample are
replicates, and a sample's population weighs every track the same,
whichever movie it came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from napari.qt.threading import thread_worker
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from napari_gemscape2.diffusion import ENSEMBLE_MSD_BOOT, ENSEMBLE_MSD_OFFSETS, MSD_MIN_LAG
from napari_gemscape2.pooling import PoolSettings, guess_sample, load_pooled_bundle, run_pooling, write_pool
from napari_gemscape2.results import DIFFUSION_SUMMARY_FILENAME

_OFFSET_LABELS = {"provided": "from the SDs", "fit": "fitted (intercept)"}


def has_saved_analysis(result_dir: Path) -> bool:
    """Whether a movie's bundle has a saved diffusion analysis to pool."""
    return (Path(result_dir) / DIFFUSION_SUMMARY_FILENAME).is_file()


@dataclass
class PoolPlan:
    results_root: Path
    # (image_path, result_dir, sample), in the list's order.
    inputs: list[tuple[Path, Path, str]]
    settings: PoolSettings
    out_dir: Path
    write_config: bool


class PoolDialog(QDialog):
    """Pick the movies to pool and name each one's sample. `entries` are
    the list's rows with a saved analysis, as `(image_path, result_dir)`."""

    def __init__(self, results_root: Path, entries: list[tuple[Path, Path]], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pool analyses")
        self.setMinimumWidth(560)
        self._results_root = results_root
        self._out_dir = results_root / "pooled"

        note = QLabel(
            "Pools each movie's <i>saved</i> analysis -- its posteriors and the tracks its filters "
            "passed; nothing is refitted. Movies with the same sample are replicates: a sample's "
            "distribution weighs every track the same, whichever movie it came from."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")

        self.table = QTableWidget(len(entries), 2)
        self.table.setHorizontalHeaderLabels(["Movie", "Sample"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        for row, (image_path, result_dir) in enumerate(entries):
            movie = QTableWidgetItem(result_dir.name)
            movie.setData(Qt.ItemDataRole.UserRole, (image_path, result_dir))
            movie.setToolTip(str(result_dir))
            movie.setFlags((movie.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable)
            movie.setCheckState(Qt.CheckState.Checked)
            self.table.setItem(row, 0, movie)
            # Each movie is its own sample until named otherwise, as in the CLI.
            self.table.setItem(row, 1, QTableWidgetItem(result_dir.name))

        all_button = QPushButton("All")
        all_button.clicked.connect(lambda: self._check_all(True))
        none_button = QPushButton("None")
        none_button.clicked.connect(lambda: self._check_all(False))
        guess_button = QPushButton("Samples from names")
        guess_button.setToolTip(
            "Name each movie's sample after it, less a trailing replicate\n"
            "number: wt_1 and wt_2 become wt. Check the result before pooling."
        )
        guess_button.clicked.connect(self._guess_samples)
        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("Movies with a saved analysis:"))
        select_row.addStretch(1)
        for button in (guess_button, all_button, none_button):
            select_row.addWidget(button)

        # The ensemble MSD: off unless asked for, and its windows explicit.
        self.ensemble_check = QCheckBox("Also the ensemble-averaged MSD of each sample")
        self.ensemble_check.setToolTip(
            "Recomputed from the pooled tracks with the exposure treated as 0 (the\n"
            "MSD estimators have no blur model), pair-weighted, with intervals from\n"
            "resampling whole tracks."
        )
        self.max_lag = _spin(10, "Each track's MSD is computed to this lag before averaging.")
        self.n_points = _spin(4, "The fits use the averaged curve's first this many lags\n(the usual rule: 25-40% of it).")
        self.offset = QComboBox()
        for key in ENSEMBLE_MSD_OFFSETS:
            self.offset.addItem(_OFFSET_LABELS[key], key)
        self.offset.setToolTip(
            "The localization offset the fits take off: the tracks' SDs, or the\n"
            "intercept of a linear fit to the curve (no SDs used)."
        )
        self.n_boot = _spin(ENSEMBLE_MSD_BOOT, "Bootstrap resamples of whole tracks.", minimum=0, maximum=10_000)
        ensemble_row = QHBoxLayout()
        for widget in (QLabel("to lag"), self.max_lag, QLabel("fit first"), self.n_points, QLabel("offset"),
                       self.offset, QLabel("resamples"), self.n_boot):
            ensemble_row.addWidget(widget)
        ensemble_row.addStretch(1)
        self._ensemble_widgets = (self.max_lag, self.n_points, self.offset, self.n_boot)

        self.out_label = QLabel()
        self.out_label.setWordWrap(True)
        out_button = QPushButton("Change…")
        out_button.clicked.connect(self._choose_out_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_label, 1)
        out_row.addWidget(out_button)

        self.config_check = QCheckBox("Also write the pool config (results folder)")
        self.config_check.setChecked(True)
        self.config_check.setToolTip(
            "A TOML config naming these movies and their samples, so\n"
            "`gemscape2 pool` can re-run this pooling headless."
        )

        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #f59e0b;")

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.run_button = self.buttons.addButton("Pool", QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(note)
        layout.addLayout(select_row)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.ensemble_check)
        layout.addLayout(ensemble_row)
        layout.addLayout(out_row)
        layout.addWidget(self.config_check)
        layout.addWidget(self.warning)
        layout.addWidget(self.buttons)

        self.table.itemChanged.connect(self._refresh)
        self.ensemble_check.toggled.connect(self._refresh)
        self.max_lag.valueChanged.connect(self._refresh)
        self.n_points.valueChanged.connect(self._refresh)
        self._refresh()

    def _rows(self) -> range:
        return range(self.table.rowCount())

    def _check_all(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row in self._rows():
            self.table.item(row, 0).setCheckState(state)

    def _guess_samples(self) -> None:
        for row in self._rows():
            self.table.item(row, 1).setText(guess_sample(self.table.item(row, 0).text()))

    def _choose_out_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Write the pooled tables to", str(self._out_dir.parent))
        if folder:
            self._out_dir = Path(folder)
            self._refresh()

    def inputs(self) -> list[tuple[Path, Path, str]]:
        out = []
        for row in self._rows():
            movie = self.table.item(row, 0)
            if movie.checkState() == Qt.CheckState.Checked:
                image_path, result_dir = movie.data(Qt.ItemDataRole.UserRole)
                sample = self.table.item(row, 1).text().strip() or result_dir.name
                out.append((image_path, result_dir, sample))
        return out

    def settings(self) -> PoolSettings:
        """The `[pool]` settings; raises ValueError for an impossible window."""
        if not self.ensemble_check.isChecked():
            return PoolSettings()
        return PoolSettings(
            ensemble_msd=True,
            ensemble_max_lag=self.max_lag.value(),
            ensemble_n_points=self.n_points.value(),
            ensemble_offset=self.offset.currentData(),
            n_boot=self.n_boot.value(),
        )

    def _refresh(self, *_args) -> None:
        for widget in self._ensemble_widgets:
            widget.setEnabled(self.ensemble_check.isChecked())
        self.out_label.setText(f"Write the pooled tables to: {self._out_dir}")
        inputs = self.inputs()
        samples = {sample for _image, _dir, sample in inputs}
        problem = ""
        try:
            self.settings()
        except ValueError as exc:
            problem = str(exc)
        if not inputs:
            problem = problem or "Pick at least one movie."
        self.run_button.setEnabled(bool(inputs) and not problem)
        self.run_button.setText(f"Pool {len(inputs)} into {len(samples)}" if inputs else "Pool")
        if not problem and len(samples) == len(inputs) > 1:
            problem = "Every movie is its own sample: name replicates alike to pool them."
        self.warning.setText(problem)

    def plan(self) -> PoolPlan:
        return PoolPlan(
            results_root=self._results_root,
            inputs=self.inputs(),
            settings=self.settings(),
            out_dir=self._out_dir,
            write_config=self.config_check.isChecked(),
        )


def _spin(value: int, tooltip: str, minimum: int = MSD_MIN_LAG, maximum: int = 1000) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setToolTip(tooltip)
    return box


class PoolEmitter(QObject):
    """Reports from the pooling thread; Qt queues each onto the GUI thread."""

    stage = Signal(str)


@thread_worker(start_thread=False)
def run_pool_worker(plan: PoolPlan, emitter: PoolEmitter):
    """Read each movie's saved analysis, pool them and write the tables.
    Returns the `pooling.PoolResult`; a movie that can't be read stops the
    pooling (a missing replicate would bias it), naming the movie."""
    bundles = []
    for index, (_image_path, result_dir, sample) in enumerate(plan.inputs):
        emitter.stage.emit(f"[{index + 1}/{len(plan.inputs)}] reading {result_dir.name}")
        try:
            bundles.append(load_pooled_bundle(result_dir, sample))
        except (ValueError, OSError) as exc:
            raise ValueError(f"{result_dir.name}: {exc}") from exc
    result = run_pooling(bundles, plan.settings, progress=emitter.stage.emit)
    emitter.stage.emit("writing the tables")
    write_pool(plan.out_dir, result)
    return result


def pool_config_path(results_root: Path) -> Path:
    """Where a GUI pooling's config goes: the results folder, numbered
    rather than overwriting an earlier one."""
    path, n = results_root / "pool.toml", 2
    while path.exists():
        path = results_root / f"pool_{n}.toml"
        n += 1
    return path
