"""The experiment list's "Batch from this movie…": run the saved
settings of one analyzed movie (the template) over other movies in the
same folder, in the background.

`BatchDialog` picks the movies and the steps; `run_batch_worker` runs
them through `batch.detect_track_bundle` and
`diffusion_batch.analyze_bundle` -- the same calls `gemscape2
detect-track`/`diffusion` make -- so a bundle written here is the one
the CLI would write from the config `batch.write_batch_config` records.

Only what the template *saved* is reused: detection, linking and
point/track filters from its manifest, and the diffusion analysis
settings from its `diffusion_summary.json` (written by the Diffusion
analysis widget's "Save analysis"). Regions are per movie, not the
template's: a movie with a mask saved in its result dir (painted in the
widget, see `results.save_regions`) is restricted to it, and one without
is analyzed over the whole field -- the movie list marks which is which.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from napari.qt.threading import thread_worker
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from napari_gemscape2.pipeline import (
    DetectTrackParams,
    PipelineCancelled,
    detect_track_params_from_manifest,
)
from napari_gemscape2.results import (
    has_regions,
    has_result,
    load_diffusion_summary,
    load_manifest,
    repo_shas,
)


@dataclass
class BatchJob:
    image_path: Path
    result_dir: Path


@dataclass
class BatchPlan:
    template: Path
    results_root: Path
    jobs: list[BatchJob]
    detect_track: bool
    diffusion: bool
    write_config: bool


def _detect_track_summary(params: dict) -> str:
    if params.get("sigma") is None:
        return "no PSF width (sigma) saved -- detect+track can't be batched from it"
    if params.get("max_step") is None:
        return (
            "no max step saved (linked before spotsolve 0.2) -- re-run tracking on it "
            "and save, then batch from it"
        )
    parts = [
        f"σ {params['sigma']:g} px",
        params.get("detector", "multi_emitter"),
        f"max step {params['max_step']:g} px",
    ]
    if params.get("min_track_length"):
        parts.append(f"tracks ≥ {params['min_track_length']} frames")
    for key, label in (("point_filters", "point"), ("track_filters", "track")):
        if params.get(key):
            parts.append(f"{label} filters: {', '.join(sorted(params[key]))}")
    return " · ".join(parts)


def _diffusion_summary_text(settings: dict) -> str:
    parts = [f"min_frames {settings.get('min_frames', '?')}"]
    grid = settings.get("grid") or {}
    if "D_min_um2_s" in grid and "D_max_um2_s" in grid:
        parts.append(f"D grid {grid['D_min_um2_s']:g}–{grid['D_max_um2_s']:g} µm²/s")
    if settings.get("alpha"):
        parts.append("α")
    if settings.get("msd_comparison"):
        parts.append("MSD comparison")
    if settings.get("filters"):
        parts.append(f"filters: {', '.join(sorted(settings['filters']))}")
    return " · ".join(parts)


class BatchDialog(QDialog):
    """Pick which movies to run and which steps. `entries` are the other
    rows of the list, as `(image_path, result_dir, skipped)`."""

    def __init__(
        self,
        template: Path,
        results_root: Path,
        entries: list[tuple[Path, Path, bool]],
        parent=None,
    ) -> None:
        super().__init__(parent)
        from napari_gemscape2.diffusion_batch import settings_from_summary

        self.setWindowTitle("Batch from this movie")
        self.setMinimumWidth(520)
        self._template = template
        self._results_root = results_root

        self._detect_params = detect_track_params_from_manifest(load_manifest(template))
        saved = load_diffusion_summary(template)
        self._diffusion_settings = settings_from_summary(saved) if saved is not None else None
        can_detect = all(self._detect_params.get(key) is not None for key in ("sigma", "max_step"))

        title = QLabel(f"<b>Template:</b> {template.name}")
        title.setWordWrap(True)
        note = QLabel(
            "Runs this movie's <i>saved</i> settings -- save it first if you changed anything. "
            "Each movie uses its own painted mask, if it has one, and the whole field otherwise."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")

        self.detect_check = QCheckBox("Detect + track")
        self.detect_check.setChecked(can_detect)
        self.detect_check.setEnabled(can_detect)
        detect_label = QLabel(_detect_track_summary(self._detect_params))
        detect_label.setWordWrap(True)
        detect_label.setStyleSheet("color: gray; margin-left: 22px;")

        self.diffusion_check = QCheckBox("Diffusion analysis")
        has_diffusion = self._diffusion_settings is not None
        self.diffusion_check.setChecked(has_diffusion)
        self.diffusion_check.setEnabled(has_diffusion)
        diffusion_label = QLabel(
            _diffusion_summary_text(self._diffusion_settings)
            if has_diffusion
            else "no saved analysis -- run it on this movie in the Diffusion analysis "
            "widget and press Save analysis to batch it"
        )
        diffusion_label.setWordWrap(True)
        diffusion_label.setStyleSheet("color: gray; margin-left: 22px;")

        self.file_list = QListWidget()
        for image_path, result_dir, skipped in entries:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, (image_path, result_dir))
            has_bundle = has_result(result_dir)
            tags = [tag for tag, on in (("masked", has_regions(result_dir)), ("has results", has_bundle)) if on]
            item.setText(image_path.name + (f"   ({', '.join(tags)})" if tags else ""))
            item.setToolTip(str(image_path))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # Unanalyzed, unskipped movies are the usual batch; re-running
            # an analyzed one is a deliberate tick.
            checked = not has_bundle and not skipped
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self.file_list.addItem(item)

        all_button = QPushButton("All")
        all_button.clicked.connect(lambda: self._check_all(True))
        none_button = QPushButton("None")
        none_button.clicked.connect(lambda: self._check_all(False))
        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("Movies:"))
        select_row.addStretch(1)
        select_row.addWidget(all_button)
        select_row.addWidget(none_button)

        self.config_check = QCheckBox("Also write the batch config (results folder)")
        self.config_check.setChecked(True)
        self.config_check.setToolTip(
            "A TOML config naming this template and these movies, so\n"
            "`gemscape2 detect-track` / `gemscape2 diffusion` can re-run it headless."
        )

        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #f59e0b;")

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.run_button = self.buttons.addButton("Run", QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(note)
        layout.addWidget(self.detect_check)
        layout.addWidget(detect_label)
        layout.addWidget(self.diffusion_check)
        layout.addWidget(diffusion_label)
        layout.addLayout(select_row)
        layout.addWidget(self.file_list, 1)
        layout.addWidget(self.config_check)
        layout.addWidget(self.warning)
        layout.addWidget(self.buttons)

        self.detect_check.toggled.connect(self._refresh)
        self.diffusion_check.toggled.connect(self._refresh)
        self.file_list.itemChanged.connect(self._refresh)
        self._refresh()

    def _items(self) -> list[QListWidgetItem]:
        return [self.file_list.item(i) for i in range(self.file_list.count())]

    def _check_all(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for item in self._items():
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(state)

    def _refresh(self, *_args) -> None:
        detect = self.detect_check.isChecked()
        self.file_list.blockSignals(True)
        for item in self._items():
            _image_path, result_dir = item.data(Qt.ItemDataRole.UserRole)
            # Diffusion alone needs tracks already on disk.
            usable = detect or has_result(result_dir)
            flags = item.flags()
            item.setFlags(
                flags | Qt.ItemFlag.ItemIsEnabled if usable else flags & ~Qt.ItemFlag.ItemIsEnabled
            )
            if not usable:
                item.setCheckState(Qt.CheckState.Unchecked)
        self.file_list.blockSignals(False)

        jobs = self.jobs()
        steps = detect or self.diffusion_check.isChecked()
        self.run_button.setEnabled(bool(jobs) and steps)
        self.run_button.setText(f"Run {len(jobs)}" if jobs else "Run")
        overwritten = sum(has_result(job.result_dir) for job in jobs)
        if detect and overwritten:
            self.warning.setText(
                f"{overwritten} of these already have results -- detect+track replaces them."
            )
        elif not steps:
            self.warning.setText("Pick at least one step.")
        else:
            self.warning.setText("")

    def jobs(self) -> list[BatchJob]:
        return [
            BatchJob(*item.data(Qt.ItemDataRole.UserRole))
            for item in self._items()
            if item.checkState() == Qt.CheckState.Checked
        ]

    def plan(self) -> BatchPlan:
        return BatchPlan(
            template=self._template,
            results_root=self._results_root,
            jobs=self.jobs(),
            detect_track=self.detect_check.isChecked(),
            diffusion=self.diffusion_check.isChecked(),
            write_config=self.config_check.isChecked(),
        )


class BatchEmitter(QObject):
    """Reports from the batch thread; Qt queues each onto the GUI thread."""

    job_started = Signal(int, str)
    # (done, total, stage) within the current movie.
    progress = Signal(int, int, str)
    # (job index, n_tracks): the movie's bundle is on disk now.
    bundle_written = Signal(int, int)
    # (job index, ok, message)
    job_finished = Signal(int, bool, str)


@thread_worker(start_thread=False)
def run_batch_worker(plan: BatchPlan, cancel_event: threading.Event, emitter: BatchEmitter) -> bool:
    """Run `plan` one movie at a time. A movie that fails is reported and
    the batch moves on; a cancel stops after the current step's next
    checkpoint. Returns whether it ran to the end."""
    from napari_gemscape2 import units
    from napari_gemscape2.batch import detect_track_bundle
    from napari_gemscape2.diffusion_batch import DiffusionSettings, analyze_bundle, settings_from_summary

    detect_params = (
        DetectTrackParams(**detect_track_params_from_manifest(load_manifest(plan.template)))
        if plan.detect_track
        else None
    )
    diffusion_settings = (
        DiffusionSettings(**settings_from_summary(load_diffusion_summary(plan.template) or {}))
        if plan.diffusion
        else None
    )
    if plan.detect_track:
        import spotsolve
        import napari_gemscape2

        shas = repo_shas(spotsolve, napari_gemscape2)
    else:
        shas = {}

    def detect_progress(done: int, total: int, stage: str) -> None:
        emitter.progress.emit(done, total, stage)

    def diffusion_progress(done: int, total: int) -> None:
        # analyze_bundle has no cancel of its own; raising here stops it
        # before it writes anything.
        if cancel_event.is_set():
            raise PipelineCancelled("cancelled")
        emitter.progress.emit(done, total, "diffusion posteriors")

    for index, job in enumerate(plan.jobs):
        if cancel_event.is_set():
            return False
        emitter.job_started.emit(index, job.image_path.name)
        notes: list[str] = []
        try:
            if detect_params is not None:
                extra = detect_track_bundle(
                    job.image_path,
                    job.result_dir,
                    detect_params,
                    repo_shas=shas,
                    progress_callback=detect_progress,
                    cancel_event=cancel_event,
                )
                emitter.bundle_written.emit(index, extra["n_tracks"])
                notes.append(f"{extra['n_points']} points, {extra['n_tracks']} tracks")
                notes.extend(f"! {note}" for note in extra.get("metadata_notes") or ())
            if diffusion_settings is not None:
                emitter.progress.emit(0, 0, "diffusion posteriors")
                report = analyze_bundle(job.result_dir, diffusion_settings, progress=diffusion_progress)
                notes.append(
                    f"{report.n_fitted} of {report.n_tracks} tracks fitted, "
                    f"{report.n_passing} pass the filters "
                    f"(exposure {units.fmt(report.exposure_s, 'exposure_s')})"
                )
                if not report.units_known:
                    notes.append("! no pixel size / frame interval recorded: units are px and frames")
                if diffusion_settings.alpha and not report.alpha:
                    notes.append("! α not computed: it needs exposure 0")
        except PipelineCancelled:
            emitter.job_finished.emit(index, False, "cancelled")
            return False
        except Exception as exc:
            emitter.job_finished.emit(index, False, f"{type(exc).__name__}: {exc}")
            continue
        emitter.job_finished.emit(index, True, "; ".join(notes))
    return True


def batch_config_path(results_root: Path, template: Path) -> Path:
    """Where a GUI batch's config goes: the results folder, named after
    the template, numbered rather than overwriting an earlier batch's."""
    stem = f"batch_{template.name}"
    path = results_root / f"{stem}.toml"
    n = 2
    while path.exists():
        path = results_root / f"{stem}_{n}.toml"
        n += 1
    return path

