"""Widgets the MSD tab and the pool dialog share: `labelled`,
`percent_box`, `fit_window_to_figure`, and `EnsembleWindowControls`, the
ensemble MSD's window as one row of controls -- how far each track's MSD
runs before the tracks are averaged, the share of that curve D is fitted
over, the share alpha is fitted over, and where the localization offset
comes from. The MSD tab's Ensemble window and the experiment list's pool
dialog both use it, so the two say it the same way.

D and alpha get separate windows because they want different ones: D the
first, best-measured lags (the 25-40% rule), alpha a span of lags wide
enough for a log-log slope -- by default the whole curve. Both are shares
of the curve, so they follow it when it is run further; the lags they come
to are shown beside them, and are what a run records
(`diffusion.ensemble_windows`).
"""

from __future__ import annotations

from qtkit import flow_row
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QApplication, QComboBox, QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget

from napari_gemscape2.diffusion import (
    ENSEMBLE_ALPHA_FRACTION,
    ENSEMBLE_D_FRACTION,
    ENSEMBLE_MSD_MAX_LAG,
    ENSEMBLE_MSD_OFFSETS,
    MSD_MIN_LAG,
    ensemble_windows,
)

OFFSET_LABELS = {"provided": "from the SDs", "fit": "fitted (intercept)"}


def fit_window_to_figure(window: QWidget, figure) -> None:
    """Size a plot window for `figure`'s own layout (its inches at its dpi,
    plus room for the toolbar), within the screen: the ensemble figure gains
    a column per optional analysis, which a fixed-size window squeezes."""
    width, height = (figure.get_size_inches() * figure.dpi).astype(int)
    height += 60  # the navigation toolbar
    screen = window.screen() or QApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        width, height = min(width, available.width() - 40), min(height, available.height() - 80)
    window.resize(width, height)


def percent_box(fraction: float, tooltip: str) -> QSpinBox:
    """A share of a curve, in 5% steps."""
    box = QSpinBox()
    box.setRange(5, 100)
    box.setSingleStep(5)
    box.setSuffix(" %")
    box.setValue(round(100 * fraction))
    box.setToolTip(tooltip)
    return box


def labelled(text: str, *widgets: QWidget) -> QWidget:
    """A label and its controls as one item, so a wrapping row never parts
    them. Several controls share the slot when only one shows at a time."""
    pair = QWidget()
    layout = QHBoxLayout(pair)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.addWidget(QLabel(text))
    for widget in widgets:
        layout.addWidget(widget)
    return pair


class EnsembleWindowControls(QWidget):
    """`curve to lag [10] · D over the first [30 %] · α over the first
    [100 %] · offset [..]  → D: 3 lags, α: 10`, wrapping in a narrow dock.
    `extra` (label, control) pairs join the end of the row."""

    # The curve's length changed: the curves must be recomputed.
    curve_changed = Signal()
    # A window or the offset changed: the same curves only refit.
    fit_changed = Signal()

    def __init__(self, *extra: tuple[str, QWidget], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.max_lag = QSpinBox()
        self.max_lag.setRange(MSD_MIN_LAG, 1000)
        self.max_lag.setValue(ENSEMBLE_MSD_MAX_LAG)
        self.max_lag.setToolTip(
            "How far each track's time-averaged MSD is computed before the\n"
            "tracks are averaged. The late lags rest on the long tracks alone\n"
            "(the α panel says how many reach its last lag), and long tracks\n"
            "are not a random sample: fast particles leave the focus sooner."
        )
        self.d_percent = percent_box(
            ENSEMBLE_D_FRACTION,
            "The share of the curve's first lags the linear fit (D, and the\n"
            "fitted offset) uses: the usual rule is 25-40%, the best-measured\n"
            f"lags. Never under {MSD_MIN_LAG} lags.",
        )
        self.alpha_percent = percent_box(
            ENSEMBLE_ALPHA_FRACTION,
            "The share of the curve's first lags the log-log fit (α) uses.\n"
            "Wider than D's: a slope in log-log needs a span of lags to show\n"
            "curvature, and three lags are well under a decade of τ. Lower it\n"
            "if the last lags rest on too few tracks.",
        )
        self.offset = QComboBox()
        for key in ENSEMBLE_MSD_OFFSETS:
            self.offset.addItem(OFFSET_LABELS[key], key)
        self.offset.setToolTip(
            "The localization offset the fits take off: the tracks' own position\n"
            "SDs (D through the origin of the corrected curve), or the intercept\n"
            "of the linear fit over D's window, which uses no SDs and reports the\n"
            "localization SD it implies -- a check on the SDs."
        )
        self._readout = QLabel()
        self._readout.setToolTip("The lags each fit comes to: the shares of the curve, rounded.")

        row = flow_row(
            labelled("curve to lag", self.max_lag),
            labelled("· D over the first", self.d_percent),
            labelled("· α over the first", self.alpha_percent),
            labelled("· offset", self.offset),
            *(labelled(text, widget) for text, widget in extra),
            self._readout,
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(row)

        self.max_lag.valueChanged.connect(self._update_readout)
        self.max_lag.valueChanged.connect(lambda _v: self.curve_changed.emit())
        for box in (self.d_percent, self.alpha_percent):
            box.valueChanged.connect(self._update_readout)
            box.valueChanged.connect(lambda _v: self.fit_changed.emit())
        self.offset.currentIndexChanged.connect(lambda _i: self.fit_changed.emit())
        self._update_readout()

    def windows(self) -> tuple[int, int]:
        """`(n_points, alpha_points)`: the lags D and α are fitted over."""
        return ensemble_windows(
            self.max_lag.value(), self.d_percent.value() / 100, self.alpha_percent.value() / 100
        )

    def offset_key(self) -> str:
        return self.offset.currentData()

    def _update_readout(self, *_args) -> None:
        n_points, alpha_points = self.windows()
        self._readout.setText(f"→ D: {n_points} lags, α: {alpha_points}")
