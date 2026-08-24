"""Small Qt building blocks shared across the dock widgets -- the
green/amber/red status-label color language and the section-separator
rule, originally `params_panel.py`-private, promoted here once
`diffusion_panel.py` needed the same look. Kept dependency-free of any
other widget module (no napari `Viewer`, no pipeline imports) so every
dock widget can import it without pulling in unrelated state.
"""

from __future__ import annotations

from qtpy.QtWidgets import QFrame, QLabel

STATUS_LEVEL_COLORS = {
    "neutral": "#9a9a9a",
    "ok": "#22c55e",
    "caution": "#f59e0b",
    "error": "#ef4444",
}


def style_status_label(label: QLabel, level: str = "neutral") -> None:
    color = STATUS_LEVEL_COLORS.get(level, STATUS_LEVEL_COLORS["neutral"])
    label.setStyleSheet(f"color: {color}; font-size: 11px;")


def hline() -> QFrame:
    """A thin horizontal rule for separating a widget's logical sections --
    reads faster than spacing alone once a form has several knob groups."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line
