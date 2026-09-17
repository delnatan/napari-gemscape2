"""Is a given (D, dt, density) combination trackable at all?

Two independent length scales set "how far a particle moves between frames,
relative to how close its neighbors are":

  - `rms_step_um(D, dt)`: the 2D root-mean-square displacement implied by the
    diffusion coefficient over one frame interval, from the Einstein relation
    <r^2> = 4*D*dt (2D, isotropic Brownian motion).
  - `mean_nn_spacing_um(density)`: the mean nearest-neighbor spacing of a 2D
    homogeneous Poisson point process at that density, E[r] =
    1/(2*sqrt(density)) (Clark & Evans 1954).

`crowding_ratio` is their ratio. When it's << 1, a particle's frame-to-frame
step is much shorter than the gap to its nearest neighbor, so linking is
essentially unambiguous no matter how the linker is built. As it
approaches/exceeds 1, a particle's own next position becomes comparable in
distance to *another* particle's current position -- the regime where
frame-to-frame identity is genuinely ambiguous (identity swaps) rather than
merely gate-limited.

The ratio is a property of the acquisition (frame interval) and the sample
(diffusion coefficient, label density) alone. That is what makes it worth
keeping across a change of linker: it says nothing about any particular
linker's internals, so it stays meaningful as one.

Ported from `sfwloc.tracking_diagnostics` (same author) when the pipeline
moved off sfwloc onto `spotsolve`. The two closed forms above are sample
physics and carry over unchanged. THE THRESHOLDS DO NOT, and this is the one
thing to know before reading a verdict: `RATIO_CAUTION`/`RATIO_UNRESOLVABLE`
below were read off a recall-vs-ratio sweep of sfwloc's Stage-1 LAP linker,
which gated on a single flat distance. `spotsolve.tracking.link` scores a
link by likelihood ratio under a per-track posterior over D, using each
detection's own CRLB -- so it is expected to degrade *later* than the LAP
linker did, and these cutoffs are conservative rather than calibrated for it.
Treat the verdict as an advisory ordering ("this movie is crowded relative to
its step size"), not a measured error rate, until the sweep is rerun against
the new linker.
"""

from __future__ import annotations

import numpy as np

from spt_pipeline import units


def rms_step_um(D_um2_s: float, dt_s: float) -> float:
    """2D root-mean-square displacement per frame interval, from the
    Einstein relation <r^2> = 4*D*dt."""
    return float(np.sqrt(4.0 * D_um2_s * dt_s))


def mean_nn_spacing_um(density_um2: float) -> float:
    """Mean nearest-neighbor spacing for a 2D homogeneous Poisson process at
    `density_um2` particles/um^2 (Clark & Evans 1954): E[r] =
    1/(2*sqrt(density)). Infinite at zero density -- one particle has no
    neighbor to be crowded by."""
    if density_um2 <= 0.0:
        return float("inf")
    return float(1.0 / (2.0 * np.sqrt(density_um2)))


def crowding_ratio(D_um2_s: float, dt_s: float, density_um2: float) -> float:
    """`rms_step_um` / `mean_nn_spacing_um` -- closed form:
    `4 * sqrt(D_um2_s * dt_s * density_um2)`. Zero at zero density."""
    spacing = mean_nn_spacing_um(density_um2)
    if not np.isfinite(spacing):
        return 0.0
    return float(rms_step_um(D_um2_s, dt_s) / spacing)


# Read off sfwloc's benchmark/run_crowding_ratio_validation.py -- a pooled
# recall-vs-ratio curve for its Stage-1 LAP linker under a generous,
# step-scaled gate (which isolates crowding ambiguity from gate-limited
# termination). Measured there (8x8 D x density grid spanning ratio
# ~0.06-1.6, 3 trials/cell, pooled):
#   ratio  link_recall  id_switch_rate
#   0.19       0.99          0.01
#   0.31       0.97          0.03
#   0.44       0.95          0.05
#   0.56       0.91          0.09
#   0.69       0.85          0.15
#   0.81       0.82          0.18
#   1.06-1.19  0.73          0.27
#   1.56       0.54          0.46
# Recall crosses 0.95 around ratio ~0.4 and 0.8 around ratio ~0.9. See this
# module's docstring for why these are advisory, not calibrated, now that
# `spotsolve.tracking.link` does the linking.
RATIO_CAUTION = 0.4
RATIO_UNRESOLVABLE = 0.9

_LINKER_CAVEAT = (
    "Thresholds are inherited from sfwloc's LAP linker and are conservative "
    "for spotsolve's likelihood-ratio linker -- read as an advisory, not a "
    "measured error rate."
)


def check_resolvability(D_um2_s: float, dt_s: float, density_um2: float) -> dict:
    """Advisory check: given a (measured or expected) diffusion coefficient,
    frame interval, and detection density, is frame-to-frame identity
    expected to be resolvable?

    Returns a dict with `ratio`, `rms_step_um`, `mean_nn_spacing_um`,
    `verdict` (one of `"ok"`, `"caution"`, `"unresolvable"`) and a
    human-readable `message`.
    """
    ratio = crowding_ratio(D_um2_s, dt_s, density_um2)
    step = rms_step_um(D_um2_s, dt_s)
    spacing = mean_nn_spacing_um(density_um2)

    if ratio >= RATIO_UNRESOLVABLE:
        verdict = "unresolvable"
        message = (
            f"crowding ratio {ratio:.2f} >= {RATIO_UNRESOLVABLE}: the step "
            f"({step:.3f} {units.UM}) is comparable to or larger than the mean "
            f"nearest-neighbor spacing ({spacing:.3f} {units.UM}) -- frame-to-frame "
            "identity is expected to be frequently ambiguous. Increase frame "
            f"rate (lower dt) and/or lower labeling density. {_LINKER_CAVEAT}"
        )
    elif ratio >= RATIO_CAUTION:
        verdict = "caution"
        message = (
            f"crowding ratio {ratio:.2f} in [{RATIO_CAUTION}, "
            f"{RATIO_UNRESOLVABLE}): the step ({step:.3f} {units.UM}) is a non-trivial "
            f"fraction of the mean nearest-neighbor spacing ({spacing:.3f} {units.UM}) "
            f"-- some identity swaps expected in crowded regions. {_LINKER_CAVEAT}"
        )
    else:
        verdict = "ok"
        message = (
            f"crowding ratio {ratio:.2f} < {RATIO_CAUTION}: the step "
            f"({step:.3f} {units.UM}) is small relative to the mean nearest-neighbor "
            f"spacing ({spacing:.3f} {units.UM}) -- crowding is not expected to be the "
            "limiting factor."
        )

    return {
        "ratio": ratio,
        "rms_step_um": step,
        "mean_nn_spacing_um": spacing,
        "verdict": verdict,
        "message": message,
    }
