"""What every number in this pipeline is measured in, in one place.

Every stage here converts pixels and frames into physical units at some
point -- `loctable` emits `y`/`x` (px) beside `y_um`/`x_um`,
`diffusion.tracks_to_diffusionkit_df` builds `x_um`/`sigma_x_um`,
`pipeline.track_metrics_df` adds `duration_s`/`mean_step_um`, and
diffusionkit returns `D_um2_s`/`K_um2_s_alpha`. The column names carry
most of that in a suffix, which is why a saved table is readable at all;
but a suffix is a convention, not a label, and three columns in the same
tracks-pane table (`se_x`, `se_x_um`, `radius_of_gyration_um`) sit in two
different unit systems with only `_um` to tell them apart.

So this module is the one place that answers "what is this column in?",
and every display site asks it rather than hardcoding a unit next to a
format string:

  - `label`/`header` for Qt (table headers, status lines, tooltips) --
    Unicode, "D_classical [µm²/s]".
  - `mpl_label` for matplotlib axes -- mathtext, matching the style
    diffusionkit's own plots already use (`$\\mu$m$^2$/s`), so a figure
    from `joint_plot` and one from `diffusionkit.classic.viz` don't label
    the same quantity two ways.
  - `fmt` for one value in running text -- "0.0123 µm²/s".

`unit_of` returns None for a column whose unit this module doesn't know,
and every function above degrades to the bare column name in that case.
That is deliberate: an unlabelled number is a smaller problem than a
confidently mislabelled one, and a new detector column showing up bare is
a visible prompt to add it here.

Resolution order for a column name (see `unit_of`): an exact entry, then
with a per-track aggregate suffix (`_min`/`_mean`/`_max`, added by
`widgets/diffusion_panel._qc_aggregate_table`) stripped, then a
unit-bearing name suffix (`_um2_s`, `_um`, `_s`, `_rad`, ...), then a
statistic suffix (`_median`/`_lo`/`_hi`/`_stderr`) or a fit-variant suffix
(`_map`/`_classical`/`_track_fit`) stripped, and finally the unitless and
pixel-space registries below.
"""

from __future__ import annotations

from typing import Optional

# Unicode forms, for Qt labels. "ADU" is spotsolve's own unit for `flux`
# and `bg` -- camera counts above the offset, not photons: nothing here
# converts by a gain, since spotsolve measures noise from each frame
# rather than taking a gain calibration.
UM = "µm"
UM2 = "µm²"
UM2_S = "µm²/s"
UM2_S_ALPHA = "µm²/sᵅ"
SECONDS = "s"
PX = "px"
PX2 = "px²"
PER_PX2 = "px⁻²"
PER_UM2 = "µm⁻²"
ADU = "ADU"
RAD = "rad"
FRAMES = "frames"
POINTS = "points"

# The same units as matplotlib mathtext. Keyed by the Unicode form above so
# there is one unit registry rather than two that can drift apart.
_MPL = {
    UM: r"$\mu$m",
    UM2: r"$\mu$m$^2$",
    UM2_S: r"$\mu$m$^2$/s",
    UM2_S_ALPHA: r"$\mu$m$^2$/s$^\alpha$",
    PX2: r"px$^2$",
    PER_PX2: r"px$^{-2}$",
    PER_UM2: r"$\mu$m$^{-2}$",
    "µm/px": r"$\mu$m/px",
}

# Exact column/key names. Anything derivable from a suffix rule below is
# left out, so this stays the list of genuinely irregular names rather
# than a copy of every schema.
_EXACT: dict[str, Optional[str]] = {
    # loctable positions and fit widths: pixel space, no suffix to say so.
    "y": PX,
    "x": PX,
    "se_y": PX,
    "se_x": PX,
    "se_pos": PX,
    "sigma": PX,
    "fit_sigma": PX,
    "median_se_pos": PX,
    # Brightness, in camera counts above the offset.
    "flux": ADU,
    "se_flux": ADU,
    # The on-centre model pixel value -- spotsolve's one column in the
    # units of a pixel rather than of an integrated spot, read against
    # `bg`.
    "peak": ADU,
    "bg": ADU,
    "median_flux": ADU,
    "background": ADU,
    "offset": ADU,
    # loctable's own time column (`frame * dt_s`), and `frames_df`'s.
    "t": SECONDS,
    "seconds": SECONDS,
    # Counts and indices. A frame number is an index, not a measurement,
    # so it is registered as unitless rather than "in frames".
    "frame": None,
    "track_length": POINTS,
    "n_linked_steps": "steps",
    # Named for the area they are per, not for their own unit -- the one
    # place a suffix rule would get the wrong answer (`density_um2` is a
    # count per µm², and `lam_birth_per_px2` a rate per px²).
    "density_um2": PER_UM2,
    "lam_birth_per_px2": PER_PX2,
    # Physical scalars carried on the manifest / session.
    "pixel_size_um": "µm/px",
    "dt_s": "s/frame",
    "dt_spread_s": "s",
    "sigma_px": PX,
    "sigma_init": PX,
    "sigma_estimate": PX,
    "sigma_ci_px": PX,
    "mean_localization_offset_um2": UM2,
    # Dimensionless by construction, listed so they read as "known with no
    # unit" rather than "unknown".
    "alpha": None,
    "eps": None,
    "sigma_ratio": None,
    "flux_ratio": None,
    "flux_snr": None,
    "straightness": None,
    "gyration_asymmetry": None,
    "immobile_fraction": None,
    "agg_flux_fraction": None,
    "crowding_ratio": None,
    "p_cont": None,
    "se_inflate": None,
    # Variance per unit signal, which spotsolve's own schema labels ADU
    # (it is gain-like: ADU² per ADU). Taken from there rather than
    # re-derived, on the principle that this table states what upstream
    # says and nothing it had to infer.
    "dispersion": ADU,
    "r2": None,
    "r_squared": None,
    "log_bf10": None,
    "evidence": None,
    "track_id": None,
    "loc_id": None,
    "model": None,
    "method": None,
}

# Unit-bearing name suffixes, longest first so `_um2_s_alpha` is tested
# before `_um2_s` and `_per_px2` before `_px2`.
_SUFFIX_UNITS: tuple[tuple[str, str], ...] = (
    ("_um2_s_alpha", UM2_S_ALPHA),
    ("_per_px2", PER_PX2),
    ("_um2_s", UM2_S),
    ("_um2", UM2),
    ("_px2", PX2),
    ("_rad", RAD),
    ("_um", UM),
    ("_px", PX),
    ("_s", SECONDS),
)

# Per-track aggregates of a per-point column (`flux_min`, `se_x_max`) take
# that column's unit.
_AGGREGATE_SUFFIXES = ("_min", "_mean", "_max", "_sum")

# A point estimate or interval edge of a fitted quantity, and the naming
# this project uses to keep one quantity's several fits in separate
# columns (`alpha_map` vs `alpha_classical` vs `alpha_track_fit`). Both
# kinds carry the base quantity's unit.
_STATISTIC_SUFFIXES = (
    "_median",
    "_mean",
    "_lo",
    "_hi",
    "_stderr",
    "_map",
    "_classical",
    "_classical_anom",
    "_track_fit",
    "_est",
    "_link",
    "_arith_mean",
    "_par",
    "_perp",
)

# Greek/italic forms for the quantity part of a matplotlib label, so an
# axis reads "$\alpha$ (log-log fit)" like diffusionkit's own do rather
# than spelling the letter out.
_MPL_TOKENS = {
    "D": r"$D$",
    "K": r"$K$",
    "alpha": r"$\alpha$",
    "eps": r"$\epsilon$",
    "psi": r"$\psi$",
    "sigma": r"$\sigma$",
    "tau": r"$\tau$",
    "r2": r"$R^2$",
}


def _strip_suffix(name: str, suffixes) -> Optional[str]:
    """`name` without the first matching suffix, or None if none match.
    Never strips a name down to nothing."""
    for suffix in suffixes:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return None


def unit_of(column: str) -> Optional[str]:
    """The unit of `column` as a Unicode string, or None when this module
    has no entry for it (an unknown column, or a genuinely dimensionless
    one -- `is_known` tells those apart).

    Resolution order is in this module's docstring. Case-sensitive: the
    schemas here are, and `D` is not `d`."""
    if column in _EXACT:
        return _EXACT[column]
    base = _strip_suffix(column, _AGGREGATE_SUFFIXES)
    if base is not None and base in _EXACT:
        return _EXACT[base]
    for suffix, unit in _SUFFIX_UNITS:
        # Aggregates put the stat last (`se_x_um_max`), so test the
        # unit suffix against the aggregate-stripped name too.
        for candidate in (column, base):
            if candidate is not None and candidate.endswith(suffix) and len(candidate) > len(suffix):
                return unit
    for candidate in (column, base):
        if candidate is None:
            continue
        stripped = _strip_suffix(candidate, _STATISTIC_SUFFIXES)
        if stripped is not None:
            return unit_of(stripped)
    return None


def is_known(column: str) -> bool:
    """Whether this module knows what `column` is measured in -- True for
    a dimensionless column it knows to BE dimensionless (`alpha`), False
    for one it has simply never heard of."""
    if column in _EXACT:
        return True
    if unit_of(column) is not None:
        return True
    # A dimensionless column's aggregate or statistic is equally
    # dimensionless, and equally *known*: `flux_snr_mean` and `eps_median`
    # resolve to a registered `None`, not to "never heard of it".
    base = _strip_suffix(column, _AGGREGATE_SUFFIXES)
    if base is not None and is_known(base):
        return True
    stripped = _strip_suffix(base or column, _STATISTIC_SUFFIXES)
    return stripped is not None and is_known(stripped)


def _unit_free_name(column: str) -> str:
    """`column` with its unit-bearing suffix removed, keeping any
    aggregate suffix: `D_classical_um2_s` -> `D_classical`,
    `se_x_um_max` -> `se_x_max`. What `header`/`label` show beside the
    bracketed unit, so the unit is stated once rather than twice."""
    aggregate = ""
    base = column
    stripped = _strip_suffix(column, _AGGREGATE_SUFFIXES)
    if stripped is not None:
        aggregate = column[len(stripped) :]
        base = stripped
    for suffix, _unit in _SUFFIX_UNITS:
        if base.endswith(suffix) and len(base) > len(suffix):
            base = base[: -len(suffix)]
            break
    return base + aggregate


def header(column: str) -> str:
    """A table-header label: the column name with its unit in brackets,
    e.g. `D_classical [µm²/s]`, `se_x_max [px]`, `se_x_um_max [µm]`.

    Bracketed rather than parenthesized so a scan down a 40-column header
    row reads the units as one group, and so a header is visibly not the
    exact column name (which the tooltip gives)."""
    unit = unit_of(column)
    return f"{_unit_free_name(column)} [{unit}]" if unit else column


def headers(columns) -> dict[str, str]:
    """`{column: header}` for a whole table, resolving collisions.

    `header` states the unit once by dropping the name's own unit token,
    which means two columns of the same quantity in different units --
    `se_x_max` (px) and `se_x_um_max` (µm), both in the tracks pane --
    would otherwise display as the same `se_x_max`, distinguished only by
    the bracket. Where that happens, both keep their full column name and
    the unit is simply appended, so the pair reads
    `se_x_max [px]` / `se_x_um_max [µm]`."""
    columns = list(columns)
    proposed = {column: header(column) for column in columns}
    # Grouped by the unit-free NAME, not by the finished header: the two
    # `se_x_max` columns produce different headers (one [px], one [µm])
    # and are still the same word on screen, which is the collision worth
    # resolving.
    sharing: dict[str, list[str]] = {}
    for column in columns:
        sharing.setdefault(_unit_free_name(column), []).append(column)
    for group in sharing.values():
        if len(group) < 2:
            continue
        for column in group:
            unit = unit_of(column)
            proposed[column] = f"{column} [{unit}]" if unit else column
    return proposed


def label(column: str) -> str:
    """`header`, for running text -- the same string, kept as its own
    function so a caller's intent is legible and the two can diverge if a
    header ever needs to be shorter than a sentence's label."""
    return header(column)


def tooltip(column: str) -> str:
    """What a header/filter-row hover says: the exact column name and its
    unit, or a note that the unit isn't registered."""
    unit = unit_of(column)
    if unit:
        return f"{column} — in {unit}"
    if is_known(column):
        return f"{column} — dimensionless"
    return f"{column} — unit not registered (see spt_pipeline.units)"


def mpl_label(column: str) -> str:
    """A matplotlib axis label in mathtext: `radius_of_gyration_um` ->
    `radius of gyration ($\\mu$m)`, `alpha_map` -> `$\\alpha$ map`.

    Matches the convention diffusionkit's own `viz` modules use --
    quantity, then unit in parentheses -- so this project's figures and
    diffusionkit's label the same quantity the same way."""
    unit = unit_of(column)
    words = [_MPL_TOKENS.get(token, token) for token in _unit_free_name(column).split("_")]
    quantity = " ".join(words)
    mpl_unit = _MPL.get(unit, unit) if unit else None
    return f"{quantity} ({mpl_unit})" if mpl_unit else quantity


def mpl_log_label(column: str) -> str:
    """`mpl_label` for a log10-transformed axis: the unit stays, since it
    is the unit the log was taken of -- `$\\log_{10}$ D ($\\mu$m$^2$/s)`,
    which is how diffusionkit labels its own log-D axes."""
    return r"$\log_{10}$ " + mpl_label(column)


def fmt(value, column: str, precision: int = 4) -> str:
    """One value with its unit, for a status line: `fmt(0.0123,
    "D_est_um2_s")` -> `0.0123 µm²/s`. A None/NaN value renders as "n/a"
    rather than as a number that isn't one."""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "n/a"
    text = f"{number:.{precision}g}"
    unit = unit_of(column)
    return f"{text} {unit}" if unit else text


def fmt_unit(value, unit: Optional[str], precision: int = 4) -> str:
    """`fmt` for a value that isn't a named column -- a unit given
    directly (`fmt_unit(dt, SECONDS)`)."""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "n/a"
    text = f"{number:.{precision}g}"
    return f"{text} {unit}" if unit else text
