"""Linear-interpolation crossing used for every failure temperature.

`cross` is the original helper (e1-e9 still use it; its results must stay reproducible, so it is left untouched).
The functions below are the pre-registered rules of the replication test (design doc 8.8.24, notebook 15, e12):
only downward crossings count, the start of the grid may sit exactly on 0.5, and every input gets a defined state.
"""
import numpy as np

LEVEL = 0.5
DIRECT_HIGH, DIRECT_LOW = 1.30, 0.80   # clamps of the direct estimate's extrapolation (rules 3 and 4)

REASON_DOWN = "向下交會"
REASON_NOT_FAILED = "網格內未失效"          # state 2: every AUROC above 0.5
REASON_FAILED_AT_START = "網格起點已失效"    # state 2: every AUROC below 0.5
REASON_UP_ONLY = "只有向上穿越"
REASON_FLAT = "平段"
REASON_NONFINITE = "非有限值"


def cross(x, y, level=0.0):
    """Temperature where y crosses `level`, linear interpolation on the sorted grid; None if no sign change."""
    x, y = np.asarray(x), np.asarray(y) - level
    for i in range(len(x) - 1):
        if y[i] > 0 >= y[i + 1] or y[i] < 0 <= y[i + 1]:
            return x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1])
    return None


def _prep(t, a):
    t, a = np.asarray(t, dtype=float), np.asarray(a, dtype=float)
    assert t.ndim == 1 and t.shape == a.shape and len(t) >= 2, "need two or more temperatures, one AUROC each"
    assert np.all(np.diff(t) > 0), "temperatures must be strictly increasing"
    return t, a


def crossings_down(t, a):
    """Every downward crossing of 0.5, in grid order (the first one is T*). Empty if any AUROC is not finite."""
    t, a = _prep(t, a)
    if not np.isfinite(a).all():
        return []
    s = a - LEVEL
    out = []
    if s[0] == 0 and s[1] < 0:       # the grid starts exactly on 0.5 and falls
        out.append(float(t[0]))
    for i in range(len(s) - 1):
        if s[i] > 0 and s[i + 1] <= 0:
            # s_{i+1} = 0 means the crossing is the grid point itself; return it exactly, not through rounding
            out.append(float(t[i + 1]) if s[i + 1] == 0 else float(t[i] + (t[i + 1] - t[i]) * s[i] / (s[i] - s[i + 1])))
    return out


def cross_down(t, a):
    """First downward crossing of AUROC = 0.5 (float), or None. Upward crossings never count."""
    c = crossings_down(t, a)
    return c[0] if c else None


def endpoint_state(t, a):
    """(state, T* or None, reason). State 1: a downward crossing; 2: every AUROC on one side of 0.5; 3: anything else."""
    t, a = _prep(t, a)
    if not np.isfinite(a).all():
        return 3, None, REASON_NONFINITE
    c = cross_down(t, a)
    if c is not None:
        return 1, c, REASON_DOWN
    s = a - LEVEL
    if (s > 0).all():
        return 2, None, REASON_NOT_FAILED
    if (s < 0).all():
        return 2, None, REASON_FAILED_AT_START
    # No downward crossing and not one-sided: once positive the curve stays positive, so the only ways left are an
    # upward crossing or touching 0.5 without crossing it.
    up = any(s[i] < 0 <= s[i + 1] for i in range(len(s) - 1))
    return 3, None, REASON_UP_ONLY if up else REASON_FLAT


def _extrapolate_one_sided(t, a):
    """Rules 3 and 4 of the direct estimate: straight line through the two outermost grid points on the failing side."""
    if (a > LEVEL).all():
        m = (a[-1] - a[-2]) / (t[-1] - t[-2])
        return (min(DIRECT_HIGH, float(t[-1] + (LEVEL - a[-1]) / m)) if m < 0 else DIRECT_HIGH), 3
    if (a < LEVEL).all():
        m = (a[1] - a[0]) / (t[1] - t[0])
        return (max(DIRECT_LOW, float(t[0] + (LEVEL - a[0]) / m)) if m < 0 else DIRECT_LOW), 4
    return None, 5


def direct_estimate(t, a):
    """(value or None, rule 1-5) for the direct estimate T_direct. Division only happens when the slope is negative."""
    t, a = _prep(t, a)
    if not np.isfinite(a).all():
        return None, 1
    c = cross_down(t, a)
    if c is not None:
        return c, 2
    return _extrapolate_one_sided(t, a)


def extrapolate(t, a):
    """Supplementary extrapolated value for a state-2 curve (same line as direct-estimate rules 3 and 4); None otherwise.
    Reported next to T*, never used as T*."""
    t, a = _prep(t, a)
    if not np.isfinite(a).all() or cross_down(t, a) is not None:
        return None
    v, _ = _extrapolate_one_sided(t, a)
    return v
