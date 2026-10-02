"""Acceptance tests for the pre-registered crossing rules (plan 8.8.24, implementation list item 2)."""
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cmargin.crossing import (cross, cross_down, crossings_down, direct_estimate, endpoint_state, extrapolate,
                              REASON_DOWN, REASON_FAILED_AT_START, REASON_FLAT, REASON_NONFINITE, REASON_NOT_FAILED,
                              REASON_UP_ONLY)

T3 = [0.94, 0.97, 0.99]
T11 = [0.94, 0.97, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.06, 1.09, 1.14]
TOL = 1e-9


def close(x, y):
    return x is not None and math.isclose(x, y, rel_tol=0.0, abs_tol=TOL)


def test_three_registered_examples():
    assert close(cross_down(T3, [0.4, 0.6, 0.4]), 0.98)        # the upward crossing at 0.955 is skipped
    assert close(cross_down([0.94, 0.97], [0.5, 0.4]), 0.94)   # starts exactly on 0.5 and falls
    assert close(cross_down(T3, [0.6, 0.5, 0.4]), 0.97)        # lands exactly on 0.5


def test_grid_point_on_level_is_returned_exactly():
    assert cross_down(T3, [0.6, 0.5, 0.4]) == 0.97
    assert cross_down([0.94, 0.97], [0.5, 0.4]) == 0.94


def test_first_of_several_downward_crossings():
    a = [0.6, 0.4, 0.6, 0.4] + [0.4] * 7
    c = crossings_down(T11, a)
    assert len(c) == 2 and close(c[0], 0.955) and close(c[1], 0.995)
    assert close(cross_down(T11, a), 0.955)


def test_states_on_the_eleven_point_grid():
    t = np.array(T11)
    s, v, r = endpoint_state(t, np.full(11, 0.6))
    assert (s, v, r) == (2, None, REASON_NOT_FAILED)
    s, v, r = endpoint_state(t, 0.4 + 0.05 * np.exp(-t))       # reviewer A: strictly falling, never reaches 0.5
    assert (s, v, r) == (2, None, REASON_FAILED_AT_START)
    s, v, r = endpoint_state(t, np.linspace(0.4, 0.6, 11))     # only an upward crossing
    assert (s, v, r) == (3, None, REASON_UP_ONLY)
    s, v, r = endpoint_state(t, np.r_[np.full(5, 0.6), np.full(6, 0.5)][::-1])   # 0.5 plateau, then above
    assert (s, r) == (3, REASON_FLAT)
    s, v, r = endpoint_state(t, np.full(11, 0.5))
    assert (s, r) == (3, REASON_FLAT)
    a = np.linspace(0.6, 0.4, 11); a[4] = np.nan
    s, v, r = endpoint_state(t, a)
    assert (s, v, r) == (3, None, REASON_NONFINITE)
    s, v, r = endpoint_state(t, 0.5 - 3.0 * (t - 1.012))
    assert s == 1 and r == REASON_DOWN and close(v, 1.012)


def test_touching_from_below_is_not_a_crossing():
    a = np.full(11, 0.45); a[5] = 0.5
    assert cross_down(T11, a) is None
    assert endpoint_state(T11, a)[0] == 3


def test_direct_estimate_rules():
    t = np.array(T11)
    assert direct_estimate(t, np.full(11, 0.6)) == (1.30, 3)            # flat above: slope 0 -> 1.30
    assert direct_estimate(t, np.full(11, 0.5)) == (None, 5)            # plateau on 0.5
    v, rule = direct_estimate(t, 0.5 - 3.0 * (t - 1.012))
    assert rule == 2 and close(v, 1.012)
    assert direct_estimate(t, np.linspace(0.40, 0.45, 11)) == (0.80, 4)  # below 0.5 and rising
    a = np.full(11, 0.6); a[3] = np.inf
    assert direct_estimate(t, a) == (None, 1)
    assert direct_estimate(t, np.linspace(0.40, 0.60, 11)) == (None, 5)  # upward crossing only


def test_direct_estimate_extrapolation_and_clamps():
    t = np.array(T11)
    a = 0.5 - 1.0 * (t - 1.20)             # above 0.5 on the grid, line hits 0.5 at 1.20
    v, rule = direct_estimate(t, a)
    assert rule == 3 and close(v, 1.20)
    a = 0.5 - 0.1 * (t - 2.0)              # would cross far above: clamped
    assert direct_estimate(t, a) == (1.30, 3)
    a = 0.5 - 1.0 * (t - 0.90)             # below 0.5 on the grid, line hits 0.5 at 0.90
    v, rule = direct_estimate(t, a)
    assert rule == 4 and close(v, 0.90)
    a = 0.5 - 0.1 * (t - 0.10)
    assert direct_estimate(t, a) == (0.80, 4)


def test_extrapolate_only_for_state_two():
    t = np.array(T11)
    assert close(extrapolate(t, 0.5 - 1.0 * (t - 1.20)), 1.20)
    assert close(extrapolate(t, 0.5 - 1.0 * (t - 0.90)), 0.90)
    assert extrapolate(t, 0.5 - 3.0 * (t - 1.012)) is None      # state 1
    assert extrapolate(t, np.linspace(0.4, 0.6, 11)) is None     # state 3
    assert extrapolate(t, np.full(11, np.nan)) is None


def test_old_cross_is_unchanged():
    assert close(cross(T3, [0.4, 0.6, 0.4], 0.5), 0.955)
    assert cross([0.94, 0.97], [0.5, 0.4], 0.5) is None


def test_input_checks():
    with pytest.raises(AssertionError):
        cross_down([0.97, 0.94], [0.6, 0.4])
    with pytest.raises(AssertionError):
        cross_down([0.94], [0.6])
