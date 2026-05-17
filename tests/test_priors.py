"""Prior constants are well-formed and consistent."""

import pytest

from moral_rules import COMMUNITY_CONDITIONS, CONDITION_0, GOAL_UTILITIES, GOALS
from moral_rules.priors import normalize_proportions


def test_goal_sets_are_consistent():
    assert set(GOALS) == set(GOAL_UTILITIES) == set(CONDITION_0)
    for name, cond in COMMUNITY_CONDITIONS.items():
        assert set(cond) == set(GOALS), f"{name} has mismatched goals"


def test_utilities_are_plausible():
    for goal, (mean, se) in GOAL_UTILITIES.items():
        assert 0 <= mean <= 100, goal
        assert se > 0, goal


def test_normalize_proportions_sums_to_one():
    p = normalize_proportions(CONDITION_0)
    assert sum(p.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in p.values())


def test_normalize_rejects_nonpositive():
    with pytest.raises(ValueError):
        normalize_proportions({"a": 0, "b": 0})
