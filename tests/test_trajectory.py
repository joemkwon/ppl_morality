"""Plan-utility arithmetic and shortcutiness invariants."""

import numpy as np
import pytest

from moral_rules import calculate_plan_utility, calculate_shortcutiness
from moral_rules.gridworld import (
    full_rule_following_trajectory,
    shortest_path_trajectory,
)


def _traj(actions, end_type="F"):
    steps = [{"coordinate": (i, 0), "type": "S", "action": a} for i, a in enumerate(actions)]
    steps.append(
        {
            "coordinate": (len(actions), 0),
            "type": end_type,
            "action": actions[-1] if actions else None,
        }
    )
    return steps


def test_plan_utility_costs_and_reward():
    # 2 cardinal + 1 diagonal step, then terminal step (action repeats last).
    traj = _traj(["north", "east", "north-east"])
    # path_cost: north(1) + east(1) + north-east(√2) + terminal north-east(√2)
    expected_cost = 1 + 1 + np.sqrt(2) + np.sqrt(2)
    assert calculate_plan_utility(traj, 50.0, 0.0) == pytest.approx(50.0 - expected_cost)
    # sigma=1 zeroes the path cost; only the goal reward remains.
    assert calculate_plan_utility(traj, 50.0, 1.0) == pytest.approx(50.0)
    # sigma=0.5 halves the path cost.
    assert calculate_plan_utility(traj, 50.0, 0.5) == pytest.approx(50.0 - 0.5 * expected_cost)


def test_no_reward_if_not_ending_on_goal():
    traj = _traj(["east"], end_type="S")
    assert calculate_plan_utility(traj, 99.0, 0.0) < 0  # only cost, no reward


def test_shortcutiness_in_unit_interval_and_monotone(trajectories):
    sigmas = [calculate_shortcutiness(gw, t) for gw, t in trajectories]
    for s in sigmas:
        assert 0.0 <= s <= 1.0
    assert sigmas[0] == pytest.approx(0.0)  # map1_1 = strict rule-following
    assert sigmas[-1] == pytest.approx(1.0)  # map1_8 = full shortest path
    assert sigmas == sorted(sigmas), "trajectories must be ordered by sigma"


def test_extreme_paths_have_extreme_sigma(trajectories):
    gw, _ = trajectories[0]
    start, end = (0, 9), (9, 0)
    rf = full_rule_following_trajectory(gw, start, end)
    sp = shortest_path_trajectory(gw, start, end)
    assert calculate_shortcutiness(gw, rf) == pytest.approx(0.0)
    assert calculate_shortcutiness(gw, sp) == pytest.approx(1.0)
