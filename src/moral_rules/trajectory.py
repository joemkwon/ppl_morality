"""Trajectory utilities: plan utility, shortcutiness, and trajectory I/O.

These reproduce the exact computations used by the scripts that generated the
accepted paper's figures (``models/exp2_model`` / ``models/exp4_model``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .gridworld import (
    CARDINAL_ACTIONS,
    full_rule_following_trajectory,
    shortest_path_trajectory,
)

DIAGONAL_COST = np.sqrt(2)


def calculate_plan_utility(
    trajectory: list[dict], goal_utility: float, shortcut_value: float
) -> float:
    """Agent utility of a plan: ``U_goal - (1 - sigma) * path_cost``.

    The goal reward is collected only if the trajectory actually ends on the
    destination cell (``type == "F"``). Cardinal moves cost 1, diagonal moves
    cost sqrt(2). The path cost is scaled by ``(1 - shortcut_value)``: this is
    how the model approximates the cost of a shortcut trajectory as an
    interpolation between the full rule-following cost (sigma=0) and a
    cost-free beeline (sigma=1).
    """
    reward_for_goal = goal_utility if trajectory[-1]["type"] == "F" else 0.0
    path_cost = 0.0
    for step in trajectory:
        if step["action"] in CARDINAL_ACTIONS:
            path_cost += 1.0
        else:  # diagonal move (or the appended terminal step with no action)
            path_cost += DIAGONAL_COST
    return reward_for_goal - (1.0 - shortcut_value) * path_cost


def calculate_shortcutiness(gridworld: list[list[str]], trajectory: list[dict]) -> float:
    """Degree of rule-breaking ``sigma`` in ``[0, 1]``.

    ``0`` = the strict rule-following (sidewalk-only) plan; ``1`` = the
    unrestricted shortest path. Defined by linear interpolation of trajectory
    *length* between those two extremes, clamped to ``[0, 1]``.
    """
    start = tuple(trajectory[0]["coordinate"])
    end = tuple(trajectory[-1]["coordinate"])

    rule_following_length = len(full_rule_following_trajectory(gridworld, start, end))
    shortest_length = len(shortest_path_trajectory(gridworld, start, end))
    given_length = len(trajectory)

    if rule_following_length == shortest_length:
        return 0.0
    sigma = (rule_following_length - given_length) / (rule_following_length - shortest_length)
    return max(0.0, min(1.0, sigma))


def load_gridworld_and_trajectory(file_name: str | Path) -> tuple[list[list[str]], list[dict]]:
    """Load a ``{"gridworld": ..., "trajectory": ...}`` JSON file.

    Coordinates are normalized to tuples (JSON stores them as 2-element lists).
    """
    with open(file_name, encoding="utf-8") as fh:
        data = json.load(fh)
    trajectory = [{**step, "coordinate": tuple(step["coordinate"])} for step in data["trajectory"]]
    return data["gridworld"], trajectory
