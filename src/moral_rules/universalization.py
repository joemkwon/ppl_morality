"""The universalization step: "what if everyone in my situation did this?"

This module is the heart of the model -- and the place where the *code that
produced the accepted paper* deviates from the *equation printed in the paper*.
Both forms are provided and selected via ``variant``:

``"published_stochastic"`` (default; reproduces the accepted figures)
    The implementation used by ``models/exp2_model`` / ``models/exp4_model``.
    It does **not** compute a per-agent welfare sum at all. It only simulates
    how many total grass-steps the community would take if everyone with a
    goal at least as important as mine adopted my degree of rule-breaking, and
    returns a *stochastic* community penalty:

        if total_grass_steps <= capacity:        ->  0
        else: overage = total_grass_steps - capacity
              P(grass destroyed) = 1 - exp(-rate * overage)
              return -community_cost  w.p. that probability, else 0

``"welfare_sum"`` (the paper's Eq. 2)
    The implementation used by ``models/experimental_model/fragility_search``
    (and the never-run ``main_model``):  the mean over the simulated population
    of each agent's goal utility, minus the community cost (deterministically)
    if the grass capacity is exceeded:

        U_univ = (1/N) * ( sum_i U_goal,i  -  community_cost * 1[S_total > C] )

    Note this still omits each agent's own path cost, exactly as the original
    research code did; it is provided for transparency and future work, not to
    reproduce the accepted figures.

See ``docs/MODEL.md`` for the full discussion of this discrepancy.
"""

from __future__ import annotations

import numpy as np

from .config import GRASS_OVERAGE_RATE
from .gridworld import full_rule_following_trajectory, sample_location
from .priors import GOAL_UTILITIES

VARIANTS = ("published_stochastic", "welfare_sum")


def sample_goal(goal_probabilities: dict[str, float]) -> str:
    goals = list(goal_probabilities.keys())
    probabilities = list(goal_probabilities.values())
    return str(np.random.choice(goals, p=probabilities))


def sample_goal_utility(goal: str) -> float:
    mean, se = GOAL_UTILITIES[goal]
    return float(np.random.normal(mean, se))


def _simulate_total_grass_steps(
    gridworld, my_goal_utility, my_grass_shortcutiness, goal_probabilities, num_agents
):
    """Total grass-steps taken by a universalizing population (+ their goal utilities).

    Agents whose sampled goal utility is below mine do not universalize the
    behaviour (they would not take the shortcut), so they are skipped.
    """
    total_grass_steps = 0
    total_goal_utility = 0.0
    for _ in range(num_agents):
        goal = sample_goal(goal_probabilities)
        goal_utility = sample_goal_utility(goal)
        if goal_utility < my_goal_utility:
            continue
        start = sample_location(gridworld)
        end = sample_location(gridworld)
        while end == start:
            end = sample_location(gridworld)
        rule_following_length = len(full_rule_following_trajectory(gridworld, start, end))
        total_grass_steps += int(my_grass_shortcutiness * rule_following_length)
        total_goal_utility += goal_utility
    return total_grass_steps, total_goal_utility


def universalized_plan_utility(
    gridworld: list[list[str]],
    my_goal_utility: float,
    my_grass_shortcutiness: float,
    goal_probabilities: dict[str, float],
    *,
    grass_capacity: float,
    grass_community_cost: float,
    num_agents: int,
    variant: str = "published_stochastic",
) -> float:
    """One Monte-Carlo sample of the universalized utility (see module docstring)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

    total_grass_steps, total_goal_utility = _simulate_total_grass_steps(
        gridworld,
        my_goal_utility,
        my_grass_shortcutiness,
        goal_probabilities,
        num_agents,
    )

    if variant == "published_stochastic":
        if total_grass_steps <= grass_capacity:
            return 0.0
        overage = total_grass_steps - grass_capacity
        prob_destroyed = 1.0 - np.exp(-GRASS_OVERAGE_RATE * overage)
        return -grass_community_cost if np.random.random() < prob_destroyed else 0.0

    # variant == "welfare_sum"  (paper Eq. 2)
    total_utility = total_goal_utility
    if total_grass_steps > grass_capacity:
        total_utility -= grass_community_cost
    return total_utility / num_agents
