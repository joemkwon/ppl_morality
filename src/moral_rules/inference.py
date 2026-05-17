"""Forward model (rational rule-follower) and Bayesian inference of myopia.

Pipeline (faithful to ``models/exp2_model`` / ``models/exp4_model``):

1. ``generate_forward_model_table`` -- for a discretized grid of (myopia
   ``lambda``, shortcut ``sigma``) values, compute the overall utility

       U_overall = lambda * U_agent(sigma)  +  (1 - lambda) * U_univ(sigma)

   then convert each *shortcut column* to a distribution over myopia with a
   numerically-stable softmax. (The softmax is per-column: it answers "given
   this much rule-breaking, which myopia values rationalize it?".)
2. Average several such tables over resampled goal utilities, renormalizing
   each shortcut column.
3. ``infer_myopia_parameter`` -- read off the column for the observed
   trajectory's shortcutiness; that column *is* the posterior over myopia.
4. Summaries: posterior-mean myopia, and the model's moral judgment
   ``100 - 100 * mean_myopia`` (higher myopia -> less permissible).
"""

from __future__ import annotations

import numpy as np

from .gridworld import full_rule_following_trajectory
from .priors import normalize_proportions
from .trajectory import calculate_plan_utility, calculate_shortcutiness
from .universalization import sample_goal_utility, universalized_plan_utility

ForwardModelTable = dict[tuple[float, float], float]


def generate_forward_model_table(
    gridworld: list[list[str]],
    trajectory: list[dict],
    my_goal_utility: float,
    goal_proportions: dict[str, float],
    *,
    num_bins: int,
    num_agents: int,
    num_simulations: int,
    grass_capacity: float,
    grass_community_cost: float,
    variant: str = "published_stochastic",
    progress: bool = False,
) -> ForwardModelTable:
    """P(myopia | shortcut) over the discretized grid, one shortcut column at a time."""
    goal_probabilities = normalize_proportions(goal_proportions)
    myopia_values = np.linspace(0, 1, num_bins)
    shortcut_values = np.linspace(0, 1, num_bins)

    start = trajectory[0]["coordinate"]
    end = trajectory[-1]["coordinate"]
    rule_following_trajectory = full_rule_following_trajectory(gridworld, start, end)

    iterator = shortcut_values
    if progress:
        from tqdm import tqdm

        iterator = tqdm(shortcut_values, desc="forward model")

    table: ForwardModelTable = {}
    for shortcut in iterator:
        # U_agent does not depend on myopia, so compute it once per column.
        my_trajectory_utility = calculate_plan_utility(
            rule_following_trajectory, my_goal_utility, shortcut
        )
        avg_universalized_utility = float(
            np.mean(
                [
                    universalized_plan_utility(
                        gridworld,
                        my_goal_utility,
                        shortcut,
                        goal_probabilities,
                        grass_capacity=grass_capacity,
                        grass_community_cost=grass_community_cost,
                        num_agents=num_agents,
                        variant=variant,
                    )
                    for _ in range(num_simulations)
                ]
            )
        )

        utilities = {
            myopia: myopia * my_trajectory_utility + (1.0 - myopia) * avg_universalized_utility
            for myopia in myopia_values
        }
        # Numerically-stable softmax over myopia, within this shortcut column.
        max_utility = max(utilities.values())
        exp_utilities = {m: np.exp(v - max_utility) for m, v in utilities.items()}
        total = sum(exp_utilities.values())
        for myopia, exp_u in exp_utilities.items():
            table[(float(myopia), float(shortcut))] = float(exp_u / total)
    return table


def average_forward_model_tables(tables: list[ForwardModelTable]) -> ForwardModelTable:
    """Mean of several tables, then renormalize within each shortcut column."""
    keys: set[tuple[float, float]] = set()
    for t in tables:
        keys.update(t.keys())
    averaged = {k: float(np.mean([t.get(k, 0.0) for t in tables])) for k in keys}

    columns: dict[float, list[tuple[float, float]]] = {}
    for (myopia, shortcut), prob in averaged.items():
        columns.setdefault(shortcut, []).append((myopia, prob))

    normalized: ForwardModelTable = {}
    for shortcut, myopia_probs in columns.items():
        total = sum(p for _, p in myopia_probs)
        if total > 0:
            for myopia, prob in myopia_probs:
                normalized[(myopia, shortcut)] = prob / total
        else:
            uniform = 1.0 / len(myopia_probs)
            for myopia, _ in myopia_probs:
                normalized[(myopia, shortcut)] = uniform
    return normalized


def infer_myopia_parameter(
    gridworld: list[list[str]],
    trajectory: list[dict],
    table: ForwardModelTable,
) -> tuple[list[float], list[float]]:
    """Posterior over myopia for ``trajectory``: the table column nearest its sigma."""
    actual_shortcut = calculate_shortcutiness(gridworld, trajectory)
    shortcut_values = sorted({k[1] for k in table})
    closest = min(shortcut_values, key=lambda s: abs(s - actual_shortcut))

    myopia_values = sorted({k[0] for k in table if k[1] == closest})
    probabilities = [table[(m, closest)] for m in myopia_values]
    total = sum(probabilities)
    if total > 0:
        posterior = [p / total for p in probabilities]
    else:
        posterior = [1.0 / len(probabilities)] * len(probabilities)
    return myopia_values, posterior


def posterior_mean_myopia(myopia_values: list[float], posterior: list[float]) -> float:
    return float(np.sum(np.array(myopia_values) * np.array(posterior)))


def moral_judgment_from_myopia(mean_myopia: float) -> float:
    """Model's predicted permissibility on a 0-100 scale (less myopic = more OK)."""
    return 100.0 - mean_myopia * 100.0


def infer_myopia_for_trajectory(
    gridworld: list[list[str]],
    trajectory: list[dict],
    goal_name: str,
    goal_proportions: dict[str, float],
    *,
    num_bins: int,
    num_agents: int,
    num_simulations: int,
    num_goal_utility_samples: int,
    grass_capacity: float,
    grass_community_cost: float,
    variant: str = "published_stochastic",
) -> dict:
    """End-to-end inference for one (trajectory, goal, community) configuration.

    Resamples the agent's own goal utility ``num_goal_utility_samples`` times,
    builds a forward-model table for each, averages them, and returns the
    posterior over myopia plus its summary statistics.
    """
    tables = [
        generate_forward_model_table(
            gridworld,
            trajectory,
            sample_goal_utility(goal_name),
            goal_proportions,
            num_bins=num_bins,
            num_agents=num_agents,
            num_simulations=num_simulations,
            grass_capacity=grass_capacity,
            grass_community_cost=grass_community_cost,
            variant=variant,
        )
        for _ in range(num_goal_utility_samples)
    ]
    averaged = average_forward_model_tables(tables)
    myopia_values, posterior = infer_myopia_parameter(gridworld, trajectory, averaged)
    mean_myopia = posterior_mean_myopia(myopia_values, posterior)
    return {
        "goal": goal_name,
        "myopia_values": myopia_values,
        "posterior": posterior,
        "mean_myopia": mean_myopia,
        "moral_judgment": moral_judgment_from_myopia(mean_myopia),
        "shortcutiness": calculate_shortcutiness(gridworld, trajectory),
    }
