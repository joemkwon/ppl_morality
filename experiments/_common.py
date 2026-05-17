"""Shared helpers for the experiment drivers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from moral_rules import (
    GOALS,
    infer_myopia_for_trajectory,
    load_gridworld_and_trajectory,
)
from moral_rules.config import TRAJECTORY_DIR, TRAJECTORY_FILES


@dataclass
class RunParams:
    """Model hyperparameters. ``paper()`` = exact accepted-paper settings."""

    num_bins: int = 50
    num_agents: int = 100
    num_simulations: int = 5
    num_goal_utility_samples: int = 3
    grass_community_cost: float = 1_000_000
    variant: str = "published_stochastic"

    @classmethod
    def paper(cls) -> RunParams:
        return cls()

    @classmethod
    def fast(cls) -> RunParams:
        """Coarse settings for an end-to-end smoke run (minutes, not hours)."""
        return cls(num_bins=15, num_agents=30, num_simulations=2, num_goal_utility_samples=2)


def sweep_goals_and_trajectories(
    goal_proportions: dict[str, float],
    grass_capacity: float,
    params: RunParams,
    goals: list[str] | None = None,
) -> list[dict]:
    """Run inference for every (goal, trajectory) at one community/capacity.

    Returns one record per (goal, trajectory) with the posterior and summaries.
    The "Steps on grass" axis is the trajectory file ordinal (``map1_k`` -> k-1),
    matching the paper's CSV convention; the true continuous shortcutiness is
    also recorded.
    """
    goals = goals or GOALS
    records: list[dict] = []
    for goal in goals:
        for idx, fname in enumerate(TRAJECTORY_FILES):
            gridworld, trajectory = load_gridworld_and_trajectory(TRAJECTORY_DIR / fname)
            result = infer_myopia_for_trajectory(
                gridworld,
                trajectory,
                goal,
                goal_proportions,
                num_bins=params.num_bins,
                num_agents=params.num_agents,
                num_simulations=params.num_simulations,
                num_goal_utility_samples=params.num_goal_utility_samples,
                grass_capacity=grass_capacity,
                grass_community_cost=params.grass_community_cost,
                variant=params.variant,
            )
            result["map"] = Path(fname).stem
            result["steps_on_grass"] = idx
            records.append(result)
    return records


def records_to_csv(records: list[dict], path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Goal": r["goal"],
            "Steps on grass": r["steps_on_grass"],
            "myopia": r["mean_myopia"],
            "moral judgment": r["moral_judgment"],
            "shortcutiness": r["shortcutiness"],
        }
        for r in records
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
