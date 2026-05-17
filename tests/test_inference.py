"""Forward-model normalization, posterior recovery, and seeded determinism."""

import numpy as np
import pytest

from moral_rules import (
    CONDITION_0,
    generate_forward_model_table,
    infer_myopia_for_trajectory,
    infer_myopia_parameter,
    seed_everything,
)

KW = dict(
    num_bins=12,
    num_agents=20,
    num_simulations=2,
    grass_capacity=120,
    grass_community_cost=1_000_000,
)


def test_softmax_columns_sum_to_one(trajectories):
    seed_everything(0)
    gw, traj = trajectories[3]
    table = generate_forward_model_table(gw, traj, 50.0, CONDITION_0, **KW)
    columns: dict[float, float] = {}
    for (_, shortcut), prob in table.items():
        columns[shortcut] = columns.get(shortcut, 0.0) + prob
    for total in columns.values():
        assert total == pytest.approx(1.0, abs=1e-9)


def test_infer_returns_normalized_posterior(trajectories):
    seed_everything(0)
    gw, traj = trajectories[5]
    table = generate_forward_model_table(gw, traj, 50.0, CONDITION_0, **KW)
    myopia_values, posterior = infer_myopia_parameter(gw, traj, table)
    assert len(myopia_values) == len(posterior) == KW["num_bins"]
    assert sum(posterior) == pytest.approx(1.0)
    assert all(p >= 0 for p in posterior)


def test_injected_posterior_peaks_at_dominant_myopia(trajectories):
    """A hand-built column with one dominant myopia must be recovered."""
    gw, traj = trajectories[4]
    myopias = list(np.linspace(0, 1, 5))
    shortcut = 0.5
    table = {(m, shortcut): (0.9 if i == 3 else 0.025) for i, m in enumerate(myopias)}
    _, posterior = infer_myopia_parameter(gw, traj, table)
    assert int(np.argmax(posterior)) == 3


def test_seeded_determinism(trajectories):
    gw, traj = trajectories[2]
    kw = dict(
        num_bins=10,
        num_agents=20,
        num_simulations=2,
        num_goal_utility_samples=2,
        grass_capacity=120,
        grass_community_cost=1_000_000,
    )
    seed_everything(123)
    a = infer_myopia_for_trajectory(gw, traj, "pain", CONDITION_0, **kw)
    seed_everything(123)
    b = infer_myopia_for_trajectory(gw, traj, "pain", CONDITION_0, **kw)
    assert a["mean_myopia"] == b["mean_myopia"]
    assert a["posterior"] == b["posterior"]


def test_welfare_sum_variant_runs(trajectories):
    seed_everything(0)
    gw, traj = trajectories[1]
    table = generate_forward_model_table(gw, traj, 50.0, CONDITION_0, variant="welfare_sum", **KW)
    assert len(table) == KW["num_bins"] ** 2
