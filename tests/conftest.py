import pytest

from moral_rules import load_gridworld_and_trajectory
from moral_rules.config import TRAJECTORY_DIR, TRAJECTORY_FILES


@pytest.fixture(scope="session")
def trajectories():
    """All 8 canonical (gridworld, trajectory) pairs, ordered map1_1..map1_8."""
    return [load_gridworld_and_trajectory(TRAJECTORY_DIR / f) for f in TRAJECTORY_FILES]
