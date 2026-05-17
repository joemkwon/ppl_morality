"""Global configuration: paths, default hyperparameters, and reproducible seeding.

The original research code seeded nothing, so every run produced different numbers
(the universalization step is genuinely stochastic -- see ``universalization.py``).
For a public release we make runs reproducible *without changing the model logic*:
:func:`seed_everything` seeds both the ``random`` and ``numpy`` global RNGs, which
are the only sources of randomness the model uses.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --- Paths -----------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
DATA_DIR = REPO_ROOT / "data"
TRAJECTORY_DIR = DATA_DIR / "trajectories"
HUMAN_DATA_DIR = DATA_DIR / "human"
FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"

# --- Default hyperparameters ----------------------------------------------
# These match the values used by the scripts that produced the accepted paper's
# figures (``models/exp2_model`` and ``models/exp4_model``).
DEFAULT_SEED = 0
NUM_BINS = 50  # discretization of the myopia x shortcut grid
NUM_AGENTS = 100  # population size in the universalization step
NUM_SIMULATIONS = 5  # universalization Monte-Carlo samples per cell
NUM_GOAL_UTILITY_SAMPLES = 3  # goal-utility resamples averaged per trajectory
GRASS_COMMUNITY_COST = 1_000_000  # penalty when the grass is destroyed
GRASS_OVERAGE_RATE = 0.15  # lambda in P(destroyed) = 1 - exp(-rate * overage)

# The eight canonical trajectories ("map1_1" .. "map1_8"), ordered by increasing
# amount of grass shortcutting. Filename ordinal k -> "steps on grass" axis = k-1.
TRAJECTORY_FILES = [f"map1_{i}.json" for i in range(1, 9)]


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Seed every RNG the model touches so a run is bit-for-bit reproducible.

    Reproducibility additionally requires a single thread and a fixed NumPy
    version (NumPy's Gaussian/`choice` algorithms can change across major
    versions); both are pinned in ``pyproject.toml``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
