"""Computational model of moral-rule interpretation ("keep off the grass").

Kwon, Wong, Tenenbaum & Levine, "What do moral rules mean?", CogSci 2025.

Public API mirrors the model pipeline: build a forward model of a rational
rule-follower, then invert it to infer how *myopic* an observed agent is, and
hence how permissible their rule-breaking trajectory looks.
"""

from .config import DEFAULT_SEED, seed_everything
from .inference import (
    generate_forward_model_table,
    infer_myopia_for_trajectory,
    infer_myopia_parameter,
    moral_judgment_from_myopia,
    posterior_mean_myopia,
)
from .priors import COMMUNITY_CONDITIONS, CONDITION_0, GOAL_UTILITIES, GOALS
from .trajectory import (
    calculate_plan_utility,
    calculate_shortcutiness,
    load_gridworld_and_trajectory,
)
from .universalization import universalized_plan_utility

__version__ = "1.0.0"

__all__ = [
    "seed_everything",
    "DEFAULT_SEED",
    "GOALS",
    "GOAL_UTILITIES",
    "CONDITION_0",
    "COMMUNITY_CONDITIONS",
    "load_gridworld_and_trajectory",
    "calculate_plan_utility",
    "calculate_shortcutiness",
    "universalized_plan_utility",
    "generate_forward_model_table",
    "infer_myopia_parameter",
    "infer_myopia_for_trajectory",
    "posterior_mean_myopia",
    "moral_judgment_from_myopia",
]
