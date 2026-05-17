"""Human-derived priors: goal utilities, goal frequencies, and community conditions.

PROVENANCE
----------
These constants were elicited from ~50 Prolific participants and are the values
*reported in the paper* (Kwon, Wong, Tenenbaum & Levine, CogSci 2025). The raw
per-participant survey export is **not** part of this release (it is not retained
in a clean, de-identified form); these aggregates are therefore the single source
of truth and are reproduced here verbatim from the analysis behind the paper.

``GOAL_UTILITIES`` -- per goal, the (mean, standard error) of its importance /
urgency on a 0-100 scale. Agents sample a goal utility ~ Normal(mean, se).

``CONDITION_0`` -- the baseline community goal-frequency distribution (the
"expected % of the community pursuing each goal" survey question). Used by the
grass-fragility sweep.

``COMMUNITY_CONDITIONS`` -- five hand-constructed community compositions used to
study how identical actions carry different moral weight depending on community
needs (the "more low urgency" vs "more high urgency" communities in the paper's
community-context analysis). Condition_0 is the empirical baseline; Conditions
1-4 deliberately re-weight high-urgency goals (pain, porta-potty) up or down.
Values are raw "percent of community" estimates; they are renormalized to a
probability distribution at sampling time.
"""

from __future__ import annotations

# Goal -> (mean utility, standard error), 0-100 scale.
GOAL_UTILITIES: dict[str, tuple[float, float]] = {
    "a friend": (52.7, 3.87),
    "pain": (82.6, 3.87),
    "ice cream": (46.0, 3.87),
    "vac clinic": (48.0, 3.87),
    "porta-potty": (65.8, 3.87),
    "police car": (63.1, 3.87),
}

GOALS: list[str] = list(GOAL_UTILITIES)

# Baseline empirical community composition (expected % pursuing each goal).
CONDITION_0: dict[str, float] = {
    "a friend": 50.5,
    "pain": 34.9,
    "ice cream": 42.6,
    "vac clinic": 28.7,
    "porta-potty": 30.7,
    "police car": 11.1,
}

# Five community conditions used in the community-context analysis (Fig. 6/7).
# Condition_0 here uses the rounded survey values used by the exp4 driver; the
# unrounded baseline above is used by the grass-fragility sweep (exp2 driver).
COMMUNITY_CONDITIONS: dict[str, dict[str, float]] = {
    "Condition_0": {  # empirical baseline
        "a friend": 44,
        "pain": 35,
        "ice cream": 44,
        "vac clinic": 23,
        "porta-potty": 34,
        "police car": 9,
    },
    "Condition_1": {  # more high-urgency goals (pain & porta-potty up)
        "a friend": 44,
        "pain": 60,
        "ice cream": 44,
        "vac clinic": 23,
        "porta-potty": 50,
        "police car": 9,
    },
    "Condition_2": {  # high-urgency goals near zero
        "a friend": 44,
        "pain": 0,
        "ice cream": 44,
        "vac clinic": 23,
        "porta-potty": 1,
        "police car": 9,
    },
    "Condition_3": {  # mostly low-urgency goals
        "a friend": 60,
        "pain": 0,
        "ice cream": 70,
        "vac clinic": 23,
        "porta-potty": 1,
        "police car": 9,
    },
    "Condition_4": {  # almost everyone pursues ice cream (lowest urgency)
        "a friend": 4,
        "pain": 0,
        "ice cream": 96,
        "vac clinic": 0,
        "porta-potty": 0,
        "police car": 2,
    },
}


def normalize_proportions(proportions: dict[str, float]) -> dict[str, float]:
    """Turn raw 'percent of community' estimates into a probability distribution."""
    total = sum(proportions.values())
    if total <= 0:
        raise ValueError("Goal proportions must sum to a positive value.")
    return {goal: value / total for goal, value in proportions.items()}
