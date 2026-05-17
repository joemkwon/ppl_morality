"""Grass-fragility analysis (Fig. 4), using the paper's Eq. 2 (welfare_sum).

Shows how the universalized utility of cutting across the grass depends on how
fragile the grass is (its capacity). The paper's three regimes:

* extremely fragile  (capacity ~ a handful of steps)  -> rule is uninformative
* intermediate       (capacity ~ 150)                  -> rule is most informative
* extremely robust   (capacity > 10000)                -> rule is uninformative

This driver intentionally uses ``variant="welfare_sum"`` -- the paper's printed
Eq. 2 -- because it is the principled per-agent-welfare form; the inferred-myopia
figures (Fig. 6/7) instead use the published stochastic variant. See
``docs/MODEL.md``.
"""

from __future__ import annotations

import argparse

import numpy as np

from moral_rules import CONDITION_0, load_gridworld_and_trajectory, seed_everything
from moral_rules.config import DEFAULT_SEED, FIGURES_DIR, TRAJECTORY_DIR
from moral_rules.gridworld import full_rule_following_trajectory  # noqa: F401  (warm import)
from moral_rules.priors import normalize_proportions
from moral_rules.universalization import universalized_plan_utility


def main(seed: int = DEFAULT_SEED, fast: bool = False, out_path: str | None = None) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed_everything(seed)
    gw, _ = load_gridworld_and_trajectory(TRAJECTORY_DIR / "map1_5.json")
    probs = normalize_proportions(CONDITION_0)

    num_agents = 30 if fast else 100
    num_sims = 5 if fast else 20
    my_goal_utility = 50.0
    # Log-spaced capacities spanning fragile -> robust (the paper's regimes).
    capacities = np.unique(np.round(np.logspace(0.5, 4.2, 14 if fast else 30)))
    shortcut_levels = [0.1, 0.3, 0.6, 1.0]

    fig, ax = plt.subplots(figsize=(9, 6))
    for sigma in shortcut_levels:
        utilities = []
        for cap in capacities:
            samples = [
                universalized_plan_utility(
                    gw,
                    my_goal_utility,
                    sigma,
                    probs,
                    grass_capacity=cap,
                    grass_community_cost=10_000,
                    num_agents=num_agents,
                    variant="welfare_sum",
                )
                for _ in range(num_sims)
            ]
            utilities.append(float(np.mean(samples)))
        ax.plot(capacities, utilities, marker="o", label=f"shortcut sigma={sigma}")

    for cap, lbl in [(4, "fragile"), (150, "intermediate"), (10000, "robust")]:
        ax.axvline(cap, color="gray", linestyle=":", alpha=0.6)
        ax.text(
            cap, ax.get_ylim()[1], lbl, rotation=90, va="top", ha="right", fontsize=8, color="gray"
        )

    ax.set_xscale("log")
    ax.set_xlabel("grass capacity (steps before destruction) - log scale")
    ax.set_ylabel("universalized utility (welfare_sum, Eq. 2)")
    ax.set_title("Grass fragility shapes the universalized utility (Fig. 4)")
    ax.legend()
    fig.tight_layout()

    out_path = out_path or str(FIGURES_DIR / "fig4_fragility.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(seed=args.seed, fast=args.fast, out_path=args.out)
