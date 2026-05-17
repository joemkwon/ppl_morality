"""Experiment 4: inferred myopia across five community compositions (Fig. 6/7).

Mirrors ``models/exp4_model``: for each of the five ``COMMUNITY_CONDITIONS``
it runs inference for all 6 goals x 8 trajectories and produces a violin plot
of inferred myopia by trajectory and goal. This is the community-context
result: identical actions imply different myopia depending on whether the
community pursues mostly low- or high-urgency goals.
"""

from __future__ import annotations

import argparse

from _common import RunParams, records_to_csv, sweep_goals_and_trajectories

from moral_rules import COMMUNITY_CONDITIONS, seed_everything
from moral_rules.config import DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR
from moral_rules.viz import plot_myopia_violins


def main(
    seed: int = DEFAULT_SEED,
    fast: bool = False,
    grass_capacity: int = 275,
    conditions: list[str] | None = None,
) -> list[str]:
    seed_everything(seed)
    params = RunParams.fast() if fast else RunParams.paper()
    conditions = conditions or list(COMMUNITY_CONDITIONS)
    outputs = []
    for name in conditions:
        print(f"[exp4] {name} (grass_capacity={grass_capacity})")
        records = sweep_goals_and_trajectories(COMMUNITY_CONDITIONS[name], grass_capacity, params)
        records_to_csv(
            records,
            RESULTS_DIR / f"exp4_{name}_grass_capacity_{grass_capacity}.csv",
        )
        out = str(FIGURES_DIR / "fig6_7" / f"myopia_violin_{name}_grass_{grass_capacity}.png")
        plot_myopia_violins(
            records,
            out,
            title=f"Inferred myopia - {name} (grass capacity {grass_capacity})",
        )
        print(f"  wrote {out}")
        outputs.append(out)
    return outputs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--grass-capacity", type=int, default=275)
    args = ap.parse_args()
    main(seed=args.seed, fast=args.fast, grass_capacity=args.grass_capacity)
