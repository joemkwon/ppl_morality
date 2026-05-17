"""Experiment 2: grass-capacity x community-cost sweep on the baseline community.

Mirrors ``models/exp2_model``: for each (grass_capacity, community_cost) it
runs inference for all 6 goals x 8 trajectories under the empirical baseline
community (``CONDITION_0``) and writes
``results/myopia_moral_judgment_grass_capacity_<C>_community_cost_<cost>.csv``.
These CSVs are the inputs behind the paper's grass-fragility analysis (Fig. 4).
"""

from __future__ import annotations

import argparse
import itertools

from _common import RunParams, records_to_csv, sweep_goals_and_trajectories

from moral_rules import CONDITION_0, seed_everything
from moral_rules.config import DEFAULT_SEED, RESULTS_DIR


def main(
    seed: int = DEFAULT_SEED,
    fast: bool = False,
    grass_capacities: list[int] | None = None,
    community_costs: list[int] | None = None,
) -> None:
    seed_everything(seed)
    params = RunParams.fast() if fast else RunParams.paper()
    grass_capacities = grass_capacities or ([180] if fast else [100, 200])
    community_costs = community_costs or (
        [10_000] if fast else [100, 2000, 4000, 6000, 8000, 10000]
    )

    for capacity, cost in itertools.product(grass_capacities, community_costs):
        params.grass_community_cost = cost
        print(f"[exp2] grass_capacity={capacity} community_cost={cost}")
        records = sweep_goals_and_trajectories(CONDITION_0, capacity, params)
        out = (
            RESULTS_DIR / f"myopia_moral_judgment_grass_capacity_{capacity}"
            f"_community_cost_{cost}.csv"
        )
        records_to_csv(records, out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    main(seed=args.seed, fast=args.fast)
