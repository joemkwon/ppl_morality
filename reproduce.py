#!/usr/bin/env python3
"""One command to regenerate the paper's figures.

    python reproduce.py --seed 0            # representative run (minutes)
    python reproduce.py --seed 0 --fast     # quick end-to-end smoke (~1-2 min)
    python reproduce.py --seed 0 --full     # exact accepted-paper settings (hours)

Outputs land in ``figures/`` and ``results/``:

* ``figures/fig1_trajectories.png``      - Fig. 1 (the 8 trajectories)
* ``figures/fig4_fragility.png``         - Fig. 4 (grass-fragility regimes)
* ``figures/fig6_7/*.png``               - Fig. 6/7 (myopia by community)
* ``figures/model_vs_human.png``         - model overlaid on the pilot data
* ``results/*.csv``                      - per-condition inference tables

Figures 3 and 5 of the paper are human-survey plots whose raw 50-participant
data is not part of this release; they are documented, not regenerated. See
``README.md`` and ``docs/MODEL.md``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

import compare_model_human  # noqa: E402
import make_fig1_trajectories  # noqa: E402
import run_experiment2  # noqa: E402
import run_experiment4  # noqa: E402
import run_fragility  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="coarse settings; quick smoke run")
    ap.add_argument("--full", action="store_true", help="exact accepted-paper settings (slow)")
    args = ap.parse_args()
    fast = args.fast and not args.full

    steps = [
        ("Fig 1  trajectories", lambda: make_fig1_trajectories.main()),
        ("Fig 4  grass fragility", lambda: run_fragility.main(seed=args.seed, fast=fast)),
        ("Fig 6/7 community myopia", lambda: run_experiment4.main(seed=args.seed, fast=fast)),
        ("model vs human", lambda: compare_model_human.main(seed=args.seed, fast=fast)),
    ]
    if args.full:
        steps.append(
            ("Exp 2 capacity x cost CSVs", lambda: run_experiment2.main(seed=args.seed, fast=False))
        )

    for label, fn in steps:
        print(f"\n=== {label} ===")
        t0 = time.time()
        fn()
        print(f"    ({time.time() - t0:.1f}s)")
    print("\nDone. See figures/ and results/.")


if __name__ == "__main__":
    main()
