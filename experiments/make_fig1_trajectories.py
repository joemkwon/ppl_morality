"""Figure 1: the eight agent trajectories, from no-shortcut to maximal shortcut.

Reproduces the paper's Fig. 1 (2x4 grid of grid-world panels).
"""

from __future__ import annotations

import argparse

from moral_rules import calculate_shortcutiness, load_gridworld_and_trajectory
from moral_rules.config import FIGURES_DIR, TRAJECTORY_DIR, TRAJECTORY_FILES
from moral_rules.viz import plot_trajectory_grid


def main(out_path: str | None = None) -> str:
    grids, titles = [], []
    for fname in TRAJECTORY_FILES:
        gw, traj = load_gridworld_and_trajectory(TRAJECTORY_DIR / fname)
        grids.append((gw, traj))
        sigma = calculate_shortcutiness(gw, traj)
        titles.append(f"{fname.split('.')[0]}  (sigma={sigma:.2f})")
    out_path = out_path or str(FIGURES_DIR / "fig1_trajectories.png")
    plot_trajectory_grid(grids, out_path, titles=titles)
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    main(ap.parse_args().out)
