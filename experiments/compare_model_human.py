"""Model vs. human moral judgments (overlay on the pilot data).

The original project never shipped a model<->human comparison script (that
analysis was done off-repo). This adds a faithful one: it aggregates the human
``Experiment2_Pilot.csv`` (mean rating by goal x steps-on-grass, pooled over the
sign manipulation) and overlays the model's predicted moral judgment
(``100 - 100*mean_myopia``) for the same goals and trajectories.

Caveat: the human "Steps on Grass" axis and the model's trajectory-ordinal axis
are aligned by index (0..7), matching the paper's CSV convention; the model's
continuous shortcutiness is reported separately by the experiment CSVs.
"""

from __future__ import annotations

import argparse

import pandas as pd
from _common import RunParams, sweep_goals_and_trajectories

from moral_rules import CONDITION_0, GOALS, seed_everything
from moral_rules.config import DEFAULT_SEED, FIGURES_DIR, HUMAN_DATA_DIR

# Human CSV uses different casing than the model's goal keys.
HUMAN_TO_MODEL_GOAL = {
    "a friend": "a friend",
    "Ice Cream": "ice cream",
    "Pain": "pain",
    "vac clinic": "vac clinic",
    "porta-potty": "porta-potty",
    "police car": "police car",
}


def main(seed: int = DEFAULT_SEED, fast: bool = False, out_path: str | None = None) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed_everything(seed)

    human_csv = HUMAN_DATA_DIR / "Experiment2_Pilot.csv"
    if not human_csv.exists():
        print(
            "[skip] Human comparison needs Experiment2_Pilot.csv, which is not "
            "part of the public release (human data is held privately by the "
            "authors; see data/human/README.md). Skipping model-vs-human."
        )
        return ""

    human = pd.read_csv(human_csv)
    human["model_goal"] = human["Goal"].map(HUMAN_TO_MODEL_GOAL)
    human_agg = (
        human.groupby(["model_goal", "Steps on Grass"])["Moral Judgment"].mean().reset_index()
    )

    params = RunParams.fast() if fast else RunParams.paper()
    records = sweep_goals_and_trajectories(CONDITION_0, 180, params)
    model = pd.DataFrame(
        {
            "model_goal": r["goal"],
            "Steps on Grass": r["steps_on_grass"],
            "moral judgment": r["moral_judgment"],
        }
        for r in records
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    for ax, goal in zip(axes.ravel(), GOALS, strict=True):
        h = human_agg[human_agg["model_goal"] == goal].sort_values("Steps on Grass")
        m = model[model["model_goal"] == goal].sort_values("Steps on Grass")
        ax.plot(h["Steps on Grass"], h["Moral Judgment"], marker="o", label="human (mean)")
        ax.plot(m["Steps on Grass"], m["moral judgment"], marker="s", linestyle="--", label="model")
        ax.set_title(goal)
        ax.set_ylim(0, 100)
        ax.set_xlabel("steps on grass")
        ax.set_ylabel("moral judgment")
        ax.legend(fontsize=8)
    fig.suptitle("Model vs. human moral judgment by goal")
    fig.tight_layout()

    out_path = out_path or str(FIGURES_DIR / "model_vs_human.png")
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
