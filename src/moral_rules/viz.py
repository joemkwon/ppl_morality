"""Plotting: trajectories (Fig. 1), myopia posteriors, and community violins (Fig. 6/7)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

GRASS_COLOR = "#4f9d4f"
SIDEWALK_COLOR = "#d9d9d9"
PATH_COLOR = "#d62728"


def render_trajectory(
    gridworld: list[list[str]],
    trajectory: list[dict],
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Draw one grid-world trajectory (sidewalk vs grass + the agent's path)."""
    height = len(gridworld)
    width = len(gridworld[0])
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    for y in range(height):
        for x in range(width):
            color = GRASS_COLOR if gridworld[y][x] == "G" else SIDEWALK_COLOR
            ax.add_patch(
                plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white", linewidth=0.5)
            )

    xs = [step["coordinate"][0] + 0.5 for step in trajectory]
    ys = [step["coordinate"][1] + 0.5 for step in trajectory]
    ax.plot(xs, ys, color=PATH_COLOR, linewidth=2.5, zorder=3)
    ax.scatter([xs[0]], [ys[0]], c="black", s=60, zorder=4, label="start")
    ax.scatter(
        [xs[-1]], [ys[-1]], marker="*", c="gold", s=300, edgecolors="black", zorder=4, label="goal"
    )

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_trajectory_grid(
    grids_and_trajectories: list[tuple], out_path: str | Path, titles: list[str] | None = None
) -> None:
    """Grid of trajectory panels (the paper's Fig. 1 layout: 2 rows x 4)."""
    n = len(grids_and_trajectories)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, (gw, traj) in enumerate(grids_and_trajectories):
        render_trajectory(gw, traj, ax=axes[i], title=titles[i] if titles else None)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_posterior(
    myopia_values: list[float], posterior: list[float], out_path: str | Path, title: str = ""
) -> None:
    """Single inferred myopia posterior with its mean marked."""
    mean_myopia = float(np.sum(np.array(myopia_values) * np.array(posterior)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(myopia_values, posterior, label="posterior")
    ax.axvline(mean_myopia, color="red", linestyle="--", label=f"mean = {mean_myopia:.2f}")
    ax.set_xlabel("myopia parameter (lambda)")
    ax.set_ylabel("posterior probability")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_myopia_violins(records: list[dict], out_path: str | Path, title: str = "") -> None:
    """Violin plot of inferred myopia by trajectory and goal (Fig. 6/7 style).

    ``records`` items: ``{"map", "goal", "myopia_values", "posterior"}``.
    Each posterior is resampled to approximate its distribution for the violin.
    """
    rows = []
    for rec in records:
        samples = np.random.choice(rec["myopia_values"], size=1000, p=rec["posterior"])
        rows.extend({"Map": rec["map"], "Goal": rec["goal"], "Myopia": float(s)} for s in samples)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(x="Map", y="Myopia", hue="Goal", data=df, palette="Set2", cut=0, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("trajectory (increasing grass shortcut ->)")
    ax.set_ylabel("inferred myopia")
    plt.xticks(rotation=45)
    ax.legend(title="Goal", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_moral_judgment_lines(
    df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Moral judgment vs steps on grass",
    value_col: str = "moral judgment",
) -> None:
    """Line plot of (model or human) judgment vs steps on grass, per goal."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for goal in sorted(df["Goal"].unique()):
        g = df[df["Goal"] == goal].sort_values("Steps on grass")
        ax.plot(g["Steps on grass"], g[value_col], marker="o", label=goal)
    ax.set_title(title)
    ax.set_xlabel("steps on grass")
    ax.set_ylabel(value_col)
    ax.set_ylim(0, 100)
    ax.legend(title="Goal", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
