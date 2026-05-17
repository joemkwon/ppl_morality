# What do moral rules mean?

Model code for **Kwon, Wong, Tenenbaum & Levine, "What do moral rules mean?",
*Proceedings of CogSci 2025*.**

People interpret a simple rule like *"keep off the grass"* by reasoning about
the rule's *purpose*. This repository implements that idea as a computational
model in a grid-world community park: a rational agent weighs its immediate goal
against the universalized consequence of everyone shortcutting across the grass,
and an observer inverts this model to infer how *myopic* an agent who took a
given shortcut must be — which predicts how permissible the shortcut seems.

> **Provenance / cleanup note.** This is a cleaned, deduplicated, reproducible
> release. The accepted paper's figures were produced by the Python code in the
> old `models/exp2_model/` and `models/exp4_model/` (now unified into the
> `moral_rules` package). The pre-cleanup state is preserved in git under the
> tag **`legacy-2024`**. An unrelated abandoned Julia repository
> (`joemkwon/line_modeling`) is **not** part of this work.

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # Python >= 3.10
pip install -e ".[dev]"        # add ",llm" for the exploratory GPT-4o baseline
```

## Reproduce the figures

```bash
python reproduce.py --seed 0           # representative run (minutes)
python reproduce.py --seed 0 --fast    # quick end-to-end smoke (~1-2 min)
python reproduce.py --seed 0 --full    # exact accepted-paper settings (slow)
```

Outputs land in `figures/` and `results/`:

| Output | Paper figure |
|---|---|
| `figures/fig1_trajectories.png` | **Fig. 1** — the 8 trajectories (σ = 0 → 1) |
| `figures/fig4_fragility.png` | **Fig. 4** — grass-fragility regimes |
| `figures/fig6_7/*.png` | **Fig. 6/7** — inferred myopia by community |
| `figures/model_vs_human.png` | model overlaid on the pilot human data |
| `results/*.csv` | per-condition inference tables |

**Not regenerated:** the paper's **Fig. 3** (community goal distributions) and
**Fig. 5** (priors by sign condition) are plots of the raw 50-participant survey,
which is *not* part of this release (see `data/human/README.md`). The survey
aggregates are reproduced verbatim as documented constants in
`src/moral_rules/priors.py`.

Results are bit-for-bit reproducible for a fixed `--seed` (single thread, pinned
NumPy). Without a seed the model is genuinely stochastic — see
[`docs/MODEL.md`](docs/MODEL.md), which also documents a **discrepancy between
the paper's printed Eq. 2 and the code that produced the figures**, and how this
release handles it (both forms are provided; the published one is the default).

## Layout

```
src/moral_rules/   the model: gridworld, trajectory, universalization, inference
experiments/       thin drivers for each analysis (Exp 2, Exp 4, fragility, ...)
reproduce.py       one command -> all figures
data/trajectories/ the 8 canonical trajectories (one deduplicated copy)
data/human/        de-identified pilot ratings + provenance
LLM/               exploratory GPT-4o baseline (NOT in the paper)
tests/             invariant + determinism tests
docs/MODEL.md      equations, paper<->code map, the Eq. 2 discrepancy
```

## Tests

```bash
pytest          # or: make test
make check-no-pii
```

## Citation

See [`CITATION.cff`](CITATION.cff). Code under MIT; `data/` under CC-BY-4.0.
