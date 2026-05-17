# Changelog

## 1.0.0 — 2026-05-17 — Public release cleanup

First public, reproducible release. The pre-cleanup state is preserved in git
under the tag **`legacy-2024`**.

### Repository archaeology
- Confirmed **`ppl_morality`** is the paper's codebase; the accepted figures
  were produced by `models/exp2_model` and `models/exp4_model`.
- The separate **`joemkwon/line_modeling`** repo (Julia PDDL multi-agent
  water-collection task) is an **unrelated abandoned predecessor** and is *not*
  part of this release.

### Added
- `moral_rules` package: one deduplicated, documented, importable core
  (`gridworld`, `trajectory`, `universalization`, `inference`, `priors`, `viz`,
  `config`) replacing three ~660-line divergent copies of the model.
- `reproduce.py` (one command → Fig. 1/4/6/7), thin experiment drivers, a new
  model↔human comparison (`compare_model_human.py`), `pyproject.toml`,
  `Makefile`, pinned dependencies, test suite, `docs/MODEL.md`.
- `universalization` `variant=` switch exposing **both** the published
  stochastic form (default; reproduces the paper) and the paper's printed
  Eq. 2 welfare-sum form (`docs/MODEL.md` §3).

### Fixed / reproducibility
- **Deterministic seeding** (`config.seed_everything`) — the original code
  seeded nothing; runs are now bit-for-bit reproducible per `--seed` **with no
  change to model logic**.
- Removed global mutable state (`grass_capacity`/`population`) and hardcoded
  absolute `/Users/...` paths.
- Deleted the dead, buggy `main_model/model.py` (broken `__main__`; stray
  module-global `goal_utility`) — never used for the paper; bugs are moot.

### Removed (kept only under the `legacy-2024` tag)
- Superseded model code: `models/main_model`, `models/astar_model`,
  `models/value_iteration_model`, `models/experimental_model`, `models/old`,
  `v0/v1/v2` CSV/plot dirs, `backup_sep20.py`, `node_modules/`.
- **All PII**: `old_pilot_results/` / `old_pilot_stimuli/` jsPsych exports
  contained Prolific IDs, IP addresses, and user agents. The single
  human-data file shipped (`data/human/Experiment2_Pilot.csv`) has none;
  enforced by `make check-no-pii`.
- ≈5 MB of non-deterministic cached GPT-4o responses (regenerable).

### Notes
- Paper Fig. 3 and Fig. 5 are raw-survey plots; the survey is not part of this
  release, so those figures are documented (priors verbatim in `priors.py`),
  not regenerated.
- The four GPT-4o scripts were consolidated into one `LLM/gpt4_baseline.py`,
  clearly marked exploratory and **not part of the accepted paper**.
