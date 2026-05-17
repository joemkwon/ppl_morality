# LLM baseline (exploratory — NOT in the paper)

> **This directory is not part of the accepted CogSci 2025 paper
> ("What do moral rules mean?").** No LLM result appears in the paper. It is
> retained only for transparency and possible future work.

A GPT-4o baseline that is shown the same park-map stimuli used in the human
experiments and asked the same questions (grass fragility, community impact,
per-goal frequency/urgency, permissibility), 50 samples per question at
temperature 1.0.

## Contents

- `gpt4_baseline.py` — single parametrized script. It replaces the four original
  near-identical scripts (`gpt4_experiment{1,2}_{with,without}_sign.py`), which
  remain in git history under the `legacy-2024` tag.
- `experiment_stimuli/` — the four stimulus sets
  (`exp{1,2}_stimuli_{with_sign,no_sign}/`), each with an `experiment_config.json`
  and its map images.

Cached GPT-4o responses from the original exploration are **not shipped** (≈5 MB,
non-deterministic, regenerable); they remain in the `legacy-2024` tag.

## Running

```bash
pip install -e ".[llm]"
export OPENAI_API_KEY=...        # never commit this; .env is git-ignored
python LLM/gpt4_baseline.py --experiment 1 --sign
python LLM/gpt4_baseline.py --experiment 2 --no-sign
```

Results are written to `LLM/experiment_results/` (git-ignored). Calling a paid
API with `temperature=1.0` and no seed is inherently non-reproducible.
