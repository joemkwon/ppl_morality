# The model: equations, code map, and a known paper↔code discrepancy

## 1. Setup

A 10×10 grid-world park. Cells are `S` (sidewalk), `G` (grass), or the goal
`F`. Agents start and end on perimeter sidewalk cells; movement is 8-connected
(cardinal cost `1`, diagonal cost `√2`).

Six destinations, each with a utility (mean, SE) and a community-frequency,
elicited from ~50 Prolific participants and reported in the paper
(`src/moral_rules/priors.py`):

| goal | utility mean ± SE |
|---|---|
| a friend | 52.7 ± 3.87 |
| pain | 82.6 ± 3.87 |
| ice cream | 46.0 ± 3.87 |
| vac clinic | 48.0 ± 3.87 |
| porta-potty | 65.8 ± 3.87 |
| police car | 63.1 ± 3.87 |

## 2. Equations and where they live in the code

**Degree of rule-breaking σ ∈ [0, 1]** — trajectory length interpolated
between the strict sidewalk path (σ=0) and the unrestricted shortest path
(σ=1). → `trajectory.calculate_shortcutiness`.

**Agent utility (paper Eq. 1)** `U_agent = U_goal − (1 − σ)·path_cost`
(goal reward only if the trajectory ends on `F`; the `(1−σ)` factor
approximates a shortcut's cost as an interpolation toward a cost-free
beeline). → `trajectory.calculate_plan_utility`.

**Universalized utility (paper Eq. 2)** — see §3; this is where code and paper
differ. → `universalization.universalized_plan_utility`.

**Myopia mixture (paper Eq. 3)**
`U_overall = λ·U_agent + (1 − λ)·U_univ`, λ ∈ [0, 1]
(λ=1 fully self-interested, λ=0 fully community-minded).
→ inside `inference.generate_forward_model_table`.

**Forward model + inference (paper Eq. 4)** — for each shortcut column, a
numerically-stable softmax over λ turns utilities into `P(λ | σ)`; the column
nearest the observed σ *is* the posterior over λ. → `inference.generate_forward_model_table`,
`inference.infer_myopia_parameter`. Summaries: posterior-mean λ and the model's
moral judgment `100 − 100·E[λ]` (`inference.moral_judgment_from_myopia`).

The forward-model table is averaged over several resampled goal-utilities and
renormalized per column (`inference.infer_myopia_for_trajectory`), exactly as
the paper-producing scripts did.

## 3. The paper↔code discrepancy in the universalization term (read this)

The repository contained **three divergent implementations** of the
universalized utility. This matters because Eq. 2 is the entire `(1−λ)` half of
the objective.

| Source (legacy path) | What it computed | Used for the paper? |
|---|---|---|
| `models/main_model/model.py` | `mean_i U_goal,i` minus a **deterministic** `−cost` if grass over capacity (no path cost). Its `__main__` was **dead** (`sample_goal_utility('friend')` → `KeyError`; valid key is `'a friend'`), and it read a stray module-global `goal_utility` in the forward model. | **No** — dead, never run |
| `models/experimental_model/fragility_search.py` | Same welfare-mean form, cleanly parameterized. Closest to the **printed Eq. 2**. | Supporting (fragility) |
| `models/exp2_model`, `models/exp4_model` | **No welfare sum at all.** `0` if total grass-steps ≤ capacity, else `−cost` **stochastically** with `P = 1 − e^(−0.15·overage)`. Per-shortcut-column softmax. | **YES — produced the accepted figures** |

So the equation printed in the paper,

> `U_univ = (1/N)·( Σ_i U_goal,i − cost · 1[S_total > C] )`

is **not** the function that generated the paper's figures. The figure-producing
code instead returns a *stochastic penalty only*:

> `U_univ = 0` if `S_total ≤ C`, else `−cost` with probability `1 − e^(−0.15·(S_total − C))`.

### Findings on the originally-flagged "bugs"

1. **Global-`goal_utility` leak** — real, but **only in the dead
   `main_model.py`**. The paper code (`exp2`/`exp4`) threads the agent's own
   goal utility correctly. Not a paper bug; the file was deleted.
2. **"Universalization ignores path cost"** — reframed: the paper-producing
   code computes *no agent welfare at all*. This is a **modelling /
   documentation discrepancy**, not a code error, and it is the substantive
   issue here.
3. **No RNG seeds** — real and consequential: the universalization is a
   Bernoulli draw averaged over only 5×3 samples, so qualitative trends are
   robust but exact numbers were not reproducible run-to-run. Fixed by
   `config.seed_everything` **without changing any model logic**.

### How this release handles it (chosen policy: preserve + document)

`universalization.universalized_plan_utility` exposes a `variant` argument:

- **`"published_stochastic"` (default)** — the exact `exp2`/`exp4` computation
  that produced the **accepted paper's figures**. Used by `reproduce.py` for
  Fig. 6/7 and the model↔human comparison.
- **`"welfare_sum"`** — the paper's printed **Eq. 2** (the
  `fragility_search` form; still omits per-agent path cost, exactly as the
  original did). Used by `experiments/run_fragility.py` for the Fig. 4
  fragility narrative, and available for future work.

The accepted results are preserved exactly; the principled equation is
faithfully available and clearly labelled. Nothing silently "fixes" the
published model. Future work that wants U_univ to match Eq. 2 (including a
per-agent path cost) should build on `"welfare_sum"`.

## 4. Faithfully-preserved cleanups (no effect on results)

- BFS `visited` set seeded as `{(x, y)}` instead of the original
  `set((x, y))` (a set of two ints). BFS still dequeues the goal on a shortest
  path first, so path *lengths* — and therefore every trajectory's σ — are
  unchanged. Verified in `tests/test_gridworld.py`.
- `U_agent` is computed once per shortcut column instead of redundantly inside
  the λ loop (identical value, faster).
- Global mutable `grass_capacity` / `population` and hardcoded
  `/Users/...` paths removed; all parameters are explicit arguments.
