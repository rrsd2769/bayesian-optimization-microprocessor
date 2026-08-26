# Multi-Objective BO Upgrade: Defensible Pareto Results Implementation Plan

## Overview

The current results are numerically correct but structurally indefensible: a
single-objective optimizer is being plotted as if it produced a Pareto front,
the two plotted objectives are 88% rank-correlated, the accuracy surrogate is
non-monotone, two of six search dimensions are inert, and the headline p-value
sits exactly at its arithmetic floor. This plan replaces the problem
formulation with true multi-objective BO, replaces the accuracy surrogate with
a monotone scaling law, makes the scheduler variables load-bearing via a
latency constraint, and scales the statistics up by ~3 orders of magnitude —
which is essentially free, because `evaluate()` takes 3.8 ms, not the 5 s its
docstring assumes.

## Current State Analysis

All numbers below were measured against `HEAD = a5ac4ec`, not assumed.

### Defect 1 — Single-objective optimizer plotted as a Pareto front

[`src/bo_experiment.py:48`](../../src/bo_experiment.py) declares
`OBJECTIVES = {"energy_mj": ObjectiveProperties(minimize=True)}` with
`accuracy >= 0.75` as an outcome constraint. This is single-objective
optimization. A converged single-objective optimizer *collapses to one point* —
that is correct behaviour, not a bug.

The consequence, measured in `results/bo_results.csv`:

- The config `L4 / w=0.250 / q=4 / dvfs=0.500` was evaluated **7 times out of 30 trials** (23% of budget).
- 8 of 22 feasible BO points are duplicates at that single location.
- Random Search, which explores rather than converges, therefore yields a *visually wider* Pareto front than BO.

The Pareto figure structurally undercuts the result it is meant to support.

### Defect 2 — The plotted objectives barely trade off

Over all 85 evaluated configurations (BO + Random + Grid):

| Objective pair | Spearman ρ |
|---|---|
| memory_kb vs energy_mj | **0.881** |
| accuracy vs memory_kb | 0.673 |
| latency_ms vs energy_mj | 0.621 |
| **accuracy vs energy_mj** | **0.521** |

`figure3_pareto_frontier` plots memory vs energy — the *most* collinear pair
available. Both quantities are essentially `params × bits`. A Pareto front
between near-collinear objectives is a short diagonal with no tension in it.
Accuracy vs energy is the pair with a real tradeoff.

### Defect 3 — The accuracy surrogate is non-monotone (highest-severity)

[`src/objective.py:65`](../../src/objective.py) `accuracy_proxy()` uses
inverse-distance weighting over 24 calibration points. IDW has no monotonicity
guarantee, and here it visibly fails:

- At `num_layers=2`, predicted accuracy **peaks at w=0.75 (0.7097), then falls to 0.6612 at w=1.00** — overshooting *both* measured endpoints (`(2,0.25)=0.5249`, `(2,1.00)=0.6642`) by up to 4.5 pp. The overshoot is contamination from the higher-accuracy L3/L4 calibration points.
- At `num_layers=4`, accuracy **decreases** from w=0.25 (0.7543) to w=0.35 (0.7536).
- Across a dense sweep, **9 of 45 width steps are non-monotone**.

A surrogate that predicts a wider network is less accurate is not defensible in
review.

### Defect 4 — Two of six search dimensions are inert

Verified by direct evaluation: holding architecture fixed and sweeping
`time_slice ∈ {5.0, 27.5, 50.0} × priority ∈ {0,1,2}`, the values of
`accuracy`, `memory_kb`, and `energy_mj` are **bit-identical** across all 9
combinations. These two parameters move only `latency_ms`, which is neither an
objective nor a constraint. BO spends its budget searching two dimensions of
pure noise.

Their effect on latency is also strongly architecture-dependent:

| Architecture | latency range over scheduler vars | swing |
|---|---|---|
| L2 w=0.40 | 11.35 → 11.35 ms | 0.0% |
| L4 w=0.25 | 12.32 → 12.32 ms | 0.0% |
| L3 w=0.60 | 47.59 → 54.59 ms | 14.7% |
| L4 w=0.90 | 310.10 → 486.90 ms | **57.0%** |

Scheduling only matters once the workload saturates the processor — which is
physically correct, and becomes a *finding* rather than dead weight once
latency is constrained.

### Defect 5 — The optimum is a box corner

`results/bo_best_config.json` reports `num_layers=4` (max), `width_mult=0.25`
(min), `quant_bits=4` (min), `dvfs_factor=0.5` (min) — **4 of 6 variables
pinned at their bounds**, with `bo_std_mj = 0.0` across all 5 seeds. The
problem as posed is too easy to justify BO.

### Defect 6 — The headline p-value is at its arithmetic floor

`robustness_summary.json` reports `p_value = 0.0312`. The one-sided Wilcoxon
signed-rank test with n=5 pairs has a **minimum achievable p-value of exactly
1/32 = 0.03125**. The test is saturated: BO won on all 5 seeds and this is the
strongest statement the test can make. With n=20 the same clean sweep yields
p ≈ 3.9e-6.

### Enabling fact — evaluation is ~1300× cheaper than assumed

[`src/objective.py:8`](../../src/objective.py) states "Performance target: <5
seconds per call." Measured: **3.8 ms/call**. The entire 30-trial BO run costs
0.1 s of objective evaluation. Every statistical weakness above is
cheap to fix.

### Defect 7 — "Grid Search" is not a grid search

[`src/baselines.py:37`](../../src/baselines.py) `_make_grid()` builds the
3×3×4×3×3×3 = 972-point lattice and then takes a **seeded random sample of 25
points** from it. This is randomized lattice sampling, not systematic grid
search, and the docstring's "Grid: 3 values per param" invites a reviewer to
assume otherwise.

## Desired End State

A results package where:

1. The Pareto front is the object actually being optimized (qNEHVI over accuracy and energy), not a post-hoc projection of a converged point cloud.
2. The headline comparison metric is **hypervolume**, the standard multi-objective measure, reported against an exhaustively-computed **reference Pareto front** so absolute optimality gap is known — not just relative ranking.
3. The accuracy surrogate is monotone by construction and reports honest generalization error (leave-one-out).
4. Every search dimension provably affects the optimization problem.
5. Statistics have real power: ≥30 seeds, bootstrap confidence intervals, and a p-value that is not floor-limited.
6. Figures show tension and uncertainty rather than overplotted duplicates.

### Key Discoveries

- Ax 1.3.1 + BoTorch 0.18.1 support MOO in this environment. **Smoke-tested successfully**: 14 trials produced a 6-point Pareto front spanning accuracy 0.721→0.878 and energy 0.00327→0.01229 mJ. This already dominates the current 30-trial result in spread and interpretability.
- `ax_client.get_pareto_optimal_parameters()` works; `botorch.utils.multi_objective.hypervolume.Hypervolume` imports cleanly.
- **A monotone log-linear scaling law fits the existing calibration data well and needs no retraining.** Candidate forms fitted to the 6 measured architectures:

| Form | k | R² | RMSE | **LOO RMSE** | LOO max | non-monotone |
|---|---|---|---|---|---|---|
| `a + b·ln(L) + c·ln(w)` | 3 | 0.9858 | 1.37 pp | **2.40 pp** | 3.82 pp | **0** |
| `1 − B·L^−p·w^−q` | 3 | 0.9748 | 1.82 pp | 9.75 pp | 21.39 pp | 0 |
| logistic in ln L, ln w | 4 | 0.9921 | 1.02 pp | 7.56 pp | 16.74 pp | 0 |
| `Amax − B·L^−p·w^−q` | 4 | 0.9748 | 1.82 pp | 15.20 pp | 35.68 pp | 0 |
| *current IDW* | — | (interpolates) | 0 | — | — | **9/45** |

  The **log-linear form wins decisively on leave-one-out** because it is linear
  in its parameters and therefore not ill-conditioned by only 6 architectures.
  The power-law forms overfit badly (LOO max error up to 35 pp).

- **Quantization has no measurable accuracy effect in this data.** Across all 6 architectures, the maximum accuracy spread over `quant_bits ∈ {32,16,8,4}` is **0.30 pp** — smaller than the surrogate's own 2.40 pp LOO error. This must be stated as a finding, not modelled as signal.
- A latency budget of 100 ms leaves **31.0%** of the space jointly feasible (with `accuracy ≥ 0.75`); 50 ms leaves 17.2%; 200 ms leaves 44.2%. Latency percentiles over 400 random configs: p25=23.7 ms, p50=46.1 ms, p75=109.0 ms, p95=475.6 ms.
- `evaluate()`'s 3.8 ms is dominated by rebuilding the torch model on every call. Architecture stats depend only on `(num_layers, width_mult)` and are therefore memoizable, which is what makes exhaustive reference-front enumeration tractable.

## What We're NOT Doing

- **Not retraining any networks.** The scaling law is fitted to the existing 24 calibration measurements. `src/collect_calibration.py` stays available but is not run. (Per the handoff, this machine trains at ~28-33 s/epoch and stalls badly on lid-close.)
- **Not deleting the single-objective constrained experiment.** It stays as a secondary "targeted search" result and as an ablation against the MOO formulation.
- **Not changing the energy model.** The linear `quant_bits/32` scaling was deliberately chosen over quadratic in a prior session for defensibility without a hardware citation. Out of scope.
- **Not changing the trained baseline model, the CIFAR-10 data pipeline, or `src/train.py`.**
- **Not introducing a test framework.** `pytest` is not installed; the repo's convention is `__main__` self-test blocks (see `src/energy_proxy.py`, `src/objective.py`). Follow that convention.
- **Not adding new search dimensions.** The six stay; two of them merely become active.
- **Not restyling figures that are already sound** (`convergence_curve.pdf`, `comparison_bar.pdf`) beyond what the new data requires.

## Implementation Approach

Fix the problem definition before touching any plotting code. The Pareto
figure is bad because the *experiment* is bad; re-plotting the same degenerate
data differently would only disguise it. Order: surrogate → problem
well-posedness → optimizer → statistics → figures → docs. Each phase leaves
the repo runnable.

Because every metric changes in Phases 1–2, do **not** regenerate figures until
Phase 5. Expect all existing `results/*.csv` to be superseded.

---

## Phase 1: Replace the accuracy surrogate with a monotone scaling law

### Overview

Swap IDW for a 3-parameter log-linear scaling law, fitted by least squares to
the existing calibration points, with quantization handled as an explicitly
measured null effect. Add a validation script that reports fit quality and
proves monotonicity.

### Changes Required

#### 1. Accuracy surrogate

**File**: `src/objective.py`
**Changes**: Replace `_normalize()` and `accuracy_proxy()` with a fitted
scaling law. Keep `CALIBRATION_POINTS` unchanged — it remains the single source
of truth. Fit once at import time and cache the coefficients.

```python
# Accuracy scaling law: acc = a + b*ln(num_layers) + c*ln(width_mult)
#
# Fitted by least squares to the 6 measured architectures in
# CALIBRATION_POINTS (averaged over quant_bits — see QUANT_ACCURACY_NOTE).
# Monotone non-decreasing in both depth and width by construction, since
# b, c > 0. Functional form follows the neural scaling-law literature
# (Hestness et al. 2017; Kaplan et al. 2020), specialised to the two
# architecture degrees of freedom in this search space.
#
# Validity is claimed ONLY inside the search box
# num_layers in [2,4], width_mult in [0.25,1.0]; predictions are clipped
# to ACCURACY_CLIP to prevent extrapolation past physical limits.

ACCURACY_CLIP = (0.40, 0.95)

def _fit_scaling_law() -> tuple[float, float, float]:
    """Least-squares fit of (a, b, c). Called once at import."""
    arch = {}
    for nl, wm, _qb, acc in CALIBRATION_POINTS:
        arch.setdefault((nl, wm), []).append(acc)
    X = np.array([[1.0, np.log(nl), np.log(wm)] for (nl, wm) in arch])
    y = np.array([np.mean(v) for v in arch.values()])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return tuple(float(c) for c in coef)

_A, _B, _C = _fit_scaling_law()

def accuracy_proxy(num_layers: int, width_mult: float, quant_bits: int) -> float:
    """
    Estimate accuracy from a monotone scaling law fitted to measured data.

    quant_bits is accepted for interface compatibility but does not affect
    the result: across all 6 measured architectures the accuracy spread over
    quant_bits in {32,16,8,4} is at most 0.30 pp, well below the surrogate's
    own 2.40 pp leave-one-out error. Post-training quantisation to >=4 bits
    is lossless for this model family at measurement resolution.
    """
    acc = _A + _B * np.log(num_layers) + _C * np.log(width_mult)
    return float(np.clip(acc, *ACCURACY_CLIP))
```

Note: `_fit_scaling_law` must use `np.linalg.lstsq`, not `scipy.optimize.curve_fit` —
the model is linear in its parameters, so the closed-form solution is exact,
deterministic, and has no initialization or convergence risk.

#### 2. Surrogate validation script

**File**: `src/validate_surrogate.py` (new)
**Changes**: Standalone script producing the numbers the paper must cite.

Must report:
- Fitted coefficients `a, b, c`.
- R², RMSE, and per-architecture residuals against all 6 measured points.
- **Leave-one-out RMSE and max error** (expected ≈ 2.40 pp / 3.82 pp).
- A dense monotonicity sweep over `num_layers × width_mult` asserting **zero** violations.
- The measured quantization spread (expected ≤ 0.30 pp), justifying the null treatment.
- A side-by-side comparison against the old IDW surrogate's 9/45 violations.

Write results to `results/surrogate_validation.json` for the paper to cite.

#### 3. Update the existing self-test

**File**: `src/objective.py` `__main__` block
**Changes**: The baseline assertion at line ~183 checks the proxy reproduces
`CALIBRATION_POINTS[0][3] = 0.8170` within 1%. **This assertion will now fail** —
the scaling law fits rather than interpolates, so the baseline architecture
`(3, 1.00)` is predicted with a residual (≈1.5 pp for the power law; recompute
for log-linear). Loosen the tolerance to the surrogate's reported RMSE and
state why in the message. Do not delete the check.

### Success Criteria

#### Automated Verification:
- [ ] `.venv/bin/python -m src.validate_surrogate` runs clean and writes `results/surrogate_validation.json`
- [ ] Monotonicity sweep reports **0** violations in both `num_layers` and `width_mult`
- [ ] LOO RMSE ≤ 3.0 pp and LOO max error ≤ 5.0 pp
- [ ] `.venv/bin/python -m src.objective` self-test passes with the loosened, documented tolerance
- [ ] `accuracy_proxy` returns identical values for all four `quant_bits` at fixed architecture

#### Manual Verification:
- [ ] Predicted accuracy at `(2, 1.00)` no longer exceeds the measured value at `(2, 0.75)` — the specific IDW overshoot is gone
- [ ] Fitted `b` and `c` are both positive, and `b > c` (depth dominates width, matching the measurements)

**Implementation Note**: After this phase, pause. Every downstream number
changes here; confirm the fit is acceptable before regenerating anything.

---

## Phase 2: Make the problem well-posed

### Overview

Add a latency constraint so the scheduler variables become load-bearing, and
rename the misdescribed grid baseline.

### Changes Required

#### 1. Latency budget

**File**: `config/config.py`
**Changes**: Add the constraint and the MOO reference point.

```python
# ── Constraints ───────────────────────────────────────────────────────────────
ACCURACY_THRESHOLD = 0.75      # unchanged
LATENCY_BUDGET_MS  = 100.0     # real-time inference deadline (10 FPS).
                               # Leaves 31.0% of the search space jointly
                               # feasible with the accuracy threshold.
LATENCY_SENSITIVITY_MS = [50.0, 100.0, 200.0]   # for the sensitivity figure

# ── Multi-objective BO ────────────────────────────────────────────────────────
# Hypervolume reference point. Must be dominated by every feasible point and
# must stay FIXED across all methods and seeds, or hypervolumes are not
# comparable.
HV_REF_ENERGY_MJ = 0.10        # > worst observed feasible energy
HV_REF_ACCURACY  = 0.70        # < accuracy threshold
MOO_TRIALS = 100
```

The 100 ms budget is defensible as a 10 FPS real-time deadline and is binding
without being crippling. Report the sensitivity sweep so the choice is not
load-bearing on its own.

#### 2. Thread the constraint through feasibility

**Files**: `src/bo_experiment.py`, `src/baselines.py`, `src/robustness_check.py`
**Changes**: `is_feasible` currently means `accuracy >= ACCURACY_THRESHOLD` in
three separate places. Replace all three with a single shared helper so the
definition cannot drift:

```python
# in src/objective.py
def is_feasible(result: dict) -> bool:
    """Joint feasibility: accuracy floor AND latency budget."""
    return (result["accuracy"] >= ACCURACY_THRESHOLD
            and result["latency_ms"] <= LATENCY_BUDGET_MS)
```

Add `latency_ms <= LATENCY_BUDGET_MS` to `OUTCOME_CONSTRAINTS` in
`src/bo_experiment.py:49` so the single-objective run respects it too.

#### 3. Rename the grid baseline honestly

**File**: `src/baselines.py`
**Changes**: `_make_grid()` samples 25 points at random from a 972-point
lattice. Either evaluate the **full** lattice (972 × 3.8 ms ≈ 3.7 s — now
affordable, and makes it a genuine exhaustive grid), or rename the method to
`"LatticeSample"` in the CSV `method` column and docstring. **Prefer the full
lattice** — it is cheap, removes a reviewer objection, and strengthens the
baseline rather than weakening it.

If the full lattice is used, note in the paper that Grid receives 972
evaluations vs BO's 100 — i.e. BO wins despite a ~10× *disadvantage* in budget.
That is a much stronger claim than the current equal-budget framing.

### Success Criteria

#### Automated Verification:
- [ ] `is_feasible()` is defined once and imported in all three call sites — `grep -rn "ACCURACY_THRESHOLD" src/` shows no remaining inline feasibility checks
- [ ] A config with `accuracy=0.90, latency_ms=500` is correctly infeasible
- [ ] `src/baselines.py` full-lattice run completes in < 30 s

#### Manual Verification:
- [ ] Sweeping `time_slice`/`priority` at a high-FLOPs architecture (e.g. `L4 w=0.90`) now flips feasibility — confirming the scheduler variables became active
- [ ] The CSV `method` column no longer says "Grid" for a randomized sample

---

## Phase 3: Multi-objective BO, hypervolume, and a reference Pareto front

### Overview

Add a true MOO experiment alongside the existing single-objective one, and
compute the ground-truth Pareto front by exhaustive enumeration so absolute
optimality gap is reportable.

### Changes Required

#### 1. Memoize architecture stats

**File**: `src/objective.py`
**Changes**: `build_model()` + `get_model_stats()` dominate the 3.8 ms and
depend only on `(num_layers, width_mult)`. Memoize to make exhaustive
enumeration tractable.

```python
from functools import lru_cache

@lru_cache(maxsize=4096)
def _arch_stats(num_layers: int, width_mult_q: float) -> tuple:
    """Cached FLOPs/params. width_mult_q must be pre-quantised by the caller."""
```

Cache correctness is verifiable because `evaluate()` is deterministic
(`torch.manual_seed(RANDOM_SEED)` is set before every `build_model`).
**Verify explicitly**: run 200 random configs with and without the cache and
assert bit-identical outputs. Do not assume.

#### 2. Multi-objective experiment

**File**: `src/moo_experiment.py` (new)
**Changes**: qNEHVI over `(accuracy ↑, energy_mj ↓)` subject to the latency
budget. Structure mirrors `src/bo_experiment.py` so the two are directly
comparable.

```python
OBJECTIVES = {
    "energy_mj": ObjectiveProperties(minimize=True,  threshold=HV_REF_ENERGY_MJ),
    "accuracy":  ObjectiveProperties(minimize=False, threshold=HV_REF_ACCURACY),
}
OUTCOME_CONSTRAINTS = [f"latency_ms <= {LATENCY_BUDGET_MS}"]
```

The `threshold` values are the hypervolume reference point and **must** match
`config.py`. Smoke-tested: this API path works in Ax 1.3.1.

Emit the same CSV schema as `bo_experiment.py` plus a `hypervolume` column
holding the running HV after each trial — that column is what Phase 5's
convergence figure plots.

#### 3. Hypervolume utilities

**File**: `src/pareto_metrics.py` (new)
**Changes**: Shared metric code, so BO / Random / Grid are all scored
identically.

- `dominates(a, b)`, `pareto_front(rows)` — pure-Python, no BoTorch dependency, for use in plotting.
- `hypervolume(rows, ref_point)` — wraps `botorch.utils.multi_objective.hypervolume.Hypervolume`.
- `igd(front, reference_front)` — inverted generational distance to the true front.
- `hv_trace(rows, ref_point)` — running hypervolume per trial index.

**Critical**: the reference point must be identical across every method and
seed. Read it from `config.py`; never derive it from the observed data, or
hypervolumes become incomparable across runs.

#### 4. Reference Pareto front by exhaustive enumeration

**File**: `src/reference_front.py` (new)
**Changes**: With memoization, enumerate the objective-relevant space
exhaustively. Because `accuracy` and `energy_mj` are provably independent of
`time_slice`/`priority` (verified in Defect 4), the `(accuracy, energy)` front
needs only `num_layers × width_mult × quant_bits × dvfs_factor`:

- `num_layers`: 3 values
- `width_mult`: 151 values (0.25→1.00 step 0.005)
- `quant_bits`: 4 values
- `dvfs_factor`: 51 values (0.50→1.00 step 0.01)

= 92,412 evaluations. For latency feasibility, take the **minimum achievable
latency** over the 15 `time_slice × priority` combinations per architecture
(architecture count is only 3 × 151 = 453, so this is 6,795 extra evaluations,
not 1.4 M).

Write `results/reference_pareto_front.csv`. This unlocks the strongest claim in
the paper: *"BO recovers X% of the achievable hypervolume within N
evaluations"* — an absolute statement, not a relative one.

### Success Criteria

#### Automated Verification:
- [ ] Cache correctness: 200 random configs produce bit-identical results with and without `lru_cache`
- [ ] `.venv/bin/python -m src.reference_front` completes in < 10 min and writes a non-empty front
- [ ] `hypervolume()` of the reference front ≥ hypervolume of any method's front (sanity: nothing beats ground truth)
- [ ] `pareto_front()` output contains no dominated pair — assert pairwise
- [ ] `.venv/bin/python -m src.moo_experiment` produces ≥ 15 distinct Pareto points at 100 trials

#### Manual Verification:
- [ ] The MOO front spans a visibly wide accuracy range (smoke test reached 0.721→0.878 in just 14 trials)
- [ ] No configuration is duplicated more than twice in the MOO run — contrast with 7× in the current single-objective run

**Implementation Note**: Pause here. Inspect the reference front before running
the full seed sweep; if it looks wrong, every subsequent number is wrong.

---

## Phase 4: Statistics with real power

### Overview

Raise seed count from 5 to 30, fix the floor-limited significance test, and
report bootstrap confidence intervals.

### Changes Required

#### 1. Seed count

**File**: `config/config.py`
**Changes**:

```python
ROBUSTNESS_SEEDS = list(range(1, 31))   # 30 seeds.
# Rationale: one-sided Wilcoxon with n=5 has a minimum achievable p-value of
# exactly 1/32 = 0.03125 — the previously reported 0.0312 was the floor, not a
# measurement. n=30 permits p < 1e-6. Cost: ~100 trials x 30 seeds x 3.8 ms
# ~= 11 s of objective evaluation per method.
```

#### 2. Rewrite the robustness analysis

**File**: `src/robustness_check.py`
**Changes**: Currently reports only best feasible energy. Extend to:

- **Primary metric**: final hypervolume per method per seed (this is the multi-objective headline).
- **Secondary**: best feasible energy (preserves continuity with the existing result).
- **Bootstrap 95% CI** on the difference in means (10,000 resamples) — more informative than std alone, and robust to the non-normality of these distributions.
- **Wilcoxon** retained, now with n=30 so it is not floor-limited. **Report the floor explicitly** (`min_achievable_p`) in the JSON so no future reader mistakes a floor for a measurement.
- **Effect size**: Cliff's delta or rank-biserial correlation. A p-value alone says nothing about magnitude, and reviewers ask.
- **IGD to the reference front** per method per seed.

Guard the `scipy` import as the current code does; `scipy` is present in
`.venv` (used above for `curve_fit`/`wilcoxon`), so the fallback path is
unlikely but should stay.

#### 3. Budget-matched comparison

**File**: `src/robustness_check.py`
**Changes**: Report hypervolume at matched evaluation counts (10/25/50/100
trials) so the comparison is not a single endpoint. This is what makes the
convergence figure quantitative rather than decorative.

### Success Criteria

#### Automated Verification:
- [ ] `results/robustness_summary.json` contains 30 seeds per method
- [ ] Reported `p_value` is strictly greater than the reported `min_achievable_p` — proving the test is not saturated
- [ ] `bo_hv_std > 0` — with a real multi-objective metric, seed variance should no longer be identically zero
- [ ] Bootstrap CI on the HV difference excludes 0
- [ ] Full sweep completes in < 10 minutes

#### Manual Verification:
- [ ] The effect size is reported alongside the p-value
- [ ] BO's advantage holds at *every* matched budget, not just the final one — if it only wins at the end, say so honestly

---

## Phase 5: Rebuild the figure set

### Overview

Replace the degenerate Pareto figure and add the figures the new experiment
makes possible. `src/plot_results.py` is 980 lines with a large embedded
change-log comment block at lines 295–312 — clean that up while editing.

### Changes Required

#### 1. Pareto frontier figure — rebuild

**File**: `src/plot_results.py` `figure3_pareto_frontier()`
**Changes**:

- **Axes**: accuracy (x) vs energy (y, **log scale** — energy spans 0.00076 → 0.094 mJ, nearly 3 orders of magnitude; the current linear axis crushes the interesting low-energy region into the baseline).
- Plot the **reference Pareto front** as a filled boundary — this is the figure's most valuable addition, since it shows how close each method got to ground truth.
- Draw per-method fronts as staircases over the *union across seeds*, with the single-seed front as a lighter overlay.
- Mark the latency-infeasible region with hatching.
- **Fix overplotting**: the current figure silently stacks 7 identical BO points. Either jitter, or scale marker area by multiplicity and note it in the caption. Do not hide duplication — it is a real property of the single-objective run and worth showing in the SOO-vs-MOO ablation.
- Delete the stale instruction comment block at lines 295–312.

#### 2. Hypervolume convergence — new

**File**: `src/plot_results.py` `figure_hypervolume_convergence()` (new)
**Changes**: HV vs evaluation count, mean across 30 seeds, with bootstrap 95%
CI bands. Horizontal dashed line at the reference front's HV. This becomes the
paper's primary results figure — it shows sample efficiency, uncertainty, and
absolute optimality gap in one panel.

#### 3. SOO vs MOO ablation — new

**File**: `src/plot_results.py` `figure_soo_vs_moo()` (new)
**Changes**: Directly contrast the two formulations — the single-objective run
converging to one point vs the multi-objective run covering the front. This
turns Defect 1 from an embarrassment into a deliberate methodological finding.

#### 4. Latency sensitivity — new

**File**: `src/plot_results.py` `figure_latency_sensitivity()` (new)
**Changes**: Final HV vs `LATENCY_BUDGET_MS ∈ {50, 100, 200}`, demonstrating
that conclusions do not hinge on the chosen budget.

#### 5. Parallel coordinates of the Pareto set — new

**File**: `src/plot_results.py` `figure_pareto_parallel_coords()` (new)
**Changes**: All 6 parameters on parallel axes, one line per Pareto-optimal
config, coloured by energy. Answers the question a reviewer actually cares
about — *what do the good configurations look like?* — and visually confirms
that the scheduler variables now vary along the front.

#### 6. Retire the misleading figure

**File**: `src/plot_results.py`
**Changes**: The memory-vs-energy view (ρ=0.881) should not be a headline
figure. Keep it only as a supplementary panel with the correlation stated in
the caption, or drop it.

### Success Criteria

#### Automated Verification:
- [ ] `.venv/bin/python -m src.plot_results` regenerates all figures without error
- [ ] Every figure listed in `__main__` exists in `figures/` with a nonzero size
- [ ] No figure function silently emits an empty plot — each logs its point count
- [ ] The embedded change-log comment block (lines 295–312) is gone

#### Manual Verification:
- [ ] Open every PDF. The Pareto figure shows visible separation between methods and the reference front.
- [ ] Log-scale energy axis makes the low-energy region legible.
- [ ] CI bands are visible and do not fully overlap between BO and Random.
- [ ] Parallel-coordinates plot shows `time_slice`/`priority` varying — visual proof they are no longer inert.
- [ ] Colours remain distinguishable in greyscale.

---

## Phase 6: Update the written record

### Overview

Every claim in the docs is now stale. Update them together, once, at the end.

### Changes Required

#### 1. LaTeX results table

**File**: `src/plot_results.py` `generate_latex_table()`
**Changes**: Add hypervolume (mean ± 95% CI), IGD to the reference front, and
final Pareto-set size per method. Keep best-feasible-energy for continuity.
Report n=30 seeds.

#### 2. Methodology

**File**: `docs/methodology.txt`
**Changes**: Document the scaling-law surrogate (form, coefficients, R², LOO
error, monotonicity proof, and citations to Hestness 2017 / Kaplan 2020), the
latency constraint and its justification, the MOO formulation, the fixed
hypervolume reference point, and the exhaustive reference front.

State explicitly, as findings rather than omissions:
- Quantization to ≥4 bits is accuracy-neutral for this model family (≤0.30 pp, below surrogate error).
- Scheduling parameters affect latency only above a FLOPs threshold (0% swing at L2 w=0.40; 57% at L4 w=0.90).
- The surrogate is claimed valid **only inside the search box**.

#### 3. README

**File**: `README.md`
**Changes**: Update the results table and add `moo_experiment.py`,
`reference_front.py`, `validate_surrogate.py`, and `pareto_metrics.py` to the
pipeline description and run instructions.

#### 4. Stale docstrings

**Files**: `src/objective.py:8`, `src/bo_experiment.py`
**Changes**: `src/objective.py` claims "Performance target: <5 seconds per
call" — the measured value is 3.8 ms. Correct it; the discrepancy is what
justified the entire budget increase, so it deserves a sentence rather than a
silent edit.

### Success Criteria

#### Automated Verification:
- [ ] `figures/table_main_results.tex` compiles standalone
- [ ] No file references `memory_vs_energy.pdf` or other deleted artifacts — `grep -rn "memory_vs_energy" .` returns nothing outside git history

#### Manual Verification:
- [ ] Every number in `README.md` and `docs/methodology.txt` traces to a regenerated file in `results/`
- [ ] No claim about accuracy/quantization tradeoffs survives — the data does not support one
- [ ] The methodology states the surrogate's LOO error wherever accuracy figures are quoted

---

## Testing Strategy

The repo convention is `__main__` self-test blocks, not a test framework
(`pytest` is not installed). Follow it.

### Self-tests to add or extend:

- `src/objective.py` — monotonicity assertion over a dense grid; `quant_bits` invariance; loosened baseline-residual check with a documented tolerance.
- `src/pareto_metrics.py` — `dominates()` on hand-worked cases; `pareto_front()` on a known 5-point set; hypervolume of a unit square against a known reference point.
- `src/reference_front.py` — assert no point on the emitted front dominates another; assert every front point is feasible.
- `src/moo_experiment.py` — assert the HV trace is monotone non-decreasing (hypervolume can never decrease as points are added; if it does, the reference point is being derived from data instead of fixed).

### Manual testing steps:

1. Run `src/validate_surrogate.py`; confirm 0 monotonicity violations and LOO RMSE ≈ 2.4 pp.
2. Run one 100-trial MOO seed; confirm ≥15 distinct Pareto points and no config repeated >2×.
3. Run `src/reference_front.py`; spot-check that its extreme points match the known measured corners — `(4, 1.00)` should be the max-accuracy end.
4. Run the full 30-seed sweep; confirm `bo_hv_std > 0`.
5. Regenerate all figures and open each PDF.

## Risks

- **The scaling law fits 6 architectures with 3 parameters.** LOO error (2.40 pp) is honest but is estimated from only 6 points. If a reviewer presses, the answer is `src/collect_calibration.py` — deliberately deferred here to avoid retraining cost. State the limitation in the methodology rather than hiding it.
- **The 100 ms latency budget is a modelling choice.** Mitigated by the sensitivity sweep (Phase 5.4); if conclusions flip across {50, 100, 200} ms, that is itself a finding and must be reported.
- **Hypervolume is reference-point dependent.** Mitigated by fixing it in `config.py` and asserting the HV trace is monotone. This is the single most common way MOO comparisons are silently invalidated.
- **Ax may still propose near-duplicate points.** MOO reduces this but does not eliminate it. The success criterion (no config repeated >2×) will catch it if qNEHVI is misconfigured.

## References

- Prior session handoff: `~/.claude/handoffs/bayesian-optimization-microprocessor/general/2026-08-26_13-15-07_general_energy-model-accuracy-proxy-fix.md`
- Current surrogate: [`src/objective.py:65`](../../src/objective.py) `accuracy_proxy()`
- Current SOO formulation: [`src/bo_experiment.py:48`](../../src/bo_experiment.py)
- Misdescribed grid baseline: [`src/baselines.py:37`](../../src/baselines.py) `_make_grid()`
- Pareto figure to rebuild: [`src/plot_results.py:387`](../../src/plot_results.py) `figure3_pareto_frontier()`
- Scaling-law form: Hestness et al. (2017), "Deep Learning Scaling is Predictable, Empirically"; Kaplan et al. (2020), "Scaling Laws for Neural Language Models"
- qNEHVI: Daulton et al. (2021), "Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement"
- Random search baseline: Bergstra & Bengio (2012)
