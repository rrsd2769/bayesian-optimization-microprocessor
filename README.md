# Bayesian Optimization for Energy-Efficient Microprocessor Scheduling

Joint optimization of a CNN's architecture and an embedded scheduler's
runtime parameters to minimize inference energy on a simulated ARM
Cortex-M4, subject to accuracy and latency constraints. Both a
single-objective (energy, accuracy-constrained) and a **multi-objective**
(accuracy vs. energy) Bayesian Optimization formulation are implemented
via [Ax](https://ax.dev)/[BoTorch](https://botorch.org), and compared
against Grid Search, Random Search, and an exhaustively-enumerated
reference Pareto front.

See [docs/methodology.txt](docs/methodology.txt) for the full write-up
of the evaluation pipeline, search space, surrogate validation, and
experimental design.

## Problem

Two formulations share the same search space and `evaluate()` function:

```
x = (num_layers, width_mult, quant_bits, time_slice, priority, dvfs_factor)

# Single-objective (SOO), src/bo_experiment.py:
minimize   energy_mj(x)
subject to accuracy(x) >= ACCURACY_THRESHOLD, latency_ms(x) <= LATENCY_BUDGET_MS

# Multi-objective (MOO, headline), src/moo_experiment.py:
maximize   accuracy(x)
minimize   energy_mj(x)
subject to latency_ms(x) <= LATENCY_BUDGET_MS
```

A single-objective optimizer necessarily converges to one point once it
has explored enough of the space — that's correct behavior, not a Pareto
front. The MOO formulation is what actually produces a Pareto front, and
is the headline result (see `figures/pareto_frontier.pdf` and
`figures/soo_vs_moo.pdf` for the direct contrast).

Six decision variables span both neural network architecture and
scheduling policy:

| Variable      | Domain                | Meaning                               |
|---------------|------------------------|----------------------------------------|
| `num_layers`  | {2, 3, 4}              | Conv block depth                       |
| `width_mult`  | [0.25, 1.0]            | Channel width multiplier               |
| `quant_bits`  | {4, 8, 16, 32}         | Weight/activation precision            |
| `time_slice`  | [5.0, 50.0] ms         | Round-robin scheduler time slice       |
| `priority`    | {0, 1, 2}              | Task priority (0 selects priority scheduling) |
| `dvfs_factor` | [0.5, 1.0]             | CPU clock scaling (DVFS)               |

## Pipeline

Every candidate configuration is evaluated by [`src/objective.py`](src/objective.py),
which chains together:

1. **[`src/model.py`](src/model.py)** — builds a `ConfigurableCNN` (2-4 conv blocks) for CIFAR-10 from `num_layers`/`width_mult` (cached per architecture — see `_arch_model_and_stats`)
2. **[`src/quantization.py`](src/quantization.py)** — applies fp32/fp16/int8/simulated-4-bit quantization, reports memory footprint
3. **Accuracy surrogate** — a monotone scaling law, `accuracy = a + b·ln(num_layers) + c·ln(width_mult)`, fitted by least squares to 6 measured architectures (`CALIBRATION_POINTS` in `objective.py`, collected via [`src/collect_calibration.py`](src/collect_calibration.py)); validated in [`src/validate_surrogate.py`](src/validate_surrogate.py) (R²=0.986, leave-one-out RMSE=2.40pp, 0 monotonicity violations vs. 34 for the interpolant it replaced)
4. **[`src/energy_proxy.py`](src/energy_proxy.py)** — analytical CMOS energy model: `E = FLOPs × V² × C × dvfs_factor × (quant_bits/32) × 1000`
5. **[`src/scheduler_sim.py`](src/scheduler_sim.py)** — SimPy discrete-event simulation of Round-Robin/Priority scheduling with Poisson task arrivals, reports latency

`evaluate(config) -> {accuracy, memory_kb, energy_mj, latency_ms}` is the
single entry point every search method calls. `is_feasible(result)` (also
in `objective.py`) is the single shared definition of feasibility:
`accuracy >= ACCURACY_THRESHOLD and latency_ms <= LATENCY_BUDGET_MS`.

## Experiments

| Script | Method | Budget |
|---|---|---|
| [`src/moo_experiment.py`](src/moo_experiment.py) | **Multi-objective BO (qNEHVI) — headline** | 100 trials |
| [`src/bo_experiment.py`](src/bo_experiment.py) | Single-objective BO, accuracy-constrained | 30 trials |
| [`src/baselines.py`](src/baselines.py) | Grid Search (exhaustive, not sampled) | 972 configs |
| [`src/baselines.py`](src/baselines.py) | Random Search | matched to whichever method it's compared against |
| [`src/reference_front.py`](src/reference_front.py) | Exhaustive reference Pareto front | 92,412 grid points |
| [`src/robustness_check.py`](src/robustness_check.py) | MOO vs. Random (primary, hypervolume) | 20 trials × 15 seeds |
| [`src/robustness_check.py`](src/robustness_check.py) | BO vs. Random (secondary, best energy) | 30 trials × 30 seeds |
| [`src/plot_results.py`](src/plot_results.py) | Figures + LaTeX tables for the paper | — |

All methods share the same 6-D search space and the same `evaluate()`
function, so results are directly comparable.

## Running it

```bash
# One-time: train the baseline model + calibration architectures
python -m src.train
python -m src.collect_calibration
python -m src.validate_surrogate      # confirms the accuracy surrogate

# Ground truth: exhaustive reference Pareto front (~5 min)
python -m src.reference_front

# Multi-objective headline run (~10-15 min) + single-objective baselines
python -m src.moo_experiment
python -m src.bo_experiment
python -m src.baselines

# Multi-seed statistics (primary MOO sweep ~15-20 min, secondary SOO sweep fast)
python -m src.robustness_check

# Generate figures + LaTeX tables
python -m src.plot_results
```

`robustness_check.py` is resumable: each seed's raw CSV is written as soon
as that seed completes, and a re-run skips any seed whose file already has
the expected row count. Outputs land in `results/` (CSVs + JSON),
`figures/` (PDFs + LaTeX tables), and `logs/`.

## Results

### Multi-objective (headline)

A single 100-trial MOO run reaches **hypervolume 0.016553 — 98.4% of the
exhaustive reference ceiling** (0.016821, from enumerating all 92,412
grid points), with a 10-point Pareto set spanning accuracy 0.72–0.87 and
energy 0.0011–0.0123 mJ. Unlike the single-objective corner solution
below, this front shows `time_slice`, `priority`, and `dvfs_factor`
genuinely varying across configurations.

**Robustness (15 seeds × 20 trials, MOO vs. Random at matched budget):**

| Method | Hypervolume | IGD to reference front |
|---|---|---|
| **Multi-Obj. BO** | **0.015891 ± 0.000348** | **0.185** |
| Random Search | 0.011774 ± 0.002225 | 0.535 |
| Reference (exhaustive) | 0.016821 | 0.000 |

Cliff's delta = **1.0** — MOO's hypervolume exceeded Random's on every one
of the 15 seeds (zero distributional overlap). Bootstrap 95% CI on the
difference: [0.00315, 0.00523] (excludes zero). A budget-matched check
shows the two methods statistically tied during Sobol initialization
(5 trials) and MOO pulling ahead once the GP surrogate takes over.

### Single-objective (secondary)

BO's best feasible configuration (30 evaluations, accuracy ≥ 0.75,
latency ≤ 100ms): `num_layers=4, width_mult=0.25, quant_bits=4,
dvfs_factor=0.5` — the deepest-but-narrowest architecture pushed to
minimum precision and clock scaling. This is expected, correct behavior
for a single-objective optimizer under a hard constraint — see
[`figures/soo_vs_moo.pdf`](figures/soo_vs_moo.pdf) for the direct contrast
with the MOO front above.

| Method | Best energy (mJ) | Feasible found (of budget) |
|---|---|---|
| **Bayesian Opt.** | **0.000759** | 18/30 |
| Random Search | 0.003801 | 6/30 |
| Grid Search (exhaustive, 972 configs) | 0.000759 | 264/972 |

Grid Search's exhaustive 972-config sweep matches BO's result exactly —
BO finds the same optimum at ~3% of Grid's evaluation budget.

**Robustness (30 seeds × 30 trials, BO vs. Random):** BO converges to the
exact same optimum on every seed (0.000759 ± 0.000000 mJ), a **78.3%
energy reduction** over Random Search's mean (0.003496 ± 0.002468 mJ);
bootstrap 95% CI on the difference [0.0019, 0.0037] mJ (excludes zero).
The Wilcoxon test is at its floor (p≈0, n=30) because BO has zero variance
across seeds — a reflection that this landscape is simple enough for BO to
reliably find the true global optimum, not a statistical weakness.

See [`figures/table_main_results.tex`](figures/table_main_results.tex),
[`results/robustness_summary.json`](results/robustness_summary.json), and
[`results/surrogate_validation.json`](results/surrogate_validation.json)
for the full numbers, and `figures/*.pdf` for the Pareto frontier,
hypervolume convergence, SOO-vs-MOO contrast, Pareto-set parallel
coordinates, latency-budget sensitivity, convergence, and
feasibility-discovery plots.

## Repo layout

```
config/      global config: paths, seeds, search space, hyperparameters
src/         pipeline modules + experiment scripts (see Pipeline/Experiments above)
data/        CIFAR-10 (downloaded on first run, gitignored)
checkpoints/ trained model weights
results/     per-trial CSVs, best-config JSON, robustness summary
figures/     IEEE-style PDFs + LaTeX table for the paper
docs/        methodology write-up
logs/        run logs (gitignored)
```
