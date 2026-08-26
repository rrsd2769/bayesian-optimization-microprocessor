# Bayesian Optimization for Energy-Efficient Microprocessor Scheduling

Joint optimization of a CNN's architecture and an embedded scheduler's
runtime parameters to minimize inference energy on a simulated ARM
Cortex-M4, subject to a hard accuracy constraint. Bayesian Optimization
(via [Ax](https://ax.dev)) is compared against Grid Search and Random
Search under identical search spaces and evaluation budgets.

See [docs/methodology.txt](docs/methodology.txt) for the full write-up
of the evaluation pipeline, search space, and experimental design.

## Problem

```
x = (num_layers, width_mult, quant_bits, time_slice, priority, dvfs_factor)

minimize   energy_mj(x)
subject to accuracy(x) >= ACCURACY_THRESHOLD
```

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

1. **[`src/model.py`](src/model.py)** — builds a `ConfigurableCNN` (2-4 conv blocks) for CIFAR-10 from `num_layers`/`width_mult`
2. **[`src/quantization.py`](src/quantization.py)** — applies fp32/fp16/int8/simulated-4-bit quantization, reports memory footprint
3. **Accuracy proxy** — inverse-distance-weighted interpolation over real measured `(num_layers, width_mult, quant_bits) → accuracy` calibration points (see `CALIBRATION_POINTS` in `objective.py`, collected via [`src/collect_calibration.py`](src/collect_calibration.py))
4. **[`src/energy_proxy.py`](src/energy_proxy.py)** — analytical CMOS energy model: `E = FLOPs × V² × C × dvfs_factor × (quant_bits/32) × 1000`
5. **[`src/scheduler_sim.py`](src/scheduler_sim.py)** — SimPy discrete-event simulation of Round-Robin/Priority scheduling with Poisson task arrivals, reports latency

`evaluate(config) -> {accuracy, memory_kb, energy_mj, latency_ms}` is the
single entry point all three search methods call.

## Experiments

| Script | Method | Budget |
|---|---|---|
| [`src/bo_experiment.py`](src/bo_experiment.py) | Bayesian Optimization (Ax/BoTorch) | 30 trials |
| [`src/baselines.py`](src/baselines.py) | Grid Search | 25 sampled configs |
| [`src/baselines.py`](src/baselines.py) | Random Search | 30 trials |
| [`src/robustness_check.py`](src/robustness_check.py) | BO vs. Random across 5 seeds | 30 trials × 5 seeds |
| [`src/plot_results.py`](src/plot_results.py) | Figures + LaTeX table for the paper | — |

All methods share the same 6-D search space and the same `evaluate()`
function, so results are directly comparable.

## Running it

```bash
# One-time: train the baseline model + calibration architectures
python -m src.train
python -m src.collect_calibration

# Run the search methods
python -m src.bo_experiment
python -m src.baselines
python -m src.robustness_check

# Generate figures + LaTeX table
python -m src.plot_results
```

Outputs land in `results/` (CSVs + JSON), `figures/` (PDFs + LaTeX
table), and `logs/`.

## Results

BO's best feasible configuration (30 evaluations, accuracy ≥ 0.75):
`num_layers=4, width_mult=0.25, quant_bits=4, dvfs_factor=0.5` —
the deepest-but-narrowest architecture pushed to minimum precision and
clock scaling. Depth drives accuracy far more than width for this CNN
family, so BO learns to go deep-and-narrow rather than shallow-and-wide
to clear the accuracy constraint at minimum energy.

| Method | Best energy (mJ) | Feasible found (of budget) |
|---|---|---|
| **Bayesian Opt.** | **0.000759** | 22/30 |
| Random Search | 0.003801 | 18/30 |
| Grid Search | 0.001519 | 16/25 |

**5-seed robustness** (BO vs. Random, identical seeds/budget):
BO converges to the exact same optimum on every seed
(0.000759 ± 0.000000 mJ), a **82.5% energy reduction** over Random
Search's mean (0.004329 ± 0.001427 mJ), Wilcoxon p = 0.0312.

See [`figures/table_main_results.tex`](figures/table_main_results.tex)
and [`results/robustness_summary.json`](results/robustness_summary.json)
for the full numbers, and `figures/*.pdf` for the convergence,
comparison, Pareto frontier, and feasibility-discovery plots.

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
