"""
Global configuration.
Single source of truth for all hyperparameters, paths, and seeds.
"""
from pathlib import Path
# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
 d.mkdir(parents=True, exist_ok=True)
# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET = "CIFAR10"
NUM_CLASSES = 10
IMAGE_SIZE = 32
NUM_WORKERS = 2
PIN_MEMORY = True
# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
DEVICE = "mps" # change to "cuda" if GPU available
# ── Search Space (Ax will search over these) ──────────────────────────────────
SEARCH_SPACE = {
 "num_layers": {"type": "choice", "values": [2, 3, 4]},
 "width_mult": {"type": "range", "bounds": [0.25, 1.0]},
 "quant_bits": {"type": "choice", "values": [4, 8, 16, 32]},
 "time_slice": {"type": "range", "bounds": [5.0, 50.0]},
 "priority": {"type": "choice", "values": [0, 1, 2]},
 "dvfs_factor": {"type": "range", "bounds": [0.5, 1.0]},
}
# ── Scheduler Simulation ──────────────────────────────────────────────────────
SIM_RAM_KB = 512
SIM_CLOCK_HZ = 120_000_000 # ARM Cortex-M4: 120 MHz
SIM_VOLTAGE = 3.3 # Volts
SIM_CAPACITANCE = 1e-12 # Farads
SIM_NUM_TASKS = 50
SIM_ARRIVAL_RATE = 0.2 # Mean inter-arrival (seconds)

# ── Constraints ────────────────────────────────────────────────────────────────
ACCURACY_THRESHOLD = 0.75 # Hard constraint: acc >= this
# Real-time inference deadline (10 FPS). Leaves ~31% of the search space
# jointly feasible with ACCURACY_THRESHOLD (measured via random sampling).
# See LATENCY_SENSITIVITY_MS for the sensitivity sweep — conclusions must
# not hinge on this single value.
LATENCY_BUDGET_MS = 100.0
LATENCY_SENSITIVITY_MS = [50.0, 100.0, 200.0]

# ── Bayesian Optimization (single-objective, constrained) ───────────────────────
BO_TRIALS = 30
RANDOM_TRIALS = 30 # Must equal BO_TRIALS for fair comparison
OBJECTIVE_METRIC = "energy_mj"
CONSTRAINT_METRIC = "accuracy"

# ── Grid baseline ────────────────────────────────────────────────────────────
# Full lattice: 3 (num_layers) x 3 (width_mult) x 4 (quant_bits) x
# 3 (time_slice) x 3 (priority) x 3 (dvfs_factor) = 972 configs.
# Exhaustive, not a random sample — evaluate() is ~3.8ms/call so the full
# lattice costs seconds, not hours.
GRID_LATTICE = {
 "num_layers": [2, 3, 4],
 "width_mult": [0.25, 0.625, 1.0],
 "quant_bits": [4, 8, 16, 32],
 "time_slice": [5.0, 27.5, 50.0],
 "priority": [0, 1, 2],
 "dvfs_factor": [0.5, 0.75, 1.0],
}

# ── Multi-objective Bayesian Optimization ────────────────────────────────────
# Hypervolume reference point: must be dominated by every feasible point and
# MUST stay fixed across all methods/seeds, or hypervolumes are not
# comparable. Never derive this from observed data.
HV_REF_ACCURACY = 0.70 # < ACCURACY_THRESHOLD
HV_REF_ENERGY_MJ = 0.10 # > worst observed feasible energy
# Headline single-seed run (feeds the main Pareto/hypervolume-convergence
# figures). Measured cost on this machine: qNEHVI's per-trial cost is NOT
# flat — it grows with the size of the discovered Pareto front (box
# decomposition cost), observed at trial 30 -> 129s cumulative, trial 50
# -> 225s cumulative wall time for ONE seed. 100 trials is a one-time
# ~10-15 minute cost, affordable once.
MOO_TRIALS = 100

# ── Robustness / statistics ──────────────────────────────────────────────────
# n=5 gives a one-sided Wilcoxon a minimum achievable p-value of exactly
# 1/32 = 0.03125 (a floor, not a measurement). n=30 permits p < 1e-6 and
# supports a bootstrap CI. Used for the SECONDARY (single-objective,
# best-feasible-energy) statistic, which is cheap: plain BO (qLogEI) has
# no box-decomposition cost and runs 30 trials in well under a second.
ROBUSTNESS_SEEDS = list(range(1, 31))
# The PRIMARY (hypervolume) statistic uses qNEHVI, which is far more
# expensive per trial (see MOO_TRIALS comment above) and does not scale to
# 30 seeds x 100 trials (~5+ hours, measured). MOO_ROBUSTNESS_SEEDS reuses
# the first 15 of ROBUSTNESS_SEEDS (for comparability with the secondary
# statistic) at a reduced per-seed budget, keeping the full sweep to
# roughly 15-20 minutes.
MOO_ROBUSTNESS_SEEDS = ROBUSTNESS_SEEDS[:15]
MOO_ROBUSTNESS_TRIALS = 20
# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"