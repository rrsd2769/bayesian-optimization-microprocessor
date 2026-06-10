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

# ── Bayesian Optimization ─────────────────────────────────────────────────────
BO_TRIALS = 30
RANDOM_TRIALS = 30 # Must equal BO_TRIALS for fair comparison
GRID_CONFIGS = 25
ROBUSTNESS_SEEDS = [42, 7, 13, 99, 2024]
ACCURACY_THRESHOLD = 0.75 # Hard constraint: BO minimizes energy s.t. acc>= this
OBJECTIVE_METRIC = "energy_mj"
CONSTRAINT_METRIC = "accuracy"
# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"