"""
Global configuration: paths and compute settings only.
Science constants (device, workload, limits, search space, calibration) live in the Scenario and in
src/search/space.py, not here.
"""
from pathlib import Path
# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "v2"
RESULTS_DIR = PROJECT_ROOT / "results" / "v2"   # results/v1 is the read-only v1 archive
FIGURES_DIR = PROJECT_ROOT / "figures" / "v2"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
 d.mkdir(parents=True, exist_ok=True)
# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
DEVICE = "mps" # change to "cuda" if GPU available
# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
