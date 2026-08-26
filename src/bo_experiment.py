"""
Bayesian Optimization experiment using Ax Platform.
Objective: minimize energy_mj
Constraint: accuracy >= ACCURACY_THRESHOLD (0.75)
Space: 6 parameters (3 architecture + 3 scheduling)
Trials: 30 (BO_TRIALS from config)
Outputs:
 results/bo_results.csv — trial-by-trial results
 results/bo_best_config.json — best feasible configuration
 logs/bo_experiment.log — full run log
"""
import os, sys, csv, json, logging, time
from pathlib import Path
from typing import List, Dict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.config import (
 RANDOM_SEED, RESULTS_DIR, BO_TRIALS, ACCURACY_THRESHOLD, LATENCY_BUDGET_MS,
 OBJECTIVE_METRIC, LOG_FORMAT, LOGS_DIR
)
from src.objective import evaluate, is_feasible
from ax.service.ax_client import AxClient, ObjectiveProperties
logging.basicConfig(
 level=logging.INFO, format=LOG_FORMAT,
 handlers=[
  logging.FileHandler(LOGS_DIR / "bo_experiment.log"),
  logging.StreamHandler(sys.stdout),
 ]
)
logger = logging.getLogger(__name__)

# CRITICAL: These bounds must be identical in bo_experiment.py and baselines.py.
# Changing these after experiments are run invalidates the comparison.
PARAMETERS = [
 {"name": "num_layers", "type": "choice", "values": [2, 3, 4],
 "value_type": "int", "is_ordered": True},
 {"name": "width_mult", "type": "range", "bounds": [0.25, 1.0],
 "value_type": "float"},
 {"name": "quant_bits", "type": "choice", "values": [4, 8, 16, 32],
 "value_type": "int", "is_ordered": True},
 {"name": "time_slice", "type": "range", "bounds": [5.0, 50.0],
 "value_type": "float"},
 {"name": "priority", "type": "choice", "values": [0, 1, 2],
 "value_type": "int", "is_ordered": True},
 {"name": "dvfs_factor", "type": "range", "bounds": [0.5, 1.0],
 "value_type": "float"},
]

OBJECTIVES = {"energy_mj": ObjectiveProperties(minimize=True)}
OUTCOME_CONSTRAINTS = [
 f"accuracy >= {ACCURACY_THRESHOLD}",
 f"latency_ms <= {LATENCY_BUDGET_MS}",
]
# ── CSV columns — shared with baselines.py 

FIELDNAMES = [
 "trial_id", "method", "seed",
 "num_layers", "width_mult", "quant_bits",
 "time_slice", "priority", "dvfs_factor",
 "accuracy", "memory_kb", "energy_mj", "latency_ms",
 "is_feasible", "cumulative_best_energy",
]

def run_bo_experiment(
 n_trials: int = BO_TRIALS,
 seed: int = RANDOM_SEED,
 output_csv: Path = None,
 verbose: bool = True,
) -> List[Dict]:
 """
 Run full Bayesian Optimization experiment.
 Args:
 n_trials: Number of BO trials.
 seed: Random seed (used for both Ax and reproducibility).
 output_csv: Path for results CSV.
 verbose: Log each trial to stdout.
 Returns:
 List of trial result dicts (same format as CSV rows).
 """
 if output_csv is None:
  output_csv = RESULTS_DIR / "bo_results.csv"

 logger.info("=" * 60)
 logger.info("BAYESIAN OPTIMIZATION EXPERIMENT")
 logger.info(f"Trials={n_trials} | Seed={seed} | Objective=minimize {OBJECTIVE_METRIC}")
 logger.info(f"Constraint: accuracy >= {ACCURACY_THRESHOLD}")
 logger.info(f"Output: {output_csv}")
 logger.info("=" * 60)
 ax_client = AxClient(random_seed=seed, verbose_logging=False)
 ax_client.create_experiment(
  name = "bo_energy_scheduling",
  parameters = PARAMETERS,
  objectives = OBJECTIVES,
  outcome_constraints = OUTCOME_CONSTRAINTS,
 )
 results = []
 best_energy = float("inf")
 best_config = None

 with open(output_csv, "w", newline="") as f:
  writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
  writer.writeheader()
  for trial_idx in range(n_trials):
   t0 = time.perf_counter()
   # Ax proposes next config
   parameters, trial_id = ax_client.get_next_trial()
   # Evaluate (this calls model→quantize→energy→scheduler)
   raw = evaluate(parameters)
   elapsed = time.perf_counter() - t0
   # Report back to Ax (updates GP surrogate)
   ax_client.complete_trial(
    trial_index=trial_id,
    raw_data={
        "energy_mj": (raw["energy_mj"], 0.0),
        "accuracy": (raw["accuracy"], 0.0),
        "latency_ms": (raw["latency_ms"], 0.0),
    },
)
   feasible = is_feasible(raw)
   if feasible and raw["energy_mj"] < best_energy:
    best_energy = raw["energy_mj"]
    best_config = parameters.copy()
   row = {
 "trial_id": trial_idx + 1,
 "method": "BO",
 "seed": seed,
 "num_layers": parameters["num_layers"],
 "width_mult": round(float(parameters["width_mult"]),
4),
 "quant_bits": parameters["quant_bits"],
 "time_slice": round(float(parameters["time_slice"]),
2),
 "priority": parameters["priority"],
 "dvfs_factor":
round(float(parameters["dvfs_factor"]), 4),
 "accuracy": raw["accuracy"],
 "memory_kb": raw["memory_kb"],
 "energy_mj": raw["energy_mj"],
 "latency_ms": raw["latency_ms"],
 "is_feasible": feasible,
 "cumulative_best_energy": round(best_energy, 6) if best_energy <
float("inf") else "",
 }
   results.append(row)
   writer.writerow(row)
   f.flush() # Flush after every row — protect against crashes
   if verbose:
    logger.info(
 f"[BO {trial_idx+1:02d}/{n_trials}] "
 f"E={raw['energy_mj']:.6f}mJ | "
 f"acc={raw['accuracy']:.4f} | "
 f"mem={raw['memory_kb']:.1f}KB | "
 f"lat={raw['latency_ms']:.2f}ms | "
 f"feasible={feasible} | "
 f"best={best_energy:.6f}mJ | "
 f"{elapsed:.1f}s"
 )
# Save best config
 best_path = RESULTS_DIR / "bo_best_config.json"
 with open(best_path, "w") as f:
  json.dump({"best_energy_mj": best_energy, "config": best_config,
 "seed": seed, "n_trials": n_trials}, f, indent=2)
 logger.info("\n" + "=" * 60)
 logger.info("BO EXPERIMENT COMPLETE")
 logger.info(f"Best feasible energy: {best_energy:.6f} mJ")
 logger.info(f"Best config: {json.dumps(best_config)}")
 logger.info(f"Results CSV: {output_csv}")
 logger.info(f"Best config JSON: {best_path}")
 logger.info("=" * 60)
 return results
if __name__ == "__main__":
 results = run_bo_experiment()
 feasible = [r for r in results if r["is_feasible"]]
 logger.info(f"Summary: {len(results)} trials, {len(feasible)} feasible")

