"""
Multi-objective Bayesian Optimization experiment using Ax Platform (qNEHVI).
Objectives: maximize accuracy, minimize energy_mj
Constraint: latency_ms <= LATENCY_BUDGET_MS
Space: 6 parameters (3 architecture + 3 scheduling) — identical to
bo_experiment.py and baselines.py.
Trials: 100 (MOO_TRIALS from config)
Outputs:
 results/moo_results.csv — trial-by-trial results, with a running
 hypervolume column
 results/moo_pareto_front.json — final Pareto-optimal set (from Ax)
 logs/moo_experiment.log — full run log

This is the primary multi-objective result. Unlike bo_experiment.py (which
optimizes a single objective under an accuracy constraint and therefore
converges to one point), this experiment produces a genuine Pareto front —
see src/plot_results.py figure_soo_vs_moo() for the direct comparison.
"""
import os, sys, csv, json, logging, time
from pathlib import Path
from typing import List, Dict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.config import (
 RANDOM_SEED, RESULTS_DIR, MOO_TRIALS, LATENCY_BUDGET_MS,
 HV_REF_ENERGY_MJ, HV_REF_ACCURACY, LOG_FORMAT, LOGS_DIR
)
from src.objective import evaluate, is_feasible
from src.pareto_metrics import hypervolume, _ref_point_tensor
from ax.service.ax_client import AxClient, ObjectiveProperties
logging.basicConfig(
 level=logging.INFO, format=LOG_FORMAT,
 handlers=[
  logging.FileHandler(LOGS_DIR / "moo_experiment.log"),
  logging.StreamHandler(sys.stdout),
 ]
)
logger = logging.getLogger(__name__)

# CRITICAL: These bounds must be identical across bo_experiment.py,
# moo_experiment.py, and baselines.py. Changing them after experiments are
# run invalidates the comparison.
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

# Thresholds double as the hypervolume reference point — MUST match
# HV_REF_ENERGY_MJ / HV_REF_ACCURACY in config.py, or the "final HV" logged
# here and the HV computed by pareto_metrics.py diverge.
OBJECTIVES = {
 "energy_mj": ObjectiveProperties(minimize=True, threshold=HV_REF_ENERGY_MJ),
 "accuracy": ObjectiveProperties(minimize=False, threshold=HV_REF_ACCURACY),
}
OUTCOME_CONSTRAINTS = [f"latency_ms <= {LATENCY_BUDGET_MS}"]

FIELDNAMES = [
 "trial_id", "method", "seed",
 "num_layers", "width_mult", "quant_bits",
 "time_slice", "priority", "dvfs_factor",
 "accuracy", "memory_kb", "energy_mj", "latency_ms",
 "is_feasible", "hypervolume",
]


def run_moo_experiment(
 n_trials: int = MOO_TRIALS,
 seed: int = RANDOM_SEED,
 output_csv: Path = None,
 verbose: bool = True,
) -> List[Dict]:
 """
 Run full multi-objective Bayesian Optimization experiment (qNEHVI).
 Returns:
 List of trial result dicts (same format as CSV rows), each carrying
 the running hypervolume after that trial.
 """
 if output_csv is None:
  output_csv = RESULTS_DIR / "moo_results.csv"

 logger.info("=" * 60)
 logger.info("MULTI-OBJECTIVE BAYESIAN OPTIMIZATION EXPERIMENT (qNEHVI)")
 logger.info(f"Trials={n_trials} | Seed={seed} | Objectives=maximize accuracy, minimize energy_mj")
 logger.info(f"Constraint: latency_ms <= {LATENCY_BUDGET_MS}")
 logger.info(f"HV reference point: accuracy>={HV_REF_ACCURACY}, energy<={HV_REF_ENERGY_MJ}")
 logger.info(f"Output: {output_csv}")
 logger.info("=" * 60)

 ax_client = AxClient(random_seed=seed, verbose_logging=False)
 ax_client.create_experiment(
  name="moo_energy_accuracy",
  parameters=PARAMETERS,
  objectives=OBJECTIVES,
  outcome_constraints=OUTCOME_CONSTRAINTS,
 )

 results = []
 feasible_so_far = []
 ref_point = _ref_point_tensor()

 with open(output_csv, "w", newline="") as f:
  writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
  writer.writeheader()
  for trial_idx in range(n_trials):
   t0 = time.perf_counter()
   parameters, trial_id = ax_client.get_next_trial()
   raw = evaluate(parameters)
   elapsed = time.perf_counter() - t0
   ax_client.complete_trial(
    trial_index=trial_id,
    raw_data={
     "energy_mj": (raw["energy_mj"], 0.0),
     "accuracy": (raw["accuracy"], 0.0),
     "latency_ms": (raw["latency_ms"], 0.0),
    },
   )
   feasible = is_feasible(raw)
   if feasible:
    feasible_so_far.append(raw)
   hv = hypervolume(feasible_so_far, ref_point)
   row = {
    "trial_id": trial_idx + 1,
    "method": "MOO",
    "seed": seed,
    "num_layers": parameters["num_layers"],
    "width_mult": round(float(parameters["width_mult"]), 4),
    "quant_bits": parameters["quant_bits"],
    "time_slice": round(float(parameters["time_slice"]), 2),
    "priority": parameters["priority"],
    "dvfs_factor": round(float(parameters["dvfs_factor"]), 4),
    "accuracy": raw["accuracy"],
    "memory_kb": raw["memory_kb"],
    "energy_mj": raw["energy_mj"],
    "latency_ms": raw["latency_ms"],
    "is_feasible": feasible,
    "hypervolume": round(hv, 8),
   }
   results.append(row)
   writer.writerow(row)
   f.flush() # Flush after every row — protect against crashes
   if verbose:
    logger.info(
     f"[MOO {trial_idx+1:03d}/{n_trials}] "
     f"E={raw['energy_mj']:.6f}mJ | "
     f"acc={raw['accuracy']:.4f} | "
     f"lat={raw['latency_ms']:.2f}ms | "
     f"feasible={feasible} | "
     f"HV={hv:.6f} | "
     f"{elapsed*1000:.1f}ms"
    )

 pareto = ax_client.get_pareto_optimal_parameters()
 pareto_set = [
  {
   "params": params,
   "energy_mj": means["energy_mj"],
   "accuracy": means["accuracy"],
  }
  for _idx, (params, (means, _covs)) in pareto.items()
 ]
 pareto_path = RESULTS_DIR / "moo_pareto_front.json"
 with open(pareto_path, "w") as f:
  json.dump({"seed": seed, "n_trials": n_trials, "final_hypervolume": results[-1]["hypervolume"],
   "pareto_set": pareto_set}, f, indent=2)

 logger.info("\n" + "=" * 60)
 logger.info("MOO EXPERIMENT COMPLETE")
 logger.info(f"Final hypervolume: {results[-1]['hypervolume']:.6f}")
 logger.info(f"Pareto set size: {len(pareto_set)}")
 logger.info(f"Results CSV: {output_csv}")
 logger.info(f"Pareto front JSON: {pareto_path}")
 logger.info("=" * 60)
 return results


if __name__ == "__main__":
 results = run_moo_experiment()
 feasible = [r for r in results if r["is_feasible"]]
 logger.info(f"Summary: {len(results)} trials, {len(feasible)} feasible, "
 f"final HV={results[-1]['hypervolume']:.6f}")
