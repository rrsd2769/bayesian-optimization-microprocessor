"""
Exhaustive reference Pareto front over (accuracy, energy_mj) — the ground
truth that BO/MOO/Random/Grid are compared against.

accuracy and energy_mj do not depend on time_slice/priority (verified:
holding those fixed at any value leaves accuracy, memory_kb, and energy_mj
bit-identical — see the time_slice/priority invariance check in
src/objective.py's docstring history). So the (accuracy, energy) grid only
needs num_layers x width_mult x quant_bits x dvfs_factor:
 num_layers: 3 values {2,3,4}
 width_mult: 151 values (0.25 -> 1.00, step ~0.005)
 quant_bits: 4 values {4,8,16,32}
 dvfs_factor: 51 values (0.50 -> 1.00, step 0.01)
 = 92,412 grid points.

Feasibility additionally requires latency_ms <= LATENCY_BUDGET_MS for SOME
choice of (time_slice, priority) — these are free scheduling parameters
that trade off nothing against accuracy or energy, so a grid point counts
as latency-feasible if the best available scheduling choice meets the
budget. Latency depends on (num_layers, width_mult, dvfs_factor) but not
quant_bits, so it is computed once per (num_layers, width_mult,
dvfs_factor) triple (23,103 combos) over the 9 (time_slice, priority)
pairs in GRID_LATTICE (207,927 scheduler simulations, ~0.8ms each,
~3 minutes total).

Output:
 results/reference_pareto_front.csv — Pareto-optimal feasible points only
 results/reference_front_summary.json — grid size, feasible count,
 hypervolume of the reference front, wall-clock cost
"""
import os, sys, csv, json, logging, time
from functools import lru_cache
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
 RESULTS_DIR, ACCURACY_THRESHOLD, LATENCY_BUDGET_MS, LATENCY_SENSITIVITY_MS,
 GRID_LATTICE, LOG_FORMAT
)
from src.objective import _arch_model_and_stats, accuracy_proxy
from src.energy_proxy import estimate_energy_mj
from src.quantization import apply_quantization
from src.scheduler_sim import run_scheduler_sim
from src.pareto_metrics import pareto_front, hypervolume, dominates

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

NUM_LAYERS = [2, 3, 4]
WIDTH_GRID = [float(w) for w in np.round(np.linspace(0.25, 1.0, 151), 4)]
QUANT_BITS = [4, 8, 16, 32]
DVFS_GRID = [float(d) for d in np.round(np.linspace(0.5, 1.0, 51), 4)]
TS_PR_COMBOS = [(ts, pr) for ts in GRID_LATTICE["time_slice"] for pr in GRID_LATTICE["priority"]]


@lru_cache(maxsize=None)
def _best_latency_ms(num_layers: int, width_mult: float, dvfs_factor: float) -> float:
 """
 Minimum latency achievable over all (time_slice, priority) scheduling
 choices at this (architecture, dvfs_factor). quant_bits is irrelevant
 to latency (verified in scheduler_sim.py) so a fixed placeholder value
 is passed.
 """
 _, stats = _arch_model_and_stats(num_layers, width_mult)
 flops = stats["flops"]
 best = float("inf")
 for ts, pr in TS_PR_COMBOS:
  cfg = {"time_slice": ts, "priority": pr, "dvfs_factor": dvfs_factor, "quant_bits": 32}
  lat = run_scheduler_sim(cfg, flops)["mean_latency_ms"]
  best = min(best, lat)
 return best


def enumerate_reference_front():
 t0 = time.perf_counter()
 rows = []
 total = len(NUM_LAYERS) * len(WIDTH_GRID) * len(QUANT_BITS) * len(DVFS_GRID)
 logger.info(f"Enumerating {total:,} grid points "
 f"({len(NUM_LAYERS)}x{len(WIDTH_GRID)}x{len(QUANT_BITS)}x{len(DVFS_GRID)})...")
 n_done = 0
 for nl in NUM_LAYERS:
  for w in WIDTH_GRID:
   model, stats = _arch_model_and_stats(nl, w)
   for dv in DVFS_GRID:
    lat = _best_latency_ms(nl, w, dv)
    for qb in QUANT_BITS:
     acc = accuracy_proxy(nl, w, qb)
     energy = estimate_energy_mj(stats["flops"], dv, qb)
     _, mem_kb = apply_quantization(model, qb)
     feasible = acc >= ACCURACY_THRESHOLD and lat <= LATENCY_BUDGET_MS
     rows.append({
      "num_layers": nl, "width_mult": round(w, 4),
      "quant_bits": qb, "dvfs_factor": round(dv, 4),
      "accuracy": round(acc, 4), "memory_kb": mem_kb,
      "energy_mj": round(energy, 6),
      "latency_ms": round(lat, 4),
      "is_feasible": feasible,
     })
     n_done += 1
  logger.info(f"  num_layers={nl} done ({n_done:,}/{total:,})")
 elapsed = time.perf_counter() - t0
 logger.info(f"Grid enumeration complete in {elapsed:.1f}s")
 return rows, elapsed


def compute_latency_sensitivity(rows) -> dict:
 """
 Reference-front hypervolume at each budget in LATENCY_SENSITIVITY_MS,
 reusing the SAME enumerated rows (each already stores raw accuracy and
 latency_ms) — no re-enumeration, no extra scheduler simulations. Shows
 whether the paper's conclusions depend on the specific 100ms choice.
 This is a property of the SEARCH SPACE itself (what's achievable),
 not of any particular optimizer.
 """
 result = {}
 for budget in LATENCY_SENSITIVITY_MS:
  feasible = [r for r in rows
  if r["accuracy"] >= ACCURACY_THRESHOLD and r["latency_ms"] <= budget]
  front = pareto_front(feasible)
  hv = hypervolume(front) if front else 0.0
  result[str(budget)] = {
   "feasible_count": len(feasible),
   "feasible_fraction": round(len(feasible) / len(rows), 4),
   "pareto_front_size": len(front),
   "hypervolume": round(hv, 6),
  }
  logger.info(f"  latency<={budget}ms: feasible={len(feasible):,} "
  f"({100*len(feasible)/len(rows):.1f}%), front={len(front)}, HV={hv:.6f}")
 return result


def main():
 rows, elapsed = enumerate_reference_front()
 feasible_rows = [r for r in rows if r["is_feasible"]]
 logger.info(f"Feasible: {len(feasible_rows):,}/{len(rows):,} "
 f"({100*len(feasible_rows)/len(rows):.1f}%)")

 front = pareto_front(feasible_rows)
 front_sorted = sorted(front, key=lambda r: r["accuracy"])
 logger.info(f"Reference Pareto front size: {len(front_sorted)}")

 assert len(front_sorted) > 0, \
  "Reference front is empty — check ACCURACY_THRESHOLD/LATENCY_BUDGET_MS"
 for i in range(len(front_sorted)):
  for j in range(len(front_sorted)):
   if i != j:
    a = (front_sorted[i]["accuracy"], front_sorted[i]["energy_mj"])
    b = (front_sorted[j]["accuracy"], front_sorted[j]["energy_mj"])
    assert not dominates(a, b), \
     f"Reference front contains a dominated pair: {a} dominates {b}"

 hv = hypervolume(front_sorted)

 out_csv = RESULTS_DIR / "reference_pareto_front.csv"
 fieldnames = ["num_layers", "width_mult", "quant_bits", "dvfs_factor",
 "accuracy", "memory_kb", "energy_mj", "latency_ms", "is_feasible"]
 with open(out_csv, "w", newline="") as f:
  writer = csv.DictWriter(f, fieldnames=fieldnames)
  writer.writeheader()
  for r in front_sorted:
   writer.writerow(r)

 logger.info("Computing latency sensitivity (reusing enumerated rows, no re-simulation)...")
 latency_sensitivity = compute_latency_sensitivity(rows)

 summary = {
  "grid_size": len(rows),
  "feasible_count": len(feasible_rows),
  "feasible_fraction": round(len(feasible_rows) / len(rows), 4),
  "pareto_front_size": len(front_sorted),
  "reference_hypervolume": round(hv, 6),
  "enumeration_seconds": round(elapsed, 1),
  "grid_spec": {
   "num_layers": NUM_LAYERS,
   "width_mult_points": len(WIDTH_GRID),
   "quant_bits": QUANT_BITS,
   "dvfs_factor_points": len(DVFS_GRID),
  },
  "latency_sensitivity": latency_sensitivity,
 }
 summary_path = RESULTS_DIR / "reference_front_summary.json"
 with open(summary_path, "w") as f:
  json.dump(summary, f, indent=2)

 logger.info("=" * 60)
 logger.info("REFERENCE PARETO FRONT COMPLETE")
 logger.info(f"Hypervolume: {hv:.6f}")
 logger.info(f"Front size: {len(front_sorted)}")
 logger.info(f"CSV: {out_csv}")
 logger.info(f"Summary: {summary_path}")
 logger.info("=" * 60)
 return summary


if __name__ == "__main__":
 main()
