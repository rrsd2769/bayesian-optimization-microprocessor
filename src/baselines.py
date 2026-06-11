"""
Baseline search methods: Grid Search and Random Search.
CRITICAL DESIGN CONSTRAINT:
 Both methods use the SAME evaluate() function and SAME search space
 as bo_experiment.py. Same budget (RANDOM_TRIALS = BO_TRIALS = 30).
 Same CSV column format for direct comparison in plot_results.py.
References:
 Bergstra & Bengio (2012) "Random Search for Hyper-parameter Optimization"
 → Random search baseline is essential for BO comparison papers.
"""
import os, sys, csv, logging, itertools, time
from pathlib import Path
from typing import List, Dict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.config import (
 RANDOM_SEED, RESULTS_DIR, GRID_CONFIGS, RANDOM_TRIALS,
 ACCURACY_THRESHOLD, LOG_FORMAT, LOGS_DIR
)
from src.objective import evaluate
logging.basicConfig(
 level=logging.INFO, format=LOG_FORMAT,
 handlers=[
 logging.FileHandler(LOGS_DIR / "baselines.log"),
 logging.StreamHandler(sys.stdout),
 ]
)
logger = logging.getLogger(__name__)
FIELDNAMES = [
 "trial_id", "method", "seed",
 "num_layers", "width_mult", "quant_bits",
 "time_slice", "priority", "dvfs_factor",
 "accuracy", "memory_kb", "energy_mj", "latency_ms",
 "is_feasible", "cumulative_best_energy",
]
def _make_grid(n: int, seed: int) -> List[dict]:
 """
 Sample n configs from a coarse grid over the search space.
 Grid: 3 values per param → 3^6 = 729 total → random sample of n.
 Seeded sampling ensures reproducibility.
 """
 rng = np.random.default_rng(seed)
 grid = {
 "num_layers": [2, 3, 4],
 "width_mult": [0.25, 0.625, 1.0],
 "quant_bits": [4, 8, 16, 32], # all values
 "time_slice": [5.0, 27.5, 50.0],
 "priority": [0, 1, 2],
 "dvfs_factor": [0.5, 0.75, 1.0],
 }
 all_combos = list(itertools.product(*grid.values()))
 keys = list(grid.keys())
 idxs = rng.choice(len(all_combos), size=min(n, len(all_combos)),
replace=False)
 return [dict(zip(keys, all_combos[i])) for i in idxs]

def _run_search(configs: List[dict], method: str, seed: int,
 output_csv: Path) -> List[Dict]:
 """Generic search loop used by both grid and random search."""
 results = []
 best_energy = float("inf")
 with open(output_csv, "w", newline="") as f:
  writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
  writer.writeheader()
  for i, config in enumerate(configs):
   t0 = time.perf_counter()
   raw = evaluate(config)
   elapsed = time.perf_counter() - t0
   is_feasible = raw["accuracy"] >= ACCURACY_THRESHOLD
   if is_feasible and raw["energy_mj"] < best_energy:
    best_energy = raw["energy_mj"]
   row = {
 "trial_id": i + 1,
 "method": method,
 "seed": seed,
 "num_layers": config["num_layers"],
 "width_mult": round(float(config["width_mult"]), 4),
 "quant_bits": config["quant_bits"],
 "time_slice": round(float(config["time_slice"]), 2),
 "priority": config["priority"],
 "dvfs_factor": round(float(config["dvfs_factor"]),
4),
 "accuracy": raw["accuracy"],
 "memory_kb": raw["memory_kb"],
 "energy_mj": raw["energy_mj"],
 "latency_ms": raw["latency_ms"],
 "is_feasible": is_feasible,
 "cumulative_best_energy": round(best_energy, 6) if best_energy <
float("inf") else "",
 }
   results.append(row)
   writer.writerow(row)
   f.flush()

   logger.info(
 f"[{method} {i+1:02d}/{len(configs)}] "
 f"E={raw['energy_mj']:.6f}mJ | acc={raw['accuracy']:.4f} | "
 f"feasible={is_feasible} | best={best_energy:.6f}mJ | {elapsed:.1f}s"
 )
 logger.info(f"{method} complete. Best energy: {best_energy:.6f} mJ → {output_csv}")
 return results
def run_grid_search(n: int = GRID_CONFIGS, seed: int = RANDOM_SEED,
 output_csv: Path = None) -> List[Dict]:
 if output_csv is None:
  output_csv = RESULTS_DIR / "grid_results.csv"
 logger.info("=" * 50)
 logger.info(f"GRID SEARCH | n={n} | seed={seed}")
 logger.info("=" * 50)
 configs = _make_grid(n, seed)
 return _run_search(configs, "Grid", seed, output_csv)
def run_random_search(n: int = RANDOM_TRIALS, seed: int = RANDOM_SEED,
 output_csv: Path = None) -> List[Dict]:
 """
 Sample uniformly from the SAME search space as BO.
 Budget = RANDOM_TRIALS = BO_TRIALS (fair comparison).
 """
 if output_csv is None:
  output_csv = RESULTS_DIR / "random_results.csv"
 logger.info("=" * 50)
 logger.info(f"RANDOM SEARCH | n={n} | seed={seed}")
 logger.info("=" * 50)
 rng = np.random.default_rng(seed)
 configs = [
 {
 "num_layers": int(rng.choice([2, 3, 4])),
 "width_mult": float(rng.uniform(0.25, 1.0)),
 "quant_bits": int(rng.choice([4, 8, 16, 32])),
 "time_slice": float(rng.uniform(5.0, 50.0)),
 "priority": int(rng.choice([0, 1, 2])),
 "dvfs_factor": float(rng.uniform(0.5, 1.0)),
 }
  for _ in range(n)
 ]
 return _run_search(configs, "Random", seed, output_csv)
if __name__ == "__main__":
 logger.info("Running Grid Search baseline...")
 run_grid_search()
 logger.info("Running Random Search baseline...")
 run_random_search()
 logger.info("All baselines complete.")