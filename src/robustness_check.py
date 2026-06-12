"""
5-seed robustness experiment.
Runs BO and Random Search for each seed in ROBUSTNESS_SEEDS.
Computes mean ± std of best feasible energy per method.
Output: results/robustness_summary.json
Paper use: "BO achieves X.XX ± Y.YY mJ (n=5 seeds) vs
 Random X.XX ± Y.YY mJ, a Z.Z% reduction."
"""
import os, sys, json, logging
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
 ROBUSTNESS_SEEDS, BO_TRIALS, RANDOM_TRIALS,
 RESULTS_DIR, ACCURACY_THRESHOLD, LOG_FORMAT, LOGS_DIR
)
from src.bo_experiment import run_bo_experiment
from src.baselines import run_random_search
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
 handlers=[logging.FileHandler(LOGS_DIR / "robustness.log"),
 logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
def best_feasible_energy(rows) -> float:
 feasible = [r for r in rows if str(r.get("is_feasible","")).lower() in
("true","1")]
 if not feasible:
  return float("nan")
 return min(float(r["energy_mj"]) for r in feasible)
def run_robustness_check():
 bo_bests, random_bests = [], []
 for seed in ROBUSTNESS_SEEDS:
  logger.info(f"\n{'='*50}\nSeed: {seed}\n{'='*50}")
  bo_rows = run_bo_experiment(
  n_trials = BO_TRIALS,
  seed = seed,
  output_csv = RESULTS_DIR / f"bo_seed{seed}.csv",
 verbose = True,
 )
  bo_bests.append(best_feasible_energy(bo_rows))
  rand_rows = run_random_search(
  n = RANDOM_TRIALS,
  seed = seed,
  output_csv = RESULTS_DIR / f"random_seed{seed}.csv",
 )
  random_bests.append(best_feasible_energy(rand_rows))
  logger.info(f"Seed {seed}: BO={bo_bests[-1]:.6f}mJ | "
 f"Random={random_bests[-1]:.6f}mJ")
  
 bo_mean, bo_std = float(np.nanmean(bo_bests)), float(np.nanstd(bo_bests))
 random_mean, random_std = float(np.nanmean(random_bests)), float(np.nanstd(random_bests))
 improvement_pct = (random_mean - bo_mean) / random_mean * 100 if random_mean > 0 else float("nan")
 # Non-parametric comparison (no normality assumption)
 # Wilcoxon signed-rank test if scipy available, else skip
 try:
  from scipy.stats import wilcoxon
  stat, p_val = wilcoxon(bo_bests, random_bests, alternative="less")
  statistical = {"wilcoxon_stat": round(stat, 4), "p_value": round(p_val,
4)}
 except ImportError:
  statistical = {"note": "scipy not installed — install for p-value"}
 summary = {
 "seeds": ROBUSTNESS_SEEDS,
 "bo_bests_mj": [round(v, 6) for v in bo_bests],
 "random_bests_mj": [round(v, 6) for v in random_bests],
 "bo_mean_mj": round(bo_mean, 6),
 "bo_std_mj": round(bo_std, 6),
 "random_mean_mj": round(random_mean, 6),
 "random_std_mj": round(random_std, 6),
 "improvement_pct": round(improvement_pct, 2),
 "statistics": statistical,
 }
 path = RESULTS_DIR / "robustness_summary.json"
 with open(path, "w") as f:
  json.dump(summary, f, indent=2)
 logger.info("\n" + "=" * 60)
 logger.info("ROBUSTNESS CHECK COMPLETE")
 logger.info(f"BO: {bo_mean:.6f} ± {bo_std:.6f} mJ (n={len(ROBUSTNESS_SEEDS)})")
 logger.info(f"Random: {random_mean:.6f} ± {random_std:.6f} mJ")
 logger.info(f"Energy reduction: {improvement_pct:.1f}%")
 if "p_value" in statistical:
  logger.info(f"Wilcoxon p-value: {statistical['p_value']:.4f}")
 logger.info(f"Results: {path}")
 logger.info("=" * 60)
 return summary
if __name__ == "__main__":
 run_robustness_check()