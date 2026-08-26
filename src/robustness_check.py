"""
Multi-seed robustness experiment.
Primary metric: final hypervolume (accuracy, energy_mj) per method per seed
— MOO (qNEHVI) vs Random Search at matched budget. Secondary: best feasible
energy from the single-objective constrained run (BO vs Random Search),
kept for continuity with the SOO formulation.

The two loops use DIFFERENT seed counts / budgets because qNEHVI's per-trial
cost is not flat — it grows with the discovered Pareto front's size (box
decomposition), measured at ~130s cumulative for 30 trials and ~225s for 50
trials on this machine. 30 seeds x 100 trials of qNEHVI would take ~5+
hours. MOO_ROBUSTNESS_SEEDS/MOO_ROBUSTNESS_TRIALS (config.py) keep the
primary sweep to ~15-20 minutes; the secondary (plain qLogEI, no box
decomposition) affords the full 30 seeds x 30 trials cheaply.

Also reports:
 - Bootstrap 95% CI (10,000 resamples) on the mean difference
 - Wilcoxon signed-rank test, WITH the test's own floor reported alongside
   the p-value so a floor is never mistaken for a measurement
 - Cliff's delta effect size
 - IGD to the exhaustive reference Pareto front (src/reference_front.py)
 - Hypervolume at matched evaluation budgets, so the comparison is not a
   single endpoint

Resumable: each seed's raw results are written to their own CSV as soon as
that seed completes. Re-running this script skips any seed whose CSV
already has the expected row count, so an interrupted run can continue
rather than restart from scratch.

Output: results/robustness_summary.json
"""
import os, sys, csv, json, logging
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
 ROBUSTNESS_SEEDS, BO_TRIALS, RANDOM_TRIALS,
 MOO_ROBUSTNESS_SEEDS, MOO_ROBUSTNESS_TRIALS,
 RESULTS_DIR, ACCURACY_THRESHOLD, LOG_FORMAT, LOGS_DIR
)
from src.bo_experiment import run_bo_experiment
from src.moo_experiment import run_moo_experiment
from src.baselines import run_random_search
from src.pareto_metrics import hypervolume, igd, pareto_front, _ref_point_tensor

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
 handlers=[logging.FileHandler(LOGS_DIR / "robustness.log"),
 logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

BUDGET_CHECKPOINTS = sorted({
 max(1, MOO_ROBUSTNESS_TRIALS // 4), MOO_ROBUSTNESS_TRIALS // 2,
 (3 * MOO_ROBUSTNESS_TRIALS) // 4, MOO_ROBUSTNESS_TRIALS,
})
BOOTSTRAP_RESAMPLES = 10_000
MIN_ACHIEVABLE_WILCOXON_P_HV = 1.0 / (2 ** len(MOO_ROBUSTNESS_SEEDS))
MIN_ACHIEVABLE_WILCOXON_P_ENERGY = 1.0 / (2 ** len(ROBUSTNESS_SEEDS))


def best_feasible_energy(rows) -> float:
 feasible = [r for r in rows if str(r.get("is_feasible", "")).lower() in
("true", "1")]
 if not feasible:
  return float("nan")
 return min(float(r["energy_mj"]) for r in feasible)


def _feasible_rows(rows):
 return [r for r in rows if str(r.get("is_feasible", "")).lower() in ("true", "1")]


def hv_at_budget(rows, budget: int, ref_point) -> float:
 """Hypervolume using only the first `budget` trials of `rows`."""
 truncated = rows[:budget]
 return hypervolume(_feasible_rows(truncated), ref_point)


def _load_or_run(csv_path, expected_rows: int, run_fn):
 """
 Resumability helper: if csv_path already has `expected_rows` rows,
 load and return them instead of re-running run_fn(). Guards against
 losing multi-minute qNEHVI progress to an interrupted session.
 """
 if csv_path.exists():
  with open(csv_path) as f:
   existing = list(csv.DictReader(f))
  if len(existing) == expected_rows:
   logger.info(f"  [resume] {csv_path.name} already complete ({expected_rows} rows) — skipping")
   return existing
 return run_fn()


def bootstrap_ci_diff(a: list, b: list, n_resamples: int = BOOTSTRAP_RESAMPLES,
 rng: np.random.Generator = None) -> dict:
 """
 Bootstrap 95% CI on mean(a) - mean(b), via resampling PAIRED
 differences (a and b share seeds, so this respects the pairing).
 """
 if rng is None:
  rng = np.random.default_rng(0)
 a = np.asarray(a, dtype=float)
 b = np.asarray(b, dtype=float)
 diffs = a - b
 n = len(diffs)
 boot_means = np.empty(n_resamples)
 for i in range(n_resamples):
  sample = rng.choice(diffs, size=n, replace=True)
  boot_means[i] = sample.mean()
 lo, hi = np.percentile(boot_means, [2.5, 97.5])
 return {
  "mean_diff": float(diffs.mean()),
  "ci_95_lo": float(lo),
  "ci_95_hi": float(hi),
  "excludes_zero": bool(lo > 0 or hi < 0),
 }


def cliffs_delta(a: list, b: list) -> float:
 """
 Cliff's delta effect size in [-1, 1]. |delta| < 0.147 negligible,
 < 0.33 small, < 0.474 medium, else large (Romano et al. 2006).
 """
 a = np.asarray(a, dtype=float)
 b = np.asarray(b, dtype=float)
 gt = sum((ai > bj) for ai in a for bj in b)
 lt = sum((ai < bj) for ai in a for bj in b)
 return (gt - lt) / (len(a) * len(b))


def _wilcoxon_report(x, y, alternative, min_achievable_p):
 try:
  from scipy.stats import wilcoxon
  stat, p = wilcoxon(x, y, alternative=alternative)
  return {
   "wilcoxon_stat": round(float(stat), 4),
   "p_value": round(float(p), 6),
   "min_achievable_p": round(min_achievable_p, 6),
   "test_saturated": bool(p <= min_achievable_p * 1.001),
  }
 except ImportError:
  return {"note": "scipy not installed — install for p-value"}


def run_robustness_check():
 ref_point = _ref_point_tensor()
 reference_front_path = RESULTS_DIR / "reference_pareto_front.csv"
 reference_front = None
 if reference_front_path.exists():
  with open(reference_front_path) as f:
   reference_front = list(csv.DictReader(f))
 else:
  logger.warning("results/reference_pareto_front.csv not found — "
  "IGD will be omitted. Run src/reference_front.py first.")

 # ── Primary: MOO (qNEHVI) vs Random at matched MOO_ROBUSTNESS_TRIALS budget
 moo_hvs, moo_rand_hvs = [], []
 moo_igds, moo_rand_igds = [], []
 moo_budget_hvs = {b: [] for b in BUDGET_CHECKPOINTS}
 rand_budget_hvs = {b: [] for b in BUDGET_CHECKPOINTS}

 logger.info(f"PRIMARY sweep: {len(MOO_ROBUSTNESS_SEEDS)} seeds x "
 f"{MOO_ROBUSTNESS_TRIALS} trials (qNEHVI)")
 for seed in MOO_ROBUSTNESS_SEEDS:
  logger.info(f"\n{'='*50}\n[Primary/HV] Seed: {seed}\n{'='*50}")
  moo_csv = RESULTS_DIR / f"moo_seed{seed}.csv"
  moo_rows = _load_or_run(moo_csv, MOO_ROBUSTNESS_TRIALS, lambda: run_moo_experiment(
   n_trials=MOO_ROBUSTNESS_TRIALS, seed=seed, output_csv=moo_csv, verbose=False,
  ))
  moo_rand_csv = RESULTS_DIR / f"moo_random_seed{seed}.csv"
  moo_rand_rows = _load_or_run(moo_rand_csv, MOO_ROBUSTNESS_TRIALS, lambda: run_random_search(
   n=MOO_ROBUSTNESS_TRIALS, seed=seed, output_csv=moo_rand_csv,
  ))

  moo_hv = hypervolume(_feasible_rows(moo_rows), ref_point)
  moo_rand_hv = hypervolume(_feasible_rows(moo_rand_rows), ref_point)
  moo_hvs.append(moo_hv)
  moo_rand_hvs.append(moo_rand_hv)

  if reference_front is not None:
   moo_igds.append(igd(pareto_front(_feasible_rows(moo_rows)), reference_front))
   moo_rand_igds.append(igd(pareto_front(_feasible_rows(moo_rand_rows)), reference_front))

  for b in BUDGET_CHECKPOINTS:
   moo_budget_hvs[b].append(hv_at_budget(moo_rows, b, ref_point))
   rand_budget_hvs[b].append(hv_at_budget(moo_rand_rows, b, ref_point))

  logger.info(f"Seed {seed}: MOO_HV={moo_hv:.6f} | Random_HV={moo_rand_hv:.6f}")

 # ── Secondary: SOO-constrained BO vs Random at matched RANDOM_TRIALS budget
 bo_bests, random_bests = [], []
 logger.info(f"\nSECONDARY sweep: {len(ROBUSTNESS_SEEDS)} seeds x {BO_TRIALS} trials (qLogEI)")
 for seed in ROBUSTNESS_SEEDS:
  bo_csv = RESULTS_DIR / f"bo_seed{seed}.csv"
  bo_rows = _load_or_run(bo_csv, BO_TRIALS, lambda: run_bo_experiment(
   n_trials=BO_TRIALS, seed=seed, output_csv=bo_csv, verbose=False,
  ))
  bo_bests.append(best_feasible_energy(bo_rows))

  rand_csv = RESULTS_DIR / f"random_seed{seed}.csv"
  rand_rows = _load_or_run(rand_csv, RANDOM_TRIALS, lambda: run_random_search(
   n=RANDOM_TRIALS, seed=seed, output_csv=rand_csv,
  ))
  random_bests.append(best_feasible_energy(rand_rows))

  logger.info(f"[Secondary/energy] Seed {seed}: BO_best={bo_bests[-1]:.6f}mJ | "
  f"Random_best={random_bests[-1]:.6f}mJ")

 # ── Primary: hypervolume statistics ──────────────────────────────────────
 moo_hv_mean, moo_hv_std = float(np.mean(moo_hvs)), float(np.std(moo_hvs))
 rand_hv_mean, rand_hv_std = float(np.mean(moo_rand_hvs)), float(np.std(moo_rand_hvs))
 hv_bootstrap = bootstrap_ci_diff(moo_hvs, moo_rand_hvs)
 hv_cliffs_delta = cliffs_delta(moo_hvs, moo_rand_hvs)
 hv_statistical = _wilcoxon_report(moo_hvs, moo_rand_hvs, "greater", MIN_ACHIEVABLE_WILCOXON_P_HV)

 # ── Secondary: best-feasible-energy statistics (continuity with SOO) ────
 bo_mean, bo_std = float(np.nanmean(bo_bests)), float(np.nanstd(bo_bests))
 random_mean, random_std = float(np.nanmean(random_bests)), float(np.nanstd(random_bests))
 energy_improvement_pct = (random_mean - bo_mean) / random_mean * 100 if random_mean > 0 else float("nan")
 energy_bootstrap = bootstrap_ci_diff(random_bests, bo_bests) # positive = BO better (lower energy)
 energy_statistical = _wilcoxon_report(bo_bests, random_bests, "less", MIN_ACHIEVABLE_WILCOXON_P_ENERGY)

 # ── Budget-matched comparison ─────────────────────────────────────────────
 budget_comparison = {
  str(b): {
   "moo_hv_mean": round(float(np.mean(moo_budget_hvs[b])), 6),
   "moo_hv_std": round(float(np.std(moo_budget_hvs[b])), 6),
   "random_hv_mean": round(float(np.mean(rand_budget_hvs[b])), 6),
   "random_hv_std": round(float(np.std(rand_budget_hvs[b])), 6),
  }
  for b in BUDGET_CHECKPOINTS
 }

 summary = {
  "primary_hypervolume": {
   "n_seeds": len(MOO_ROBUSTNESS_SEEDS),
   "seeds": MOO_ROBUSTNESS_SEEDS,
   "trials_per_seed": MOO_ROBUSTNESS_TRIALS,
   "moo_hvs": [round(v, 6) for v in moo_hvs],
   "random_hvs": [round(v, 6) for v in moo_rand_hvs],
   "moo_hv_mean": round(moo_hv_mean, 6),
   "moo_hv_std": round(moo_hv_std, 6),
   "random_hv_mean": round(rand_hv_mean, 6),
   "random_hv_std": round(rand_hv_std, 6),
   "bootstrap_ci_diff": hv_bootstrap,
   "cliffs_delta": round(hv_cliffs_delta, 4),
   "statistics": hv_statistical,
   "igd_to_reference": {
    "moo_igd_mean": round(float(np.nanmean(moo_igds)), 6) if moo_igds else None,
    "random_igd_mean": round(float(np.nanmean(moo_rand_igds)), 6) if moo_rand_igds else None,
   } if reference_front is not None else None,
   "budget_matched_comparison": budget_comparison,
  },
  "secondary_best_feasible_energy": {
   "n_seeds": len(ROBUSTNESS_SEEDS),
   "seeds": ROBUSTNESS_SEEDS,
   "trials_per_seed": BO_TRIALS,
   "bo_bests_mj": [round(v, 6) for v in bo_bests],
   "random_bests_mj": [round(v, 6) for v in random_bests],
   "bo_mean_mj": round(bo_mean, 6),
   "bo_std_mj": round(bo_std, 6),
   "random_mean_mj": round(random_mean, 6),
   "random_std_mj": round(random_std, 6),
   "improvement_pct": round(energy_improvement_pct, 2),
   "bootstrap_ci_diff": energy_bootstrap,
   "statistics": energy_statistical,
  },
 }

 path = RESULTS_DIR / "robustness_summary.json"
 with open(path, "w") as f:
  json.dump(summary, f, indent=2)

 logger.info("\n" + "=" * 60)
 logger.info("ROBUSTNESS CHECK COMPLETE")
 logger.info(f"[PRIMARY, n={len(MOO_ROBUSTNESS_SEEDS)}] MOO HV: {moo_hv_mean:.6f} ± {moo_hv_std:.6f} vs "
 f"Random HV: {rand_hv_mean:.6f} ± {rand_hv_std:.6f}")
 logger.info(f"Cliff's delta: {hv_cliffs_delta:.4f} | "
 f"Bootstrap 95% CI on diff: [{hv_bootstrap['ci_95_lo']:.6f}, {hv_bootstrap['ci_95_hi']:.6f}]")
 if "p_value" in hv_statistical:
  logger.info(f"Wilcoxon p={hv_statistical['p_value']:.6f} "
  f"(floor={hv_statistical['min_achievable_p']:.6f}, "
  f"saturated={hv_statistical['test_saturated']})")
 logger.info(f"[SECONDARY, n={len(ROBUSTNESS_SEEDS)}] BO best energy: {bo_mean:.6f} ± {bo_std:.6f} mJ vs "
 f"Random: {random_mean:.6f} ± {random_std:.6f} mJ ({energy_improvement_pct:.1f}% reduction)")
 logger.info(f"Results: {path}")
 logger.info("=" * 60)
 return summary


if __name__ == "__main__":
 run_robustness_check()
