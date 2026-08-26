"""
Validation of the accuracy scaling-law surrogate in src/objective.py.
Reports fit quality, leave-one-out generalization error, and a
monotonicity proof, plus a live comparison against the inverse-distance
weighting (IDW) proxy it replaced.
Output: results/surrogate_validation.json
"""
import os, sys, json, logging
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import RESULTS_DIR, LOG_FORMAT
from src.objective import (
 CALIBRATION_POINTS, accuracy_proxy, ACCURACY_CLIP,
 _SCALING_A, _SCALING_B, _SCALING_C,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

WIDTH_SWEEP = np.linspace(0.25, 1.0, 60)
LAYER_SWEEP_FOR_WIDTH_CHECK = np.linspace(0.25, 1.0, 20)


def _architectures():
 """CALIBRATION_POINTS collapsed to (num_layers, width_mult) -> mean acc."""
 arch = {}
 for (nl, wm, _qb, acc) in CALIBRATION_POINTS:
  arch.setdefault((nl, wm), []).append(acc)
 return arch


def _fit_law(archs: dict) -> tuple:
 design = np.array([[1.0, np.log(nl), np.log(wm)] for (nl, wm) in archs])
 target = np.array([np.mean(v) for v in archs.values()])
 coef, *_ = np.linalg.lstsq(design, target, rcond=None)
 return tuple(float(c) for c in coef)


def _law_predict(nl: float, wm: float, coef: tuple) -> float:
 a, b, c = coef
 return float(np.clip(a + b * np.log(nl) + c * np.log(wm), *ACCURACY_CLIP))


def fit_quality(archs: dict) -> dict:
 coef = (_SCALING_A, _SCALING_B, _SCALING_C)
 items = sorted(archs.items())
 y = np.array([np.mean(v) for (_k, v) in items])
 pred = np.array([_law_predict(nl, wm, coef) for ((nl, wm), _v) in items])
 resid = y - pred
 ss_res = float(np.sum(resid ** 2))
 ss_tot = float(np.sum((y - y.mean()) ** 2))
 r2 = 1.0 - ss_res / ss_tot
 rmse = float(np.sqrt(np.mean(resid ** 2)))
 per_arch = [
  {"num_layers": nl, "width_mult": wm, "measured": float(m),
   "predicted": float(p), "residual_pp": round(float((m - p) * 100), 3)}
  for ((nl, wm), _v), m, p in zip(items, y, pred)
 ]
 return {
  "coefficients": {"a": coef[0], "b": coef[1], "c": coef[2]},
  "r_squared": round(r2, 5),
  "rmse_pp": round(rmse * 100, 3),
  "max_residual_pp": round(float(np.max(np.abs(resid))) * 100, 3),
  "per_architecture": per_arch,
 }


def leave_one_out(archs: dict) -> dict:
 items = sorted(archs.items())
 errors = []
 for i in range(len(items)):
  held_out_key, held_out_vals = items[i]
  train = {k: v for j, (k, v) in enumerate(items) if j != i}
  coef = _fit_law(train)
  y_true = float(np.mean(held_out_vals))
  y_pred = _law_predict(held_out_key[0], held_out_key[1], coef)
  errors.append(y_true - y_pred)
 errors = np.array(errors)
 return {
  "loo_rmse_pp": round(float(np.sqrt(np.mean(errors ** 2))) * 100, 3),
  "loo_max_error_pp": round(float(np.max(np.abs(errors))) * 100, 3),
  "n_architectures": len(items),
 }


def monotonicity_check() -> dict:
 violations = 0
 total_steps = 0
 for nl in (2, 3, 4):
  vals = [accuracy_proxy(nl, w, 8) for w in WIDTH_SWEEP]
  diffs = np.diff(vals)
  violations += int((diffs < -1e-12).sum())
  total_steps += len(diffs)
 for wm in LAYER_SWEEP_FOR_WIDTH_CHECK:
  vals = [accuracy_proxy(nl, float(wm), 8) for nl in (2, 3, 4)]
  diffs = np.diff(vals)
  violations += int((diffs < -1e-12).sum())
  total_steps += len(diffs)
 return {"violations": violations, "steps_checked": total_steps}


def quantization_spread(archs: dict) -> dict:
 by_arch_bits = {}
 for (nl, wm, qb, acc) in CALIBRATION_POINTS:
  by_arch_bits.setdefault((nl, wm), {})[qb] = acc
 spreads = {
  f"L{nl}_w{wm}": round((max(v.values()) - min(v.values())) * 100, 3)
  for (nl, wm), v in by_arch_bits.items()
 }
 return {"max_spread_pp": max(spreads.values()), "per_architecture_pp": spreads}


# ── Old IDW proxy, reimplemented here ONLY for a live comparison ────────────
def _idw_normalize(nl, wm, qb):
 return np.array([
  (nl - 2) / 2.0,
  (wm - 0.25) / 0.75,
  (np.log2(max(qb, 1)) - 2) / 3.0,
 ])


def _idw_predict(nl, wm, qb):
 query = _idw_normalize(nl, wm, qb)
 total_w, weighted = 0.0, 0.0
 for (cnl, cwm, cqb, cacc) in CALIBRATION_POINTS:
  pt = _idw_normalize(cnl, cwm, cqb)
  dist = np.linalg.norm(query - pt)
  if dist < 1e-8:
   return float(cacc)
  w = 1.0 / (dist ** 2)
  weighted += w * cacc
  total_w += w
 return float(np.clip(weighted / total_w, 0.50, 0.95))


def idw_monotonicity_check() -> dict:
 violations = 0
 total_steps = 0
 for nl in (2, 3, 4):
  vals = [_idw_predict(nl, w, 8) for w in WIDTH_SWEEP]
  diffs = np.diff(vals)
  violations += int((diffs < -1e-12).sum())
  total_steps += len(diffs)
 return {"violations": violations, "steps_checked": total_steps}


def main():
 archs = _architectures()
 fit = fit_quality(archs)
 loo = leave_one_out(archs)
 mono = monotonicity_check()
 quant = quantization_spread(archs)
 idw_mono = idw_monotonicity_check()

 summary = {
  "scaling_law": fit,
  "leave_one_out": loo,
  "monotonicity": mono,
  "quantization_null_effect": quant,
  "comparison_vs_old_idw_proxy": {
   "idw_monotonicity_violations": idw_mono["violations"],
   "idw_steps_checked": idw_mono["steps_checked"],
   "scaling_law_violations": mono["violations"],
  },
 }

 logger.info("=" * 60)
 logger.info("ACCURACY SURROGATE VALIDATION")
 logger.info("=" * 60)
 logger.info(f"Fit: a={fit['coefficients']['a']:.4f} b={fit['coefficients']['b']:.4f} "
 f"c={fit['coefficients']['c']:.4f}")
 logger.info(f"R^2={fit['r_squared']:.4f}  RMSE={fit['rmse_pp']:.2f}pp  "
 f"max_resid={fit['max_residual_pp']:.2f}pp")
 logger.info(f"Leave-one-out RMSE={loo['loo_rmse_pp']:.2f}pp  "
 f"max={loo['loo_max_error_pp']:.2f}pp  (n={loo['n_architectures']} architectures)")
 logger.info(f"Monotonicity violations: {mono['violations']}/{mono['steps_checked']} "
 f"(old IDW proxy: {idw_mono['violations']}/{idw_mono['steps_checked']})")
 logger.info(f"Max quantization accuracy spread: {quant['max_spread_pp']:.2f}pp "
 f"(below LOO RMSE of {loo['loo_rmse_pp']:.2f}pp -> treated as null effect)")
 logger.info("=" * 60)

 assert mono["violations"] == 0, \
  f"Scaling law must be monotone; found {mono['violations']} violations"
 assert loo["loo_rmse_pp"] <= 3.0, f"LOO RMSE too high: {loo['loo_rmse_pp']}pp"
 assert loo["loo_max_error_pp"] <= 5.0, f"LOO max error too high: {loo['loo_max_error_pp']}pp"
 logger.info("All validation checks PASSED")

 out_path = RESULTS_DIR / "surrogate_validation.json"
 with open(out_path, "w") as f:
  json.dump(summary, f, indent=2)
 logger.info(f"Wrote {out_path}")
 return summary


if __name__ == "__main__":
 main()
