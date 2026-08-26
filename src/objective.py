"""
Unified evaluation function for the BO pipeline.
evaluate(config) → {accuracy, memory_kb, energy_mj, latency_ms}
Called by:
 - bo_experiment.py, moo_experiment.py (SOO / MOO trial loops)
 - baselines.py (grid + random search)
 - robustness_check.py (30 seeds x multiple trials)
Measured cost: ~3.8ms/call (dominated by model construction in build_model()).
Accuracy: scaling-law proxy fitted to real architecture measurements —
see accuracy_proxy() and src/validate_surrogate.py.
"""
import os, sys, time, logging
from functools import lru_cache
import torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
 RANDOM_SEED, ACCURACY_THRESHOLD, LATENCY_BUDGET_MS, LOG_FORMAT
)
from src.model import build_model, get_model_stats
from src.quantization import apply_quantization
from src.energy_proxy import estimate_energy_mj
from src.scheduler_sim import run_scheduler_sim

logger = logging.getLogger(__name__)
# ── Accuracy Proxy Calibration
#─────────────────────────────────────────────────
# Ground-truth measurements: baseline (Days 1-2) + full architecture sweep
# (src/collect_calibration.py) — 6 architectures spanning num_layers x
# width_mult, each measured at quant_bits in {32, 16, 8, 4}.
# Format: (num_layers, width_mult, quant_bits, measured_accuracy)
CALIBRATION_POINTS = [
 # (num_layers, width_mult, quant_bits, accuracy)
 (3, 1.00, 32, 0.8170), # baseline
 (3, 1.00, 16, 0.8170),
 (3, 1.00, 8, 0.8172),
 (3, 1.00, 4, 0.8172),
 (2, 0.25, 32, 0.5249), # collect_calibration.py
 (2, 0.25, 16, 0.5249),
 (2, 0.25, 8, 0.5227),
 (2, 0.25, 4, 0.5227),
 (2, 1.00, 32, 0.6642),
 (2, 1.00, 16, 0.6621),
 (2, 1.00, 8, 0.6612),
 (2, 1.00, 4, 0.6612),
 (4, 0.25, 32, 0.7542),
 (4, 0.25, 16, 0.7535),
 (4, 0.25, 8, 0.7543),
 (4, 0.25, 4, 0.7543),
 (4, 1.00, 32, 0.8786),
 (4, 1.00, 16, 0.8779),
 (4, 1.00, 8, 0.8782),
 (4, 1.00, 4, 0.8782),
 (3, 0.625, 32, 0.7778),
 (3, 0.625, 16, 0.7762),
 (3, 0.625, 8, 0.7768),
 (3, 0.625, 4, 0.7768),
]


# ── Accuracy scaling law ──────────────────────────────────────────────────────
# acc = a + b*ln(num_layers) + c*ln(width_mult)
#
# Fitted by ordinary least squares to the 6 measured architectures in
# CALIBRATION_POINTS (accuracy averaged over quant_bits — see the docstring
# on accuracy_proxy() for why quantization is not a fit variable). Monotone
# non-decreasing in both depth and width by construction whenever b, c > 0,
# which holds for this data (depth dominates: b > c). Functional form follows
# the neural scaling-law literature (Hestness et al. 2017; Kaplan et al.
# 2020), specialised to the two architecture degrees of freedom searched
# here. This replaces an inverse-distance-weighting proxy that was
# non-monotone (9/45 violations on a dense width sweep) — see
# src/validate_surrogate.py for the comparison.
#
# Validity is claimed ONLY inside the search box num_layers in [2,4],
# width_mult in [0.25,1.0]; predictions are clipped to ACCURACY_CLIP to
# prevent nonsensical extrapolation past the box.
ACCURACY_CLIP = (0.40, 0.95)

def _fit_scaling_law() -> tuple:
 """
 Least-squares fit of (a, b, c) to CALIBRATION_POINTS, averaged over
 quant_bits per architecture. Linear in its parameters, so this is an
 exact closed-form fit — no initialization or convergence risk.
 """
 arch = {}
 for (nl, wm, _qb, acc) in CALIBRATION_POINTS:
  arch.setdefault((nl, wm), []).append(acc)
 design = np.array([[1.0, np.log(nl), np.log(wm)] for (nl, wm) in arch])
 target = np.array([np.mean(v) for v in arch.values()])
 coef, *_ = np.linalg.lstsq(design, target, rcond=None)
 return tuple(float(c) for c in coef)

_SCALING_A, _SCALING_B, _SCALING_C = _fit_scaling_law()

def accuracy_proxy(num_layers: int, width_mult: float, quant_bits: int) ->float:
 """
 Estimate accuracy from a monotone scaling law fitted to measured
 architecture data.

 quant_bits is accepted for interface compatibility but does not affect
 the result: across all 6 measured architectures in CALIBRATION_POINTS,
 the accuracy spread over quant_bits in {32,16,8,4} is at most 0.30
 percentage points — below the surrogate's own ~2.4pp leave-one-out
 error (see results/surrogate_validation.json). Post-training
 quantization to >=4 bits is accuracy-neutral for this model family at
 measurement resolution, so it is treated as a null effect rather than
 fit as signal.
 """
 acc = _SCALING_A + _SCALING_B * np.log(num_layers) + _SCALING_C * np.log(width_mult)
 return float(np.clip(acc, *ACCURACY_CLIP))

@lru_cache(maxsize=4096)
def _arch_model_and_stats(num_layers: int, width_mult: float):
 """
 Cached (model, stats) for a given architecture. FLOPs/params depend only
 on (num_layers, width_mult), and build_model() is deterministic
 (torch.manual_seed(RANDOM_SEED) is set before every construction), so
 the cached model is bit-identical to a freshly-built one. This is the
 dominant cost in evaluate() (torchinfo's summary() forward pass) and
 becomes essential for exhaustive enumeration (src/reference_front.py)
 and repeated architectures within a single BO/MOO run.
 """
 torch.manual_seed(RANDOM_SEED)
 model = build_model({"num_layers": num_layers, "width_mult": width_mult})
 model.eval()
 stats = get_model_stats(model)
 return model, stats

def evaluate(config: dict) -> dict:
 """
 Evaluate one configuration across all 4 paper metrics.
 Args:
 config (dict): must contain all 6 keys:
 num_layers (int) — {2, 3, 4}
 width_mult (float) — [0.25, 1.0]
 quant_bits (int) — {4, 8, 16, 32}
 time_slice (float) — [5.0, 50.0] ms
 priority (int) — {0, 1, 2}
 dvfs_factor (float) — [0.5, 1.0]
 Returns:
 dict: {accuracy, memory_kb, energy_mj, latency_ms}
 Example:
 >>> cfg = {"num_layers":3, "width_mult":0.5, "quant_bits":8,
 ... "time_slice":10.0, "priority":1, "dvfs_factor":0.75}
 >>> evaluate(cfg)
 {"accuracy": 0.7326, "memory_kb": 23.14, "energy_mj": 0.005741,
"latency_ms": 35.7168}
 """
 t0 = time.perf_counter()
 num_layers = int(config["num_layers"])
 width_mult = float(config["width_mult"])
 quant_bits = int(config["quant_bits"])
 dvfs_factor = float(config["dvfs_factor"])
# Step 1: Architecture → FLOPs + param count (cached)
 model, stats = _arch_model_and_stats(num_layers, width_mult)
 # Step 2: Memory after quantization
 _, memory_kb = apply_quantization(model, quant_bits)
 # Step 3: Accuracy from proxy
 accuracy = accuracy_proxy(num_layers, width_mult, quant_bits)
 # Step 4: Energy from FLOPs + DVFS + quantization bit-width
 energy_mj = estimate_energy_mj(stats["flops"], dvfs_factor, quant_bits)
 # Step 5: Latency from scheduler simulation
 sched = run_scheduler_sim(config, flops=stats["flops"])
 latency_ms = sched["mean_latency_ms"]
 elapsed_ms = (time.perf_counter() - t0) * 1000
 logger.debug(
  f"evaluate() | layers={num_layers}, width={width_mult:.2f}, "
  f"bits={quant_bits}, dvfs={dvfs_factor:.2f} | "
  f"acc={accuracy:.4f}, mem={memory_kb:.1f}KB, "
  f"E={energy_mj:.6f}mJ, lat={latency_ms:.2f}ms | {elapsed_ms:.0f}ms"
 )
 return {
 "accuracy": round(accuracy, 4),
 "memory_kb": round(memory_kb, 2),
 "energy_mj": round(energy_mj, 6),
 "latency_ms": round(latency_ms, 4),
 }

def is_feasible(result: dict) -> bool:
 """
 Joint feasibility: accuracy floor AND latency budget.
 Single source of truth — used by bo_experiment.py, moo_experiment.py,
 baselines.py, and robustness_check.py so the definition cannot drift
 between them.
 """
 return (result["accuracy"] >= ACCURACY_THRESHOLD
  and result["latency_ms"] <= LATENCY_BUDGET_MS)

def evaluate_for_ax(parameterization: dict) -> dict:
 """
 Ax-compatible wrapper.
 Ax requires {metric: (mean, SEM)} tuples. SEM=0.0 = deterministic.
Pass this function to AxClient or use inside the BO loop.
 """
 result = evaluate(parameterization)
 return {k: (v, 0.0) for k, v in result.items()}
if __name__ == "__main__":
 import logging
 logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
 print("\n=== evaluate() Integration Test ===\n")
 print("All 5 test configs must produce positive non-zero values for all 4 metrics.\n")
 test_configs = [
 {"num_layers": 2, "width_mult": 0.25, "quant_bits": 8,
 "time_slice": 5.0, "priority": 0, "dvfs_factor": 0.50},
 {"num_layers": 3, "width_mult": 0.50, "quant_bits": 8,
 "time_slice": 10.0, "priority": 1, "dvfs_factor": 0.75},
 {"num_layers": 3, "width_mult": 1.00, "quant_bits": 32,
 "time_slice": 20.0, "priority": 1, "dvfs_factor": 1.00}, # baseline
 {"num_layers": 4, "width_mult": 0.75, "quant_bits": 4,
 "time_slice": 10.0, "priority": 0, "dvfs_factor": 0.60},
 {"num_layers": 2, "width_mult": 1.00, "quant_bits": 16,
 "time_slice": 50.0, "priority": 2, "dvfs_factor": 0.80},
 ]
 header = f"{'layers':<8} {'width':<8} {'bits':<6} {'acc':<10} {'mem(KB)':<12} {'E(mJ)':<14} {'lat(ms)'}"
 print(header)
 print("-" * 70)

 t_calls = []
 for cfg in test_configs:
  t = time.perf_counter()
  r = evaluate(cfg)
  t_calls.append((time.perf_counter() - t) * 1000)
  print(f"{cfg['num_layers']:<8} {cfg['width_mult']:<8.2f} {cfg['quant_bits']:<6} "
   f"{r['accuracy']:<10.4f} {r['memory_kb']:<12.1f} "
   f"{r['energy_mj']:<14.6f} {r['latency_ms']:.4f}")
  assert all(v > 0 for v in r.values()), f"Non-positive metric in: {r}"
 print(f"\nMean evaluation time: {sum(t_calls)/len(t_calls):.0f}ms per call")
 # The scaling-law surrogate FITS the calibration data rather than
 # interpolating it, so the baseline architecture is not reproduced
 # exactly. Tolerance is the surrogate's own leave-one-out RMSE (see
 # results/surrogate_validation.json, ~2.4pp) with headroom, not 1%.
 SURROGATE_LOO_TOLERANCE = 0.03
 baseline_r = evaluate({"num_layers": 3, "width_mult": 1.0, "quant_bits": 32,
 "time_slice": 20.0, "priority": 1, "dvfs_factor":
1.0})
 expected_acc = CALIBRATION_POINTS[0][3] # Day 1 measured accuracy
 assert abs(baseline_r["accuracy"] - expected_acc) < SURROGATE_LOO_TOLERANCE, \
 f"Proxy mismatch: got {baseline_r['accuracy']:.4f}, expected {expected_acc:.4f}"
 # Test Ax wrapper format
 ax_r = evaluate_for_ax(test_configs[0])
 assert all(isinstance(v, tuple) and len(v) == 2 for v in ax_r.values()),  "evaluate_for_ax() must return {metric: (float, float)} tuples"
 print("\nIntegration test: ALL PASSED ") 
 print("evaluate_for_ax() format: VERIFIED ✓")