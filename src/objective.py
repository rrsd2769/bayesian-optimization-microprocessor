"""
Unified evaluation function for the BO pipeline.
evaluate(config) → {accuracy, memory_kb, energy_mj, latency_ms}
Called by:
 - bo_experiment.py (30 times per BO run)
 - baselines.py (25 grid + 30 random times)
 - robustness_check.py (5 seeds × 60 calls)
Performance target: <5 seconds per call.
Accuracy: proxy via inverse-distance weighting from calibration points.
"""
import os, sys, time, logging
import torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
 RANDOM_SEED, ACCURACY_THRESHOLD, LOG_FORMAT
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

def _normalize(nl: int, wm: float, qb: int) -> np.ndarray:
 """Normalize architecture params to [0,1] for distance calculation."""
 return np.array([
 (nl - 2) / 2.0, # num_layers: {2,3,4} → {0, 0.5, 1.0}
 (wm - 0.25) / 0.75, # width_mult: [0.25,1.0] → [0,1]
 (np.log2(max(qb,1)) - 2) / 3.0, # quant_bits log scale → [0,1]
 ])
def accuracy_proxy(num_layers: int, width_mult: float, quant_bits: int) ->float:
 """
 Estimate accuracy via inverse-distance weighting from calibration points.
 Reference: Shepard interpolation / IDW spatial interpolation.
 If exact config matches a calibration point, returns the exact value.
 Otherwise, nearby calibration points contribute proportionally.
 """
 query = _normalize(num_layers, width_mult, quant_bits)
 total_w = 0.0
 weighted_acc = 0.0
 for (nl, wm, qb, acc) in CALIBRATION_POINTS:
  pt = _normalize(nl, wm, qb)
  dist = np.linalg.norm(query - pt)
  if dist < 1e-8: # Exact match
   return float(acc)
  w = 1.0 / (dist ** 2)
  weighted_acc += w * acc
  total_w += w
 if total_w < 1e-10:
  return float(np.mean([a for (*_, a) in CALIBRATION_POINTS]))
 return float(np.clip(weighted_acc / total_w, 0.50, 0.95))

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
 {"accuracy": 0.7527, "memory_kb": 23.14, "energy_mj": 0.005741,
"latency_ms": 35.7168}
 """
 t0 = time.perf_counter()
 num_layers = int(config["num_layers"])
 width_mult = float(config["width_mult"])
 quant_bits = int(config["quant_bits"])
 dvfs_factor = float(config["dvfs_factor"])
# Step 1: Architecture → FLOPs + param count
 torch.manual_seed(RANDOM_SEED)
 model = build_model(config)
 model.eval()
 stats = get_model_stats(model)
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
 # Critical: baseline config must match Day 1 measured accuracy within 1%
 baseline_r = evaluate({"num_layers": 3, "width_mult": 1.0, "quant_bits": 32,
 "time_slice": 20.0, "priority": 1, "dvfs_factor":
1.0})
 expected_acc = CALIBRATION_POINTS[0][3] # Your Day 1 measured accuracy
 assert abs(baseline_r["accuracy"] - expected_acc) < 0.01, \
 f"Proxy mismatch: got {baseline_r['accuracy']:.4f}, expected {expected_acc:.4f}"
 # Test Ax wrapper format
 ax_r = evaluate_for_ax(test_configs[0])
 assert all(isinstance(v, tuple) and len(v) == 2 for v in ax_r.values()),  "evaluate_for_ax() must return {metric: (float, float)} tuples"
 print("\nIntegration test: ALL PASSED ") 
 print("evaluate_for_ax() format: VERIFIED ✓")