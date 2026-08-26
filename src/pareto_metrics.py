"""
Shared multi-objective metrics: Pareto dominance, hypervolume, IGD.
Objectives: accuracy (maximize), energy_mj (minimize).
Used by moo_experiment.py, reference_front.py, robustness_check.py, and
plot_results.py so BO/Random/Grid are all scored identically.
CRITICAL: the hypervolume reference point must be IDENTICAL across every
method and seed, or hypervolumes are not comparable. It is read from
config.py (HV_REF_ENERGY_MJ, HV_REF_ACCURACY) and must never be derived
from observed data.
"""
import os, sys, logging
from typing import List, Dict, Tuple
import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import HV_REF_ENERGY_MJ, HV_REF_ACCURACY, LOG_FORMAT

logger = logging.getLogger(__name__)


def dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
 """
 a=(accuracy, energy_mj), b=(accuracy, energy_mj).
 a dominates b iff a is at least as good on both objectives and strictly
 better on at least one (accuracy: higher is better, energy: lower is
 better).
 """
 acc_a, e_a = a
 acc_b, e_b = b
 not_worse = acc_a >= acc_b and e_a <= e_b
 strictly_better = acc_a > acc_b or e_a < e_b
 return not_worse and strictly_better


def pareto_front(rows: List[Dict]) -> List[Dict]:
 """
 Non-dominated subset of rows, in the (accuracy, energy_mj) plane.
 Assumes rows are already feasibility-filtered by the caller.

 O(n log n) sort-and-sweep rather than the naive O(n^2) all-pairs
 comparison: sort by accuracy descending, then energy ascending within
 ties, and sweep tracking the minimum energy seen among strictly
 higher-accuracy groups. A point survives iff its group's minimum
 energy beats every previous (strictly higher accuracy) group's
 minimum. Exact duplicates (same accuracy AND same energy) are all
 kept, matching pairwise-dominance semantics, since neither dominates
 the other. This scale matters: reference-front enumeration produces
 tens of thousands of candidate points, where O(n^2) is impractically
 slow.
 """
 if not rows:
  return []
 acc = np.array([float(r["accuracy"]) for r in rows])
 energy = np.array([float(r["energy_mj"]) for r in rows])
 order = np.lexsort((energy, -acc)) # primary: acc desc, secondary: energy asc
 sorted_acc = acc[order]
 sorted_energy = energy[order]

 keep_mask = np.zeros(len(rows), dtype=bool)
 best_energy_so_far = np.inf
 n = len(rows)
 i = 0
 while i < n:
  j = i
  while j < n and sorted_acc[j] == sorted_acc[i]:
   j += 1
  group_min_energy = sorted_energy[i:j].min() # group sorted asc by energy -> == sorted_energy[i]
  if group_min_energy < best_energy_so_far:
   for k in range(i, j):
    if sorted_energy[k] == group_min_energy:
     keep_mask[order[k]] = True
   best_energy_so_far = group_min_energy
  i = j
 return [r for r, keep in zip(rows, keep_mask) if keep]


def _ref_point_tensor() -> torch.Tensor:
 """
 (accuracy, -energy) in maximize-both convention — same column order
 used everywhere in this module. Fixed by config — never derived from
 observed data.
 """
 return torch.tensor([HV_REF_ACCURACY, -HV_REF_ENERGY_MJ], dtype=torch.double)


def hypervolume(rows: List[Dict], ref_point: torch.Tensor = None) -> float:
 """
 Hypervolume dominated by the feasible rows' (accuracy, energy_mj)
 front, relative to a fixed reference point. Rows need not be
 pre-filtered to the Pareto front — the algorithm is correct (if less
 efficient) with dominated points included.
 Returns 0.0 for an empty or fully-dominated-by-ref-point input.
 """
 if not rows:
  return 0.0
 if ref_point is None:
  ref_point = _ref_point_tensor()
 pareto_Y = torch.tensor(
  [[float(r["accuracy"]), -float(r["energy_mj"])] for r in rows],
  dtype=torch.double,
 )
 hv = Hypervolume(ref_point=ref_point)
 return float(hv.compute(pareto_Y))


def hv_trace(rows: List[Dict], ref_point: torch.Tensor = None) -> List[float]:
 """
 Running hypervolume after each trial (rows assumed in trial order).
 Only feasible rows contribute; a trial's HV equals the previous
 trial's HV if that trial was infeasible. Monotone non-decreasing by
 construction — adding a point can never shrink the dominated region.
 """
 if ref_point is None:
  ref_point = _ref_point_tensor()
 trace = []
 feasible_so_far = []
 for r in rows:
  if r.get("is_feasible") in (True, "True", "true", "1", 1):
   feasible_so_far.append(r)
  trace.append(hypervolume(feasible_so_far, ref_point))
 return trace


def igd(front: List[Dict], reference_front: List[Dict]) -> float:
 """
 Inverted Generational Distance: for each point in reference_front,
 the distance to its nearest neighbour in `front`, averaged. Lower is
 better (0 = front covers the reference front exactly).
 Objectives are min-max normalized by the reference front's own range
 before distance is computed, so accuracy and energy contribute
 comparably despite different units/scales.
 """
 if not front or not reference_front:
  return float("nan")
 ref_pts = np.array([[float(r["accuracy"]), float(r["energy_mj"])] for r in reference_front])
 front_pts = np.array([[float(r["accuracy"]), float(r["energy_mj"])] for r in front])
 lo = ref_pts.min(axis=0)
 span = np.maximum(ref_pts.max(axis=0) - lo, 1e-12)
 ref_norm = (ref_pts - lo) / span
 front_norm = (front_pts - lo) / span
 dists = np.min(
  np.linalg.norm(ref_norm[:, None, :] - front_norm[None, :, :], axis=-1),
  axis=1,
 )
 return float(np.mean(dists))


if __name__ == "__main__":
 logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
 print("\n=== pareto_metrics self-test ===\n")

 # dominates() on hand-worked cases
 assert dominates((0.90, 0.001), (0.80, 0.002)), "better on both must dominate"
 assert dominates((0.90, 0.002), (0.80, 0.002)), "better accuracy, equal energy must dominate"
 assert not dominates((0.80, 0.002), (0.90, 0.001)), "worse on both must not dominate"
 assert not dominates((0.90, 0.003), (0.80, 0.002)), "better acc but worse energy must not dominate"
 print("dominates(): PASSED")

 # pareto_front() on a known 5-point set
 rows = [
  {"accuracy": 0.90, "energy_mj": 0.010}, # non-dominated (best acc)
  {"accuracy": 0.85, "energy_mj": 0.005}, # non-dominated (tradeoff)
  {"accuracy": 0.70, "energy_mj": 0.002}, # non-dominated (best energy)
  {"accuracy": 0.80, "energy_mj": 0.006}, # dominated by row 1 (0.90,0.010)? NO: worse energy AND worse acc vs row1? acc0.80<0.90 but energy0.006<0.010 -> not dominated by row1. Check vs row2 (0.85,0.005): acc0.80<0.85 AND energy0.006>0.005 -> dominated by row2.
  {"accuracy": 0.60, "energy_mj": 0.020}, # dominated by everything
 ]
 front = pareto_front(rows)
 front_pts = {(r["accuracy"], r["energy_mj"]) for r in front}
 expected = {(0.90, 0.010), (0.85, 0.005), (0.70, 0.002)}
 assert front_pts == expected, f"pareto_front mismatch: got {front_pts}, expected {expected}"
 print(f"pareto_front(): PASSED ({len(front)}/{len(rows)} non-dominated)")

 # hypervolume() of a simple case with a known reference point
 ref = torch.tensor([0.70, -0.10], dtype=torch.double) # accuracy, -energy
 simple = [{"accuracy": 0.90, "energy_mj": 0.05}]
 hv = hypervolume(simple, ref)
 expected_hv = (0.90 - 0.70) * (0.10 - 0.05) # box area under maximize convention
 assert abs(hv - expected_hv) < 1e-9, f"hv mismatch: got {hv}, expected {expected_hv}"
 print(f"hypervolume(): PASSED (single-point box area = {hv:.6f})")

 # hv_trace() must be monotone non-decreasing
 trace_rows = [
  {"accuracy": 0.80, "energy_mj": 0.02, "is_feasible": True},
  {"accuracy": 0.75, "energy_mj": 0.03, "is_feasible": True}, # dominated, HV unchanged
  {"accuracy": 0.85, "energy_mj": 0.01, "is_feasible": True}, # improves HV
  {"accuracy": 0.99, "energy_mj": 0.20, "is_feasible": False}, # infeasible, HV unchanged
 ]
 trace = hv_trace(trace_rows, ref)
 diffs = np.diff(trace)
 assert (diffs >= -1e-12).all(), f"hv_trace not monotone: {trace}"
 print(f"hv_trace(): PASSED (monotone non-decreasing: {[round(t,6) for t in trace]})")

 print("\nAll pareto_metrics checks: PASSED ✓")
