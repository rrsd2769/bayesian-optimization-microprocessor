"""
SimPy discrete-event simulation of ARM Cortex-M4 CPU scheduling.
Models: Round Robin and Priority Scheduling.
Outputs: mean_latency_ms, p95_latency_ms, energy_mj, throughput_tps
"""
import os, sys, logging
from dataclasses import dataclass, field
from typing import List
import simpy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (SIM_CLOCK_HZ, SIM_NUM_TASKS, SIM_ARRIVAL_RATE,
 RANDOM_SEED, LOG_FORMAT)
from src.energy_proxy import estimate_energy_mj
logger = logging.getLogger(__name__)
@dataclass
class Task:
 task_id: int; arrival_time: float; priority: int; flops: int
 start_time: float = -1.0; finish_time: float = -1.0
 @property
 def latency_ms(self) -> float:
  if self.finish_time < 0: return -1.0
  return (self.finish_time - self.arrival_time) * 1000.0
@dataclass
class SimResult:
 policy: str; time_slice_ms: float; priority_level: int; dvfs_factor: float
 latencies_ms: List[float] = field(default_factory=list)
 total_energy_mj: float = 0.0
 @property
 def mean_latency_ms(self): return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0
 @property
 def p95_latency_ms(self): return float(np.percentile(self.latencies_ms,
95)) if self.latencies_ms else 0.0
 @property
 def throughput_tps(self):
  if not self.latencies_ms: return 0.0
  t = max(self.latencies_ms) / 1000.0
  return len(self.latencies_ms) / t if t > 0 else 0.0
 def to_dict(self):
  return {"policy": self.policy, "time_slice_ms": self.time_slice_ms,
 "mean_latency_ms": round(self.mean_latency_ms, 4),
 "p95_latency_ms": round(self.p95_latency_ms, 4),
 "throughput_tps": round(self.throughput_tps, 4),
 "energy_mj": round(self.total_energy_mj, 6),
 "num_tasks": len(self.latencies_ms)}
def _make_tasks(rng, flops: int, dvfs_factor: float) -> List[Task]:
 """Generate SIM_NUM_TASKS inference tasks with Poisson arrivals."""
 arrivals = np.cumsum(rng.exponential(SIM_ARRIVAL_RATE, SIM_NUM_TASKS))
 task_flops = [max(1, int(flops * rng.uniform(0.95, 1.05))) for _ in
range(SIM_NUM_TASKS)]
 return [Task(i, float(arrivals[i]), 0, task_flops[i]) for i in
range(SIM_NUM_TASKS)]
def _run_rr(env, cpu, tasks, result, time_slice_s, dvfs_factor, quant_bits):
 eff_clock = SIM_CLOCK_HZ * dvfs_factor
 for task in tasks:
  yield env.timeout(max(0, task.arrival_time - env.now))
  def rr_proc(t):
   remaining = t.flops; t.start_time = env.now
   while remaining > 0:
    with cpu.request() as req:
     yield req
     slice_f = min(remaining, int(eff_clock * time_slice_s))
     yield env.timeout(slice_f / eff_clock)
     remaining -= slice_f
   t.finish_time = env.now
   result.latencies_ms.append(t.latency_ms)
   result.total_energy_mj += estimate_energy_mj(t.flops, dvfs_factor, quant_bits)
  env.process(rr_proc(task))
def _run_priority(env, cpu, tasks, result, dvfs_factor, quant_bits):
 eff_clock = SIM_CLOCK_HZ * dvfs_factor
 for task in tasks:
  yield env.timeout(max(0, task.arrival_time - env.now))
  def pri_proc(t):
   with cpu.request(priority=t.priority) as req:
    yield req; t.start_time = env.now
    yield env.timeout(t.flops / eff_clock)
    t.finish_time = env.now
   result.latencies_ms.append(t.latency_ms)
   result.total_energy_mj += estimate_energy_mj(t.flops, dvfs_factor, quant_bits)
  env.process(pri_proc(task))
def run_scheduler_sim(config: dict, flops: int) -> dict:
 """
 Run scheduling simulation for a config dict.
 Called by evaluate() in objective.py.
 Args:
 config: must contain time_slice, priority, dvfs_factor
 flops: FLOPs of the model being scheduled
 Returns:
 dict: mean_latency_ms, p95_latency_ms, energy_mj, throughput_tps,
num_tasks
 """
 rng = np.random.default_rng(RANDOM_SEED)
 time_slice_ms = float(config["time_slice"])
 priority = int(config["priority"])
 dvfs_factor = float(config["dvfs_factor"])
 quant_bits = int(config["quant_bits"])
 policy = "priority" if priority == 0 else "round_robin"
 tasks = _make_tasks(rng, flops, dvfs_factor)
 env = simpy.Environment()
 result = SimResult(policy, time_slice_ms, priority, dvfs_factor)
 if policy == "round_robin":
  cpu = simpy.Resource(env, capacity=1)
  env.process(_run_rr(env, cpu, tasks, result, time_slice_ms/1000.0,dvfs_factor, quant_bits))
 else:
  cpu = simpy.PriorityResource(env, capacity=1)
  env.process(_run_priority(env, cpu, tasks, result, dvfs_factor, quant_bits))
 env.run()
 return result.to_dict()
if __name__ == "__main__":
 logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
 BASELINE_FLOPS = 3_096_576
 print("\n=== Scheduler Simulation Validation ===\n")
 cases = [
  ({"time_slice": 10.0, "priority": 1, "dvfs_factor": 1.0, "quant_bits": 32}, "RR full speed"),
  ({"time_slice": 10.0, "priority": 1, "dvfs_factor": 0.75, "quant_bits": 32}, "RR 75%"),
  ({"time_slice": 10.0, "priority": 1, "dvfs_factor": 0.5, "quant_bits": 32}, "RR 50%"),
  ({"time_slice": 20.0, "priority": 0, "dvfs_factor": 1.0, "quant_bits": 32}, "Priority full"),
  ({"time_slice": 50.0, "priority": 2, "dvfs_factor": 0.5, "quant_bits": 32}, "RR slow lowprio"),
 ]
 print(f"{'Config':<25} {'Latency(ms)':<14} {'Energy(mJ)':<14} {'Thruput'}")
 print("-" * 70)
 for cfg, desc in cases:
  r = run_scheduler_sim(cfg, BASELINE_FLOPS)
  print(f"{desc:<25} {r['mean_latency_ms']:<14.4f} {r['energy_mj']:<14.4f} {r['throughput_tps']:.2f}")
 # Sanity: lower DVFS → lower energy AND higher latency
  r_full = run_scheduler_sim({"time_slice": 10.0, "priority": 1,
"dvfs_factor": 1.0, "quant_bits": 32}, BASELINE_FLOPS)
  r_half = run_scheduler_sim({"time_slice": 10.0, "priority": 1,
"dvfs_factor": 0.5, "quant_bits": 32}, BASELINE_FLOPS)
  assert r_half["energy_mj"] < r_full["energy_mj"], "FAIL: lower DVFS should reduce energy"
  assert r_half["mean_latency_ms"] > r_full["mean_latency_ms"], "FAIL: lower DVFS should increase latency"
 # Sanity: lower quant_bits → lower energy, latency unaffected (bits don't change clock cycles)
  r_bits32 = run_scheduler_sim({"time_slice": 10.0, "priority": 1,
"dvfs_factor": 1.0, "quant_bits": 32}, BASELINE_FLOPS)
  r_bits8 = run_scheduler_sim({"time_slice": 10.0, "priority": 1,
"dvfs_factor": 1.0, "quant_bits": 8}, BASELINE_FLOPS)
  assert r_bits8["energy_mj"] < r_bits32["energy_mj"], "FAIL: lower quant_bits should reduce energy"
print("\nSanity checks: PASSED ✓")