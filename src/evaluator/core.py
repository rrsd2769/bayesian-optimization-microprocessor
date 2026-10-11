"""
evaluate(design, scenario) -> Outcome: the one function the search code calls.

A design is (L, w, b, tau, pi, f_idx). The pipeline inside is
    Arch(L, w).profile() -> flash and SRAM (memory) -> measured accuracy (table)
    -> if flash or SRAM exceeds the limit: stop, latency inf, miss ratio 1, energy inf (no simulation)
    -> cycles at the clock's flash wait states -> RTOS simulation -> energy over the simulated horizon,
       divided by the number of inference requests.
Everything constant lives in the Scenario. The simulation cache is keyed by (workload, cycles, f, tau, pi) and is
invisible to callers.
"""
import math
from dataclasses import dataclass
from functools import lru_cache

from src.evaluator.cycles import inference_cycles
from src.evaluator.energy import total_energy_uj
from src.evaluator.memory import flash_bytes, sram_bytes
from src.evaluator.rtos_sim import PI_LEVELS, simulate
from src.evaluator.scenario import Scenario
from src.network.arch import Arch


@dataclass(frozen=True)
class Design:
    num_layers: int
    width_mult: float
    quant_bits: int
    time_slice_ms: float
    priority: str            # one of PI_LEVELS
    freq_idx: int            # index into the device's operating points

    def __post_init__(self):
        if self.priority not in PI_LEVELS:
            raise ValueError(f"priority must be one of {PI_LEVELS}, got {self.priority}")


@dataclass(frozen=True)
class Outcome:
    accuracy: float
    energy_uj: float
    latency_ms: float
    miss_ratio: float
    flash_kb: float
    sram_kb: float
    acceptable: bool
    violated: tuple          # names of the broken constraints: accuracy, flash, sram, latency, miss_ratio


@lru_cache(maxsize=None)
def _profile(num_layers: int, width_mult: float) -> list:
    return Arch(num_layers, width_mult).profile()


@lru_cache(maxsize=None)
def _simulate(workload, cycles: int, f_hz: int, tau_ms: float, pi: str):
    return simulate(cycles, f_hz, tau_ms, pi, workload)


def evaluate(design: Design, scenario: Scenario = None) -> Outcome:
    sc = scenario if scenario is not None else Scenario.nominal()
    if sc.accuracy is None:
        raise ValueError("the scenario has no accuracy table: run stage 'accuracy' or pass Scenario(accuracy=...)")
    if not 0 <= design.freq_idx < len(sc.device.points):
        raise ValueError(f"freq_idx must be in 0..{len(sc.device.points) - 1}, got {design.freq_idx}")
    lim, b = sc.limits, design.quant_bits
    profile = _profile(design.num_layers, design.width_mult)
    flash = flash_bytes(profile, b, sc.memory_cal)
    sram = sram_bytes(profile, b, sc.memory_cal)
    acc = sc.accuracy.acc(design.num_layers, design.width_mult, b)
    violated = [name for name, bad in (("accuracy", acc < lim.accuracy_floor), ("flash", flash > lim.flash_bytes),
                                       ("sram", sram > lim.sram_bytes)) if bad]
    if "flash" in violated or "sram" in violated:
        return Outcome(acc, math.inf, math.inf, 1.0, flash / 1024.0, sram / 1024.0, False, tuple(violated))
    point = sc.device.points[design.freq_idx]
    cycles = inference_cycles(profile, b, point.flash_ws, sc.cycle_cal)
    sim = _simulate(sc.workload, cycles, int(point.f_hz), design.time_slice_ms, design.priority)
    energy = total_energy_uj(sc.device, point, sim.active_cycles, sim.horizon_s, sc.idle_mode) / sc.workload.n_requests
    latency = sim.p95_latency_ms if lim.latency_stat == "p95" else sim.mean_latency_ms
    violated += [name for name, bad in (("latency", latency > lim.latency_ms), ("miss_ratio", sim.miss_ratio > lim.miss_ratio)) if bad]
    return Outcome(acc, energy, latency, sim.miss_ratio, flash / 1024.0, sram / 1024.0, not violated, tuple(violated))
