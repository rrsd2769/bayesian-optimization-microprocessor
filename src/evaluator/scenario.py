"""
Scenario: every science constant the evaluator uses, as one frozen value.

Cost models take their constants from a Scenario, never from module-level constants, so the sensitivity
runs (kernel costs x0.5 / x2, other task sets, Stop-mode idle, other limits) are new Scenario values and
need no code changes. The workload holds the task set and the kernel costs.
"""
from dataclasses import dataclass, field, replace

from src.evaluator.accuracy import AccuracyTable
from src.evaluator.cycles import CycleCalibration
from src.evaluator.device import STM32F446RE, Device
from src.evaluator.energy import IDLE_MODES
from src.evaluator.memory import MemoryCalibration
from src.evaluator.rtos_sim import Workload


@dataclass(frozen=True)
class Limits:
    latency_ms: float = 100.0
    miss_ratio: float = 0.01
    flash_bytes: int = STM32F446RE.flash_bytes
    sram_bytes: int = STM32F446RE.sram_bytes
    accuracy_floor: float = 0.75     # scoring floor, as in v1
    latency_stat: str = "p95"        # which inference latency the limit applies to: "p95" or "mean"

    def __post_init__(self):
        if self.latency_stat not in ("p95", "mean"):
            raise ValueError(f"latency_stat must be 'p95' or 'mean', got {self.latency_stat}")


@dataclass(frozen=True)
class Scenario:
    device: Device = STM32F446RE
    idle_mode: str = "sleep"
    limits: Limits = field(default_factory=Limits)
    cycle_cal: CycleCalibration = field(default_factory=CycleCalibration)
    memory_cal: MemoryCalibration = field(default_factory=MemoryCalibration)
    workload: Workload = field(default_factory=Workload)
    accuracy: AccuracyTable = None     # measured table; evaluate() raises without one

    def __post_init__(self):
        if self.idle_mode not in IDLE_MODES:
            raise ValueError(f"idle_mode must be one of {IDLE_MODES}, got {self.idle_mode}")

    @staticmethod
    def nominal() -> "Scenario":
        """The paper's main case, with the Phase 2 fitted calibration when the calibrate stage has run."""
        from src import results
        from src.evaluator.calibrate import from_json
        table = AccuracyTable(results.load("accuracy_measurements")) if results.exists("accuracy_measurements") else None
        if not (results.exists("cycle_calibration") and results.exists("memory_calibration")):
            return Scenario(accuracy=table)
        return Scenario(cycle_cal=from_json(CycleCalibration, results.load("cycle_calibration")),
                        memory_cal=from_json(MemoryCalibration, results.load("memory_calibration")), accuracy=table)

    def but(self, **changes) -> "Scenario":
        """A sensitivity variant: same scenario with some fields replaced."""
        return replace(self, **changes)

    def with_limits(self, **changes) -> "Scenario":
        return replace(self, limits=replace(self.limits, **changes))

    def with_kernel_scale(self, k: float) -> "Scenario":
        """Context-switch and tick costs times k."""
        return replace(self, workload=self.workload.with_kernel_scale(k))

    def with_overhead_scale(self, *, r16: float = 1.0, r4: float = 1.0) -> "Scenario":
        """Scale the literature overheads r16 - 1 and r4 - 1 (int16 over int8, int4 over int8) by the given factors."""
        c = self.cycle_cal
        return replace(self, cycle_cal=replace(c, r16=1 + r16 * (c.r16 - 1), r4=1 + r4 * (c.r4 - 1)))
