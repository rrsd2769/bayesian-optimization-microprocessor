"""
Scenario: every science constant the evaluator uses, as one frozen value.

Cost models take their constants from a Scenario, never from module-level constants, so the sensitivity
runs (kernel costs x0.5 / x2, other task sets, Stop-mode idle, other limits) are new Scenario values and
need no code changes. The workload (task set and kernel costs) is added in Phase 3.
"""
from dataclasses import dataclass, field, replace

from src.evaluator.cycles import CycleCalibration
from src.evaluator.device import STM32F446RE, Device
from src.evaluator.energy import IDLE_MODES
from src.evaluator.memory import MemoryCalibration


@dataclass(frozen=True)
class Limits:
    latency_ms: float = 100.0
    miss_ratio: float = 0.01
    flash_bytes: int = STM32F446RE.flash_bytes
    sram_bytes: int = STM32F446RE.sram_bytes
    accuracy_floor: float = 0.75     # scoring floor, as in v1


@dataclass(frozen=True)
class Scenario:
    device: Device = STM32F446RE
    idle_mode: str = "sleep"
    limits: Limits = field(default_factory=Limits)
    cycle_cal: CycleCalibration = field(default_factory=CycleCalibration)
    memory_cal: MemoryCalibration = field(default_factory=MemoryCalibration)

    def __post_init__(self):
        if self.idle_mode not in IDLE_MODES:
            raise ValueError(f"idle_mode must be one of {IDLE_MODES}, got {self.idle_mode}")

    @staticmethod
    def nominal() -> "Scenario":
        """The paper's main case. Phase 2 makes this load the fitted calibration."""
        return Scenario()

    def but(self, **changes) -> "Scenario":
        """A sensitivity variant: same scenario with some fields replaced."""
        return replace(self, **changes)
