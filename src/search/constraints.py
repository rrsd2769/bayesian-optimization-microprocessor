"""
The constraint list every search method sees. This is the one place to change it.

A constraint is a name, a value read from an Outcome and an upper bound from the scenario's Limits. Flash and SRAM
are merged into one memory-utilisation constraint, mem_util = max(flash / limit, sram / limit) <= 1 (the plan's
fallback for a slow qNEHVI pilot; the pilot showed 5-30 s per Gaussian-process step with six modelled outputs).
The accuracy floor is a constraint too (acc_gap = floor - accuracy <= 0, a separate metric because Ax does not allow
a constraint on an objective metric), so every method searches the same feasible set the reference front and the
hypervolume use. Accuracy is also an objective.
"""
import math
from dataclasses import dataclass
from typing import Callable

from src.evaluator import Limits, Outcome

VIOLATION_CAP = 1e3     # stands in for an infinite violation (memory-infeasible designs have latency inf)


@dataclass(frozen=True)
class Constraint:
    name: str
    value: Callable[[Outcome, Limits], float]
    bound: Callable[[Limits], float]
    scale: Callable[[Limits], float]      # violation = (value - bound) / scale, so violations are unit-free

    def violation(self, o: Outcome, limits: Limits) -> float:
        v = self.value(o, limits)
        if math.isinf(v):
            return VIOLATION_CAP
        return min((v - self.bound(limits)) / self.scale(limits), VIOLATION_CAP)


def _mem_util(o: Outcome, l: Limits) -> float:
    return max(o.flash_kb / (l.flash_bytes / 1024.0), o.sram_kb / (l.sram_bytes / 1024.0))


CONSTRAINTS = (
    Constraint("latency", lambda o, l: o.latency_ms, lambda l: l.latency_ms, lambda l: l.latency_ms),
    Constraint("miss_ratio", lambda o, l: o.miss_ratio, lambda l: l.miss_ratio, lambda l: max(l.miss_ratio, 0.01)),
    Constraint("mem_util", _mem_util, lambda l: 1.0, lambda l: 1.0),
    Constraint("acc_gap", lambda o, l: l.accuracy_floor - o.accuracy, lambda l: 0.0, lambda l: l.accuracy_floor),
)


def violations(o: Outcome, limits: Limits) -> list:
    """One number per constraint, <= 0 when satisfied."""
    return [c.violation(o, limits) for c in CONSTRAINTS]


def values(o: Outcome, limits: Limits) -> dict:
    """Raw constraint values by name (latency is inf for memory-infeasible designs)."""
    return {c.name: c.value(o, limits) for c in CONSTRAINTS}
