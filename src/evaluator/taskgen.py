"""
Generated periodic task sets for the Phase 5 sensitivity runs (Kim 2021, J. Comput. Sci. Eng. 15(2):72-77,
10.5626/jcse.2021.15.2.72, method harmonicT2, MATLAB code in its Fig. 2d):

    utilizations   UUniFast (Bini and Buttazzo 2005): n values, uniformly distributed, summing to U
    periods        T = f1 * f2 * f3, each f_k drawn uniformly from row k of [1 2 4; 1 5 10; 1 6 12],
                   redrawn until T >= 3 (the paper's rule; here T is in milliseconds)
    execution      C = T x U_i

Choices the paper does not fix (made by the user for this study): n = 2 tasks, total utilization defined at a
reference clock of 180 MHz (C in ms is converted to cycles at that clock, so slower clocks load the core more),
deadlines equal to periods (as in the nominal workload), rate-monotonic priorities (shorter period, higher
priority). The inference priorities follow: below = 1, equal = the lowest periodic priority (2), above = n + 2.
The same seed is used for every utilization level, so the periods are identical and only the loads scale.
"""
from dataclasses import replace

import numpy as np

from src.evaluator.rtos_sim import PeriodicTask, Workload

FACTORS = ((1, 2, 4), (1, 5, 10), (1, 6, 12))
MIN_PERIOD_MS = 3


def uunifast(n: int, total: float, rng: np.random.Generator) -> list:
    out, rest = [], total
    for i in range(1, n):
        nxt = rest * rng.random() ** (1.0 / (n - i))
        out.append(rest - nxt)
        rest = nxt
    return out + [rest]


def harmonic_period(rng: np.random.Generator) -> int:
    while True:
        t = 1
        for row in FACTORS:
            t *= row[int(rng.integers(0, 3))]
        if t >= MIN_PERIOD_MS:
            return t


def generated_workload(base: Workload, n: int, utilization: float, f_ref_hz: float, seed: int) -> Workload:
    """`base` with its periodic tasks replaced by a harmonicT2 set of total utilization `utilization` at f_ref_hz."""
    rng = np.random.default_rng(seed)
    utils = uunifast(n, utilization, rng)
    periods = [harmonic_period(rng) for _ in range(n)]
    order = sorted(range(n), key=lambda i: (periods[i], i))          # rate monotonic: shortest period first
    tasks = tuple(PeriodicTask(f"gen{rank + 1}", periods[i], max(1, round(periods[i] * utils[i] * f_ref_hz / 1000.0)),
                               n + 1 - rank)
                  for rank, i in enumerate(order))
    return replace(base, periodic=tasks, inference_priority=(("below", 1), ("equal", 2), ("above", n + 2)))
