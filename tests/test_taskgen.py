"""Generated task sets (Kim 2021 harmonicT2): utilizations, periods, priorities, determinism, effect on evaluate()."""
import itertools

import numpy as np
import pytest

from src.evaluator import Design, Workload, evaluate
from src.evaluator.taskgen import FACTORS, generated_workload, harmonic_period, uunifast
from tests.test_evaluator import scenario

ALLOWED = {a * b * c for a, b, c in itertools.product(*FACTORS) if a * b * c >= 3}


def test_uunifast_sums_to_total_and_is_nonnegative():
    rng = np.random.default_rng(1)
    for n in (1, 2, 5):
        u = uunifast(n, 0.5, rng)
        assert len(u) == n and sum(u) == pytest.approx(0.5) and min(u) >= 0


def test_periods_are_products_of_the_three_factor_rows_and_at_least_3():
    rng = np.random.default_rng(2)
    ps = {harmonic_period(rng) for _ in range(500)}
    assert ps <= ALLOWED and min(ps) >= 3 and len(ps) > 5


def test_generated_workload_utilization_priorities_and_levels():
    base = Workload()
    lo, hi = generated_workload(base, 2, 0.2, 180e6, 7), generated_workload(base, 2, 0.5, 180e6, 7)
    for w, u in ((lo, 0.2), (hi, 0.5)):
        assert sum(t.cycles / (t.period_ms * 180e3) for t in w.periodic) == pytest.approx(u, rel=1e-3)
        assert [t.priority for t in w.periodic] == [3, 2]
        assert w.periodic[0].period_ms <= w.periodic[1].period_ms          # rate monotonic
        assert dict(w.inference_priority) == {"below": 1, "equal": 2, "above": 4}
    assert [t.period_ms for t in lo.periodic] == [t.period_ms for t in hi.periodic]       # same seed, same periods
    assert generated_workload(base, 2, 0.2, 180e6, 7) == lo


def test_scenario_with_generated_tasks_changes_the_outcome_and_loads_slow_clocks_more():
    sc = scenario()
    d = Design(3, 0.625, 8, 5.0, "below", 0)
    low, high = sc.with_generated_tasks(0.2), sc.with_generated_tasks(0.5)
    assert evaluate(d, low) != evaluate(d, sc) and evaluate(d, high).latency_ms >= evaluate(d, low).latency_ms
    slow = evaluate(Design(2, 0.25, 8, 5.0, "above", 0), high)
    assert slow.miss_ratio > 0       # 0.5 at 180 MHz is 3x overload at 30 MHz
