"""
RTOS simulator (Phase 3): hand-computed cases on small workloads, ordering properties on the nominal one,
determinism and speed.
"""
import math
import time
from dataclasses import replace

import pytest

from src.evaluator import energy
from src.evaluator.cycles import inference_cycles
from src.evaluator.rtos_sim import PI_LEVELS, PeriodicTask, Workload, _requests, simulate
from src.evaluator.scenario import Scenario
from src.network.arch import Arch

NOMINAL = Workload()
SW, TK = NOMINAL.switch_cycles, NOMINAL.tick_cycles
# the same requests, replayed, but with one request only and no jitter, for hand cases
ALONE = replace(NOMINAL, periodic=(), n_requests=1, jitter=0.0)


def test_kernel_costs_are_the_decided_values():
    assert (NOMINAL.switch_cycles, NOMINAL.tick_cycles) == (390, 100)     # derived from Ungurean 2020 / assumption
    half, double = NOMINAL.with_kernel_scale(0.5), NOMINAL.with_kernel_scale(2)
    assert (half.switch_cycles, half.tick_cycles, double.switch_cycles, double.tick_cycles) == (195, 50, 780, 200)


def test_inference_alone_latency_is_cycles_plus_one_tick_and_one_switch():
    f, cycles = 10_000_000, 3000           # tick = 10,000 cycles, so no tick falls inside the run
    arrivals, _ = _requests(ALONE)
    k = math.ceil(arrivals[0])             # the request is released at the next tick
    tick = f // 1000
    for pi in PI_LEVELS:
        r = simulate(cycles, f, 1, pi, ALONE)
        assert r.completed
        finish = k * tick + TK + SW + cycles
        assert r.mean_latency_ms == pytest.approx((finish - arrivals[0] * tick) / f * 1000)
        assert r.n_switches == 1 and r.miss_ratio == 0.0
        # one tick per ms up to and including tick k: that is k + 1 tick costs, then the switch and the work
        assert r.active_cycles == (k + 1) * TK + SW + cycles
        assert r.horizon_s == pytest.approx(finish / f)


def test_inference_above_everything_makes_periodic_tasks_miss():
    w = replace(NOMINAL, n_requests=3)
    r = simulate(10_000_000, 60_000_000, 1, "above", w)          # 167 ms of work vs a 10 ms sensor period
    assert r.miss_ratio > 0.1


def test_inference_below_causes_no_periodic_miss_while_utilization_is_below_one():
    w = replace(NOMINAL, n_requests=3)
    r = simulate(10_000_000, 60_000_000, 1, "below", w)
    sensor, comms = NOMINAL.periodic
    util = (sensor.cycles / (sensor.period_ms * 60_000) + comms.cycles / (comms.period_ms * 60_000))
    assert util < 1 and r.miss_ratio == 0.0


def test_smaller_slice_gives_more_switches_among_equal_priorities():
    w = replace(NOMINAL, n_requests=5)                            # inference ties with comms at priority 2
    a = simulate(10_000_000, 60_000_000, 1, "equal", w)
    b = simulate(10_000_000, 60_000_000, 20, "equal", w)
    assert a.n_switches > b.n_switches


def test_overlapping_requests_are_served_in_order_by_one_task():
    # no periodic tasks, requests of 20 ms at 60 MHz arriving about every 200 ms never overlap here; make them overlap
    w = replace(ALONE, n_requests=4, mean_interarrival_ms=5.0, jitter=0.0)
    fast = simulate(1_200_000, 60_000_000, 1, "equal", w)          # 20 ms of work each, arrivals about 5 ms apart
    slow = simulate(1_200_000, 60_000_000, 20, "equal", w)
    assert fast == slow                                             # nobody to share the slice with
    assert fast.n_switches == 1                                     # one dispatch; the task continues without switches
    assert fast.horizon_s * 1000 > 4 * 20                           # four 20 ms requests, strictly one after another


def test_slice_has_no_effect_without_an_equal_priority_competitor():
    w = replace(NOMINAL, n_requests=5)
    for pi in ("below", "above"):
        a = simulate(10_000_000, 60_000_000, 1, pi, w)
        b = simulate(10_000_000, 60_000_000, 20, pi, w)
        assert a.n_switches == b.n_switches and a.mean_latency_ms == b.mean_latency_ms


def test_the_three_priorities_give_three_distinct_results_on_a_mid_size_design():
    cycles = inference_cycles(Arch(3, 0.625).profile(), 8, 0, Scenario.nominal().cycle_cal)
    res = [simulate(cycles, 60_000_000, 5, pi, NOMINAL) for pi in PI_LEVELS]
    assert len({(r.mean_latency_ms, r.miss_ratio, r.n_switches) for r in res}) == 3


def test_bit_identical_across_calls_and_same_requests_for_every_design():
    a = simulate(5_000_000, 120_000_000, 3, "equal", NOMINAL)
    assert a == simulate(5_000_000, 120_000_000, 3, "equal", NOMINAL)
    assert _requests(NOMINAL)[0].tolist() == _requests(NOMINAL)[0].tolist()


def test_kernel_cost_scale_changes_active_cycles_only_upward():
    base = simulate(5_000_000, 120_000_000, 3, "equal", NOMINAL)
    more = simulate(5_000_000, 120_000_000, 3, "equal", NOMINAL.with_kernel_scale(2))
    less = simulate(5_000_000, 120_000_000, 3, "equal", NOMINAL.with_kernel_scale(0.5))
    assert less.active_cycles < base.active_cycles < more.active_cycles


def test_overload_stops_at_the_cap_and_reports_incomplete():
    w = replace(NOMINAL, max_horizon_s=5.0)
    r = simulate(135_807_251, 30_000_000, 1, "equal", w)          # 4.5 s per request, far beyond 5 s in total
    assert not r.completed and r.horizon_s <= 5.5 and r.mean_latency_ms > 1000


def test_periodic_deadlines_count_only_inside_the_horizon():
    w = replace(NOMINAL, periodic=(PeriodicTask("p", 10, 1000, 3),), n_requests=1)
    r = simulate(2_000_000, 60_000_000, 1, "below", w)
    assert r.miss_ratio == 0.0


def test_unknown_priority_level_raises():
    with pytest.raises(ValueError):
        simulate(1000, 60_000_000, 1, "middle", NOMINAL)


def test_worst_design_simulates_within_30_ms(capsys):
    cal = Scenario.nominal().cycle_cal
    cycles = inference_cycles(Arch(5, 1.0).profile(), 8, 0, cal)
    best = min(_timed(cycles, tau, pi) for tau in (1, 5, 20) for pi in PI_LEVELS)
    worst = max(_timed(cycles, tau, pi) for tau in (1, 5, 20) for pi in PI_LEVELS)
    print(f"\nL5_w1 int8 at 60 MHz: simulation takes {best * 1000:.1f} to {worst * 1000:.1f} ms")
    assert best <= 0.030


def _timed(cycles, tau, pi):
    t = []
    for _ in range(3):
        t0 = time.perf_counter()
        simulate(cycles, 60_000_000, tau, pi, NOMINAL)
        t.append(time.perf_counter() - t0)
    return min(t)


def test_design_table(capsys):
    """3 designs x 3 priorities x 3 slices at 60 MHz: latency, miss ratio, switches, energy per inference."""
    sc = Scenario.nominal()
    dev, point = sc.device, sc.device.points[1]
    print(f"\n{'design':<10}{'pi':<7}{'tau':>4}{'lat ms':>9}{'p95 ms':>9}{'miss':>7}{'switches':>9}{'E/inf uJ':>10}")
    for L, w in ((2, 0.25), (3, 0.625), (4, 1.0)):
        cycles = inference_cycles(Arch(L, w).profile(), 8, 0, sc.cycle_cal)
        for pi in PI_LEVELS:
            for tau in (1, 5, 20):
                r = simulate(cycles, point.f_hz, tau, pi, sc.workload)
                e = energy.energy_per_inference_uj(dev, point, r.active_cycles, r.horizon_s,
                                                   sc.workload.n_requests, sc.idle_mode)
                print(f"{Arch(L, w).name:<10}{pi:<7}{tau:>4}{r.mean_latency_ms:>9.1f}{r.p95_latency_ms:>9.1f}"
                      f"{r.miss_ratio:>7.3f}{r.n_switches:>9}{e:>10.0f}")
                assert r.completed and e > 0
