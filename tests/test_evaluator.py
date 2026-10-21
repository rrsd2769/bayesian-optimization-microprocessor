"""evaluate(design, scenario) -> Outcome, tested through the public interface only."""
import math

import pandas as pd
import pytest

from src.evaluator import (AccuracyTable, CycleCalibration, Design, Limits, MemoryCalibration, PeriodicTask, Scenario,
                           Workload, evaluate)
from src.network.arch import LAYER_COUNTS, WIDTH_GRID

MEM = MemoryCalibration(kflash8=(1000.0, 10.0, 100.0), kflash32=(500.0, 20.0), kram8=(200.0, 5.0), kram32=(300.0, 6.0),
                        act8=(0.5, 100.0), act32=2.0, rtos_reserve=32768)


def table(acc=0.9, overrides=None):
    rows = [(f"L{L}_w{w:g}", 1, b, acc, 50, 1.0) for L in LAYER_COUNTS for w in WIDTH_GRID for b in (4, 8, 16, 32)]
    df = pd.DataFrame(rows, columns=["arch", "seed", "b", "val_acc", "epochs", "wall_s"])
    for (arch, b), v in (overrides or {}).items():
        df.loc[(df["arch"] == arch) & (df["b"] == b), "val_acc"] = v
    return AccuracyTable(df)


def scenario(**kw):
    kw.setdefault("accuracy", table())
    kw.setdefault("memory_cal", MEM)
    return Scenario(**kw)


def D(L=2, w=0.25, b=8, tau=5.0, pi="below", f=0):
    return Design(L, w, b, tau, pi, f)


# ── flash and SRAM, hand-computed for Arch(2, 0.25) ──────────────────────────
# conv 3->8: 216 w, 8 bias; conv 8->16: 1152 w, 16 bias; linear 16->128: 2048 w, 128 bias; linear 128->10: 1280 w, 10 bias
# weights 4696, bias 162. Conv outputs: 8*32*32 = 8192 and 16*16*16 = 4096 elements. L = 2, w = 0.25.

@pytest.mark.parametrize("b,expected", [
    (4, 4696 * 0.5 + 162 * 4 + 1000 + 10 * 2 + 100 * 0.25),
    (8, 4696 * 1 + 162 * 4 + 1000 + 10 * 2 + 100 * 0.25),
    (16, 4696 * 2 + 162 * 4 + 1000 + 10 * 2 + 100 * 0.25),
    (32, 4696 * 4 + 162 * 4 + 500 + 20 * 2),
])
def test_flash_hand_values(b, expected):
    assert evaluate(D(b=b), scenario()).flash_kb * 1024 == pytest.approx(expected)


@pytest.mark.parametrize("b,expected", [
    (8, 0.5 * (8 * 32 * 32 + 16 * 16 * 16) + 100 * 2 + 200 + 5 * 2),
    (16, 2 * (0.5 * (8 * 32 * 32 + 16 * 16 * 16) + 100 * 2) + 200 + 5 * 2),
    (32, 2.0 * 8 * 32 * 32 + 300 + 6 * 2),
])
def test_sram_hand_values(b, expected):
    assert evaluate(D(b=b), scenario()).sram_kb * 1024 == pytest.approx(expected + 32768)


def test_int4_sram_equals_int8_sram_and_memory_grows_with_bits():
    sc = scenario()
    assert evaluate(D(b=4), sc).sram_kb == evaluate(D(b=8), sc).sram_kb
    f = [evaluate(D(4, 0.625, b), sc).flash_kb for b in (4, 8, 16, 32)]
    assert f == sorted(f)


# ── cycles, seen through latency: one request, no periodic tasks, free kernel ──
# Arch(2, 0.25) at int8, cpm8 = 2, c_fixed = 1000: macs 221184 + 294912 + 2048 + 1280 = 519424
# cycles = 519424 * 2 + 1000 = 1,039,848; at 30 MHz that is 34.6616 ms, plus up to 1 ms until the next tick.

def test_latency_equals_cycles_over_clock_plus_tick_wait():
    wl = Workload(periodic=(), n_requests=1, jitter=0.0, switch_cycles=0, tick_cycles=0)
    sc = scenario(workload=wl, cycle_cal=CycleCalibration(cpm8=2.0, c_layer=0.0, c_fixed=1000.0))
    o = evaluate(D(b=8, f=0), sc)
    run_ms = 1_039_848 / 30e6 * 1000
    assert run_ms <= o.latency_ms < run_ms + 1.0 + 1e-9


def test_higher_clock_lowers_latency_and_changes_energy():
    sc = scenario()
    lat = [evaluate(D(3, 0.625, 8, f=f), sc).latency_ms for f in range(5)]
    assert lat == sorted(lat, reverse=True)
    assert len({evaluate(D(3, 0.625, 8, f=f), sc).energy_uj for f in range(5)}) == 5


# ── constraints ──────────────────────────────────────────────────────────────

def test_memory_infeasible_design_short_circuits():
    o = evaluate(D(5, 1.0, 32), Scenario(accuracy=table()))     # ~4 MB of fp32 weights
    assert not o.acceptable and "flash" in o.violated
    assert o.latency_ms == math.inf and o.miss_ratio == 1.0 and o.energy_uj == math.inf
    assert o.flash_kb > 512


def test_sram_violation_alone():
    sc = scenario(memory_cal=MemoryCalibration(rtos_reserve=200_000, kflash8=(0.0, 0.0, 0.0), kflash32=(0.0, 0.0)))
    o = evaluate(D(), sc)
    assert o.violated == ("sram",) and o.sram_kb > 128 and o.latency_ms == math.inf


def test_accuracy_floor_is_a_constraint():
    sc = scenario(accuracy=table(0.9, {("L2_w0.25", 8): 0.7}))
    o = evaluate(D(b=8, f=4), sc)
    assert "accuracy" in o.violated and not o.acceptable and math.isfinite(o.energy_uj)


def test_acceptable_design_has_six_finite_outputs():
    o = evaluate(D(2, 0.25, 8, 5.0, "below", 4), scenario())
    assert o.acceptable and o.violated == ()
    for v in (o.accuracy, o.energy_uj, o.latency_ms, o.miss_ratio, o.flash_kb, o.sram_kb):
        assert math.isfinite(v)


def test_latency_and_miss_limits_are_reported():
    heavy = D(2, 0.25, 8, 5.0, "above", 0)
    sc = scenario(workload=Workload(periodic=(PeriodicTask("sensor", 10, 90_000, 3),)))
    o = evaluate(heavy, sc.with_limits(latency_ms=1.0))
    assert "latency" in o.violated
    o2 = evaluate(heavy, sc.with_limits(latency_ms=1e6, miss_ratio=0.0))
    assert o2.miss_ratio > 0 and "miss_ratio" in o2.violated


def test_latency_statistic_is_selectable():
    sc = scenario()
    d = D(3, 0.625, 8, 5.0, "below", 4)
    p95 = evaluate(d, sc).latency_ms
    mean = evaluate(d, sc.with_limits(latency_stat="mean")).latency_ms
    assert mean < p95
    with pytest.raises(ValueError):
        Limits(latency_stat="max")


# ── determinism, scenario control, errors ─────────────────────────────────────

def test_evaluate_is_deterministic():
    sc, d = scenario(), D(3, 0.625, 8, 7.3, "equal", 3)
    assert evaluate(d, sc) == evaluate(d, sc)


def test_changing_one_scenario_field_changes_the_outcome_without_code_changes():
    sc, d = scenario(), D(3, 0.625, 8, 2.0, "equal", 4)
    base = evaluate(d, sc)
    assert evaluate(d, sc.with_kernel_scale(2.0)) != base
    assert evaluate(d, sc.but(idle_mode="stop")).energy_uj < base.energy_uj
    assert evaluate(d, sc.with_overhead_scale(r16=2.0)) == base            # int8 design: r16 not involved
    d16 = D(3, 0.625, 16, 2.0, "equal", 4)
    assert evaluate(d16, sc.with_overhead_scale(r16=2.0)).latency_ms > evaluate(d16, sc).latency_ms


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        evaluate(D(f=5), scenario())
    with pytest.raises(ValueError):
        Design(2, 0.25, 8, 5.0, "top", 0)
    with pytest.raises(ValueError):
        evaluate(D(), Scenario(memory_cal=MEM))        # no accuracy table
    with pytest.raises(ValueError):
        evaluate(D(w=0.3), scenario())                 # off the width grid


# ── accuracy table ───────────────────────────────────────────────────────────

def _acc_table():
    rows = [(f"L{L}_w{w:g}", 1, b, 0.5 + 0.01 * L + 0.1 * w + 0.001 * b, 50, 1.0)
            for L in (2, 3) for w in (0.25, 0.4375) for b in (4, 8, 16, 32)]
    rows.append(("L2_w0.25", 2, 32, 0.123, 50, 1.0))      # another seed: not in the table
    return AccuracyTable(pd.DataFrame(rows, columns=["arch", "seed", "b", "val_acc", "epochs", "wall_s"]))


def test_accuracy_lookup_returns_measured_values_exactly():
    t = _acc_table()
    assert len(t) == 16
    for L in (2, 3):
        for w in (0.25, 0.4375):
            for b in (4, 8, 16, 32):
                assert t.acc(L, w, b) == 0.5 + 0.01 * L + 0.1 * w + 0.001 * b
    assert t.acc(2, 0.25, 32) != 0.123


def test_accuracy_lookup_raises_off_grid_and_for_unmeasured():
    t = _acc_table()
    with pytest.raises(ValueError):
        t.acc(2, 0.3, 8)
    with pytest.raises(ValueError):
        t.acc(6, 0.25, 8)
    with pytest.raises(ValueError):
        t.acc(2, 0.25, 12)
    with pytest.raises(KeyError):
        t.acc(4, 0.25, 8)
