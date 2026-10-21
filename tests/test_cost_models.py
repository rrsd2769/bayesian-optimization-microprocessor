"""
Internal seam of the evaluator (Phase 1): hand-computed cases for the cost-model formulas themselves, with
calibrations that evaluate() cannot expose (cycles, energy), plus datasheet constant checks. The same flash and
SRAM hand cases are checked again through evaluate() in test_evaluator.py.
"""
import math
import pytest

from src.evaluator import cycles, energy, memory
from src.evaluator.cycles import CycleCalibration
from src.evaluator.device import STM32F446RE, flash_wait_states
from src.evaluator.memory import MemoryCalibration
from src.evaluator.scenario import Scenario
from src.network.arch import Arch, LAYER_COUNTS, WIDTH_GRID

MEM = MemoryCalibration(kflash8=(1000.0, 10.0, 100.0), kflash32=(500.0, 20.0), kram8=(200.0, 5.0), kram32=(300.0, 6.0),
                        act8=(0.5, 100.0), act32=2.0, rtos_reserve=32768)
DEV = STM32F446RE


# ── device constants (datasheet DS10693 Rev 11) ────────────────────────────────

def test_operating_points_are_complete_and_increasing():
    pts = DEV.points
    assert len(pts) == 5
    assert all(a.f_mhz < b.f_mhz for a, b in zip(pts, pts[1:]))
    assert all(a.i_run_ma < b.i_run_ma for a, b in zip(pts, pts[1:]))
    assert all(a.i_sleep_ma < b.i_sleep_ma for a, b in zip(pts, pts[1:]))
    for p in pts:
        assert p.f_mhz > 0 and p.i_run_ma > 0 and p.i_sleep_ma > 0 and p.vos_scale in (1, 2, 3)
        assert p.i_sleep_ma < p.i_run_ma
    assert DEV.vdd == 3.3 and DEV.i_stop_ma > 0 and DEV.flash_bytes == 524288 and DEV.sram_bytes == 131072


def test_datasheet_values_spot_check():
    by_f = {p.f_mhz: p for p in DEV.points}
    assert (by_f[30].i_run_ma, by_f[60].i_run_ma, by_f[120].i_run_ma, by_f[150].i_run_ma, by_f[180].i_run_ma) \
        == (10.0, 18.0, 32.0, 41.0, 46.0)                       # Table 24, p. 83
    assert by_f[30].i_sleep_ma == 4.43 and by_f[180].i_sleep_ma == 11.36   # Table 26, p. 86
    assert DEV.i_stop_ma == 0.234                                # Table 27, p. 88


def test_flash_wait_states_follow_table_17():
    assert [flash_wait_states(f) for f in (30, 60, 120, 150, 180)] == [0, 1, 3, 4, 5]
    assert [p.flash_ws for p in DEV.points] == [0, 1, 3, 4, 5]
    assert flash_wait_states(180) == 5      # Table 17: 180 MHz with 5 wait states at 2.7-3.6 V


def test_voltage_scaling_follows_section_6_3_7():
    assert [(p.vos_scale, p.overdrive) for p in DEV.points] == [(3, False), (3, False), (3, False), (1, False), (1, True)]


# ── memory: hand-computed for Arch(2, 0.25) ───────────────────────────────────
# conv 3->8: 216 w, 8 bias; conv 8->16: 1152 w, 16 bias; linear 16->128: 2048 w, 128 bias; linear 128->10: 1280 w, 10 bias
# weights 4696, bias 162. Conv outputs: 8*32*32 = 8192 and 16*32*32 = 16384 elements (pooling follows the conv).
# L = 2 conv layers, w = 8/32 = 0.25.

@pytest.mark.parametrize("b,expected", [
    (4, 4696 * 0.5 + 162 * 4 + 1000 + 10 * 2 + 100 * 0.25),
    (8, 4696 * 1 + 162 * 4 + 1000 + 10 * 2 + 100 * 0.25),
    (16, 4696 * 2 + 162 * 4 + 1000 + 10 * 2 + 100 * 0.25),
    (32, 4696 * 4 + 162 * 4 + 500 + 20 * 2),
])
def test_flash_hand_values_small_net(b, expected):
    assert memory.flash_bytes(Arch(2, 0.25).profile(), b, MEM) == expected


@pytest.mark.parametrize("b,expected", [
    (8, 0.5 * (8 * 32 * 32 + 16 * 16 * 16) + 100 * 2 + 200 + 5 * 2),
    (16, 2 * (0.5 * (8 * 32 * 32 + 16 * 16 * 16) + 100 * 2) + 200 + 5 * 2),
    (32, 2.0 * 8 * 32 * 32 + 300 + 6 * 2),
])
def test_sram_hand_values_small_net(b, expected):
    assert memory.sram_bytes(Arch(2, 0.25).profile(), b, MEM) == expected + 32768


def test_sram_int4_uses_int8_activations():
    p = Arch(2, 0.25).profile()
    assert memory.sram_bytes(p, 4, MEM) == memory.sram_bytes(p, 8, MEM)


def test_flash_hand_value_second_architecture():
    # Arch(3, 0.25): channels 8, 16, 32. weights: 216 + 1152 + 4608 + 32*128 + 1280 = 11352; bias 8+16+32+128+10 = 194
    assert memory.flash_bytes(Arch(3, 0.25).profile(), 8, MEM) == 11352 + 194 * 4 + 1000 + 10 * 3 + 100 * 0.25


def test_memory_grows_with_bit_width():
    p = Arch(4, 0.625).profile()
    f = [memory.flash_bytes(p, b, MEM) for b in (4, 8, 16)]
    s = [memory.sram_bytes(p, b, MEM) for b in (4, 8, 16)]
    assert f == sorted(f) and s == sorted(s)


def test_unsupported_bits_raise():
    with pytest.raises(ValueError):
        memory.flash_bytes(Arch(2, 0.25).profile(), 2, MEM)


# ── cycles ────────────────────────────────────────────────────────────────────

CAL = CycleCalibration(cpm8=2.0, cpm32=5.0, c_layer=100.0, c_fixed=1000.0, k_flash=0.0, r16=1.5, r4=0.75)
ONE_CONV = [dict(kind="conv", macs=1000, weights=100)]


def test_cycles_hand_values():
    assert cycles.inference_cycles(ONE_CONV, 8, 0, CAL) == 1000 * 2.0 + 100 + 1000
    assert cycles.inference_cycles(ONE_CONV, 32, 0, CAL) == 1000 * 5.0 + 100 + 1000
    assert cycles.inference_cycles(ONE_CONV, 16, 0, CAL) == 1000 * 2.0 * 1.5 + 100 + 1000
    assert cycles.inference_cycles(ONE_CONV, 4, 0, CAL) == 1000 * 2.0 * 0.75 + 100 + 1000


def test_flash_term_only_when_k_flash_set():
    cal = CycleCalibration(cpm8=2.0, c_layer=100.0, c_fixed=1000.0, k_flash=0.5)
    base = cycles.inference_cycles(ONE_CONV, 8, 0, cal)
    assert cycles.inference_cycles(ONE_CONV, 8, 2, cal) == base + 100 * 1 * 0.5 * 2   # weights * bytes * k * ws


def test_cycles_monotone_in_macs_for_fixed_bit_width():
    order = sorted(((L, w) for L in LAYER_COUNTS for w in WIDTH_GRID),
                   key=lambda a: sum(r["macs"] for r in Arch(*a).profile()))
    for b in (4, 8, 16, 32):
        c = [cycles.inference_cycles(Arch(*a).profile(), b, 0, CAL) for a in order]
        assert all(x <= y for x, y in zip(c, c[1:]))


def test_literature_ratios_are_the_sourced_values():
    cal = CycleCalibration()
    assert cal.r16 == pytest.approx(1223.513 / 1034.033)    # Novac 2021
    assert cal.r4 == 1.14                                     # Capotondi 2020, "up to 14%"
    assert cal.cpm(16) == pytest.approx(cal.cpm8 * 1.183, rel=1e-3) and cal.cpm(4) == pytest.approx(cal.cpm8 * 1.14)


def test_first_conv_and_weight_terms():
    cal = CycleCalibration(cpm8=2.0, cpm32=5.0, c_layer=0.0, c_fixed=0.0, c_first8=3.0, c_first32=4.0,
                           c_weight8=0.5, c_weight32=0.25, c_fixed32=7.0, r16=1.5, r4=0.75)
    prof = [dict(kind="pool", macs=0, weights=0), dict(kind="conv", macs=1000, weights=100),
            dict(kind="conv", macs=2000, weights=200)]
    # the first conv only gets the extra term; weights in bytes; fp32 uses its own constants and fixed cost
    assert cycles.inference_cycles(prof, 8, 0, cal) == 3000 * 2.0 + 1000 * 3.0 + 300 * 0.5
    assert cycles.inference_cycles(prof, 32, 0, cal) == 3000 * 5.0 + 1000 * 4.0 + 300 * 4 * 0.25 + 7.0
    assert cycles.inference_cycles(prof, 16, 0, cal) == 3000 * 3.0 + 1000 * 3.0 * 1.5 + 300 * 2 * 0.5


def test_cycles_cover_layers_and_unsupported_bits():
    two = ONE_CONV * 2
    assert cycles.inference_cycles(two, 8, 0, CAL) == 2 * (1000 * 2.0 + 100) + 1000
    with pytest.raises(ValueError):
        CAL.cpm(2)


# ── energy ────────────────────────────────────────────────────────────────────

P30, P180 = DEV.points[0], DEV.points[-1]


def test_energy_at_zero_active_cycles_is_sleep_energy():
    e = energy.total_energy_uj(DEV, P30, 0, 1.0, "sleep")
    assert e == pytest.approx(3.3 * 4.43 * 1.0 * 1000)             # V * mA * s = mJ -> uJ
    assert energy.total_energy_uj(DEV, P30, 0, 1.0, "stop") == pytest.approx(3.3 * 0.234 * 1000)


def test_energy_hand_value_with_activity():
    # 18e6 cycles at 180 MHz = 0.1 s active; 0.9 s idle in Sleep
    e = energy.total_energy_uj(DEV, P180, 18e6, 1.0, "sleep")
    assert e == pytest.approx(3.3 * (46.0 * 0.1 + 11.36 * 0.9) * 1000)
    assert energy.energy_per_inference_uj(DEV, P180, 18e6, 1.0, 4, "sleep") == pytest.approx(e / 4)


def test_energy_rejects_overfull_horizon_and_bad_args():
    with pytest.raises(ValueError):
        energy.total_energy_uj(DEV, P180, 190e6, 1.0)           # 1.06 s of work in 1 s
    with pytest.raises(ValueError):
        energy.total_energy_uj(DEV, P180, 0, 1.0, "standby")
    with pytest.raises(ValueError):
        energy.energy_per_inference_uj(DEV, P180, 0, 1.0, 0)


def test_stop_idle_is_cheaper_than_sleep_idle():
    assert energy.total_energy_uj(DEV, P180, 1e6, 1.0, "stop") < energy.total_energy_uj(DEV, P180, 1e6, 1.0, "sleep")


def test_race_to_idle_table(capsys):
    """Energy per inference across the five frequencies for three architectures. The minimum need not sit at
    the lowest frequency; the printout records where it does. Stand-in workload: 50 inferences, 400 ms apart.
    A cell is 'overload' when the inference work alone does not fit in the horizon."""
    cal = Scenario.nominal().cycle_cal
    print("\nenergy per inference (uJ), int8, 50 inferences per 20 s horizon, Sleep idle, fitted cycle constants")
    print(f"{'arch':<12}" + "".join(f"{p.f_mhz:>9} MHz" for p in DEV.points) + "   argmin f")
    for L, w in ((2, 0.25), (3, 0.625), (5, 1.0)):
        c = cycles.inference_cycles(Arch(L, w).profile(), 8, 0, cal)
        e = []
        for p in DEV.points:
            try:
                e.append(energy.energy_per_inference_uj(DEV, p, 50 * c, 20.0, 50))
            except ValueError:
                e.append(None)
        ok = [x for x in e if x is not None]
        best = DEV.points[e.index(min(ok))].f_mhz
        print(f"{Arch(L, w).name:<12}" + "".join(f"{('overload' if x is None else f'{x:.0f}'):>13}" for x in e) + f"   {best} MHz")
        assert ok and all(math.isfinite(x) and x > 0 for x in ok)
