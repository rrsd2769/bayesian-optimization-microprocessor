"""Calibration fit (Phase 2): recovers known coefficients from synthetic data, and the real fit meets its targets."""
import numpy as np
import pandas as pd
import pytest

from src import results
from src.evaluator import cycles, memory
from src.evaluator.calibrate import fit_cycles, fit_memory, fit_nnls, from_json, parse_arch, to_json
from src.evaluator.cycles import CycleCalibration
from src.evaluator.memory import MemoryCalibration
from src.evaluator.scenario import Scenario
from src.network.arch import Arch


def synthetic_cloud(cal, mem):
    rows = []
    for L in (2, 3, 4, 5):
        for w in (0.25, 0.625, 1.0):
            a = Arch(L, w)
            p = a.profile()
            for prec, b in (("int8", 8), ("fp32", 32)):
                rows.append(dict(arch=a.name, precision=prec, cycles=cycles.inference_cycles(p, b, 0, cal),
                                 total_flash_bytes=0, total_ram_bytes=0, weights_bytes=0,
                                 activations_bytes=memory.activation_bytes(p, b, mem) + memory.HARNESS_RAM,
                                 kernel_flash_bytes=memory.kernel_flash(p, b, mem) + memory.HARNESS_FLASH,
                                 kernel_ram_bytes=memory.kernel_ram(p, b, mem)))
    return pd.DataFrame(rows)


CAL = CycleCalibration(cpm8=2.0, cpm32=7.0, c_layer=0.0, c_fixed=1e6, c_first8=0.0, c_first32=15.0,
                       c_weight8=8.0, c_weight32=0.0, c_fixed32=2e5)
MEM = MemoryCalibration(kflash8=(20000.0, 1000.0, 3000.0), kflash32=(6000.0, 400.0), kram8=(1500.0, 500.0),
                        kram32=(600.0, 200.0), act8=(0.4, 1200.0), act32=4.5)


def test_fit_nnls_recovers_exact_linear_data_and_is_nonnegative():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(1, 10, 12), rng.uniform(1, 10, 12), np.ones(12)])
    y = X @ np.array([3.0, 0.0, 5.0]) + 1.0
    coef, m = fit_nnls(X, y)
    assert (coef >= 0).all() and m["r2"] > 0.999 and m["max_rel_err"] < 0.05


def test_fit_recovers_synthetic_calibration():
    cloud = synthetic_cloud(CAL, MEM)
    cal, rep = fit_cycles(cloud)
    assert cal.cpm8 == pytest.approx(2.0, rel=1e-3) and cal.c_weight8 == pytest.approx(8.0, rel=1e-3)
    assert cal.cpm32 == pytest.approx(7.0, rel=1e-3) and cal.c_first32 == pytest.approx(15.0, rel=1e-3)
    assert rep["cycles_int8"]["loo_max_rel_err"] < 1e-6
    mem, rep = fit_memory(cloud)
    assert mem.act8 == pytest.approx(MEM.act8, rel=1e-3) and mem.act32 == pytest.approx(4.5, rel=1e-3)
    assert mem.kflash8 == pytest.approx(MEM.kflash8, rel=1e-3) and mem.kram32 == pytest.approx(MEM.kram32, rel=1e-3)


def test_calibration_round_trips_through_json():
    cal, rep = fit_cycles(synthetic_cloud(CAL, MEM))
    assert from_json(CycleCalibration, to_json(cal, rep)) == cal
    mem, rep = fit_memory(synthetic_cloud(CAL, MEM))
    assert from_json(MemoryCalibration, to_json(mem, rep)) == mem


def test_parse_arch():
    assert parse_arch("L4_w0.625") == Arch(4, 0.625) and parse_arch("L2_w1") == Arch(2, 1.0)


def test_nominal_scenario_loads_fitted_calibration(tmp_path):
    results.set_root(tmp_path)
    assert Scenario.nominal().cycle_cal.source == "placeholder"
    cloud = synthetic_cloud(CAL, MEM)
    cal, rep = fit_cycles(cloud)
    results.save("cycle_calibration", to_json(cal, rep))
    mem, rep = fit_memory(cloud)
    results.save("memory_calibration", to_json(mem, rep))
    sc = Scenario.nominal()
    assert sc.cycle_cal.cpm8 == pytest.approx(2.0, rel=1e-3) and sc.memory_cal.act32 == pytest.approx(4.5, rel=1e-3)
