"""Search space, Pareto/hypervolume and the reference front (small hand cases; the full enumeration is a pipeline stage)."""
import math

import pandas as pd
import pytest

from src.evaluator import Design, evaluate
from src.search.pareto import hypervolume, pareto_indices
from src.search.reference_front import (TAUS, blocks, clean_ceil, evaluate_block, front_of, hv_at, hv_corners,
                                        rejection_counts, summarise)
from src.search.space import SPACE, to_design
from tests.test_evaluator import scenario


def test_pareto_keeps_non_dominated_and_first_of_duplicates():
    pts = [(0.8, 10.0), (0.9, 20.0), (0.7, 5.0), (0.75, 12.0), (0.9, 20.0), (0.85, 30.0)]
    assert pareto_indices(pts) == [0, 1, 2]


def test_hypervolume_hand_case():
    # reference (acc 0.5, energy 100); points (0.8, 20) and (0.9, 60): 80 * 0.3 + 40 * 0.1 = 28
    assert hypervolume([(0.8, 20.0), (0.9, 60.0)], 0.5, 100.0) == pytest.approx(28.0)
    assert hypervolume([(0.8, 20.0), (0.9, 60.0), (0.7, 50.0)], 0.5, 100.0) == pytest.approx(28.0)   # dominated point adds nothing
    assert hypervolume([(0.4, 20.0), (0.9, 120.0)], 0.5, 100.0) == 0.0                              # outside the corner


def test_clean_ceil_rounds_up_to_two_significant_figures():
    assert clean_ceil(123456.0) == 130000.0 and clean_ceil(0.0123) == pytest.approx(0.013) and clean_ceil(99.0) == 99.0


def test_corners_are_derived_once_from_the_worst_energy():
    c = hv_corners(20000.0)
    assert c["A"] == dict(accuracy=0.70, energy_uj=22000.0) and c["B"] == dict(accuracy=0.75, energy_uj=30000.0)


def test_space_matches_the_plan_and_converts_to_designs():
    assert [p.name for p in SPACE] == ["num_layers", "width_mult", "quant_bits", "time_slice", "priority", "freq_idx"]
    by = {p.name: p for p in SPACE}
    assert by["time_slice"].bounds == (1.0, 20.0) and not by["priority"].ordered and len(by["freq_idx"].values) == 5
    d = to_design(dict(num_layers=3, width_mult=0.625, quant_bits=8, time_slice=4.5, priority=1, freq_idx=2))
    assert (d.priority, d.freq_idx, d.quant_bits) == ("equal", 2, 8)


def test_block_grid_and_rows_match_evaluate():
    assert len(blocks()) == 16 and len(TAUS) == 10 and TAUS[0] == 1.0 and TAUS[-1] == pytest.approx(20.0)
    sc = scenario()
    df = evaluate_block(2, 8, sc)
    assert len(df) == 5 * 5 * 10 * 3
    row = df.iloc[7]
    o = evaluate(Design(int(row.L), row.w, int(row.b), row.tau, row.pi, int(row.f_idx)), sc)
    assert o.energy_uj == row.energy_uj and o.acceptable == row.acceptable


def test_front_summary_counts_and_hv():
    sc = scenario()
    df = pd.concat([evaluate_block(2, b, sc) for b in (4, 8)], ignore_index=True)
    front = front_of(df)
    ok = df[df["acceptable"]]
    assert 0 < len(front) <= len(ok)
    for _, f in front.iterrows():      # nothing acceptable dominates a front member
        assert not ((ok["accuracy"] >= f.accuracy) & (ok["energy_uj"] < f.energy_uj)).any()
    corners = hv_corners(float(ok["energy_uj"].max()))
    s = summarise(df, corners)
    assert s["grid_size"] == len(df) and s["n_front"] == len(front)
    assert math.isfinite(hv_at(df, corners)["A"]) and s["hypervolume"]["A"] > 0
    assert rejection_counts(df)["memory_infeasible"] == 0


def test_sensitivity_case_list():
    from src.pipeline import sensitivity_cases
    cases = sensitivity_cases(scenario())
    assert len(cases) == 1 + 8 + 1 + 1 + 6 + 2      # nominal, other limit pairs, stop idle, mean latency, kernel/r16/r4 x0.5,x2, two task sets
    assert cases["latency_mean"].limits.latency_stat == "mean" and cases["idle_stop"].idle_mode == "stop"
    assert cases["lat50_miss0"].limits.latency_ms == 50 and cases["lat50_miss0"].limits.miss_ratio == 0.0
    assert cases["kernel_x2"].workload.switch_cycles == 780
