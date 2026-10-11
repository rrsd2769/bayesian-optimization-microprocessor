"""
The exhaustive reference: every design on the grid, evaluated through evaluate() only.

Grid: L (4) x w (5) x b (4) x f (5) x tau (10, log-spaced 1-20 ms) x pi (3) = 12,000 designs. The grid is
exhaustive over every variable except tau, which is continuous in the searches. Memory-infeasible designs are
dropped by the evaluator before simulation. One block = one (L, b) pair, 750 designs.

Hypervolume reference corners (fixed once, from the nominal enumeration, saved as the hv_reference artifact):
    corner A: accuracy 0.70, energy = worst acceptable energy x 1.1
    corner B: accuracy 0.75, energy = worst acceptable energy x 1.5
each rounded up to two significant figures. Results are reported at both because method rankings can depend
on the reference point (Pang 2024, IEEE TEVC, 10.1109/tevc.2023.3287399).
"""
import math
import time
from itertools import product

import numpy as np
import pandas as pd

from src.evaluator import PI_LEVELS, Design, Scenario, evaluate
from src.network.arch import LAYER_COUNTS, WIDTH_GRID
from src.search.pareto import hypervolume, pareto_indices

BITS = (4, 8, 16, 32)
TAUS = tuple(float(t) for t in np.geomspace(1.0, 20.0, 10))
COLUMNS = ["L", "w", "b", "tau", "pi", "f_idx", "accuracy", "energy_uj", "latency_ms", "miss_ratio", "flash_kb",
           "sram_kb", "acceptable", "violated", "block_wall_s"]
CORNER_A = (0.70, 1.1)     # (accuracy, energy multiple of the worst acceptable energy)
CORNER_B = (0.75, 1.5)


def blocks() -> list:
    return [(L, b) for L in LAYER_COUNTS for b in BITS]


def evaluate_block(L: int, b: int, scenario: Scenario) -> pd.DataFrame:
    t0 = time.time()
    rows = []
    for w, f, tau, pi in product(WIDTH_GRID, range(len(scenario.device.points)), TAUS, PI_LEVELS):
        o = evaluate(Design(L, w, b, tau, pi, f), scenario)
        rows.append((L, w, b, tau, pi, f, o.accuracy, o.energy_uj, o.latency_ms, o.miss_ratio, o.flash_kb, o.sram_kb,
                     o.acceptable, "+".join(o.violated), 0.0))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["block_wall_s"] = time.time() - t0
    return df


def clean_ceil(x: float) -> float:
    """Round up to two significant figures."""
    e = math.floor(math.log10(x)) - 1
    return math.ceil(x / 10 ** e - 1e-9) * 10 ** e


def hv_corners(worst_acceptable_energy_uj: float) -> dict:
    return dict(worst_acceptable_energy_uj=worst_acceptable_energy_uj,
                A=dict(accuracy=CORNER_A[0], energy_uj=clean_ceil(worst_acceptable_energy_uj * CORNER_A[1])),
                B=dict(accuracy=CORNER_B[0], energy_uj=clean_ceil(worst_acceptable_energy_uj * CORNER_B[1])))


def front_of(df: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated acceptable designs (accuracy up, energy down), in enumeration order."""
    ok = df[df["acceptable"]].reset_index(drop=True)
    return ok.iloc[pareto_indices(list(zip(ok["accuracy"], ok["energy_uj"])))].reset_index(drop=True)


def hv_at(df: pd.DataFrame, corners: dict) -> dict:
    ok = df[df["acceptable"]]
    pts = list(zip(ok["accuracy"], ok["energy_uj"]))
    return {k: hypervolume(pts, corners[k]["accuracy"], corners[k]["energy_uj"]) for k in ("A", "B")}


def rejection_counts(df: pd.DataFrame) -> dict:
    out = {name: int(df["violated"].fillna("").str.contains(name).sum())
           for name in ("accuracy", "flash", "sram", "latency", "miss_ratio")}
    out["memory_infeasible"] = int(df["violated"].fillna("").str.contains("flash|sram").sum())
    return out


def summarise(df: pd.DataFrame, corners: dict) -> dict:
    front = front_of(df)
    comp = {c: {str(k): int(v) for k, v in front[c].value_counts().sort_index().items()} for c in ("L", "b", "f_idx", "pi")}
    return dict(grid_size=int(len(df)), n_acceptable=int(df["acceptable"].sum()), n_front=int(len(front)),
                rejected=rejection_counts(df), hypervolume=hv_at(df, corners), front_composition=comp)
