"""
Accuracy lookup: measured validation accuracy keyed by (L, w, b), as in tabular NAS benchmarks
(NATS-Bench, Dong 2021, IEEE TPAMI). No interpolation: a width that is not one of the five grid
widths raises.

Each architecture is trained once with seed TABLE_SEED for the table; the extra seeds of the repeated
architectures only measure the table's noise (report/accuracy_validation).
"""
import re

import numpy as np
import pandas as pd

from src.network.arch import LAYER_COUNTS, WIDTH_GRID

BITS = (4, 8, 16, 32)
TABLE_SEED = 1
_NAME = re.compile(r"^L(\d+)_w([0-9.]+)$")


def parse_arch(name: str):
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"not an architecture name: {name}")
    return int(m.group(1)), float(m.group(2))


class AccuracyTable:
    def __init__(self, measurements: pd.DataFrame, seed: int = TABLE_SEED):
        df = measurements[measurements["seed"] == seed]
        self._acc = {}
        for name, b, v in zip(df["arch"], df["b"], df["val_acc"]):
            L, w = parse_arch(name)
            self._acc[(L, w, int(b))] = float(v)

    def acc(self, L: int, w: float, b: int) -> float:
        if L not in LAYER_COUNTS:
            raise ValueError(f"L must be in {LAYER_COUNTS}, got {L}")
        if b not in BITS:
            raise ValueError(f"b must be in {BITS}, got {b}")
        grid = [g for g in WIDTH_GRID if np.isclose(g, w, rtol=0, atol=1e-9)]
        if not grid:
            raise ValueError(f"w must be one of the grid widths {WIDTH_GRID}, got {w}")
        try:
            return self._acc[(L, grid[0], b)]
        except KeyError:
            raise KeyError(f"no measurement for L={L} w={grid[0]} b={b}: run stage 'accuracy'") from None

    def __len__(self):
        return len(self._acc)
