"""
Uncertainty of the accuracy table, from the measurements.

seed_noise: for each architecture trained with several seeds and each b, the sample standard deviation
            (ddof = 1) of validation accuracy, with the individual values.
drop_32_to_4: for every architecture (table seed), val_acc(b=32) - val_acc(b=4), in accuracy points.
"""
import numpy as np
import pandas as pd

from src.evaluator import TABLE_SEED


def validate(m: pd.DataFrame) -> dict:
    seeds = m.groupby("arch")["seed"].nunique()
    noise = {}
    for arch in seeds[seeds > 1].index:
        sub = m[m["arch"] == arch]
        noise[arch] = {str(int(b)): dict(std=float(np.std(g["val_acc"], ddof=1)), mean=float(g["val_acc"].mean()),
                                         values={str(int(s)): float(v) for s, v in zip(g["seed"], g["val_acc"])})
                       for b, g in sub.groupby("b")}
    t = m[m["seed"] == TABLE_SEED].pivot(index="arch", columns="b", values="val_acc")
    drop = {a: float((r[32] - r[4]) * 100) for a, r in t.iterrows()}
    return dict(table_seed=TABLE_SEED, n_architectures=int(len(t)), seed_noise=noise, drop_32_to_4_points=drop,
                n_with_drop_above_zero=int(sum(v > 0 for v in drop.values())))
