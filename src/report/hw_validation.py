"""
Held-out silicon check: predicted against measured Cloud cycles, flash and RAM for architectures that the
calibration did not use. Error is |predicted / measured - 1|, reported as MAE % and max %, with the signed
per-model errors alongside. Also reports the same for the calibration set as an in-sample reference.
"""
import numpy as np
import pandas as pd

from src.evaluator import predict_benchmark

QUANTITIES = (("cycles", "cycles"), ("flash_bytes", "total_flash_bytes"), ("ram_bytes", "total_ram_bytes"))


def _compare(cloud: pd.DataFrame, scenario) -> dict:
    rows, errs = [], {q: [] for q, _ in QUANTITIES}
    for _, r in cloud.sort_values("arch").iterrows():
        layers, width = r["arch"][1:].split("_w")
        b = 8 if r["precision"] == "int8" else 32
        pred = predict_benchmark(int(layers), float(width), b, scenario)
        row = dict(arch=r["arch"], precision=r["precision"])
        for q, col in QUANTITIES:
            err = pred[q] / float(r[col]) - 1
            errs[q].append(err)
            row[q] = dict(predicted=float(pred[q]), measured=float(r[col]), error_pct=float(err * 100))
        rows.append(row)
    summary = {q: dict(mae_pct=float(np.mean(np.abs(e)) * 100), max_pct=float(np.max(np.abs(e)) * 100),
                       mean_signed_pct=float(np.mean(e) * 100)) for q, e in errs.items()}
    return dict(n=len(rows), summary=summary, models=rows)


def validate(validation: pd.DataFrame, calibration: pd.DataFrame, scenario) -> dict:
    return dict(held_out=_compare(validation, scenario), in_sample=_compare(calibration, scenario))
