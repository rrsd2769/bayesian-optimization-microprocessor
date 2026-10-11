"""
Fit the cycle and memory cost models to the ST Edge AI Developer Cloud benchmarks (Phase 2).

Each target is a linear combination of per-model features with non-negative coefficients, fitted by non-negative
least squares on relative error (every row weighted 1/target), separately for int8 and fp32. A term the data does
not support comes out as 0. Reported per target: in-sample R^2 and maximum relative error, and the
leave-one-architecture-out (LOO) mean and maximum relative error, which is the number to quote.

    cycles    = cpm * macs + c_first * first_conv_macs + c_weight * weight_bytes + c_fixed
    kernel flash (int8) = const + per_layer * L + per_width * w;  (fp32) = const + per_layer * L
    kernel RAM          = const + per_layer * L
    activations (int8)  = per_out * sum_conv_output_elements + per_layer * L;  (fp32) = per_elem * first_conv_output
"""
import dataclasses

import numpy as np
from scipy.optimize import nnls

from src.evaluator.cycles import CycleCalibration
from src.evaluator.memory import HARNESS_FLASH, HARNESS_RAM, MemoryCalibration, shape_features
from src.network.arch import Arch


def parse_arch(name: str) -> Arch:
    layers, width = name[1:].split("_w")
    return Arch(int(layers), float(width))


def fit_nnls(X: np.ndarray, y: np.ndarray):
    """Relative-error NNLS. Returns coefficients and a metrics dict (in-sample and leave-one-out)."""
    W = 1.0 / y
    A = X * W[:, None]
    coef, _ = nnls(A, y * W)
    pred = X @ coef
    loo = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        ci, _ = nnls(A[keep], (y * W)[keep])
        loo.append(abs(X[i] @ ci - y[i]) / y[i])
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return coef, dict(n=len(y), r2=float(r2), max_rel_err=float(np.abs(pred / y - 1).max()),
                      loo_mean_rel_err=float(np.mean(loo)), loo_max_rel_err=float(np.max(loo)))


def _rows(cloud, precision):
    s = cloud[cloud["precision"] == precision].sort_values("arch")
    return s, [parse_arch(a).profile() for a in s["arch"]]


def fit_cycles(cloud):
    """cloud: the cloud_calibration table. Returns (CycleCalibration, report)."""
    report, coefs = {}, {}
    for prec, b in (("int8", 8), ("fp32", 32)):
        s, profiles = _rows(cloud, prec)
        macs = np.array([sum(r["macs"] for r in p) for p in profiles], float)
        first = np.array([next(r["macs"] for r in p if r["kind"] == "conv") for p in profiles], float)
        wbytes = np.array([sum(r["weights"] for r in p) * b / 8 for p in profiles], float)
        X = np.column_stack([macs, first, wbytes, np.ones(len(s))])
        coefs[prec], report[f"cycles_{prec}"] = fit_nnls(X, s["cycles"].to_numpy(float))
    i, f = coefs["int8"], coefs["fp32"]
    cal = CycleCalibration(cpm8=i[0], c_first8=i[1], c_weight8=i[2], c_fixed=i[3],
                           cpm32=f[0], c_first32=f[1], c_weight32=f[2], c_fixed32=f[3],
                           c_layer=0.0, k_flash=0.0, source="fit: Cloud STM32F469I-DISCO, 180 MHz, 12+12 models")
    return cal, report


def fit_memory(cloud):
    """Returns (MemoryCalibration, report). Benchmark-harness offsets are removed from the targets."""
    report, out = {}, {}
    # name, precision, shape features after the constant, Cloud column, harness offset to remove
    specs = (("kflash8", "int8", ("L", "w"), "kernel_flash_bytes", HARNESS_FLASH),
             ("kflash32", "fp32", ("L",), "kernel_flash_bytes", HARNESS_FLASH),
             ("kram8", "int8", ("L",), "kernel_ram_bytes", 0),
             ("kram32", "fp32", ("L",), "kernel_ram_bytes", 0),
             ("act8", "int8", ("sum_out", "L"), "activations_bytes", HARNESS_RAM),
             ("act32", "fp32", ("first_out",), "activations_bytes", HARNESS_RAM))
    for name, prec, extra, col, off in specs:
        s, profiles = _rows(cloud, prec)
        feats = [shape_features(p) for p in profiles]
        columns = [[ft[c] for ft in feats] for c in extra]
        if not name.startswith("act"):
            columns = [np.ones(len(s))] + columns          # constant term; activations have none
        coef, report[name] = fit_nnls(np.column_stack(columns), s[col].to_numpy(float) - off)
        out[name] = float(coef[0]) if name == "act32" else tuple(float(c) for c in coef)
    return MemoryCalibration(source="fit: Cloud STM32F469I-DISCO, 12+12 models", **out), report


def to_json(cal, report) -> dict:
    return dict(cal=dataclasses.asdict(cal), report=report)


def from_json(cls, data: dict):
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in data["cal"].items() if k in fields}
    return cls(**kw)
