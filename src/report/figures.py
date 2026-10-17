"""
Figures from the results store (grayscale PDF, 3.5 in column width, 300 DPI).

    fig_front        reference front and each method's median-seed front (accuracy against energy)
    fig_convergence  hypervolume % of the reference against evaluations, median and IQR per method
    fig_cycles       predicted against measured Cloud cycles on the calibration and held-out models

The median seed of a method is the seed whose final hv_A is closest to the median over seeds; ties go to the lowest
seed (qNEHVI has 18 of 20 seeds at 100 %, so its median seed is seed 1).
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import results
from src.search.pareto import pareto_indices

METHOD_LABELS = {"qnehvi": "qNEHVI", "qnparego": "qNParEGO", "sequential": "Sequential", "motpe": "MOTPE",
                 "random": "Random", "nsga2": "NSGA-II"}
ORDER = tuple(METHOD_LABELS)
STYLE = {   # gray level, marker, line style
    "qnehvi": ("0.0", "o", "-"), "qnparego": ("0.25", "s", "--"), "sequential": ("0.45", "^", "-."),
    "motpe": ("0.5", "D", ":"), "random": ("0.6", "v", "-"), "nsga2": ("0.35", "P", "--"),
}
WIDTH_IN = 3.5
DPI = 300


def _setup():
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                         "axes.linewidth": 0.6, "lines.linewidth": 1.0, "pdf.fonttype": 42, "savefig.dpi": DPI})


def _runs(method: str) -> dict:
    seeds = results.load("comparison_summary")["seeds_per_method"][method]
    return {s: results.load("run", method=method, seed=s) for s in seeds}


def median_seed(final_hv: dict) -> int:
    """Seed whose final hypervolume is closest to the median over seeds; ties go to the lowest seed."""
    med = float(np.median(list(final_hv.values())))
    return min(final_hv, key=lambda s: (abs(final_hv[s] - med), s))


def front_points(run) -> np.ndarray:
    """(accuracy, energy) of the non-dominated acceptable evaluations of one run, ascending energy."""
    ok = run[run["acceptable"].astype(bool)]
    pts = list(zip(ok["accuracy"], ok["energy_uj"]))
    idx = pareto_indices(pts)
    return np.array(sorted((pts[i] for i in idx), key=lambda p: p[1]))


def fig_front(out: Path) -> Path:
    ref = results.load("reference_front").sort_values("energy_uj")
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.3))
    ax.step(ref["energy_uj"], ref["accuracy"], where="post", color="black", lw=1.6, label="Reference front", zorder=1)
    ax.scatter(ref["energy_uj"], ref["accuracy"], s=26, color="black", zorder=1)
    for k, m in enumerate(ORDER):
        runs = _runs(m)
        seed = median_seed({s: r["hv_A"].iloc[-1] for s, r in runs.items()})
        pts = front_points(runs[seed])
        if len(pts) == 0:
            continue
        gray, marker, _ = STYLE[m]
        dx = (k - 2.5) * 40    # small horizontal offset so coincident fronts stay visible
        ax.scatter(pts[:, 1] + dx, pts[:, 0], s=14, marker=marker, facecolors="none", edgecolors=gray, lw=0.8,
                   label=f"{METHOD_LABELS[m]} (seed {seed})", zorder=2)
    ax.set_xlabel("Energy per inference ($\\mu$J)")
    ax.set_ylabel("Accuracy")
    ax.grid(color="0.85", lw=0.4)
    ax.legend(frameon=False, loc="lower right", ncol=1)
    return _save(fig, out / "fig_front.pdf")


def fig_convergence(out: Path) -> Path:
    total = results.load("reference_front_summary")["hypervolume"]["A"]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.3))
    for m in ORDER:
        runs = _runs(m)
        hv = np.array([r["hv_A"].to_numpy() for r in runs.values()])
        pct = hv / total * 100
        x = np.arange(1, hv.shape[1] + 1)
        q1, med, q3 = np.percentile(pct, [25, 50, 75], axis=0)
        gray, marker, ls = STYLE[m]
        ax.fill_between(x, q1, q3, color=gray, alpha=0.06, lw=0)
        ax.plot(x, med, color=gray, ls=ls, marker=marker, markevery=10, ms=3, label=METHOD_LABELS[m])
    ax.axhline(90, color="0.5", lw=0.5, ls=":")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Hypervolume (% of reference)")
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 102)
    ax.grid(color="0.85", lw=0.4)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8", loc="lower right", ncol=2)
    return _save(fig, out / "fig_convergence.pdf")


def fig_cycles(out: Path) -> Path:
    hw = results.load("hw_validation")
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.5))
    lo, hi = 1e6, 6e7
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.6, label="$y = x$")
    for part, marker, face, label in (("in_sample", "o", "none", "Calibration"), ("held_out", "s", "black", "Held out")):
        rows = hw[part]["models"]
        ax.scatter([r["cycles"]["measured"] for r in rows], [r["cycles"]["predicted"] for r in rows], s=14,
                   marker=marker, facecolors=face, edgecolors="black", lw=0.8, label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured cycles")
    ax.set_ylabel("Predicted cycles")
    ax.grid(color="0.85", lw=0.4, which="both")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, out / "fig_cycles.pdf")


def _save(fig, path: Path) -> Path:
    fig.tight_layout(pad=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def make_all(out: Path) -> list:
    _setup()
    return [fig_front(out), fig_convergence(out), fig_cycles(out)]
