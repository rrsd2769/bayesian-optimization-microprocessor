"""
Publication-quality figures for IEEE conference submission.
Figures (single-objective / SOO — BO vs Random vs Grid, all under the
accuracy+latency constraint, minimizing energy):
 1. convergence_curve.pdf — best energy vs. trial number
 2. comparison_bar.pdf — best result per method on all 4 metrics
 A. energy_improvement_budget.pdf — cumulative energy improvement vs. budget
 B. feasible_discovery_rate.pdf — cumulative feasible-config discovery
Figures (multi-objective / MOO — the headline result):
 3. pareto_frontier.pdf — accuracy vs energy, MOO vs Random vs the
    exhaustive reference front (src/reference_front.py)
 C. hypervolume_convergence.pdf — hypervolume vs evaluations, with
    cross-seed uncertainty and the reference front's HV as a ceiling
 D. soo_vs_moo.pdf — single-objective collapse vs multi-objective front,
    same search space, same budget class
 E. pareto_parallel_coords.pdf — parameter values along the MOO Pareto set
LaTeX:
 table_main_results.tex — main results table for paper Section V
IEEE style: serif font, 3.5in width, 300dpi, PDF export, grayscale-safe markers.
"""
import os, sys, csv, json, logging, math
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.config import (
 RESULTS_DIR, FIGURES_DIR, ACCURACY_THRESHOLD, LATENCY_BUDGET_MS, LOG_FORMAT
)
from src.pareto_metrics import pareto_front as pm_pareto_front, hypervolume as pm_hypervolume
logger = logging.getLogger(__name__)

plt.rcParams.update({
 "font.family": "serif",
 "font.size": 9,
 "axes.titlesize": 10,
 "axes.labelsize": 9,
 "legend.fontsize": 8,
 "xtick.labelsize": 8,
 "ytick.labelsize": 8,
 "figure.dpi": 300,
 "savefig.dpi": 300,
 "savefig.bbox": "tight",
 "savefig.pad_inches": 0.05,
 "axes.spines.top": False,
 "axes.spines.right": False,
 "lines.linewidth": 1.5,
 "axes.grid": True,
 "grid.alpha": 0.3,
 "grid.linestyle": "--",
 "pdf.fonttype": 42, # Embed fonts for IEEE submission
 "ps.fonttype": 42,
})

METHOD_STYLES = {
 "BO": {"color": "#1f77b4", "linestyle": "-", "marker": "o",
 "label": "Bayesian Opt. (ours)", "hatch": ""},
 "Random": {"color": "#d62728", "linestyle": "--", "marker": "s",
 "label": "Random Search", "hatch": "//"},
 "Grid": {"color": "#2ca02c", "linestyle": ":", "marker": "^",
 "label": "Grid Search", "hatch": ".."},
 "MOO": {"color": "#9467bd", "linestyle": "-", "marker": "D",
 "label": "Multi-Obj. BO (ours)", "hatch": ""},
 "Reference": {"color": "#555555", "linestyle": "-", "marker": "",
 "label": "Reference Front (exhaustive)", "hatch": ""},
}

def load_csv(path: Path) -> List[Dict]:
 with open(path) as f:
  return list(csv.DictReader(f))
def get_feasible(rows: List[Dict]) -> List[Dict]:
 return [r for r in rows if str(r.get("is_feasible", "")).lower() in ("true",
"1")]
def best_metric(rows: List[Dict], metric: str) -> float:
 """Return best feasible value for a metric (min for energy/memory/latency,
max for accuracy)."""
 feasible = get_feasible(rows)
 if not feasible:
  return float("nan")
 vals = [float(r[metric]) for r in feasible]
 return max(vals) if metric == "accuracy" else min(vals)

def cumulative_best(rows: List[Dict], metric: str = "energy_mj") -> List[float]:
 """
 Compute cumulative best feasible energy across trials.
 Used for convergence curve — x=trial_id, y=best_so_far.
 Infeasible trials do not update the best.
 """
 best = float("inf")
 result = []
 for r in rows:
  is_f = str(r.get("is_feasible", "")).lower() in ("true", "1")
  if is_f:
   best = min(best, float(r[metric]))
  result.append(best if best < float("inf") else float("nan"))
 return result


def compute_energy_improvement_series(rows: List[Dict]) -> List[float]:
    """
    Compute per-trial cumulative energy improvement percentage.
 
    For each evaluation step i, the improvement is measured relative
    to the energy of the FIRST feasible configuration discovered:
 
        improvement_pct[i] =
            100 * (first_feasible_energy - best_feasible_energy_so_far[i])
            / first_feasible_energy
 
    Design rationale:
        Using the first feasible energy as the reference (not the worst
        overall energy) isolates the optimisation gain from the
        exploration gain. This makes the comparison fair: all methods
        start from 0% at their respective first feasible hit, and the
        y-axis shows only how much better the search gets after that
        point. A higher final value means the method found a
        substantially better configuration within the same budget.
 
    Guarantees on the returned series:
        - Values before the first feasible trial are NaN (undefined).
        - The first feasible trial always yields exactly 0.0%.
        - The series is monotonically non-decreasing (cumulative best).
        - Values are clamped to [0, 100] to handle floating-point noise.
 
    Args:
        rows (List[Dict]): All rows for one method from a results CSV,
                           in trial order. Each dict must contain
                           'energy_mj' and 'is_feasible'.
 
    Returns:
        List[float]: Per-trial improvement percentage. Length equals
                     len(rows). NaN where no feasible config found yet.
 
    Raises:
        ValueError: If rows is empty.
 
    Example:
        >>> series = compute_energy_improvement_series(bo_rows)
        >>> series[:4]
        [nan, 0.0, 0.0, 12.7]
    """
    if not rows:
        raise ValueError("compute_energy_improvement_series: rows must not be empty.")
 
    first_feasible_energy: float = None   # type: ignore[assignment]
    cumulative_best        = float("inf")
    series: List[float]   = []
 
    for r in rows:
        energy     = float(r["energy_mj"])
        is_feasible = str(r.get("is_feasible", "")).lower() in ("true", "1")
 
        if is_feasible:
            if first_feasible_energy is None:
                # First feasible hit: record the reference energy.
                first_feasible_energy = energy
            cumulative_best = min(cumulative_best, energy)
 
        if first_feasible_energy is None:
            # No feasible config found yet — improvement undefined.
            series.append(float("nan"))
        elif first_feasible_energy == 0.0:
            # Degenerate case: reference energy is zero → cannot normalise.
            # Treat as 0% improvement throughout (conservative).
            series.append(0.0)
        else:
            pct = 100.0 * (first_feasible_energy - cumulative_best) / first_feasible_energy
            series.append(float(np.clip(pct, 0.0, 100.0)))
 
    return series
 
 
def compute_feasible_discovery_series(rows: List[Dict]) -> List[int]:
    """
    Compute the cumulative count of feasible configurations discovered.
 
    A configuration is feasible iff its 'is_feasible' field is True
    (equivalently: accuracy >= ACCURACY_THRESHOLD, as set in config.py).
 
    At each evaluation step i, the value is the total number of feasible
    configurations found in evaluations 1 through i (inclusive).
 
    Scientific interpretation:
        A steeper cumulative curve indicates the search method discovers
        constraint-satisfying configurations more rapidly. A method that
        finds many feasible configurations early has a higher effective
        exploration rate under constraints — this matters for embedded
        deployment, where infeasible (too-inaccurate) models are useless.
 
    Guarantees:
        - Series is monotonically non-decreasing (cumulative count).
        - Series is integer-valued in [0, len(rows)].
        - Series[i] == 0 if no feasible config found in rows[0..i].
 
    Args:
        rows (List[Dict]): All rows for one method from a results CSV,
                           in trial order. Each dict must contain
                           'is_feasible'.
 
    Returns:
        List[int]: Cumulative feasible count. Length equals len(rows).
 
    Raises:
        ValueError: If rows is empty.
 
    Example:
        >>> series = compute_feasible_discovery_series(bo_rows)
        >>> series[-1]   # total feasible configs found
        24
    """
    if not rows:
        raise ValueError("compute_feasible_discovery_series: rows must not be empty.")
 
    count: int       = 0
    series: List[int] = []
 
    for r in rows:
        if str(r.get("is_feasible", "")).lower() in ("true", "1"):
            count += 1
        series.append(count)
 
    return series

def figure1_convergence(bo, rand, grid) -> Path:
 """
 Convergence curve: best feasible energy_mj vs. trial number.
 Story: BO finds lower energy faster (fewer evaluations).
 This is the paper's primary efficiency claim.
 Paper placement: Section V-A, single column (3.5in).
 Caption: "Convergence of best feasible inference energy (mJ) as a
 function of the number of configurations evaluated. BO
 (solid) achieves lower energy in fewer evaluations than
 Random Search (dashed) and Grid Search (dotted)."
 """
 fig, ax = plt.subplots(figsize=(3.5, 2.5))
 for method, rows in [("BO", bo), ("Random", rand), ("Grid", grid)]:
  cb = cumulative_best(rows)
  n = len(cb)
  style = METHOD_STYLES[method]
  ax.plot(range(1, n + 1), cb,
 color=style["color"], linestyle=style["linestyle"],
 marker=style["marker"], markersize=3, markevery=max(1, n//8),
 label=style["label"])
 ax.set_xlabel("Number of Evaluations (log scale)")
 ax.set_ylabel("Best Feasible Energy (mJ)")
 ax.set_title("Convergence Comparison")
 ax.legend(loc="upper right", framealpha=0.9, handlelength=2.5)
 # Log x-axis: Grid runs the full 972-config lattice while BO/Random run
 # 30 — a linear axis crams BO/Random into an invisible sliver at the
 # left edge and hides their actual convergence speed.
 ax.set_xscale("log")
 ax.set_xlim(left=1)
 # Y-axis: don't start at 0 if all values are close — use autoscale
 ax.autoscale(axis="y", tight=False)
 out = FIGURES_DIR / "convergence_curve.pdf"
 fig.savefig(out)
 plt.close(fig)
 logger.info(f"Figure 1 saved: {out}")
 return out

def figure2_comparison_bar(bo, rand, grid) -> Path:
 """
 4-panel grouped bar chart: best result per method on each metric.
 Story: BO achieves best energy with comparable accuracy/memory/latency.
 Paper placement: Section V-B, double column (7.0in).
 Caption: "Best feasible result per search method across all four metrics.
 BO achieves the lowest energy while maintaining accuracy above
 the constraint threshold of {ACCURACY_THRESHOLD}."
 """
 metrics = ["accuracy", "energy_mj", "memory_kb", "latency_ms"]
 titles = [
 f"Accuracy\n(≥{ACCURACY_THRESHOLD} required)",
 "Energy (mJ)\n↓ lower is better",
 "Memory (KB)\n↓ lower is better",
 "Latency (ms)\n↓ lower is better",
 ]
 methods = ["BO", "Random", "Grid"]
 bar_colors = [METHOD_STYLES[m]["color"] for m in methods]
 bar_hatches = [METHOD_STYLES[m]["hatch"] for m in methods]
 method_rows = {"BO": bo, "Random": rand, "Grid": grid}
 vals = {m: {met: best_metric(method_rows[m], met) for met in metrics}
         for m in methods}
 fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.5))
 for ax, metric, title in zip(axes, metrics, titles):
  v = [vals[m][metric] for m in methods]
  bars = ax.bar(methods, v, color=bar_colors, hatch=bar_hatches,
 width=0.55, edgecolor="white", linewidth=0.8)
 # Annotate bars
  for bar, val in zip(bars, v):
   if not np.isnan(val):
    fmt = f"{val:.3f}" if metric == "accuracy" else f"{val:.4f}"
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()*1.01,
 fmt, ha="center", va="bottom", fontsize=6.5)
  ax.set_title(title, fontsize=7.5)
  ax.set_xticks(range(len(methods)))
  ax.set_xticklabels(methods, fontsize=7)
 # Add constraint line for accuracy panel
  if metric == "accuracy":
   ax.axhline(ACCURACY_THRESHOLD, color="red", linestyle="--",
 linewidth=0.8, alpha=0.7,
label=f"min={ACCURACY_THRESHOLD}")
   ax.legend(fontsize=6, loc="lower right")
 fig.suptitle("Best Feasible Configuration per Search Method", fontsize=9,
y=1.02)
 fig.tight_layout()
 out = FIGURES_DIR / "comparison_bar.pdf"
 fig.savefig(out)
 plt.close(fig)
 logger.info(f"Figure 2 saved: {out}")
 return out


def load_json(path: Path) -> dict:
 with open(path) as f:
  return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Pareto Frontier — accuracy vs energy (MOO headline result)
# ─────────────────────────────────────────────────────────────────────────────

def figure3_pareto_frontier(moo: List[Dict], moo_random: List[Dict],
 reference_front: List[Dict]) -> Path:
 """
 Figure 3: accuracy vs energy_mj (log scale), MOO vs Random at matched
 budget, against the exhaustive reference front.

 Replaces a memory-vs-energy view: memory_kb and energy_mj are 88%
 rank-correlated over this search space (both are essentially params x
 bits), so that pairing shows almost no tradeoff. accuracy vs energy is
 the pair with real tension (52% correlation) and is what the MOO
 objectives actually optimize.

 Energy spans ~3 orders of magnitude (0.0008-0.09 mJ) so the y-axis is
 log-scaled — a linear axis crushes the low-energy region where the
 interesting tradeoffs live.
 """
 fig, ax = plt.subplots(figsize=(3.5, 3.2))

 # ── Reference front: filled boundary showing the achievable ceiling ──────
 if reference_front:
  ref_sorted = sorted(reference_front, key=lambda r: float(r["accuracy"]))
  ref_acc = [float(r["accuracy"]) for r in ref_sorted]
  ref_e = [float(r["energy_mj"]) for r in ref_sorted]
  ax.plot(ref_acc, ref_e, color=METHOD_STYLES["Reference"]["color"],
  linewidth=1.2, linestyle="-", zorder=3,
  label=f"{METHOD_STYLES['Reference']['label']} ({len(ref_sorted)} pts)")
  ax.fill_between(ref_acc, ref_e, [max(ref_e) * 3] * len(ref_e),
  color=METHOD_STYLES["Reference"]["color"], alpha=0.06, zorder=1)

 # ── Background scatter: all feasible points per method ───────────────────
 for method, rows in [("Random", moo_random), ("MOO", moo)]:
  feasible = get_feasible(rows)
  if not feasible:
   logger.warning(f"figure3_pareto_frontier: no feasible rows for {method}")
   continue
  st = METHOD_STYLES[method]
  ax.scatter([float(r["accuracy"]) for r in feasible],
  [float(r["energy_mj"]) for r in feasible],
  color=st["color"], marker=st["marker"], s=14, alpha=0.18,
  edgecolors="none", zorder=2)

 # ── Pareto fronts per method (accuracy/energy plane) ──────────────────────
 pareto_counts = {}
 for method, rows in [("Random", moo_random), ("MOO", moo)]:
  feasible = get_feasible(rows)
  front = pm_pareto_front(feasible)
  pareto_counts[method] = len(front)
  if not front:
   continue
  front_sorted = sorted(front, key=lambda r: float(r["accuracy"]))
  st = METHOD_STYLES[method]
  acc_p = [float(r["accuracy"]) for r in front_sorted]
  e_p = [float(r["energy_mj"]) for r in front_sorted]
  ax.plot(acc_p, e_p, color=st["color"], linestyle=st["linestyle"],
  linewidth=1.6, drawstyle="steps-pre", zorder=4,
  label=f"{st['label']} ({len(front)} pts)")
  ax.scatter(acc_p, e_p, color=st["color"], marker=st["marker"], s=50,
  zorder=5, edgecolors="white", linewidths=0.6)

 ax.set_yscale("log")
 ax.axvline(ACCURACY_THRESHOLD, color="#999999", linestyle=":", linewidth=0.8, zorder=1)
 ax.text(ACCURACY_THRESHOLD, ax.get_ylim()[1], " min acc", fontsize=6,
 color="#777777", ha="left", va="top", rotation=90)

 ax.set_xlabel("Accuracy")
 ax.set_ylabel("Inference Energy (mJ, log scale)")
 ax.set_title("Pareto Frontier: Accuracy vs. Energy")
 ax.legend(loc="lower left", fontsize=6.5, framealpha=0.9, handlelength=2.2)

 out = FIGURES_DIR / "pareto_frontier.pdf"
 fig.tight_layout()
 fig.savefig(out)
 plt.close(fig)
 logger.info(f"Figure 3 saved: {out} | Pareto sizes — "
 f"MOO: {pareto_counts.get('MOO', 0)}, Random: {pareto_counts.get('Random', 0)}, "
 f"Reference: {len(reference_front)}")
 return out


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE F: Latency-budget sensitivity (reference front only)
# ─────────────────────────────────────────────────────────────────────────────

def figure_latency_sensitivity(sensitivity: dict) -> Path:
 """
 Reference-front hypervolume and feasible fraction vs. latency budget
 (from reference_front_summary.json's "latency_sensitivity" field,
 computed by re-using the SAME enumerated grid at 3 budgets — no
 re-simulation). This is a property of the search space itself, not of
 any particular optimizer: it checks that conclusions do not hinge on
 the specific 100ms choice.
 """
 # Keys in reference_front_summary.json are stored as "50.0", "100.0", "200.0".
 budgets = sorted(float(b) for b in sensitivity.keys())
 hvs = [sensitivity[f"{b:.1f}"]["hypervolume"] for b in budgets]
 fractions = [sensitivity[f"{b:.1f}"]["feasible_fraction"] * 100 for b in budgets]

 fig, ax1 = plt.subplots(figsize=(3.5, 2.6))
 ax2 = ax1.twinx()

 l1, = ax1.plot(budgets, hvs, color=METHOD_STYLES["Reference"]["color"],
 marker="o", markersize=5, linewidth=1.6, label="Reference hypervolume")
 l2, = ax2.plot(budgets, fractions, color=METHOD_STYLES["MOO"]["color"],
 marker="s", markersize=5, linewidth=1.6, linestyle="--",
 label="Feasible fraction of search space")

 ax1.axvline(LATENCY_BUDGET_MS, color="#999999", linestyle=":", linewidth=0.8, zorder=1)
 ax1.text(LATENCY_BUDGET_MS, 0.95, "used in this paper", fontsize=6,
 color="#777777", ha="center", va="top", transform=ax1.get_xaxis_transform())

 ax1.set_xlabel("Latency Budget (ms)")
 ax1.set_ylabel("Reference Hypervolume", color=METHOD_STYLES["Reference"]["color"])
 ax2.set_ylabel("Feasible Fraction (%)", color=METHOD_STYLES["MOO"]["color"])
 ax1.set_title("Latency Budget Sensitivity")
 ax1.legend(handles=[l1, l2], loc="lower right", fontsize=6.5, framealpha=0.9)

 out = FIGURES_DIR / "latency_sensitivity.pdf"
 fig.tight_layout()
 fig.savefig(out)
 plt.close(fig)
 logger.info(f"Figure F saved: {out} | HV at budgets {budgets}: {hvs}")
 return out


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE C: Hypervolume convergence
# ─────────────────────────────────────────────────────────────────────────────

def figure_hypervolume_convergence(moo: List[Dict], reference_hv: float,
 budget_comparison: dict = None) -> Path:
 """
 Hypervolume vs. evaluation count for the headline MOO run, with a
 horizontal ceiling at the exhaustive reference front's hypervolume.
 If budget_comparison (from robustness_summary.json) is supplied, overlay
 cross-seed mean +/- std at each matched budget checkpoint as error bars
 — this is where sample-to-sample uncertainty becomes visible, which the
 single-trace headline run alone cannot show.
 """
 fig, ax = plt.subplots(figsize=(3.5, 2.6))
 trials = list(range(1, len(moo) + 1))
 hv_trace = [float(r["hypervolume"]) for r in moo]
 ax.plot(trials, hv_trace, color=METHOD_STYLES["MOO"]["color"],
 linewidth=1.6, label="MOO (this run)", zorder=3)

 if budget_comparison:
  budgets = sorted(int(b) for b in budget_comparison.keys())
  means = [budget_comparison[str(b)]["moo_hv_mean"] for b in budgets]
  stds = [budget_comparison[str(b)]["moo_hv_std"] for b in budgets]
  ax.errorbar(budgets, means, yerr=stds, fmt="o", color=METHOD_STYLES["MOO"]["color"],
  markersize=4, capsize=3, linewidth=0.8, zorder=4,
  label=f"MOO cross-seed mean +/- std (n seeds)")
  rand_means = [budget_comparison[str(b)]["random_hv_mean"] for b in budgets]
  rand_stds = [budget_comparison[str(b)]["random_hv_std"] for b in budgets]
  ax.errorbar(budgets, rand_means, yerr=rand_stds, fmt="s",
  color=METHOD_STYLES["Random"]["color"], markersize=4, capsize=3,
  linewidth=0.8, zorder=4, label="Random cross-seed mean +/- std")

 ax.axhline(reference_hv, color=METHOD_STYLES["Reference"]["color"],
 linestyle="--", linewidth=1.0, zorder=2,
 label=f"Reference front ceiling ({reference_hv:.5f})")

 ax.set_xlabel("Number of Evaluations")
 ax.set_ylabel("Hypervolume")
 ax.set_title("Hypervolume Convergence")
 ax.legend(loc="lower right", fontsize=6, framealpha=0.9, handlelength=2.0)

 out = FIGURES_DIR / "hypervolume_convergence.pdf"
 fig.tight_layout()
 fig.savefig(out)
 plt.close(fig)
 logger.info(f"Figure C saved: {out} | final HV={hv_trace[-1]:.6f}, "
 f"reference HV={reference_hv:.6f} "
 f"({100*hv_trace[-1]/reference_hv:.1f}% of ceiling)")
 return out


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE D: SOO collapse vs MOO front
# ─────────────────────────────────────────────────────────────────────────────

def figure_soo_vs_moo(bo: List[Dict], moo: List[Dict]) -> Path:
 """
 Directly contrasts the single-objective constrained formulation (which
 converges to one point, revisiting it repeatedly) against the
 multi-objective formulation (which spreads across a front). Marker area
 for BO points scales with how many times that exact configuration was
 evaluated, making the SOO collapse visible rather than hidden by
 overplotting.
 """
 fig, ax = plt.subplots(figsize=(3.5, 3.0))

 bo_feasible = get_feasible(bo)
 counts = {}
 for r in bo_feasible:
  key = (round(float(r["accuracy"]), 4), round(float(r["energy_mj"]), 6))
  counts[key] = counts.get(key, 0) + 1
 bo_acc = [k[0] for k in counts]
 bo_e = [k[1] for k in counts]
 bo_sizes = [18 + 40 * (c - 1) for c in counts.values()]
 ax.scatter(bo_acc, bo_e, s=bo_sizes, color=METHOD_STYLES["BO"]["color"],
 marker=METHOD_STYLES["BO"]["marker"], alpha=0.75, edgecolors="white",
 linewidths=0.5, zorder=4,
 label=f"SOO/BO ({len(bo_feasible)} evals -> {len(counts)} distinct)")

 moo_feasible = get_feasible(moo)
 moo_front = pm_pareto_front(moo_feasible)
 moo_front_sorted = sorted(moo_front, key=lambda r: float(r["accuracy"]))
 ax.scatter([float(r["accuracy"]) for r in moo_feasible],
 [float(r["energy_mj"]) for r in moo_feasible],
 s=10, color=METHOD_STYLES["MOO"]["color"], alpha=0.15,
 edgecolors="none", zorder=2)
 ax.plot([float(r["accuracy"]) for r in moo_front_sorted],
 [float(r["energy_mj"]) for r in moo_front_sorted],
 color=METHOD_STYLES["MOO"]["color"], linewidth=1.6, drawstyle="steps-pre",
 zorder=3, label=f"MOO front ({len(moo_front)} pts)")

 ax.set_yscale("log")
 ax.set_xlabel("Accuracy")
 ax.set_ylabel("Inference Energy (mJ, log scale)")
 ax.set_title("Single- vs. Multi-Objective Search")
 ax.legend(loc="lower left", fontsize=6.5, framealpha=0.9, handlelength=2.2)

 out = FIGURES_DIR / "soo_vs_moo.pdf"
 fig.tight_layout()
 fig.savefig(out)
 plt.close(fig)
 logger.info(f"Figure D saved: {out} | SOO: {len(bo_feasible)} evals -> "
 f"{len(counts)} distinct configs | MOO front: {len(moo_front)} pts")
 return out


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE E: Parallel coordinates of the MOO Pareto set
# ─────────────────────────────────────────────────────────────────────────────

def figure_pareto_parallel_coords(moo: List[Dict]) -> Path:
 """
 All 6 search parameters on parallel axes, one line per Pareto-optimal
 MOO configuration, colored by energy. Shows what the good configurations
 actually look like, and visually confirms time_slice/priority vary along
 the front (they no longer are inert, unlike the old single-objective
 corner solution).
 """
 feasible = get_feasible(moo)
 front = pm_pareto_front(feasible)
 if not front:
  raise RuntimeError("figure_pareto_parallel_coords: empty MOO Pareto front")
 front = sorted(front, key=lambda r: float(r["accuracy"]))

 params = ["num_layers", "width_mult", "quant_bits", "time_slice", "priority", "dvfs_factor"]
 bounds = {
  "num_layers": (2, 4), "width_mult": (0.25, 1.0), "quant_bits": (4, 32),
  "time_slice": (5.0, 50.0), "priority": (0, 2), "dvfs_factor": (0.5, 1.0),
 }

 fig, ax = plt.subplots(figsize=(7.0, 3.4))
 energies = np.array([float(r["energy_mj"]) for r in front])
 norm = matplotlib.colors.LogNorm(vmin=energies.min(), vmax=energies.max())
 cmap = matplotlib.colormaps["viridis"]

 xs = list(range(len(params)))
 for r, e in zip(front, energies):
  ys = [(float(r[p]) - bounds[p][0]) / (bounds[p][1] - bounds[p][0]) for p in params]
  ax.plot(xs, ys, color=cmap(norm(e)), linewidth=1.1, alpha=0.75)

 ax.set_xticks(xs)
 ax.set_xticklabels(params, fontsize=7)
 for i, p in enumerate(params):
  lo, hi = bounds[p]
  ax.text(i, -0.14, f"min={lo:g}", fontsize=6, ha="center", va="top", transform=ax.get_xaxis_transform())
  ax.text(i, 1.06, f"max={hi:g}", fontsize=6, ha="center", va="bottom", transform=ax.get_xaxis_transform())
 ax.set_ylim(-0.02, 1.02)
 ax.set_yticks([])
 ax.set_title(f"Pareto Set Parameters ({len(front)} configurations)", pad=18)

 sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
 cbar = fig.colorbar(sm, ax=ax, pad=0.02)
 cbar.set_label("Energy (mJ)", fontsize=7)
 cbar.ax.tick_params(labelsize=6)

 fig.subplots_adjust(top=0.80, bottom=0.24, left=0.03, right=0.90)
 out = FIGURES_DIR / "pareto_parallel_coords.pdf"
 fig.savefig(out)
 plt.close(fig)
 logger.info(f"Figure E saved: {out} | {len(front)} Pareto configs")
 return out


def figure_energy_improvement_budget(bo: List[Dict], rand: List[Dict],
                                     grid: List[Dict]) -> Path:
    """
    Figure A: Energy Improvement vs. Evaluation Budget.
 
    Saved as: figures/energy_improvement_budget.pdf
 
    Plots the cumulative percentage improvement in best feasible energy
    as a function of the number of configurations evaluated. Each method's
    curve starts at 0% when its first feasible configuration is found and
    rises as better configurations are discovered.
 
    The improvement is defined relative to each method's own first feasible
    energy (not a shared baseline), so all curves start from the same
    y-origin (0%) at the trial where the first feasible config appears.
    This isolates *optimisation efficiency* from *feasibility rate* — the
    latter is shown separately in figure_feasible_discovery_rate().
 
    Scientific purpose:
        Directly answers: "After k evaluations, by how much has each method
        improved over its starting point?" A higher final improvement
        percentage, or the same improvement reached in fewer evaluations,
        constitutes evidence of superior optimisation efficiency.
 
    Paper placement:
        Section V-A. Single column (3.5 × 2.5 in).
 
    Suggested caption (copy into Overleaf):
        "Cumulative energy improvement (\\%) relative to each method's
        first feasible configuration, as a function of the number of
        configurations evaluated. BO (solid) achieves a higher final
        improvement than Random Search (dashed) and Grid Search (dotted)
        within the same evaluation budget, demonstrating superior
        sample efficiency in locating energy-optimal configurations."
 
    Args:
        bo   (List[Dict]): All rows from bo_results.csv.
        rand (List[Dict]): All rows from random_results.csv.
        grid (List[Dict]): All rows from grid_results.csv.
 
    Returns:
        Path: Absolute path to saved energy_improvement_budget.pdf.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
 
    # Determine shared x-axis upper limit (max trials across all methods).
    n_max = max(len(bo), len(rand), len(grid))
 
    for method, rows in [("BO", bo), ("Random", rand), ("Grid", grid)]:
        series = compute_energy_improvement_series(rows)
        trials = list(range(1, len(series) + 1))
        style  = METHOD_STYLES[method]
 
        # Replace NaN with None for matplotlib gap rendering.
        # Matplotlib skips None points rather than drawing to them.
        y_plot = [v if not math.isnan(v) else None for v in series]
 
        ax.plot(
            trials, y_plot,
            color     = style["color"],
            linestyle = style["linestyle"],
            marker    = style["marker"],
            markersize  = 3,
            markevery   = max(1, len(trials) // 8),
            label       = style["label"],
            zorder      = 3,
        )
 
        # Annotate final improvement value (last non-NaN element).
        final_val = next(
            (v for v in reversed(series) if not math.isnan(v)), None
        )
        if final_val is not None:
            x_last = len(series)
            # Clamp annotation x so it does not overrun the right axis edge.
            x_ann  = min(x_last + 0.3, n_max * 0.97)
            ax.annotate(
                f"{final_val:.1f}%",
                xy       = (x_last, final_val),
                xytext   = (x_ann,  final_val),
                fontsize = 6.5,
                color    = style["color"],
                va       = "center",
                ha       = "left",
                arrowprops = dict(
                    arrowstyle  = "-",
                    color       = style["color"],
                    lw          = 0.6,
                    shrinkA     = 2,
                    shrinkB     = 2,
                ),
                zorder = 5,
            )
 
    ax.set_xlabel("Number of Evaluations (log scale)")
    ax.set_ylabel("Energy Improvement (%)")
    ax.set_title("Energy Improvement vs. Evaluation Budget")
    # Log x-axis: Grid runs the full 972-config lattice while BO/Random run
    # 30 — a linear axis crams BO/Random into an invisible sliver.
    ax.set_xscale("log")
    ax.set_xlim(left=1, right=n_max * 1.3)
    ax.set_ylim(bottom=-2)     # small margin below 0% for visual clarity
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)   # 0% reference line
    # Placed in the empty gap between BO/Random's 30-eval curves and Grid's
    # 972-eval curve (log-scale x means this band is genuinely empty),
    # rather than upper-left where it would cover BO's own annotation.
    ax.legend(loc="center", bbox_to_anchor=(0.62, 0.42), framealpha=0.9, handlelength=2.5)
 
    out = FIGURES_DIR / "energy_improvement_budget.pdf"
    fig.savefig(out)
    plt.close(fig)
 
    # Log final improvement values for paper reporting.
    for method, rows in [("BO", bo), ("Random", rand), ("Grid", grid)]:
        s = compute_energy_improvement_series(rows)
        final = next((v for v in reversed(s) if not math.isnan(v)), float("nan"))
        logger.info(
            f"Figure A | {METHOD_STYLES[method]['label']:<28} "
            f"final_improvement={final:.2f}%  "
            f"n_trials={len(rows)}"
        )
 
    logger.info(f"Figure A saved: {out}")
    return out
 
 
def figure_feasible_discovery_rate(bo: List[Dict], rand: List[Dict],
                                   grid: List[Dict]) -> Path:
    """
    Figure B: Cumulative Feasible Configuration Discovery Rate.
 
    Saved as: figures/feasible_discovery_rate.pdf
 
    Plots the cumulative count of feasible configurations (accuracy >=
    ACCURACY_THRESHOLD = {threshold}) discovered as a function of the
    number of configurations evaluated.
 
    A feasible configuration satisfies the accuracy constraint imposed
    in bo_experiment.py: accuracy >= ACCURACY_THRESHOLD. Infeasible
    configurations, while evaluated, cannot be deployed on the target
    embedded system and provide no direct utility.
 
    Scientific purpose:
        A steeper cumulative curve indicates the method discovers valid,
        deployable configurations more rapidly. This matters because:
        (1) Early feasible hits provide the "first feasible energy"
            reference used in figure_energy_improvement_budget().
        (2) In a real deployment budget, more feasible options means
            more choices in the accuracy–energy tradeoff.
        (3) A method that wastes many evaluations on infeasible regions
            is less suited to constrained optimisation — exactly the
            regime this paper targets.
 
    Paper placement:
        Section V-B. Single column (3.5 × 2.5 in).
 
    Suggested caption (copy into Overleaf):
        "Cumulative number of feasible configurations (accuracy
        $\\geq {threshold}$) discovered as a function of evaluations.
        A steeper curve indicates higher constraint-satisfaction
        efficiency. BO maintains a consistently higher feasible
        discovery rate, reflecting the advantage of surrogate-guided
        search over unguided baselines in constrained optimisation."
 
    Args:
        bo   (List[Dict]): All rows from bo_results.csv.
        rand (List[Dict]): All rows from random_results.csv.
        grid (List[Dict]): All rows from grid_results.csv.
 
    Returns:
        Path: Absolute path to saved feasible_discovery_rate.pdf.
    """.format(threshold=ACCURACY_THRESHOLD)
 
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    n_max = max(len(bo), len(rand), len(grid))

    # Plotted as a RATE (% of that method's own budget), not a raw count:
    # Grid's budget (972) is ~32x BO/Random's (30), so raw counts compress
    # BO/Random's curves into a few pixels near y=0 and collide their
    # end-of-line annotations. Rate puts all three methods on the same
    # 0-100% scale regardless of budget.
    for method, rows in [("BO", bo), ("Random", rand), ("Grid", grid)]:
        series = compute_feasible_discovery_series(rows)
        n = len(series)
        trials = list(range(1, n + 1))
        rate_pct = [100.0 * v / n for v in series]
        style  = METHOD_STYLES[method]

        ax.plot(
            trials, rate_pct,
            color      = style["color"],
            linestyle  = style["linestyle"],
            marker     = style["marker"],
            markersize  = 3,
            markevery   = max(1, n // 8),
            label       = f"{style['label']} ({series[-1]}/{n})",
            zorder      = 3,
        )

    # Reference line: 100% = every evaluation was feasible.
    ax.axhline(
        100,
        color     = "#aaaaaa",
        linestyle = (0, (1, 3)),   # loosely dotted
        linewidth = 0.8,
        zorder    = 1,
        label     = "100% (all feasible)",
    )

    ax.set_xlabel("Number of Evaluations (log scale)")
    ax.set_ylabel("Cumulative Feasible Rate (%)")
    ax.set_title(f"Feasible Configuration Discovery Rate (accuracy $\\geq$ {ACCURACY_THRESHOLD})")
    # Log x-axis: Grid runs the full 972-config lattice while BO/Random run
    # 30 — a linear axis crams BO/Random into an invisible sliver.
    ax.set_xscale("log")
    ax.set_xlim(left=1, right=n_max * 1.3)
    ax.set_ylim(0, 108)
    # Upper-left is empty (all curves stay below ~60%) apart from the
    # reference line, unlike lower-right which clips Grid's rising tail.
    ax.legend(loc="upper left", fontsize=6.5, framealpha=0.9, handlelength=2.5)

    out = FIGURES_DIR / "feasible_discovery_rate.pdf"
    fig.savefig(out)
    plt.close(fig)
 
    # Log final counts for paper reporting.
    for method, rows in [("BO", bo), ("Random", rand), ("Grid", grid)]:
        s = compute_feasible_discovery_series(rows)
        pct = 100 * s[-1] / len(rows) if rows else 0.0
        logger.info(
            f"Figure B | {METHOD_STYLES[method]['label']:<28} "
            f"feasible={s[-1]}/{len(rows)}  ({pct:.1f}%)  "
            f"threshold={ACCURACY_THRESHOLD}"
        )
 
    logger.info(f"Figure B saved: {out}")
    return out
 


def generate_latex_table(bo, rand, grid, moo=None, moo_random=None,
 reference_front=None, robustness_summary=None) -> str:
 """
 Generate Table I (SOO, unchanged) and — if MOO data is supplied —
 Table II (MOO headline: hypervolume, IGD to the exhaustive reference
 front, Pareto-set size, n=seeds from robustness_summary.json).
 Paste into Overleaf directly — requires \\usepackage{booktabs}.
 """
 methods = [("Bayesian Opt. (ours)", bo),
 ("Random Search [X]", rand),
 ("Grid Search (exhaustive, 972 configs)", grid)]
 def fmt_row(name, rows):
  acc = best_metric(rows, "accuracy"); eng = best_metric(rows, "energy_mj")
  mem = best_metric(rows, "memory_kb"); lat = best_metric(rows, "latency_ms")
  bold_open = r"\textbf{" if name.startswith("Bayesian") else ""
  bold_close = r"}" if name.startswith("Bayesian") else ""
  return (f"{bold_open}{name}{bold_close} & "
 f"{bold_open}{acc*100:.2f}{bold_close} & "
 f"{bold_open}{eng:.4f}{bold_close} & "
 f"{bold_open}{mem:.1f}{bold_close} & "
 f"{bold_open}{lat:.2f}{bold_close} \\\\")
 lines = [
 r"% Table I: Main Results Comparison (single-objective, constrained)",
 r"% Requires: \usepackage{booktabs} in preamble",
 r"\begin{table}[t]",
 r"\centering",
 r"\caption{Best feasible configuration per search method (accuracy $\geq$"
 r" threshold, latency $\leq$ budget). BO: 30 evaluations. Grid: full 972-"
 r"config lattice (a $\sim$32x evaluation-budget advantage over BO).}",
 r"\label{tab:main_results}",
 r"\begin{tabular}{lcccc}",
 r"\toprule",
 r"Method & Accuracy (\%) $\uparrow$ & Energy (mJ) $\downarrow$ & Memory (KB) $\downarrow$ & Latency (ms) $\downarrow$ \\",
 r"\midrule",
 ] + [fmt_row(n, r) for n, r in methods] + [
 r"\bottomrule",
 r"\multicolumn{5}{l}{\footnotesize Bold = proposed method. Feasibility: accuracy $\geq$ 0.75, latency $\leq$ 100ms.} \\",
 r"\end{tabular}",
 r"\end{table}",
 ]

 if moo is not None and moo_random is not None and reference_front is not None:
  moo_feasible = get_feasible(moo)
  rand_feasible = get_feasible(moo_random)
  moo_front = pm_pareto_front(moo_feasible)
  rand_front = pm_pareto_front(rand_feasible)
  moo_hv = pm_hypervolume(moo_feasible)
  rand_hv = pm_hypervolume(rand_feasible)
  ref_hv = pm_hypervolume(reference_front)

  rs = robustness_summary.get("primary_hypervolume", {}) if robustness_summary else {}
  n_seeds = rs.get("n_seeds", "?")
  moo_ci = rs.get("bootstrap_ci_diff", {})
  ci_str = (f"[{moo_ci['ci_95_lo']:.5f}, {moo_ci['ci_95_hi']:.5f}]"
  if moo_ci else "n/a")
  igd_info = rs.get("igd_to_reference") or {}

  lines += [
  "",
  r"% Table II: Multi-Objective Results (headline)",
  r"\begin{table}[t]",
  r"\centering",
  r"\caption{Multi-objective search results: hypervolume (dominated volume"
  r" relative to a fixed reference point), IGD to the exhaustive reference"
  r" Pareto front (" + f"{len(reference_front)}" + r" points from a"
  r" grid enumeration over the full search space), and Pareto-set size.}",
  r"\label{tab:moo_results}",
  r"\begin{tabular}{lccc}",
  r"\toprule",
  r"Method & Hypervolume $\uparrow$ & IGD $\downarrow$ & Pareto-set size \\",
  r"\midrule",
  r"\textbf{Multi-Obj. BO (ours)} & \textbf{" + f"{moo_hv:.6f}" + r"} & "
  + (r"\textbf{" + f"{igd_info.get('moo_igd_mean', float('nan')):.4f}" + r"}"
  if igd_info.get("moo_igd_mean") is not None else "n/a")
  + r" & \textbf{" + f"{len(moo_front)}" + r"} \\",
  r"Random Search & " + f"{rand_hv:.6f}" + r" & "
  + (f"{igd_info.get('random_igd_mean', float('nan')):.4f}"
  if igd_info.get("random_igd_mean") is not None else "n/a")
  + r" & " + f"{len(rand_front)}" + r" \\",
  r"Reference (exhaustive) & " + f"{ref_hv:.6f}" + r" & 0.0000 & "
  + f"{len(reference_front)}" + r" \\",
  r"\bottomrule",
  r"\multicolumn{4}{l}{\footnotesize Bootstrap 95\% CI on HV(MOO)$-$HV(Random), n="
  + f"{n_seeds}" + r" seeds: " + ci_str + r".} \\",
  r"\end{tabular}",
  r"\end{table}",
  ]

 table = "\n".join(lines)
 path = FIGURES_DIR / "table_main_results.tex"
 with open(path, "w") as f:
  f.write(table)
 logger.info(f"LaTeX table saved: {path}")
 return table

 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    # ── SOO data (single-objective, constrained) ──────────────────────────────
    bo   = load_csv(RESULTS_DIR / "bo_results.csv")
    rand = load_csv(RESULTS_DIR / "random_results.csv")
    grid = load_csv(RESULTS_DIR / "grid_results.csv")
    print(f"Loaded SOO: BO={len(bo)}, Random={len(rand)}, Grid={len(grid)} rows")

    # ── MOO data (multi-objective headline) ───────────────────────────────────
    moo_path = RESULTS_DIR / "moo_results.csv"
    moo_random_path = RESULTS_DIR / "random_moo_budget.csv"
    ref_front_path = RESULTS_DIR / "reference_pareto_front.csv"
    ref_summary_path = RESULTS_DIR / "reference_front_summary.json"
    robustness_path = RESULTS_DIR / "robustness_summary.json"

    moo = load_csv(moo_path) if moo_path.exists() else []
    moo_random = load_csv(moo_random_path) if moo_random_path.exists() else []
    reference_front = load_csv(ref_front_path) if ref_front_path.exists() else []
    reference_summary = load_json(ref_summary_path) if ref_summary_path.exists() else None
    robustness_summary = load_json(robustness_path) if robustness_path.exists() else None
    print(f"Loaded MOO: MOO={len(moo)}, Random(matched budget)={len(moo_random)}, "
          f"Reference front={len(reference_front)} pts")

    # ── Generate SOO figures ───────────────────────────────────────────────────
    figure1_convergence(bo, rand, grid)
    figure2_comparison_bar(bo, rand, grid)
    figure_energy_improvement_budget(bo, rand, grid) # Figure A
    figure_feasible_discovery_rate(bo, rand, grid)   # Figure B

    # ── Generate MOO figures (require moo_results.csv etc.) ───────────────────
    generated_moo = False
    if moo and moo_random and reference_front:
        figure3_pareto_frontier(moo, moo_random, reference_front)   # Figure 3
        ref_hv = pm_hypervolume(reference_front)
        budget_comparison = None
        if robustness_summary:
            budget_comparison = robustness_summary.get("primary_hypervolume", {}).get(
                "budget_matched_comparison")
        figure_hypervolume_convergence(moo, ref_hv, budget_comparison)  # Figure C
        figure_soo_vs_moo(bo, moo)                                      # Figure D
        figure_pareto_parallel_coords(moo)                              # Figure E
        generated_moo = True
    else:
        logger.warning("MOO result files missing — skipping Figures 3, C, D, E. "
                        "Run src/moo_experiment.py and src/reference_front.py first.")

    generated_sensitivity = False
    if reference_summary and reference_summary.get("latency_sensitivity"):
        figure_latency_sensitivity(reference_summary["latency_sensitivity"])  # Figure F
        generated_sensitivity = True
    else:
        logger.warning("reference_front_summary.json missing latency_sensitivity — "
                        "skipping Figure F. Run src/reference_front.py first.")

    table = generate_latex_table(bo, rand, grid, moo=moo or None,
        moo_random=moo_random or None, reference_front=reference_front or None,
        robustness_summary=robustness_summary)

    # ── Report generated files ─────────────────────────────────────────────────
    outputs = [
        "convergence_curve.pdf",
        "comparison_bar.pdf",
        "energy_improvement_budget.pdf",
        "feasible_discovery_rate.pdf",
        "table_main_results.tex",
    ]
    if generated_moo:
        outputs += [
            "pareto_frontier.pdf",
            "hypervolume_convergence.pdf",
            "soo_vs_moo.pdf",
            "pareto_parallel_coords.pdf",
        ]
    if generated_sensitivity:
        outputs.append("latency_sensitivity.pdf")
    print("\nGenerated outputs:")
    for f in outputs:
        path = FIGURES_DIR / f
        size = path.stat().st_size if path.exists() else 0
        print(f"  {'✓' if size > 0 else '✗'} {f} ({size/1024:.1f}KB)")

    print("\nLaTeX table snippet:")
    print("\n".join(table.split("\n")[6:12]))

    print("\n── Figure A diagnostic (energy improvement) ─────────────────────")
    for method, rows in [("BO", bo), ("Random", rand), ("Grid", grid)]:
        s     = compute_energy_improvement_series(rows)
        final = next((v for v in reversed(s) if not math.isnan(v)), float("nan"))
        first_hit = next((i+1 for i, v in enumerate(s) if not math.isnan(v)), None)
        print(f"  {METHOD_STYLES[method]['label']:<28} "
              f"first_feasible_at_trial={first_hit}  "
              f"final_improvement={final:.2f}%")

    print("\n── Figure B diagnostic (feasible discovery rate) ────────────────")
    for method, rows in [("BO", bo), ("Random", rand), ("Grid", grid)]:
        s   = compute_feasible_discovery_series(rows)
        pct = 100 * s[-1] / len(rows) if rows else 0.0
        print(f"  {METHOD_STYLES[method]['label']:<28} "
              f"feasible={s[-1]}/{len(rows)} ({pct:.1f}%)")

    if generated_moo:
        moo_feasible = get_feasible(moo)
        moo_hv = pm_hypervolume(moo_feasible)
        print("\n── MOO diagnostic ────────────────────────────────────────────")
        print(f"  MOO final HV={moo_hv:.6f} | Reference HV={ref_hv:.6f} "
              f"({100*moo_hv/ref_hv:.1f}% of ceiling)")
    print("─────────────────────────────────────────────────────────────────")
 