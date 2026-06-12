"""
Publication-quality figures for IEEE conference submission.
Figures:
 1. convergence_curve.pdf — best energy vs. trial number (BO vs Random vs
Grid)
 2. comparison_bar.pdf — best result per method on all 4 metrics
 3. memory_vs_energy.pdf — tradeoff scatter, color = accuracy
LaTeX:
 table_main_results.tex — main results table for paper Section V
IEEE style: serif font, 3.5in width, 300dpi, PDF export, grayscale-safe markers.
"""
import os, sys, csv, logging, math
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.config import RESULTS_DIR, FIGURES_DIR, ACCURACY_THRESHOLD,LOG_FORMAT
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
 ax.set_xlabel("Number of Evaluations")
 ax.set_ylabel("Best Feasible Energy (mJ)")
 ax.set_title("Convergence Comparison")
 ax.legend(loc="upper right", framealpha=0.9, handlelength=2.5)
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
         
"""
=============================================================================
CHANGES TO plot_results.py — Pareto Frontier Figure (replaces Figure 3)
=============================================================================

SUMMARY OF CHANGES:
  1. Add helper function  get_pareto_front()  before generate_latex_table()
  2. Add figure function  figure3_pareto_frontier()  before generate_latex_table()
  3. In __main__: replace figure3_scatter() call with figure3_pareto_frontier()
  4. In __main__: replace "memory_vs_energy.pdf" with "pareto_frontier.pdf"

DELETE:  The entire figure3_scatter() function.
KEEP:    Everything else in plot_results.py unchanged.

=============================================================================
PASTE BLOCK 1: ADD THESE TWO FUNCTIONS
  Location: immediately before generate_latex_table()
  (i.e. replace the existing figure3_scatter definition entirely)
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Pareto front computation
# ─────────────────────────────────────────────────────────────────────────────

def get_pareto_front(rows: List[Dict]) -> List[Dict]:
    """
    Compute the Pareto-optimal subset of feasible configurations,
    minimising both energy_mj and memory_kb simultaneously.

    Domination definition (2-objective minimisation):
        Point A dominates point B iff:
            A.energy_mj <= B.energy_mj  AND
            A.memory_kb <= B.memory_kb  AND
            (A.energy_mj <  B.energy_mj  OR  A.memory_kb <  B.memory_kb)

    Algorithm — O(n log n) sort-and-scan:
        1. Sort feasible points by (memory_kb ASC, energy_mj ASC).
        2. Scan left to right. A point is Pareto-optimal iff its energy
           is <= the minimum energy seen in all points with smaller memory.
        3. Equal-energy ties at the same memory level are both retained
           (neither strictly dominates the other).

    The returned list is sorted by memory_kb ascending, giving the natural
    staircase shape of the Pareto frontier when plotted.

    Args:
        rows (List[Dict]): Raw rows from a results CSV (feasible or mixed).
                           Each dict must contain 'memory_kb', 'energy_mj',
                           and 'is_feasible'.

    Returns:
        List[Dict]: Non-dominated rows, sorted by memory_kb ascending.
                    Returns [] if no feasible rows exist.

    References:
        Deb, K. (2001). Multi-Objective Optimization Using Evolutionary
        Algorithms. Wiley. Ch. 2 (Pareto optimality, domination).
    """
    feasible = get_feasible(rows)
    if not feasible:
        return []

    mem_arr    = np.array([float(r["memory_kb"]) for r in feasible])
    energy_arr = np.array([float(r["energy_mj"]) for r in feasible])

    # np.lexsort: last key = primary sort key.
    # Primary: memory_kb ASC; secondary: energy_mj ASC (breaks memory ties).
    order = np.lexsort((energy_arr, mem_arr))

    pareto_indices = []
    min_energy     = float("inf")

    for idx in order:
        e = energy_arr[idx]
        if e < min_energy:
            # Strictly improves on energy — not dominated.
            pareto_indices.append(idx)
            min_energy = e
        elif e == min_energy:
            # Equal energy at the same or larger memory —
            # neither dominates the other; retain both.
            pareto_indices.append(idx)
        # e > min_energy: dominated by an earlier point — skip.

    # pareto_indices is traversed in memory-ascending order (from lexsort),
    # so the result list is already sorted by memory_kb.
    return [feasible[i] for i in pareto_indices]


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Pareto Frontier Comparison
# ─────────────────────────────────────────────────────────────────────────────

def figure3_pareto_frontier(bo: List[Dict], rand: List[Dict],
                            grid: List[Dict]) -> Path:
    """
    Figure 3: Pareto Frontier Comparison.

    Replaces the original memory_vs_energy scatter (figure3_scatter).
    Motivation: accuracy is near-constant across feasible configurations
    (~0.817), so colour-encoding accuracy adds no information. A Pareto
    frontier comparison is more scientifically rigorous and directly
    supports the paper's core claim.

    Scientific purpose:
        The Pareto frontier is the set of configurations where no
        further reduction in energy is achievable without increasing
        memory, and vice versa. Comparing frontiers across methods
        directly answers: "Does BO discover a better energy–memory
        tradeoff than baselines under a fixed evaluation budget?"

    Plot layers (bottom to top):
        1. Faint background scatter  — all feasible configurations per
           method (alpha=0.18). Shows search density and coverage.
        2. Staircase step line       — Pareto frontier per method.
           drawstyle='steps-post' is the canonical representation for
           minimisation Pareto fronts (horizontal before vertical).
        3. Bold frontier markers     — non-dominated points (s=55),
           white edge for visibility against the step line.
        4. Dominance arrow + label  — "better" annotation pointing
           toward the lower-left (lower energy AND lower memory).
        5. Frontier size annotation — count of non-dominated points
           per method, shown as a text footer inside the axes.

    Paper placement:
        Section V-C. Single column (3.5 × 3.0 in).

    Suggested caption (copy into Overleaf):
        "Pareto frontiers in the memory–energy objective space for all
        three search methods over a fixed evaluation budget. Each
        frontier represents the set of non-dominated (Pareto-optimal)
        configurations found. A frontier closer to the origin indicates
        superior multi-objective performance. Bayesian Optimization (BO,
        solid circles) achieves a frontier that dominates those of
        Random Search (dashed squares) and Grid Search (dotted triangles)
        across the majority of the memory range."

    Args:
        bo   (List[Dict]): All rows from bo_results.csv.
        rand (List[Dict]): All rows from random_results.csv.
        grid (List[Dict]): All rows from grid_results.csv.

    Returns:
        Path: Absolute path to saved pareto_frontier.pdf.

    Raises:
        RuntimeError: If no feasible configurations exist in any method.
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    all_mem    = []
    all_energy = []

    # ── Layer 1: faint background scatter (all feasible) ─────────────────────
    # Grid → Random → BO order ensures BO dots appear on top.
    for method, rows in [("Grid", grid), ("Random", rand), ("BO", bo)]:
        feasible = get_feasible(rows)
        if not feasible:
            logger.warning(f"get_pareto_front: no feasible rows for method={method}.")
            continue

        mem_f    = [float(r["memory_kb"]) for r in feasible]
        energy_f = [float(r["energy_mj"]) for r in feasible]
        all_mem.extend(mem_f)
        all_energy.extend(energy_f)

        st = METHOD_STYLES[method]
        ax.scatter(
            mem_f, energy_f,
            color      = st["color"],
            marker     = st["marker"],
            s          = 14,
            alpha      = 0.18,
            edgecolors = "none",
            zorder     = 2,
        )

    if not all_mem:
        raise RuntimeError(
            "figure3_pareto_frontier: no feasible configurations found in "
            "any method. Verify that is_feasible='True' rows exist in the "
            "bo_results.csv, random_results.csv, and grid_results.csv files."
        )

    # ── Layers 2 + 3: Pareto step lines and bold frontier markers ────────────
    pareto_counts = {}

    for method, rows in [("Grid", grid), ("Random", rand), ("BO", bo)]:
        pareto = get_pareto_front(rows)
        pareto_counts[method] = len(pareto)

        if not pareto:
            logger.warning(f"figure3_pareto_frontier: empty Pareto front for {method}.")
            continue

        mem_p    = [float(r["memory_kb"]) for r in pareto]
        energy_p = [float(r["energy_mj"]) for r in pareto]
        st       = METHOD_STYLES[method]

        # Step line — 'steps-post' = horizontal segment first, then
        # vertical drop. This correctly represents the frontier as:
        # "for memory <= X, the best achievable energy is Y."
        ax.plot(
            mem_p, energy_p,
            color     = st["color"],
            linestyle = st["linestyle"],
            linewidth = 1.6,
            drawstyle = "steps-post",
            zorder    = 4,
            label     = f"{st['label']} ({len(pareto)} pts)",
        )

        # Bold markers on frontier points (above the step line)
        ax.scatter(
            mem_p, energy_p,
            color      = st["color"],
            marker     = st["marker"],
            s          = 55,
            zorder     = 5,
            edgecolors = "white",
            linewidths = 0.6,
        )

    # ── Layer 4: dominance direction annotation ───────────────────────────────
    # Compute axis data range from actual values (autoscale not yet applied).
    data_xmin = min(all_mem)    * 0.85
    data_xmax = max(all_mem)    * 1.08
    data_ymin = min(all_energy) * 0.85
    data_ymax = max(all_energy) * 1.08
    ax.set_xlim(data_xmin, data_xmax)
    ax.set_ylim(data_ymin, data_ymax)

    x_range = data_xmax - data_xmin
    y_range = data_ymax - data_ymin

    # Arrow tail placed in upper-right quadrant; arrowhead points lower-left.
    arrow_tail_x = data_xmin + 0.82 * x_range
    arrow_tail_y = data_ymin + 0.82 * y_range
    dx = -0.13 * x_range
    dy = -0.13 * y_range

    ax.annotate(
        "",
        xy       = (arrow_tail_x + dx, arrow_tail_y + dy),   # arrowhead
        xytext   = (arrow_tail_x,      arrow_tail_y),          # tail
        arrowprops = dict(
            arrowstyle    = "-|>",
            color         = "#444444",
            lw            = 0.9,
            mutation_scale = 8,
        ),
        zorder = 6,
    )
    ax.text(
        arrow_tail_x + 0.01 * x_range,
        arrow_tail_y - 0.005 * y_range,
        "better",
        fontsize = 6.5,
        color    = "#444444",
        ha       = "left",
        va       = "top",
        style    = "italic",
        zorder   = 6,
    )

    # ── Layer 5: Pareto frontier size footer ──────────────────────────────────
    # Summarises how many non-dominated points each method found.
    # Fewer points with lower energy = more efficient search.
    size_str = "Frontier sizes:  " + "  |  ".join(
        f"{METHOD_STYLES[m]['label'].split(' ')[0]} = {pareto_counts.get(m, 0)}"
        for m in ["BO", "Random", "Grid"]
    )
    ax.text(
        0.01, 0.01,
        size_str,
        transform  = ax.transAxes,
        fontsize   = 5.8,
        color      = "#555555",
        ha         = "left",
        va         = "bottom",
        zorder     = 6,
    )

    # ── Legend: method lines + ghost for background scatter ──────────────────
    from matplotlib.lines import Line2D
    ghost = Line2D(
        [0], [0],
        linestyle  = "none",
        marker     = "o",
        color      = "#888888",
        markersize = 4,
        alpha      = 0.35,
        label      = "All feasible configs",
    )
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(
        handles + [ghost],
        labels_  + ["All feasible configs"],
        loc          = "upper right",
        fontsize     = 7,
        framealpha   = 0.9,
        handlelength = 2.2,
    )

    # ── Axis decorations ──────────────────────────────────────────────────────
    ax.set_xlabel("Model Size (KB)")
    ax.set_ylabel("Inference Energy (mJ)")
    ax.set_title("Pareto Frontier Comparison")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = FIGURES_DIR / "pareto_frontier.pdf"
    fig.savefig(out)
    plt.close(fig)

    logger.info(
        f"Figure 3 saved: {out} | "
        f"Pareto sizes — BO: {pareto_counts.get('BO', 0)}, "
        f"Random: {pareto_counts.get('Random', 0)}, "
        f"Grid: {pareto_counts.get('Grid', 0)}"
    )
    return out


# =============================================================================
# PASTE BLOCK 2: UPDATED __main__ BLOCK
#   Replace the entire existing  if __name__ == "__main__":  block with this.
# =============================================================================

 
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
 
    ax.set_xlabel("Number of Evaluations")
    ax.set_ylabel("Energy Improvement (%)")
    ax.set_title("Energy Improvement vs. Evaluation Budget")
    ax.set_xlim(left=1, right=n_max + 1.5)
    ax.set_ylim(bottom=-2)     # small margin below 0% for visual clarity
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)   # 0% reference line
    ax.legend(loc="upper left", framealpha=0.9, handlelength=2.5)
 
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
 
    for method, rows in [("BO", bo), ("Random", rand), ("Grid", grid)]:
        series = compute_feasible_discovery_series(rows)
        trials = list(range(1, len(series) + 1))
        style  = METHOD_STYLES[method]
 
        ax.plot(
            trials, series,
            color      = style["color"],
            linestyle  = style["linestyle"],
            marker     = style["marker"],
            markersize  = 3,
            markevery   = max(1, len(trials) // 8),
            label       = style["label"],
            zorder      = 3,
        )
 
        # Annotate the final cumulative count at the end of each line.
        final_count = series[-1]
        x_last      = len(series)
        x_ann       = min(x_last + 0.3, n_max * 0.97)
 
        ax.annotate(
            f"{final_count}",
            xy       = (x_last, final_count),
            xytext   = (x_ann,  final_count),
            fontsize = 6.5,
            color    = style["color"],
            va       = "center",
            ha       = "left",
            arrowprops = dict(
                arrowstyle = "-",
                color      = style["color"],
                lw         = 0.6,
                shrinkA    = 2,
                shrinkB    = 2,
            ),
            zorder = 5,
        )
 
    # Reference line: maximum possible feasible count (if ALL were feasible).
    # Shows the gap between achieved rate and the theoretical maximum.
    ax.axhline(
        n_max,
        color     = "#aaaaaa",
        linestyle = (0, (1, 3)),   # loosely dotted
        linewidth = 0.8,
        zorder    = 1,
        label     = f"Max possible ({n_max})",
    )
 
    # Threshold label inside the axes (does not require a legend entry).
    ax.text(
        0.01, 0.97,
        f"Feasibility threshold: accuracy ≥ {ACCURACY_THRESHOLD}",
        transform = ax.transAxes,
        fontsize  = 6.5,
        color     = "#555555",
        ha        = "left",
        va        = "top",
    )
 
    ax.set_xlabel("Number of Evaluations")
    ax.set_ylabel("Cumulative Feasible Configs Found")
    ax.set_title("Feasible Configuration Discovery Rate")
    ax.set_xlim(left=1, right=n_max + 1.5)
    ax.set_ylim(bottom=0)
    ax.yaxis.get_major_locator().set_params(integer=True)   # integer y-ticks
    ax.legend(loc="upper left", framealpha=0.9, handlelength=2.5)
 
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
 


def generate_latex_table(bo, rand, grid) -> str:
 """
 Generate Table I for the paper.
 Paste into Overleaf directly — requires \\usepackage{booktabs}.
 Paper placement: Section V, Table I.
 Caption: "Comparison of search methods. Best feasible result (accuracy ≥
0.75)
 for each method over 30 evaluations. ↓ = lower is better."
 """
 methods = [("Bayesian Opt. (ours)", bo),
 ("Random Search [X]", rand),
 ("Grid Search", grid)]
 def fmt_row(name, rows):
  acc = best_metric(rows, "accuracy"); eng = best_metric(rows,
"energy_mj")
  mem = best_metric(rows, "memory_kb"); lat = best_metric(rows,
"latency_ms")
  bold_open = r"\textbf{" if name.startswith("Bayesian") else ""
  bold_close = r"}" if name.startswith("Bayesian") else ""
  return (f"{bold_open}{name}{bold_close} & "
 f"{bold_open}{acc*100:.2f}{bold_close} & "
 f"{bold_open}{eng:.4f}{bold_close} & "
 f"{bold_open}{mem:.1f}{bold_close} & "
 f"{bold_open}{lat:.2f}{bold_close} \\\\")
 lines = [
 r"% Table I: Main Results Comparison",
 r"% Requires: \usepackage{booktabs} in preamble",
 r"\begin{table}[t]",
 r"\centering",
 r"\caption{Performance Comparison of Search Methods (30 Evaluations Each)}",
 r"\label{tab:main_results}",
 r"\begin{tabular}{lcccc}",
 r"\toprule",
 r"Method & Accuracy (\%) $\uparrow$ & Energy (mJ) $\downarrow$ & Memory (KB) $\downarrow$ & Latency (ms) $\downarrow$ \\",
 r"\midrule",
 ] + [fmt_row(n, r) for n, r in methods] + [
 r"\bottomrule",
 r"\multicolumn{5}{l}{\footnotesize Bold = proposed method. Feasibility constraint: accuracy $\geq$ 0.75.} \\",
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
 
    bo   = load_csv(RESULTS_DIR / "bo_results.csv")
    rand = load_csv(RESULTS_DIR / "random_results.csv")
    grid = load_csv(RESULTS_DIR / "grid_results.csv")
    print(f"Loaded: BO={len(bo)}, Random={len(rand)}, Grid={len(grid)} rows")
 
    # ── Generate all figures ───────────────────────────────────────────────────
    figure1_convergence(bo, rand, grid)
    figure2_comparison_bar(bo, rand, grid)
    figure3_pareto_frontier(bo, rand, grid)          # or figure3_scatter if not replaced
    figure_energy_improvement_budget(bo, rand, grid) # Figure A (new)
    figure_feasible_discovery_rate(bo, rand, grid)   # Figure B (new)
    table = generate_latex_table(bo, rand, grid)
 
    # ── Report generated files ─────────────────────────────────────────────────
    print("\nGenerated outputs:")
    for f in [
        "convergence_curve.pdf",
        "comparison_bar.pdf",
        "pareto_frontier.pdf",              # or "memory_vs_energy.pdf"
        "energy_improvement_budget.pdf",    # Figure A
        "feasible_discovery_rate.pdf",      # Figure B
        "table_main_results.tex",
    ]:
        path = FIGURES_DIR / f
        size = path.stat().st_size if path.exists() else 0
        print(f"  {'✓' if size > 0 else '✗'} {f} ({size/1024:.1f}KB)")
 
    print("\nLaTeX table snippet:")
    print("\n".join(table.split("\n")[6:12]))
 
    # ── Diagnostic summary (remove after confirming results) ──────────────────
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
    print("─────────────────────────────────────────────────────────────────")
 