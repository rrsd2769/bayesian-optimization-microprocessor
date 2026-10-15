"""
The comparison summary: HV % of reference and IGD at 25, 50 and 100 evaluations (median and IQR), evaluations to
90 % of the reference hypervolume, a Friedman test across the methods, qNEHVI against every other method with a
one-sided Wilcoxon signed-rank test (Holm-corrected) and Cliff's delta, and the rejected-evaluation breakdown.
Every p-value is saved next to the smallest value the test can reach with the seeds available.
"""
import math

import numpy as np
import pandas as pd
from scipy import stats

from src import results
from src.search import metrics

CHECKPOINTS = (25, 50, 100)
CORNERS = ("A", "B")
REJECTIONS = ("accuracy", "flash", "sram", "latency", "miss_ratio")


def quantiles(x) -> dict:
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    out = dict(n=len(x), n_infinite=int((~np.isfinite(x)).sum()))
    if len(x) and not len(finite) == 0 and len(finite) == len(x):
        out.update(median=float(np.median(x)), q1=float(np.percentile(x, 25)), q3=float(np.percentile(x, 75)))
    else:     # infinities (no acceptable design yet) make the quantiles undefined; use rank-based ones
        s = np.sort(x)
        pick = lambda q: float(s[min(len(s) - 1, int(math.ceil(q * len(s))) - 1)]) if len(s) else math.nan
        out.update(median=pick(0.5), q1=pick(0.25), q3=pick(0.75))
    return out


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((np.sum(a[:, None] > b[None, :]) - np.sum(a[:, None] < b[None, :])) / (len(a) * len(b)))


def holm(pvals: dict) -> dict:
    order = sorted(pvals, key=pvals.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvals[k]))
        out[k] = running
    return out


def load_runs(methods, seeds) -> dict:
    return {(m, s): results.load("run", method=m, seed=s) for m in methods for s in seeds
            if results.exists("run", method=m, seed=s)}


def summarise(runs: dict, methods, reference_hv: dict, ref_front: pd.DataFrame, corners: dict, baseline="qnehvi") -> dict:
    """runs: {(method, seed): trace DataFrame}. Tests use only the seeds present for every method."""
    seeds = sorted({s for _, s in runs})
    common = [s for s in seeds if all((m, s) in runs for m in methods)]
    out = dict(methods=list(methods), seeds_per_method={m: sorted(s for mm, s in runs if mm == m) for m in methods},
               common_seeds=common, per_method={}, tests={})
    for m in methods:
        traces = [runs[(m, s)] for s in out["seeds_per_method"][m]]
        pm = {}
        for c in CORNERS:
            for n in CHECKPOINTS:
                pm[f"hv_pct_{c}_at_{n}"] = quantiles([metrics.hv_percent(t, c, reference_hv)[n - 1] for t in traces])
            pm[f"evals_to_90_{c}"] = quantiles([metrics.evals_to(t, c, reference_hv) for t in traces])
        for n in CHECKPOINTS:
            pm[f"igd_at_{n}"] = quantiles([metrics.igd(t, n, ref_front, corners) for t in traces])
        allrows = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame(columns=["violated"])
        viol = allrows["violated"].fillna("")
        pm["rejected"] = {k: int(viol.str.contains(k).sum()) for k in REJECTIONS}
        pm["rejected"]["total_evaluations"] = int(len(allrows))
        pm["rejected"]["unacceptable"] = int((~allrows["acceptable"]).sum()) if len(allrows) else 0
        out["per_method"][m] = pm
    if len(common) >= 2:
        for c in CORNERS:
            final = {m: [metrics.hv_percent(runs[(m, s)], c, reference_hv)[-1] for s in common] for m in methods}
            fr = stats.friedmanchisquare(*[final[m] for m in methods])
            n, k = len(common), len(methods)
            out["tests"][f"friedman_{c}"] = dict(statistic=float(fr.statistic), p=float(fr.pvalue), n_seeds=n, n_methods=k,
                                                 min_exact_p=float((1.0 / math.factorial(k)) ** (n - 1)),
                                                 note="p from the chi-square approximation; min_exact_p is the smallest exact permutation p")
            raw, detail = {}, {}
            for m in methods:
                if m == baseline:
                    continue
                diff = np.array(final[baseline]) - np.array(final[m])
                n_eff = int((diff != 0).sum())
                if n_eff == 0:
                    p = 1.0
                else:
                    p = float(stats.wilcoxon(final[baseline], final[m], alternative="greater", zero_method="wilcox").pvalue)
                raw[m] = p
                detail[m] = dict(p=p, n_effective_pairs=n_eff, min_achievable_p=float(0.5 ** n_eff) if n_eff else 1.0,
                                 cliffs_delta=cliffs_delta(final[baseline], final[m]),
                                 median_diff_pct_points=float(np.median(diff)))
            for m, adj in holm(raw).items():
                detail[m]["p_holm"] = adj
            out["tests"][f"wilcoxon_{baseline}_vs_{c}"] = detail
    return out


def run(baseline="qnehvi") -> dict:
    from src.search.harness import METHODS
    ref = results.load("reference_front_summary")["hypervolume"]
    corners = results.load("hv_reference")
    front = results.load("reference_front")
    ref_front = front.rename(columns={"accuracy": "accuracy", "energy_uj": "energy_uj"})
    runs = load_runs(METHODS, range(1, 31))
    return summarise(runs, METHODS, ref, ref_front, corners, baseline)
