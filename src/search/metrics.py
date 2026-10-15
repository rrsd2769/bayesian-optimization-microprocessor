"""
Quality measures of a search trace against the reference front.

hv_percent    running hypervolume as a percentage of the reference front's, at a corner ("A" or "B")
igd           inverted generational distance: mean distance from each reference-front design to the nearest
              design on the trace's acceptable front, in objective space scaled by the corner-A box
              (accuracy by reference maximum minus 0.70, energy by 17,000 minus the reference minimum), so the
              two objectives weigh equally
evals_to      first evaluation at which the hypervolume reaches a fraction of the reference (inf if never)
"""
import math

import numpy as np
import pandas as pd

from src.search.pareto import pareto_indices


def hv_percent(trace: pd.DataFrame, corner: str, reference_hv: dict) -> np.ndarray:
    return 100.0 * trace[f"hv_{corner}"].to_numpy() / reference_hv[corner]


def evals_to(trace: pd.DataFrame, corner: str, reference_hv: dict, fraction: float = 0.9) -> float:
    pct = hv_percent(trace, corner, reference_hv) / 100.0
    hit = np.nonzero(pct >= fraction - 1e-12)[0]
    return float(hit[0] + 1) if len(hit) else math.inf


def _scaled(points, ref_front: pd.DataFrame, corners: dict) -> np.ndarray:
    a_scale = ref_front["accuracy"].max() - corners["A"]["accuracy"]
    e_scale = corners["A"]["energy_uj"] - ref_front["energy_uj"].min()
    return np.array([(a / a_scale, e / e_scale) for a, e in points], dtype=float)


def igd(trace: pd.DataFrame, n: int, ref_front: pd.DataFrame, corners: dict) -> float:
    """IGD of the first n evaluations; inf when none of them is acceptable."""
    ok = trace[(trace["eval"] <= n) & trace["acceptable"]]
    if ok.empty:
        return math.inf
    pts = list(zip(ok["accuracy"], ok["energy_uj"]))
    found = _scaled([pts[i] for i in pareto_indices(pts)], ref_front, corners)
    ref = _scaled(zip(ref_front["accuracy"], ref_front["energy_uj"]), ref_front, corners)
    return float(np.mean([np.min(np.linalg.norm(found - r, axis=1)) for r in ref]))
