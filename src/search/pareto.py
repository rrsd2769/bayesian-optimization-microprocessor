"""Two-objective Pareto front and hypervolume for (accuracy maximised, energy minimised)."""
from typing import Sequence


def pareto_indices(points: Sequence[tuple]) -> list:
    """Indices of the non-dominated (acc, energy) points; of exact duplicates only the first is kept."""
    order = sorted(range(len(points)), key=lambda i: (-points[i][0], points[i][1], i))
    front, best_energy = [], float("inf")
    for i in order:
        if points[i][1] < best_energy:
            front.append(i)
            best_energy = points[i][1]
    return sorted(front)


def hypervolume(points: Sequence[tuple], ref_acc: float, ref_energy: float) -> float:
    """Area dominated by the points and bounded by the reference corner (acc >= ref_acc, energy <= ref_energy)."""
    pts = [(a, e) for a, e in points if a > ref_acc and e < ref_energy]
    front = sorted((pts[i] for i in pareto_indices(pts)), key=lambda p: p[1])    # ascending energy and accuracy
    area, prev_acc = 0.0, ref_acc
    for a, e in front:
        area += (ref_energy - e) * (a - prev_acc)
        prev_acc = a
    return area
