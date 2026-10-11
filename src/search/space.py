"""
SPACE: the one definition of the design space. Every method adapter (Phase 6) and the reference front read it;
none re-declares it.

    num_layers   {2,3,4,5}                            ordered choice
    width_mult   {0.25, 0.4375, 0.625, 0.8125, 1.0}   ordered choice
    quant_bits   {4,8,16,32}                          ordered choice
    time_slice   [1, 20] ms                           range
    priority     {0: below, 1: equal, 2: above}       unordered choice
    freq_idx     {0..4}                               ordered choice into the device's operating points
"""
from dataclasses import dataclass

from src.evaluator import PI_LEVELS, STM32F446RE, Design
from src.network.arch import LAYER_COUNTS, WIDTH_GRID


@dataclass(frozen=True)
class Param:
    name: str
    kind: str                 # "choice" or "range"
    values: tuple = ()        # choice
    bounds: tuple = ()        # range
    ordered: bool = True


SPACE = (
    Param("num_layers", "choice", values=LAYER_COUNTS),
    Param("width_mult", "choice", values=WIDTH_GRID),
    Param("quant_bits", "choice", values=(4, 8, 16, 32)),
    Param("time_slice", "range", bounds=(1.0, 20.0)),
    Param("priority", "choice", values=tuple(range(len(PI_LEVELS))), ordered=False),
    Param("freq_idx", "choice", values=tuple(range(len(STM32F446RE.points)))),
)
TAU_GRID = 10     # reference front: log-spaced time slices


def to_design(params: dict) -> Design:
    """A parameter dict keyed by SPACE names -> Design."""
    return Design(int(params["num_layers"]), float(params["width_mult"]), int(params["quant_bits"]),
                  float(params["time_slice"]), PI_LEVELS[int(params["priority"])], int(params["freq_idx"]))
