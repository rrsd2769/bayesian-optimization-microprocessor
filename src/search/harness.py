"""
run_search(method, seed, budget, scenario) -> Trace

The harness alone owns what must be identical across methods: the design space, the call to evaluate() and the
count against the budget (memory-short-circuited designs count too), the constraint list (constraints.py), the
trace columns with the running hypervolume at both reference corners, and the seeding. A method adapter only
implements ask() -> list of designs and tell(designs, outcomes).
"""
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import results
from src.evaluator import PI_LEVELS, Design, Scenario, evaluate
from src.search.pareto import hypervolume

METHODS = ("qnehvi", "qnparego", "nsga2", "motpe", "random", "sequential")
COLUMNS = ["method", "seed", "eval", "num_layers", "width_mult", "quant_bits", "time_slice_ms", "priority", "freq_idx",
           "accuracy", "energy_uj", "latency_ms", "miss_ratio", "flash_kb", "sram_kb", "acceptable", "violated",
           "hv_A", "hv_B", "ask_s"]


@dataclass
class Trace:
    method: str
    seed: int
    frame: pd.DataFrame


def make_method(method: str, seed: int, scenario: Scenario):
    """Method name -> adapter. Imports lazily so a missing library only breaks the method that needs it."""
    if method == "random":
        from src.search.methods.random import RandomSearch as cls
    elif method == "nsga2":
        from src.search.methods.nsga2 import Nsga2 as cls
    elif method == "motpe":
        from src.search.methods.motpe import Motpe as cls
    elif method == "qnehvi":
        from src.search.methods.qnehvi import QNehvi as cls
    elif method == "qnparego":
        from src.search.methods.qnparego import QNParego as cls
    elif method == "sequential":
        from src.search.methods.sequential import Sequential as cls
    else:
        raise ValueError(f"unknown method '{method}'; known: {METHODS}")
    return cls(seed, scenario.limits)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def run_search(method, seed: int, budget: int = 100, scenario: Scenario = None, corners: dict = None) -> Trace:
    """method: a name in METHODS, or an adapter object with ask() and tell() (the tests pass fakes)."""
    sc = scenario if scenario is not None else Scenario.nominal()
    corners = corners if corners is not None else results.load("hv_reference")
    seed_everything(seed)
    name = method if isinstance(method, str) else type(method).__name__
    adapter = make_method(method, seed, sc) if isinstance(method, str) else method
    rows, acceptable_points = [], []
    while len(rows) < budget:
        t0 = time.perf_counter()
        designs = list(adapter.ask())[: budget - len(rows)]
        ask_s = (time.perf_counter() - t0) / max(len(designs), 1)
        if not designs:
            raise RuntimeError(f"{name}: ask() returned no designs after {len(rows)} evaluations")
        outcomes = [evaluate(d, sc) for d in designs]
        for d, o in zip(designs, outcomes):
            if o.acceptable:
                acceptable_points.append((o.accuracy, o.energy_uj))
            hv = {k: hypervolume(acceptable_points, corners[k]["accuracy"], corners[k]["energy_uj"]) for k in ("A", "B")}
            rows.append((name, seed, len(rows) + 1, d.num_layers, d.width_mult, d.quant_bits, d.time_slice_ms,
                         PI_LEVELS.index(d.priority), d.freq_idx, o.accuracy, o.energy_uj, o.latency_ms, o.miss_ratio,
                         o.flash_kb, o.sram_kb, o.acceptable, "+".join(o.violated), hv["A"], hv["B"], ask_s))
        adapter.tell(designs, outcomes)
    return Trace(name, seed, pd.DataFrame(rows, columns=COLUMNS))
