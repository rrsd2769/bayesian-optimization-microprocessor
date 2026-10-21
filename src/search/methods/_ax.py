"""
One Ax Bayesian-optimisation stage: Sobol start, then a Gaussian process with an acquisition function from BoTorch.

Encoding (the Ax methods only): ordered choices (L, w, b, f) are integer ranges over the choice index, which Ax
relaxes and rounds; the unordered priority is a categorical; the time slice is a continuous range. Ax's mixed
categorical encoding of ordered choices took 12-30 s per GP step against 5-15 s for integer ranges.
Objectives: accuracy up, energy down, with objective thresholds at hypervolume corner A (qNEHVI needs a reference
point, and Ax cannot infer one while no evaluated design is feasible, which is the case for the first Sobol points). Outcome constraints: src.search.constraints. A memory-infeasible design
reports accuracy and mem_util only; its infinite energy, latency and miss ratio are left out, and Ax models the
metrics each trial has.
"""
import warnings

import numpy as np
from ax.api.client import Client
from ax.core.metric import Metric
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.core.types import ComparisonOp
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig
from ax.api.utils.generation_strategy_dispatch import GenerationStrategyDispatchStruct, choose_generation_strategy

from src import results
from src.search.constraints import CONSTRAINTS, values

N_SOBOL = 12


def param_configs(space, fixed: dict = None) -> list:
    """SPACE-like params -> Ax configs; names in `fixed` are left out."""
    out = []
    for p in space:
        if fixed and p.name in fixed:
            continue
        if p.kind == "range":
            out.append(RangeParameterConfig(name=p.name, bounds=p.bounds, parameter_type="float"))
        elif p.ordered:
            out.append(RangeParameterConfig(name=p.name, bounds=(0, len(p.values) - 1), parameter_type="int"))
        else:
            out.append(ChoiceParameterConfig(name=p.name, values=list(range(len(p.values))), parameter_type="int",
                                             is_ordered=False))
    return out


class AxStage:
    def __init__(self, seed: int, limits, space, acqf_class, n_sobol: int = N_SOBOL, fixed: dict = None):
        warnings.filterwarnings("ignore")
        self.limits = limits
        ref = results.load("hv_reference")["A"]
        self.client = Client(random_seed=seed)
        self.client.configure_experiment(parameters=param_configs(space, fixed))
        self.client.configure_optimization(
            objective="accuracy, -energy",
            outcome_constraints=[f"{c.name} <= {c.bound(limits)}" for c in CONSTRAINTS])
        cfg = self.client._experiment.optimization_config
        self.client.set_optimization_config(MultiObjectiveOptimizationConfig(
            objective=cfg.objective, outcome_constraints=cfg.outcome_constraints,
            objective_thresholds=[
                ObjectiveThreshold(metric=Metric(name="accuracy"), bound=ref["accuracy"], relative=False, op=ComparisonOp.GEQ),
                ObjectiveThreshold(metric=Metric(name="energy"), bound=ref["energy_uj"], relative=False, op=ComparisonOp.LEQ)]))
        struct = GenerationStrategyDispatchStruct(method="fast", initialization_budget=n_sobol,
                                                  initialization_random_seed=seed, initialize_with_center=False)
        self.client.set_generation_strategy(choose_generation_strategy(struct, botorch_acqf_class=acqf_class))
        self.trial = None

    def propose(self) -> dict:
        (idx, x), = self.client.get_next_trials(max_trials=1).items()
        self.trial = idx
        return x

    def observe(self, o) -> None:
        raw = dict(accuracy=float(o.accuracy))
        for name, v in values(o, self.limits).items():
            if np.isfinite(v):
                raw[name] = float(v)
        if np.isfinite(o.energy_uj):
            raw["energy"] = float(o.energy_uj)
        self.client.complete_trial(self.trial, raw_data=raw)
