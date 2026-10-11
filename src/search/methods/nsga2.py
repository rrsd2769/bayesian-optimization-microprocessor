"""NSGA-II (pymoo): mixed-variable encoding, population 10 x 10 generations, constraint-domination."""
import numpy as np
from pymoo.algorithms.moo.nsga2 import RankAndCrowding
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.variable import Choice, Integer, Real

from src.search.constraints import CONSTRAINTS, violations
from src.search.methods._encoding import design_from_indices, indices_of
from src.search.space import SPACE

POP_SIZE = 10
ENERGY_STAND_IN = 1e9     # finite energy for memory-infeasible designs; constraint-domination ranks them by violation only


def _variables() -> dict:
    out = {}
    for p in SPACE:
        if p.kind == "range":
            out[p.name] = Real(bounds=p.bounds)
        elif p.ordered:
            out[p.name] = Integer(bounds=(0, len(p.values) - 1))
        else:
            out[p.name] = Choice(options=list(range(len(p.values))))
    return out


class Nsga2:
    def __init__(self, seed: int, limits):
        self.limits = limits
        self.problem = Problem(vars=_variables(), n_obj=2, n_ieq_constr=len(CONSTRAINTS))
        self.algo = MixedVariableGA(pop_size=POP_SIZE, survival=RankAndCrowding())
        self.algo.setup(self.problem, seed=seed, verbose=False)
        self.pending = None

    def ask(self) -> list:
        self.pending = self.algo.ask()
        return [design_from_indices(ind.X) for ind in self.pending]

    def tell(self, designs, outcomes) -> None:
        if len(outcomes) != len(self.pending):     # a truncated final batch cannot complete a generation
            return
        for ind, o in zip(self.pending, outcomes):
            energy = o.energy_uj if np.isfinite(o.energy_uj) else ENERGY_STAND_IN
            ind.set("F", np.array([-o.accuracy, energy]))
            ind.set("G", np.array(violations(o, self.limits)))
        self.algo.tell(infills=Population.create(*self.pending))
