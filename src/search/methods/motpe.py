"""MOTPE: Optuna's multi-objective TPE (Ozaki 2022, JAIR), multivariate, constraints through Trial.set_constraint (the four constraints)."""
import numpy as np
import optuna

from src.search.constraints import CONSTRAINTS, violations
from src.search.methods._encoding import design_from_indices
from src.search.space import SPACE

N_STARTUP = 12
ENERGY_STAND_IN = 1e9     # finite energy for memory-infeasible designs; the constraint ranks them last


class Motpe:
    def __init__(self, seed: int, limits):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.limits = limits
        sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=N_STARTUP, seed=seed)
        self.study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
        self.trials = []

    def ask(self) -> list:
        trial = self.study.ask()
        x = {}
        for p in SPACE:
            if p.kind == "range":
                x[p.name] = trial.suggest_float(p.name, *p.bounds)
            elif p.ordered:
                x[p.name] = trial.suggest_int(p.name, 0, len(p.values) - 1)
            else:
                x[p.name] = trial.suggest_categorical(p.name, list(range(len(p.values))))
        self.trials = [trial]
        return [design_from_indices(x)]

    def tell(self, designs, outcomes) -> None:
        (o,), (trial,) = outcomes, self.trials
        for c, v in zip(CONSTRAINTS, violations(o, self.limits)):
            trial.set_constraint(c.name, v)
        energy = o.energy_uj if np.isfinite(o.energy_uj) else ENERGY_STAND_IN
        self.study.tell(trial, [o.accuracy, energy])
