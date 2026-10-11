"""
Sequential: two qNEHVI stages in turn.
Stage 1 (50 evaluations): (L, w, b) with tau = 10 ms, pi = equal, f = 180 MHz.
Stage 2 (50 evaluations): (tau, pi, f) and the network, a categorical index into the stage-1 front.
The stage-1 front is the Pareto front of the acceptable stage-1 designs; with none acceptable, the single design
with the smallest total constraint violation stands in.
"""
from src.evaluator import PI_LEVELS
from src.search.constraints import violations
from src.search.methods._ax import AxStage
from src.search.methods._encoding import design_from_indices
from src.search.methods.qnehvi import QNehvi
from src.search.pareto import pareto_indices
from src.search.space import SPACE, Param

STAGE_BUDGET = 50
STAGE1_NAMES = ("num_layers", "width_mult", "quant_bits")
FIXED1 = dict(time_slice=10.0, priority=PI_LEVELS.index("equal"), freq_idx=4)     # index 4 = 180 MHz


class Sequential:
    def __init__(self, seed: int, limits):
        self.seed, self.limits = seed, limits
        space1 = tuple(p for p in SPACE if p.name in STAGE1_NAMES)
        self.stage1 = AxStage(seed, limits, space1, QNehvi.ACQF)
        self.stage2 = None
        self.history = []        # (design, outcome) of stage 1
        self.nets = None

    def _fixed(self, x: dict) -> dict:
        return {**x, **FIXED1}

    def ask(self) -> list:
        if len(self.history) < STAGE_BUDGET:
            return [design_from_indices(self._fixed(self.stage1.propose()))]
        if self.stage2 is None:
            self._start_stage2()
        x = self.stage2.propose()
        L, w, bits = self.nets[int(x.pop("net", 0))]
        x.update(num_layers=SPACE[0].values.index(L), width_mult=SPACE[1].values.index(w),
                 quant_bits=SPACE[2].values.index(bits))
        return [design_from_indices(x)]

    def _start_stage2(self) -> None:
        ok = [(d, o) for d, o in self.history if o.acceptable]
        if ok:
            keep = pareto_indices([(o.accuracy, o.energy_uj) for _, o in ok])
            front = [ok[i][0] for i in keep]
            front.sort(key=lambda d: -next(o.accuracy for dd, o in ok if dd == d))
        else:
            front = [min(self.history, key=lambda h: sum(max(v, 0.0) for v in violations(h[1], self.limits)))[0]]
        self.nets = list(dict.fromkeys((d.num_layers, d.width_mult, d.quant_bits) for d in front))
        space2 = tuple(p for p in SPACE if p.name not in STAGE1_NAMES)
        if len(self.nets) > 1:
            space2 += (Param("net", "choice", values=tuple(range(len(self.nets))), ordered=False),)
        self.stage2 = AxStage(self.seed + 1000, self.limits, space2, QNehvi.ACQF)

    def tell(self, designs, outcomes) -> None:
        (self.stage1 if len(self.history) < STAGE_BUDGET else self.stage2).observe(outcomes[0])
        self.history.append((designs[0], outcomes[0]))
