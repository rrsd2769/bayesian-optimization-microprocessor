"""qNEHVI (Ax / BoTorch): 12 Sobol, then a GP with noisy expected hypervolume improvement (BoTorch's log variant)."""
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement

from src.search.methods._ax import AxStage
from src.search.methods._encoding import design_from_indices
from src.search.space import SPACE


class QNehvi:
    ACQF = qLogNoisyExpectedHypervolumeImprovement

    def __init__(self, seed: int, limits):
        self.stage = AxStage(seed, limits, SPACE, self.ACQF)

    def ask(self) -> list:
        return [design_from_indices(self.stage.propose())]

    def tell(self, designs, outcomes) -> None:
        self.stage.observe(outcomes[0])
