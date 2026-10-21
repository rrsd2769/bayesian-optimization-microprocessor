"""Random search: uniform over the mixed space."""
import numpy as np

from src.search.methods._encoding import design_from_indices
from src.search.space import SPACE


class RandomSearch:
    def __init__(self, seed: int, limits=None):
        self.rng = np.random.default_rng(seed)

    def ask(self) -> list:
        x = {p.name: int(self.rng.integers(len(p.values))) if p.kind == "choice" else float(self.rng.uniform(*p.bounds))
             for p in SPACE}
        return [design_from_indices(x)]

    def tell(self, designs, outcomes) -> None:
        pass
