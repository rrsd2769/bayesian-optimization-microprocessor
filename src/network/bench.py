"""Training-speed benchmark on the worst-case architecture, and the epoch count it implies."""
import tempfile
from pathlib import Path

from src.jobs import Deadline
from src.network.arch import Arch
from src.network.train import load_data, train

WORST_CASE = Arch(5, 1.0)
EPOCH_CHOICES = (50, 40, 30)
BUDGET_MIN_PER_TRAINING = 12.0


def choose_epochs(sec_per_epoch: float, budget_min: float = BUDGET_MIN_PER_TRAINING) -> dict:
    """Largest count in {50, 40, 30} whose float training fits the budget; 30 flagged if none fits."""
    fits = [e for e in EPOCH_CHOICES if e * sec_per_epoch <= budget_min * 60.0]
    return dict(epochs=fits[0] if fits else EPOCH_CHOICES[-1], fits_budget=bool(fits))


def run_bench(device: str, timed_epochs: int = 3, deadline: Deadline = None) -> dict:
    """Train L=5, w=1.0 for a few epochs; the first epoch (device warm-up) is not averaged."""
    data = load_data(device)
    with tempfile.TemporaryDirectory() as tmp:
        res = train(WORST_CASE, seed=0, epochs=timed_epochs, data=data, device=device,
                    ckpt=Path(tmp) / "bench.pt", deadline=deadline)
    steady = res.epoch_s[1:] or res.epoch_s
    sec = sum(steady) / len(steady)
    return dict(arch=WORST_CASE.name, device=device, timed_epochs=timed_epochs,
                first_epoch_s=res.epoch_s[0], sec_per_epoch=sec,
                budget_min=BUDGET_MIN_PER_TRAINING, **choose_epochs(sec))
