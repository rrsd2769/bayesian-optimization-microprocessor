"""
Training for Arch on CIFAR-10, built for short resumable chunks.

- Data is loaded once into tensors on the compute device; random-crop and flip
  are batched tensor ops, so there are no DataLoader worker processes.
- A fixed, seeded 45,000 / 5,000 train/validation split. Checkpoints are chosen
  by validation accuracy. The test split is only reachable through load_test().
- Training accuracy comes from the same logits used for the loss (one forward).
- State (model, optimizer, scheduler, best weights) is saved after every epoch,
  and a rerun continues from the next epoch. Per-epoch seeding makes the shuffle,
  augmentation and dropout of epoch k independent of whether the run was resumed.
"""
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.jobs import Deadline
from src.network.arch import Arch

logger = logging.getLogger(__name__)

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
SPLIT_SEED = 20261011      # fixed: the split never depends on the training seed
VAL_SIZE = 5000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


@dataclass
class Data:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor


@dataclass
class TrainResult:
    best_val_acc: float
    best_epoch: int
    epochs: int
    wall_s: float
    final_val_acc: float
    epoch_s: list


def _normalise(x_uint8_nhwc: torch.Tensor) -> torch.Tensor:
    x = x_uint8_nhwc.permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


def _load_cifar(root: Path, train: bool):
    import numpy as np
    import torchvision
    ds = torchvision.datasets.CIFAR10(root=str(root), train=train, download=True)
    return _normalise(torch.from_numpy(np.asarray(ds.data))), torch.tensor(ds.targets, dtype=torch.long)


def split_indices(n: int = 50_000):
    """Fixed seeded split of the CIFAR-10 train set into (45,000 train, 5,000 validation) indices."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SPLIT_SEED))
    return perm[VAL_SIZE:], perm[:VAL_SIZE]


def load_data(device: str, root: Optional[Path] = None) -> Data:
    """CIFAR-10 train split as tensors on `device`, split 45,000 / 5,000 (train / validation)."""
    if root is None:
        from config import DATA_DIR
        root = DATA_DIR
    x, y = _load_cifar(root, train=True)
    train_idx, val_idx = split_indices(len(x))
    return Data(x[train_idx].to(device), y[train_idx].to(device), x[val_idx].to(device), y[val_idx].to(device))


def load_test(device: str, root: Optional[Path] = None):
    """The CIFAR-10 test split. Used only for final reporting (Phase 7)."""
    if root is None:
        from config import DATA_DIR
        root = DATA_DIR
    x, y = _load_cifar(root, train=False)
    return x.to(device), y.to(device)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random crop (padding 4) and horizontal flip, batched on x's device."""
    b, c, h, w = x.shape
    dev = x.device
    padded = nn.functional.pad(x, (4, 4, 4, 4))
    i0 = torch.randint(0, 9, (b,), generator=gen).to(dev)
    j0 = torch.randint(0, 9, (b,), generator=gen).to(dev)
    flip = (torch.rand(b, generator=gen) < 0.5).to(dev)
    base = torch.arange(h, device=dev)
    rows = (i0[:, None] + base)[:, None, :, None]
    cols = j0[:, None] + base
    cols = torch.where(flip[:, None], cols.flip(1), cols)[:, None, None, :]
    return padded[torch.arange(b, device=dev)[:, None, None, None], torch.arange(c, device=dev)[None, :, None, None], rows, cols]


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int = 1000) -> float:
    model.eval()
    correct = torch.zeros((), device=x.device)
    for i in range(0, len(x), batch):
        correct += (model(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum()
    return correct.item() / len(x)


def train(arch: Arch, seed: int, epochs: int, data: Data, device: str, ckpt: Path,
          deadline: Optional[Deadline] = None) -> TrainResult:
    """Train, or resume from `ckpt`, and return the best-validation result.

    The checkpoint holds the resume state and the best-validation weights (load_best_weights).
    Raises DeadlineReached, after saving that epoch's checkpoint, if the deadline passes mid-run.
    """
    torch.manual_seed(seed)   # initial weights depend on the seed only
    model = arch.torch().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    state = dict(epoch=0, best_val=-1.0, best_epoch=0, best_state=None, wall_s=0.0, final_val=0.0, epoch_s=[])
    if ckpt.exists():
        saved = torch.load(ckpt, map_location=device, weights_only=False)
        if saved["epochs_total"] != epochs or saved["arch"] != arch.name or saved["seed"] != seed:
            raise ValueError(f"{ckpt.name}: checkpoint is for {saved['arch']} seed {saved['seed']} "
                             f"epochs {saved['epochs_total']}, not {arch.name} seed {seed} epochs {epochs}")
        model.load_state_dict(saved["model"]); opt.load_state_dict(saved["opt"]); sched.load_state_dict(saved["sched"])
        state = saved["state"]
        logger.info(f"resuming {ckpt.name} from epoch {state['epoch']}")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    crit = nn.CrossEntropyLoss()
    n = len(data.train_x)
    while state["epoch"] < epochs:
        epoch = state["epoch"] + 1
        t0 = time.time()
        torch.manual_seed(seed * 10_000 + epoch)
        gen = torch.Generator().manual_seed(seed * 10_000 + epoch)
        order = torch.randperm(n, generator=gen).to(device)
        model.train()
        loss_sum = torch.zeros((), device=device); correct = torch.zeros((), device=device)
        for i in range(0, n, BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            xb, yb = augment(data.train_x[idx], gen), data.train_y[idx]
            out = model(xb)
            loss = crit(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * len(idx)
            correct += (out.argmax(1) == yb).sum()
        sched.step()
        val = evaluate_accuracy(model, data.val_x, data.val_y)
        state["epoch"] = epoch
        state["final_val"] = val
        state["epoch_s"].append(time.time() - t0)
        state["wall_s"] = sum(state["epoch_s"])
        if val > state["best_val"]:
            state.update(best_val=val, best_epoch=epoch,
                         best_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        torch.save(dict(arch=arch.name, seed=seed, epochs_total=epochs, model=model.state_dict(),
                        opt=opt.state_dict(), sched=sched.state_dict(), state=state), ckpt.with_name(ckpt.name + ".tmp"))
        ckpt.with_name(ckpt.name + ".tmp").replace(ckpt)
        logger.info(f"{arch.name} seed{seed} epoch {epoch}/{epochs} loss={loss_sum.item() / n:.4f} "
                    f"train={correct.item() / n:.4f} val={val:.4f} {time.time() - t0:.1f}s")
        if deadline is not None and state["epoch"] < epochs:
            deadline.check()
    return TrainResult(state["best_val"], state["best_epoch"], epochs, state["wall_s"], state["final_val"], state["epoch_s"])


def load_best_weights(ckpt: Path, arch: Arch, device: str = "cpu") -> nn.Module:
    """The best-validation weights stored in a finished checkpoint."""
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = arch.torch()
    model.load_state_dict(saved["state"]["best_state"])
    return model.to(device)
