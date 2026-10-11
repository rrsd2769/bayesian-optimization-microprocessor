"""Training: resume after a deadline stop, deterministic split, augmentation shape (tiny synthetic data, CPU)."""
import torch
import pytest

from src.jobs import Deadline, DeadlineReached
from src.network.arch import Arch
from src.network.train import (Data, augment, evaluate_accuracy, load_best_weights, split_indices, train,
                               VAL_SIZE)


def tiny_data(n_train=256, n_val=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda n: (torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g))
    (tx, ty), (vx, vy) = mk(n_train), mk(n_val)
    return Data(tx, ty, vx, vy)


class StopAfter:
    """Deadline stand-in that expires after `n` checks."""
    def __init__(self, n):
        self.n = n

    def check(self):
        self.n -= 1
        if self.n <= 0:
            raise DeadlineReached()


def test_resume_continues_from_epoch_3(tmp_path):
    arch, data, ckpt = Arch(2, 0.25), tiny_data(), tmp_path / "a.pt"
    with pytest.raises(DeadlineReached):
        train(arch, seed=1, epochs=5, data=data, device="cpu", ckpt=ckpt, deadline=StopAfter(2))
    assert torch.load(ckpt, weights_only=False)["state"]["epoch"] == 2
    res = train(arch, seed=1, epochs=5, data=data, device="cpu", ckpt=ckpt)
    saved = torch.load(ckpt, weights_only=False)["state"]
    assert saved["epoch"] == 5 and len(res.epoch_s) == 5


def test_resumed_run_matches_uninterrupted_run(tmp_path):
    arch, data = Arch(2, 0.25), tiny_data()
    full = train(arch, 1, 4, data, "cpu", tmp_path / "full.pt")
    with pytest.raises(DeadlineReached):
        train(arch, 1, 4, data, "cpu", tmp_path / "res.pt", deadline=StopAfter(2))
    res = train(arch, 1, 4, data, "cpu", tmp_path / "res.pt")
    assert res.best_val_acc == full.best_val_acc and res.final_val_acc == full.final_val_acc


def test_checkpoint_for_other_run_is_rejected(tmp_path):
    arch, data, ckpt = Arch(2, 0.25), tiny_data(), tmp_path / "a.pt"
    train(arch, 1, 1, data, "cpu", ckpt)
    with pytest.raises(ValueError):
        train(arch, 2, 1, data, "cpu", ckpt)


def test_best_weights_reproduce_best_val_acc(tmp_path):
    arch, data, ckpt = Arch(2, 0.25), tiny_data(), tmp_path / "a.pt"
    res = train(arch, 1, 3, data, "cpu", ckpt)
    model = load_best_weights(ckpt, arch)
    assert evaluate_accuracy(model, data.val_x, data.val_y) == pytest.approx(res.best_val_acc)


def test_split_is_fixed_disjoint_and_sized():
    tr, va = split_indices()
    tr2, va2 = split_indices()
    assert len(tr) == 45_000 and len(va) == VAL_SIZE == 5_000
    assert torch.equal(tr, tr2) and torch.equal(va, va2)
    assert len(set(tr.tolist()) & set(va.tolist())) == 0


def test_augment_keeps_shape_and_is_seeded():
    x = torch.randn(16, 3, 32, 32)
    a = augment(x, torch.Generator().manual_seed(5))
    b = augment(x, torch.Generator().manual_seed(5))
    assert a.shape == x.shape and torch.equal(a, b)
    assert not torch.equal(a, augment(x, torch.Generator().manual_seed(6)))


def test_augment_outputs_are_valid_crops_and_flips():
    x = torch.randn(8, 3, 32, 32)
    out = augment(x, torch.Generator().manual_seed(3))
    padded = torch.nn.functional.pad(x, (4, 4, 4, 4))
    for k in range(len(x)):
        ok = any(torch.equal(out[k], c) for i in range(9) for j in range(9)
                 for c in (padded[k, :, i:i + 32, j:j + 32], padded[k, :, i:i + 32, j:j + 32].flip(2)))
        assert ok
