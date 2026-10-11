"""
Validation accuracy of one trained architecture at b = 32, 16, 8 and 4, and the 4-bit QAT pilot.

    measure_arch(arch, seed, epochs, qat_epochs, data, device, ckpt_dir, deadline) -> DataFrame
        columns arch, seed, b, val_acc, epochs, wall_s; one row per precision.
        epochs: training epochs behind the row (float epochs; plus the QAT epochs for b = 4).
        wall_s: seconds spent producing the row (float training for 32, calibration and evaluation
        for 16 and 8, QAT for 4).
    qat_curve(arch, seed, epochs, qat_epochs, ...) -> dict(float_val_acc, val_accs per QAT epoch)

Both train the float model first (resumable; a finished checkpoint is reused), take its best-validation
weights, and quantize with src.network.quantize. Validation accuracy is measured on the 5,000 held-out
training images; the test split is not touched here.
"""
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.jobs import Deadline
from src.network.arch import Arch
from src.network.quantize import CALIB_COUNT, qat_finetune, quantize
from src.network.train import Data, evaluate_accuracy, load_best_weights, train

COLUMNS = ["arch", "seed", "b", "val_acc", "epochs", "wall_s"]


def float_model(arch: Arch, seed: int, epochs: int, data: Data, device: str, ckpt_dir: Path,
                deadline: Optional[Deadline] = None):
    """Train (or finish training) the float network; returns (best-validation model, TrainResult)."""
    ckpt = ckpt_dir / f"{arch.name}_seed{seed}.pt"
    res = train(arch, seed, epochs, data, device, ckpt, deadline)
    return load_best_weights(ckpt, arch, device), res


def calibration_images(data: Data):
    """The first CALIB_COUNT images of the training split (no augmentation)."""
    return data.train_x[:CALIB_COUNT]


def _qat(model, arch, seed, qat_epochs, data, device, ckpt_dir, deadline, on_epoch=None):
    net = quantize(model, 4, calibration_images(data))
    return qat_finetune(net, data, seed, qat_epochs, device, ckpt_dir / f"{arch.name}_seed{seed}_qat4_e{qat_epochs}.pt",
                        deadline, on_epoch)


def measure_arch(arch: Arch, seed: int, epochs: int, qat_epochs: int, data: Data, device: str, ckpt_dir: Path,
                 deadline: Optional[Deadline] = None) -> pd.DataFrame:
    model, res = float_model(arch, seed, epochs, data, device, ckpt_dir, deadline)
    rows = [(arch.name, seed, 32, evaluate_accuracy(model, data.val_x, data.val_y), epochs, res.wall_s)]
    for b in (16, 8):
        t0 = time.time()
        net = quantize(model, b, calibration_images(data))
        rows.append((arch.name, seed, b, evaluate_accuracy(net, data.val_x, data.val_y), epochs, time.time() - t0))
    q = _qat(model, arch, seed, qat_epochs, data, device, ckpt_dir, deadline)
    rows.append((arch.name, seed, 4, q["val_accs"][-1], epochs + qat_epochs, q["wall_s"]))
    return pd.DataFrame(rows, columns=COLUMNS)


def qat_curve(arch: Arch, seed: int, epochs: int, qat_epochs: int, data: Data, device: str, ckpt_dir: Path,
              deadline: Optional[Deadline] = None) -> dict:
    model, res = float_model(arch, seed, epochs, data, device, ckpt_dir, deadline)
    ptq4 = evaluate_accuracy(quantize(model, 4, calibration_images(data)), data.val_x, data.val_y)
    q = _qat(model, arch, seed, qat_epochs, data, device, ckpt_dir, deadline)
    return dict(arch=arch.name, seed=seed, float_val_acc=evaluate_accuracy(model, data.val_x, data.val_y),
                ptq4_val_acc=ptq4, val_accs=q["val_accs"], float_epochs=epochs)
