"""
ONNX export of an Arch for the STM32Cube.AI Developer Cloud.

fp32: eval-mode export, BatchNorm folded into the convs, Dropout removed. Weights are random (seeded):
      cycles and memory do not depend on weight values.
int8: static post-training quantization, QDQ format, per-channel weights, calibrated on a few images
      (onnxruntime.quantization.quantize_static). Only 8 and 32 bits can be benchmarked on the Cloud.
"""
import warnings
from pathlib import Path

import numpy as np
import torch

from src.network.arch import Arch

OPSET = 13           # first opset with per-channel QDQ
CALIB_IMAGES = 256


def export_fp32(arch: Arch, path: Path, seed: int = 0) -> None:
    torch.manual_seed(seed)
    model = arch.torch().eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(model, torch.zeros(1, 3, 32, 32), str(path), opset_version=OPSET,
                          input_names=["input"], output_names=["logits"], dynamo=False)


def export_int8(fp32_path: Path, out_path: Path, calib: np.ndarray) -> None:
    """calib: float32 array (n, 3, 32, 32), already normalised."""
    from onnxruntime.quantization import CalibrationDataReader

    class Reader(CalibrationDataReader):
        def __init__(self):
            self._it = iter(calib[i:i + 1] for i in range(len(calib)))

        def get_next(self):
            x = next(self._it, None)
            return None if x is None else {"input": x}

    from onnxruntime.quantization.shape_inference import quant_pre_process
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepped = out_path.with_name(out_path.name + ".prep")
    quant_pre_process(str(fp32_path), str(prepped), skip_symbolic_shape=True)   # shape inference, constant folding
    try:
        _quantize(prepped, out_path, Reader())
    finally:
        prepped.unlink(missing_ok=True)


def _quantize(model_in: Path, out_path: Path, reader) -> None:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    quantize_static(str(model_in), str(out_path), reader, quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)


def calibration_images(n: int = CALIB_IMAGES) -> np.ndarray:
    """The first n CIFAR-10 training images, normalised like training."""
    from config import DATA_DIR
    from src.network.train import _load_cifar
    x, _ = _load_cifar(DATA_DIR, train=True)
    return x[:n].numpy().astype(np.float32)


def export_arch(arch: Arch, b: int, path: Path, calib: np.ndarray = None) -> None:
    """Write the model at b bits (32 or 8) to `path`."""
    if b == 32:
        export_fp32(arch, path)
    elif b == 8:
        tmp = path.with_name(path.stem + "_fp32_tmp.onnx")
        export_fp32(arch, tmp)
        try:
            export_int8(tmp, path, calib if calib is not None else calibration_images())
        finally:
            tmp.unlink(missing_ok=True)
    else:
        raise ValueError(f"only 8 and 32 bits can be exported for the Cloud, got {b}")
