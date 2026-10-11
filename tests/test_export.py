"""ONNX export: fp32 matches torch, int8 stays close, both load and run (small architecture, random calibration)."""
import numpy as np
import onnxruntime as ort
import onnx
import torch
import pytest

from src.network.arch import Arch
from src.network.export import export_fp32, export_int8

ARCH = Arch(2, 0.25)


def run(path, x):
    """The exported models take one image at a time (batch 1, as benchmarked on the Cloud)."""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.concatenate([sess.run(None, {"input": x[i:i + 1]})[0] for i in range(len(x))])


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("onnx")
    fp32, int8 = d / "a_fp32.onnx", d / "a_int8.onnx"
    export_fp32(ARCH, fp32, seed=0)
    calib = np.random.default_rng(0).standard_normal((32, 3, 32, 32)).astype(np.float32)
    export_int8(fp32, int8, calib)
    return fp32, int8, calib


def test_fp32_matches_torch(models):
    fp32, _, calib = models
    torch.manual_seed(0)
    ref = ARCH.torch().eval()
    x = calib[:4]
    with torch.no_grad():
        expected = ref(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(run(fp32, x), expected, atol=1e-4)


def test_fp32_graph_has_no_batchnorm_or_dropout(models):
    ops = {n.op_type for n in onnx.load(str(models[0])).graph.node}
    assert not ops & {"BatchNormalization", "Dropout"} and {"Conv", "Gemm"} <= ops


def test_int8_is_qdq_per_channel_and_close_to_fp32(models):
    fp32, int8, calib = models
    g = onnx.load(str(int8)).graph
    ops = [n.op_type for n in g.node]
    assert "QuantizeLinear" in ops and "DequantizeLinear" in ops
    scales = {i.name: i for i in g.initializer if "scale" in i.name}
    assert any(onnx.numpy_helper.to_array(v).size > 1 for v in scales.values())      # per-channel scales
    a, b = run(fp32, calib[:8]), run(int8, calib[:8])
    assert b.shape == a.shape and np.abs(a - b).max() < 0.5 * (np.abs(a).max() + 1e-6)


def test_arch_onnx_rejects_other_bit_widths(tmp_path):
    with pytest.raises(ValueError):
        ARCH.onnx(4, tmp_path / "x.onnx")
