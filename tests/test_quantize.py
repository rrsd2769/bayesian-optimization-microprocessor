"""Fake quantization: arithmetic equals torch's CPU ops, BN folding, grids, calibration, QAT resume (tiny CPU data)."""
import pytest
import torch

from src.jobs import DeadlineReached
from src.network.arch import Arch
from src.network.quantize import (ActQuant, QuantLayer, build, calibrate, fake_quant, qat_finetune, quantize,
                                  weight_fake_quant)
from tests.test_train import StopAfter, tiny_data


def trained_like(arch, seed=0):
    """A float model with non-trivial BatchNorm statistics, in eval mode."""
    torch.manual_seed(seed)
    m = arch.torch()
    m.train()
    for _ in range(3):
        m(torch.randn(64, 3, 32, 32) * 2 + 1)
    for mod in m:
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.normal_()
    return m.eval()


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_activation_arithmetic_matches_torch(bits):
    torch.manual_seed(0)
    x = torch.randn(1000) * 3
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scale, zp = torch.tensor(0.0371), torch.tensor(float(qmin // 3))
    mine = fake_quant(x, scale, zp, qmin, qmax)
    ref = torch.fake_quantize_per_tensor_affine(x, 0.0371, int(zp), qmin, qmax)
    # torch fuses x * inv_scale + zp, so a value within one float ulp of a rounding tie can land one step apart
    diff = (mine - ref).abs()
    assert diff.max() <= scale * (1 + 1e-4) and (diff > 1e-6).float().mean() < 0.005


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_weight_arithmetic_matches_torch_per_channel(bits):
    torch.manual_seed(0)
    w = torch.randn(6, 5, 3, 3)
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().flatten(1).amax(1) / qmax
    ref = torch.fake_quantize_per_channel_affine(w, scale, torch.zeros(6, dtype=torch.int32), 0, -qmax, qmax)
    diff = (weight_fake_quant(w, bits) - ref).abs()
    assert (diff > 1e-6).float().mean() < 0.005 and (diff <= scale.view(-1, 1, 1, 1) * (1 + 1e-4)).all()


def test_weights_land_on_the_integer_grid():
    w = torch.randn(8, 16)
    for bits in (4, 8):
        qmax = 2 ** (bits - 1) - 1
        q = weight_fake_quant(w, bits)
        scale = w.abs().amax(1, keepdim=True) / qmax
        ints = q / scale
        assert torch.allclose(ints, ints.round(), atol=1e-4) and ints.abs().max() <= qmax + 1e-4
        assert len(torch.unique(ints.round())) <= 2 * qmax + 1


def test_gradient_passes_inside_range_and_stops_outside():
    x = torch.tensor([0.0, 0.5, 20.0, -20.0], requires_grad=True)
    fake_quant(x, torch.tensor(0.1), torch.tensor(0.0), -128, 127).sum().backward()
    assert x.grad.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_bn_folding_leaves_float_function_unchanged():
    arch = Arch(2, 0.25)
    m = trained_like(arch)
    net = build(m, 24, 24)         # 24-bit weights are float-exact; observers pass values through
    x = torch.randn(16, 3, 32, 32)
    net.eval()
    assert torch.allclose(net(x), m(x), atol=1e-4)
    assert not any(isinstance(l, torch.nn.BatchNorm2d) for l in net)


def test_every_conv_and_linear_is_quantized_and_activation_points_exist():
    arch = Arch(3, 0.25)
    net = build(trained_like(arch), 8, 8)
    assert len(net.quant_layers()) == arch.num_layers + 2
    # input, one per ReLU (L convs + hidden linear), after GAP, logits
    assert len(net.act_quants()) == 1 + arch.num_layers + 1 + 1 + 1


def test_bit_widths_of_the_three_precisions():
    m = trained_like(Arch(2, 0.25))
    x = torch.randn(64, 3, 32, 32)
    for b, (wb, ab) in {16: (16, 16), 8: (8, 8), 4: (4, 8)}.items():
        net = quantize(m, b, x)
        assert {l.w_bits for l in net.quant_layers()} == {wb}
        assert {a.bits for a in net.act_quants()} == {ab}
        assert not any(a.observing for a in net.act_quants())


def test_output_error_shrinks_with_more_bits():
    m = trained_like(Arch(3, 0.4375))
    x = torch.randn(200, 3, 32, 32)
    with torch.no_grad():
        ref = m(x)
        err = {b: (quantize(m, b, x).eval()(x) - ref).abs().mean().item() for b in (4, 8, 16)}
    assert err[16] < err[8] < err[4]


def test_quantized_weights_have_at_most_2_pow_b_levels_per_channel():
    m = trained_like(Arch(2, 0.25))
    net = quantize(m, 4, torch.randn(32, 3, 32, 32))
    for layer in net.quant_layers():
        for ch in layer.quantized_weight().detach().flatten(1):
            assert len(torch.unique(ch)) <= 15


def test_activation_range_includes_zero_and_is_frozen():
    q = ActQuant(8)
    q(torch.tensor([1.0, 2.0, 3.0])); q.freeze()
    assert q.scale.item() == pytest.approx(3.0 / 255, rel=1e-5)
    y = q(torch.tensor([0.0, 10.0]))
    assert y[0] == 0.0 and y[1] <= 3.0 + 1e-6


def test_qat_resume_matches_uninterrupted_run(tmp_path):
    arch, data = Arch(2, 0.25), tiny_data()
    m = trained_like(arch)
    fresh = lambda: quantize(m, 4, data.train_x[:64])
    full = qat_finetune(fresh(), data, 1, 4, "cpu", tmp_path / "full.pt")
    with pytest.raises(DeadlineReached):
        qat_finetune(fresh(), data, 1, 4, "cpu", tmp_path / "res.pt", deadline=StopAfter(2))
    assert torch.load(tmp_path / "res.pt", weights_only=False)["state"]["epoch"] == 2
    res = qat_finetune(fresh(), data, 1, 4, "cpu", tmp_path / "res.pt")
    assert res["val_accs"] == full["val_accs"] and len(res["val_accs"]) == 4


def test_qat_changes_the_weights():
    arch, data = Arch(2, 0.25), tiny_data()
    net = quantize(trained_like(arch), 4, data.train_x[:64])
    before = [l.weight.detach().clone() for l in net.quant_layers()]
    import tempfile, pathlib
    qat_finetune(net, data, 1, 1, "cpu", pathlib.Path(tempfile.mkdtemp()) / "q.pt")
    assert any(not torch.equal(b, l.weight) for b, l in zip(before, net.quant_layers()))
