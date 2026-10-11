"""Arch profile against the torch model built from the same specs."""
import pytest
import torch
import torch.nn as nn
from torchinfo import summary

from src.network.arch import Arch, LAYER_COUNTS, WIDTH_GRID

GRID = [(L, w) for L in LAYER_COUNTS for w in WIDTH_GRID]


def hook_macs(model):
    """Independent MAC count: conv and linear multiply-accumulates seen in a forward pass."""
    total = [0]

    def conv_hook(m, inp, out):
        total[0] += out.numel() * m.in_channels * m.kernel_size[0] * m.kernel_size[1]

    def linear_hook(m, inp, out):
        total[0] += m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(conv_hook)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(linear_hook)
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32))
    return total[0]


@pytest.mark.parametrize("L,w", GRID)
def test_macs_match_forward_pass(L, w):
    arch = Arch(L, w)
    assert sum(r["macs"] for r in arch.profile()) == hook_macs(arch.torch())


@pytest.mark.parametrize("L,w", GRID)
def test_params_and_macs_match_torchinfo(L, w):
    arch = Arch(L, w)
    prof = arch.profile()
    stats = summary(arch.torch(), input_size=(1, 3, 32, 32), verbose=0)
    conv_ch = sum(r["c_out"] for r in prof if r["kind"] == "conv")
    lin_bias = sum(r["bias"] for r in prof if r["kind"] == "linear")
    weights = sum(r["weights"] for r in prof)
    # torch model: conv weights + BatchNorm (2 per channel) + linear weights and biases
    assert stats.total_params == weights + 2 * conv_ch + lin_bias
    # torchinfo also counts BatchNorm and linear-bias terms as mult-adds
    assert stats.total_mult_adds == sum(r["macs"] for r in prof) + 2 * conv_ch + lin_bias


@pytest.mark.parametrize("L,w", GRID)
def test_shapes_chain_and_output(L, w):
    arch = Arch(L, w)
    prof = arch.profile()
    for prev, cur in zip(prof, prof[1:]):
        assert (prev["c_out"], prev["h_out"], prev["w_out"]) == (cur["c_in"], cur["h_in"], cur["w_in"]) \
            or cur["kind"] == "linear"
    assert prof[-1]["c_out"] == 10
    assert arch.torch()(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


def test_l5_w1_is_near_flash_limit():
    weights = sum(r["weights"] for r in Arch(5, 1.0).profile())
    assert 0.9e6 < weights < 1.1e6   # about 0.5 MB at int4: right at the 512 KB flash limit


def test_invalid_arch_raises():
    with pytest.raises(ValueError):
        Arch(6, 1.0)
    with pytest.raises(ValueError):
        Arch(3, 0.1)
