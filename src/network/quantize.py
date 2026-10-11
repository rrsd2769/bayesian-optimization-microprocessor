"""
Fake quantization of an Arch network at b = 16, 8 or 4 bits (b = 32 is the float model itself).

    quantize(float_model, b, calib_x)            -> QuantNet (post-training for b = 16, 8; the start point of QAT for b = 4)
    qat_finetune(net, data, ..., epochs, ckpt)   -> per-epoch validation accuracy of a 4-bit QAT run (resumable)

| b | weights                     | activations                | procedure                                  |
|---|-----------------------------|----------------------------|--------------------------------------------|
| 16| int16 per-channel symmetric | int16 per-tensor affine    | post-training, 1,000 calibration images    |
| 8 | int8  per-channel symmetric | int8  per-tensor affine    | post-training, same images                 |
| 4 | int4  per-channel symmetric | int8  per-tensor affine    | QAT fine-tune from the float weights       |

- BatchNorm is folded into the convs first (each conv gains a bias). Every Conv and Linear layer is quantized,
  the first conv and the classifier included. Biases stay float: on the target they are int32 at scale
  s_w * s_x and add a rounding error far below one activation step.
- Activations are quantized at the network input, after every ReLU, after global average pooling and at the
  logits (max-pool commutes with the quantizer, so it needs none). Activation ranges are the min and max seen
  over the calibration images, always including zero (torch.ao MinMaxObserver), then frozen.
- Weights are symmetric with the restricted range +-(2^(b-1) - 1), scale = max|w| / (2^(b-1) - 1) per output
  channel, recomputed from the current weights at every forward, so QAT sees its own weight changes.
- torch.ao's FakeQuantize modules are not implemented on MPS (aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams),
  so the rounding is written here as tensor ops with a straight-through gradient. torch.ao observers still compute
  the activation parameters, and the tests check this arithmetic against torch.fake_quantize_per_tensor_affine
  and torch.fake_quantize_per_channel_affine on CPU.
"""
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import MinMaxObserver

from src.jobs import Deadline
from src.network.train import BATCH_SIZE, WEIGHT_DECAY, Data, augment, evaluate_accuracy

logger = logging.getLogger(__name__)

CALIB_COUNT = 1000
QAT_LR = 1e-4
QAT_SEED_OFFSET = 1000      # QAT epoch k of seed s is seeded s * 10_000 + 1000 + k, apart from the float epochs


def fake_quant(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    """Quantize to the integer grid [qmin, qmax] and back. Gradient: 1 inside the range, 0 outside (straight-through)."""
    lo, hi = (qmin - zero_point) * scale, (qmax - zero_point) * scale
    xc = torch.minimum(torch.maximum(x, lo), hi)
    q = torch.clamp(torch.round(x * (1.0 / scale) + zero_point), qmin, qmax)
    return xc + ((q - zero_point) * scale - xc).detach()


def weight_fake_quant(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-output-channel symmetric fake quantization of a conv or linear weight (axis 0)."""
    qmax = 2 ** (bits - 1) - 1
    shape = (-1,) + (1,) * (w.dim() - 1)
    max_abs = w.detach().abs().flatten(1).amax(1).clamp_min(1e-12)
    scale = (max_abs / qmax).view(shape)
    return fake_quant(w, scale, torch.zeros_like(scale), -qmax, qmax)


class ActQuant(nn.Module):
    """Per-tensor affine activation quantizer: observes min/max, then `freeze()` fixes scale and zero point."""

    def __init__(self, bits: int):
        super().__init__()
        self.bits = bits
        self.qmin, self.qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        self.observer = MinMaxObserver(dtype=torch.qint8 if bits <= 8 else torch.qint32, quant_min=self.qmin,
                                       quant_max=self.qmax, qscheme=torch.per_tensor_affine)
        self.observing = True
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer("zero_point", torch.zeros(()))

    def freeze(self) -> None:
        scale, zp = self.observer.calculate_qparams()
        self.scale.copy_(scale.reshape(()))
        self.zero_point.copy_(zp.reshape(()).to(self.scale.dtype))
        self.observing = False

    def forward(self, x):
        if self.observing:
            self.observer(x.detach())
            return x
        return fake_quant(x, self.scale, self.zero_point, self.qmin, self.qmax)


class QuantLayer(nn.Module):
    """Conv2d(3x3, padding 1) or Linear with per-channel symmetric fake-quantized weights."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, conv: bool, w_bits: int):
        super().__init__()
        self.weight, self.bias = nn.Parameter(weight), nn.Parameter(bias)
        self.conv, self.w_bits = conv, w_bits

    def quantized_weight(self) -> torch.Tensor:
        return weight_fake_quant(self.weight, self.w_bits)

    def forward(self, x):
        w = self.quantized_weight()
        return F.conv2d(x, w, self.bias, padding=1) if self.conv else F.linear(x, w, self.bias)


class QuantNet(nn.Sequential):
    """The folded network with weight and activation fake quantization; `w_bits` and `a_bits` as in the table above."""
    w_bits: int
    a_bits: int

    def act_quants(self):
        return [m for m in self if isinstance(m, ActQuant)]

    def quant_layers(self):
        return [m for m in self if isinstance(m, QuantLayer)]


def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Weight and bias of conv followed by an eval-mode BatchNorm."""
    k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return conv.weight * k.view(-1, 1, 1, 1), bn.bias - bn.running_mean * k


def build(float_model: nn.Sequential, w_bits: int, a_bits: int) -> QuantNet:
    """Fold BatchNorm and wrap every Conv and Linear; activation quantizers are in observe mode."""
    mods = list(float_model)
    out, i = [ActQuant(a_bits)], 0
    with torch.no_grad():
        while i < len(mods):
            m = mods[i]
            if isinstance(m, nn.Conv2d):
                bn = mods[i + 1]
                assert isinstance(bn, nn.BatchNorm2d) and m.bias is None
                w, b = _fold(m, bn)
                out.append(QuantLayer(w.detach().clone(), b.detach().clone(), True, w_bits))
                i += 2
                continue
            if isinstance(m, nn.Linear):
                out.append(QuantLayer(m.weight.detach().clone(), m.bias.detach().clone(), False, w_bits))
                if i == len(mods) - 1:
                    out.append(ActQuant(a_bits))        # logits
            elif isinstance(m, nn.ReLU):
                out += [nn.ReLU(), ActQuant(a_bits)]
            elif isinstance(m, nn.Flatten):
                out += [nn.Flatten(), ActQuant(a_bits)]   # after global average pooling
            elif isinstance(m, nn.Dropout):
                out.append(nn.Dropout(m.p))
            else:
                out.append(type(m)(*_ctor_args(m)))
            i += 1
    net = QuantNet(*out)
    net.w_bits, net.a_bits = w_bits, a_bits
    return net.to(next(float_model.parameters()).device)


def _ctor_args(m: nn.Module):
    if isinstance(m, nn.MaxPool2d):
        return (m.kernel_size, m.stride)
    if isinstance(m, nn.AdaptiveAvgPool2d):
        return (m.output_size,)
    raise TypeError(f"unsupported layer {m}")


@torch.no_grad()
def calibrate(net: QuantNet, calib_x: torch.Tensor, batch: int = 250) -> None:
    """Run the calibration images through the net with observers on, then freeze every activation quantizer."""
    net.eval()
    for i in range(0, len(calib_x), batch):
        net(calib_x[i:i + batch])
    for q in net.act_quants():
        q.freeze()


BITS = {16: (16, 16), 8: (8, 8), 4: (4, 8)}   # b -> (weight bits, activation bits)


def quantize(float_model: nn.Sequential, b: int, calib_x: torch.Tensor) -> QuantNet:
    """The calibrated fake-quantized network at b bits. For b = 4 this is the QAT starting point."""
    if b not in BITS:
        raise ValueError(f"b must be 4, 8 or 16 (32 is the float model), got {b}")
    net = build(float_model.eval(), *BITS[b])
    calibrate(net, calib_x)
    return net


def qat_finetune(net: QuantNet, data: Data, seed: int, epochs: int, device: str, ckpt: Path,
                 deadline: Optional[Deadline] = None, on_epoch: Optional[Callable[[int, float], None]] = None) -> dict:
    """Fine-tune the weights of `net` with fake quantization for `epochs` epochs (Adam, lr 1e-4, augmented).

    Resumes from `ckpt` (saved after every epoch). Returns dict(val_accs=[...per epoch...], wall_s).
    Raises DeadlineReached after saving an epoch if the deadline passes before the last epoch.
    """
    params = [p for p in net.parameters()]
    opt = torch.optim.Adam(params, lr=QAT_LR, weight_decay=WEIGHT_DECAY)
    state = dict(epoch=0, val_accs=[], epoch_s=[])
    if ckpt.exists():
        saved = torch.load(ckpt, map_location=device, weights_only=False)
        if saved["epochs_total"] != epochs or saved["seed"] != seed:
            raise ValueError(f"{ckpt.name}: checkpoint is for seed {saved['seed']} epochs {saved['epochs_total']}")
        net.load_state_dict(saved["model"]); opt.load_state_dict(saved["opt"]); state = saved["state"]
        logger.info(f"resuming {ckpt.name} from epoch {state['epoch']}")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    crit, n = nn.CrossEntropyLoss(), len(data.train_x)
    while state["epoch"] < epochs:
        epoch = state["epoch"] + 1
        t0 = time.time()
        s = seed * 10_000 + QAT_SEED_OFFSET + epoch
        torch.manual_seed(s)
        gen = torch.Generator().manual_seed(s)
        order = torch.randperm(n, generator=gen).to(device)
        net.train()
        for i in range(0, n, BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            loss = crit(net(augment(data.train_x[idx], gen)), data.train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val = evaluate_accuracy(net, data.val_x, data.val_y)
        state["epoch"] = epoch
        state["val_accs"].append(val)
        state["epoch_s"].append(time.time() - t0)
        torch.save(dict(seed=seed, epochs_total=epochs, model=net.state_dict(), opt=opt.state_dict(), state=state),
                   ckpt.with_name(ckpt.name + ".tmp"))
        ckpt.with_name(ckpt.name + ".tmp").replace(ckpt)
        logger.info(f"QAT seed{seed} epoch {epoch}/{epochs} val={val:.4f} {state['epoch_s'][-1]:.1f}s")
        if on_epoch is not None:
            on_epoch(epoch, val)
        if deadline is not None and epoch < epochs:
            deadline.check()
    return dict(val_accs=state["val_accs"], wall_s=sum(state["epoch_s"]))


QAT_GAIN_POINTS = 0.2


def pick_qat_epochs(curves: list) -> int:
    """The smallest N such that, for every curve, no later epoch beats epoch N by QAT_GAIN_POINTS or more.

    Each curve is the list of validation accuracies (fractions) after QAT epochs 1..K. If no N < K
    qualifies, K is returned (the pilot has not converged; the caller reports that).
    """
    k = min(len(c) for c in curves)
    for n in range(1, k):
        if all(max(c[m] for m in range(n, k)) - c[n - 1] < QAT_GAIN_POINTS / 100.0 for c in curves):
            return n
    return k
