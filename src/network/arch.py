"""
Arch(L, w): the single definition of the network.

An Arch is an ordered list of layer specs derived from (num_layers, width_mult).
Both views of the network come from that one list:
    Arch.profile()  per-layer shapes, MACs and weight counts, no torch needed
    Arch.torch()    the trainable nn.Module
Nothing else describes layer shapes.

Network: L blocks of Conv3x3(no bias) -> BatchNorm -> ReLU -> MaxPool2, then
global average pooling and a 128-unit hidden Linear layer, then the classifier.
BatchNorm is folded into the conv at deployment, so the profile counts a conv
with one bias per output channel instead of separate BatchNorm parameters.
"""
from dataclasses import dataclass

IMAGE_SIZE = 32
NUM_CLASSES = 10
HIDDEN = 128
BASE_CHANNELS = (32, 64, 128, 256, 256)   # fifth block repeats 256 so L=5, w=1 stays near the flash limit
LAYER_COUNTS = (2, 3, 4, 5)
WIDTH_GRID = (0.25, 0.4375, 0.625, 0.8125, 1.0)
MIN_CHANNELS = 4


@dataclass(frozen=True)
class Layer:
    kind: str            # "conv" | "pool" | "gap" | "linear"
    c_in: int
    c_out: int
    hidden: bool = False  # linear only: followed by ReLU and Dropout


@dataclass(frozen=True)
class Arch:
    num_layers: int
    width_mult: float

    def __post_init__(self):
        if self.num_layers not in LAYER_COUNTS:
            raise ValueError(f"num_layers must be in {LAYER_COUNTS}, got {self.num_layers}")
        if not (0.25 <= self.width_mult <= 1.0):
            raise ValueError(f"width_mult must be in [0.25, 1.0], got {self.width_mult}")

    @property
    def name(self) -> str:
        return f"L{self.num_layers}_w{self.width_mult:g}"

    def channels(self) -> list:
        return [max(MIN_CHANNELS, int(c * self.width_mult)) for c in BASE_CHANNELS[:self.num_layers]]

    def layers(self) -> list:
        specs, c_in = [], 3
        for c_out in self.channels():
            specs += [Layer("conv", c_in, c_out), Layer("pool", c_out, c_out)]
            c_in = c_out
        specs += [Layer("gap", c_in, c_in),
                  Layer("linear", c_in, HIDDEN, hidden=True),
                  Layer("linear", HIDDEN, NUM_CLASSES)]
        return specs

    def profile(self) -> list:
        """Per layer: kind, c_in, c_out, h_in, w_in, h_out, w_out, macs, weights, bias."""
        rows, h = [], IMAGE_SIZE
        for s in self.layers():
            if s.kind == "conv":      # 3x3, stride 1, padding 1: spatial size unchanged
                row = dict(h_in=h, w_in=h, h_out=h, w_out=h,
                           macs=h * h * s.c_out * s.c_in * 9, weights=s.c_out * s.c_in * 9, bias=s.c_out)
            elif s.kind == "pool":    # 2x2, stride 2
                row = dict(h_in=h, w_in=h, h_out=h // 2, w_out=h // 2, macs=0, weights=0, bias=0)
                h //= 2
            elif s.kind == "gap":
                row = dict(h_in=h, w_in=h, h_out=1, w_out=1, macs=0, weights=0, bias=0)
                h = 1
            else:
                row = dict(h_in=1, w_in=1, h_out=1, w_out=1,
                           macs=s.c_in * s.c_out, weights=s.c_in * s.c_out, bias=s.c_out)
            rows.append(dict(kind=s.kind, c_in=s.c_in, c_out=s.c_out, **row))
        return rows

    def torch(self):
        import torch.nn as nn
        mods = []
        for s in self.layers():
            if s.kind == "conv":
                mods += [nn.Conv2d(s.c_in, s.c_out, 3, padding=1, bias=False),
                         nn.BatchNorm2d(s.c_out), nn.ReLU(inplace=True)]
            elif s.kind == "pool":
                mods.append(nn.MaxPool2d(2, 2))
            elif s.kind == "gap":
                mods += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]
            else:
                mods.append(nn.Linear(s.c_in, s.c_out))
                if s.hidden:
                    mods += [nn.ReLU(inplace=True), nn.Dropout(0.3)]
        return nn.Sequential(*mods)

    def onnx(self, b: int, path, calib=None) -> None:
        """Export to ONNX for the Cloud: fp32 (b=32) or int8 QDQ (b=8)."""
        from src.network.export import export_arch
        export_arch(self, b, path, calib)
