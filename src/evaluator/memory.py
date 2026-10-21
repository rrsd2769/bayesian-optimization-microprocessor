"""
Flash and SRAM footprint of a deployed network, calibrated on ST Edge AI Developer Cloud reports.

flash_bytes = sum(weights * b/8) + sum(bias * 4)       weights at b bits, BatchNorm folded into the conv, int32/fp32
                                                       biases; the Cloud report matches this exactly (it adds no
                                                       per-channel scale bytes)
            + kernel_flash(b, L, w)                    runtime library and generated code, fitted

sram_bytes  = activations(b)                           the Cloud's activation buffer, fitted
            + kernel_ram(b, L)                         runtime state, fitted
            + rtos_reserve                             kernel, task stacks, heap (plan constant)

The Cloud does not expose how it plans its activation buffer (it fuses and reuses buffers in ways a
max-over-layers formula does not reproduce: the formula was off by up to 2x), so activations are a regression on
the summed conv output volume and the layer count, not a mechanism. Measured on 12 architectures at int8 and fp32;
int16 activations are the int8 estimate times 2, int4 uses int8 (4-bit weights run with int8 activations). Those
two bit-widths, and the flash/RAM of the int4 kernels, were not benchmarked.

The benchmark build adds 8 B of flash and 512 B of activation RAM over the Optimize-step report; both offsets are
removed from the fit targets, so the numbers here describe the deployed network, not the harness.
Fitted in src/evaluator/calibrate.py; every field below is a PLACEHOLDER until the calibrate stage has run.
"""
from dataclasses import dataclass

WEIGHT_BITS = (4, 8, 16, 32)
BIAS_BYTES = 4
HARNESS_FLASH = 8        # Cloud benchmark build over the Optimize report, bytes
HARNESS_RAM = 512


@dataclass(frozen=True)
class MemoryCalibration:
    kflash8: tuple = (24000.0, 0.0, 0.0)     # kernel flash int4/8/16: constant, per conv layer, per unit of width
    kflash32: tuple = (6400.0, 350.0)        # fp32: constant, per conv layer
    kram8: tuple = (1800.0, 470.0)           # kernel RAM int8: constant, per conv layer
    kram32: tuple = (620.0, 235.0)           # fp32
    act8: tuple = (0.36, 1500.0)             # activations int8: bytes per conv output element, per conv layer
    act32: float = 4.8                       # fp32: bytes per first-conv output element
    rtos_reserve: int = 32 * 1024            # plan constant, kernel + stacks + heap
    source: str = "placeholder"


def act_bytes(b: int) -> int:
    return {4: 1, 8: 1, 16: 2, 32: 4}[b]


def shape_features(profile: list) -> dict:
    """What the calibration regressions use: conv layer count, width, conv output volumes."""
    convs = [r for r in profile if r["kind"] == "conv"]
    out = [r["c_out"] * r["h_out"] * r["w_out"] for r in convs]
    return dict(L=len(convs), w=convs[0]["c_out"] / 32.0, sum_out=sum(out), first_out=out[0])


def _check(b: int):
    if b not in WEIGHT_BITS:
        raise ValueError(f"unsupported bit-width {b}")


def kernel_flash(profile: list, b: int, cal: MemoryCalibration) -> float:
    f = shape_features(profile)
    if b == 32:
        k0, kl = cal.kflash32
        return k0 + kl * f["L"]
    k0, kl, kw = cal.kflash8
    return k0 + kl * f["L"] + kw * f["w"]


def flash_bytes(profile: list, b: int, cal: MemoryCalibration) -> float:
    _check(b)
    layers = [r for r in profile if r["weights"] > 0]
    total = sum(r["weights"] for r in layers) * b / 8.0 + sum(r["bias"] for r in layers) * BIAS_BYTES
    return total + kernel_flash(profile, b, cal)


def activation_bytes(profile: list, b: int, cal: MemoryCalibration) -> float:
    f = shape_features(profile)
    if b == 32:
        return cal.act32 * f["first_out"]
    per_out, per_layer = cal.act8
    return (per_out * f["sum_out"] + per_layer * f["L"]) * (2 if b == 16 else 1)


def kernel_ram(profile: list, b: int, cal: MemoryCalibration) -> float:
    k0, kl = cal.kram32 if b == 32 else cal.kram8
    return k0 + kl * shape_features(profile)["L"]


def sram_bytes(profile: list, b: int, cal: MemoryCalibration) -> float:
    _check(b)
    return activation_bytes(profile, b, cal) + kernel_ram(profile, b, cal) + cal.rtos_reserve
