"""
Flash and SRAM footprint of a deployed network (STM32Cube.AI-style: weights in flash, activations in SRAM).

flash_bytes = sum(weights * b/8)                       weights at b bits
            + sum(bias * 4)                            biases (BatchNorm folded into the conv) as int32 / fp32
            + sum(out_channels * 4)   if b <= 16       one scale per output channel of every conv/linear layer
            + code_allowance                           runtime library and application code

sram_bytes  = max over layers (in_act + out_act + scratch)   one layer's buffers live at a time
            + rtos_reserve                                   kernel, task stacks, heap

Activations take 1 byte at b in {4, 8} (4-bit weights run with int8 activations), 2 at b = 16, 4 at b = 32.
Conv scratch is `scratch_columns` im2col columns of k*k*c_in activations (CMSIS-NN style: 2 columns).
Placeholders (code_allowance, scratch_columns) are fitted against the Cloud report in Phase 2.
"""
from dataclasses import dataclass

CONV_K = 3
SCALE_BYTES = 4
BIAS_BYTES = 4
WEIGHT_BITS = (4, 8, 16, 32)


@dataclass(frozen=True)
class MemoryCalibration:
    code_allowance: int = 32 * 1024     # PLACEHOLDER until Phase 2
    scratch_columns: float = 2.0        # PLACEHOLDER until Phase 2
    rtos_reserve: int = 32 * 1024       # plan constant, kernel + stacks + heap
    source: str = "placeholder"


def act_bytes(b: int) -> int:
    return {4: 1, 8: 1, 16: 2, 32: 4}[b]


def flash_bytes(profile: list, b: int, cal: MemoryCalibration) -> float:
    if b not in WEIGHT_BITS:
        raise ValueError(f"unsupported bit-width {b}")
    layers = [r for r in profile if r["weights"] > 0]
    total = sum(r["weights"] for r in layers) * b / 8.0
    total += sum(r["bias"] for r in layers) * BIAS_BYTES
    if b <= 16:
        total += sum(r["c_out"] for r in layers) * SCALE_BYTES
    return total + cal.code_allowance


def sram_bytes(profile: list, b: int, cal: MemoryCalibration) -> float:
    if b not in WEIGHT_BITS:
        raise ValueError(f"unsupported bit-width {b}")
    a = act_bytes(b)
    peak = 0.0
    for r in profile:
        in_act = r["c_in"] * r["h_in"] * r["w_in"] * a
        out_act = r["c_out"] * r["h_out"] * r["w_out"] * a
        scratch = cal.scratch_columns * CONV_K * CONV_K * r["c_in"] * a if r["kind"] == "conv" else 0.0
        peak = max(peak, in_act + out_act + scratch)
    return peak + cal.rtos_reserve
