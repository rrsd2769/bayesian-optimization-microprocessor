"""
Inference cycles on the Cortex-M4F.

    cycles = sum over layers (
                 macs * cpm(b)                          compute
               + weights * bytes(b) * k_flash * ws      flash read stall, only if calibration says so
               + c_layer )                              per-layer fixed cost
           + c_fixed                                    per-inference fixed cost

Calibration (Phase 2 fits these on STM32Cube.AI Developer Cloud benchmarks of int8 and fp32 models):
    cpm8, cpm32, c_layer, c_fixed, and k_flash only if the leave-one-out int8 error exceeds 10 % without it.
The other two bit-widths are literature estimates, labelled as such in the paper:
    cpm16 = cpm8 * R16   R16 = 1223.513 / 1034.033 = 1.183, int16 over int8 inference time of the same
                         framework (MicroAI) on a Cortex-M4F, Nucleo-L452RE at 48 MHz, ResNet with 80 filters.
                         Novac et al. 2021, Sensors 21(9):2984, 10.3390/s21092984, Appendix D inference-time
                         table (text of Section 6.2 quotes 1223 ms and 1034 ms). The other board in that paper
                         (SparkFun Edge, Apollo3) gives 1041.617 / 1003.365 = 1.038; at 16 filters the Nucleo
                         ratio is 44.915 / 43.003 = 1.044. Limit: that framework has no SIMD int8 path (the paper
                         says int16 and int8 run the same instructions on 32-bit registers), whereas STM32Cube.AI
                         int8 uses SIMD kernels, so R16 may understate the int16 cost relative to Cube.AI int8.
    cpm4  = cpm8 * R4    R4 = 1.14, the overhead CMix-NN, Capotondi et al. 2020, IEEE TCAS-II 67(5):871-875,
                         10.1109/tcsii.2020.2983648, reports for 4-bit weights with 8-bit activations (w4a8)
                         over w8a8: "up to 14%" extra time from unpacking 4-bit weights in the inner loop
                         (Section IV-A, text beside Figure 4, p. 4 of the accepted manuscript). Figure 4 shows
                         the whole-layer effect is smaller, about 1.07 to 1.08 (read off the chart, MobilenetV1
                         192_0.5 layers 3 and 27, STM32H743), so 1.14 is the stated upper end. Cross-check with
                         Table III (same library, per-channel ICN quantization, different networks): w4a4 on
                         224_0.75 runs at 0.30 MAC/cycle against 0.33 and 0.35 for w8a8 on 192_0.5 and 160_0.5,
                         a ratio of 1.10 to 1.17. Cortex-M7, not M4; the core difference is stated in the paper.
                         Read from the accepted manuscript (Bologna institutional repository); the published
                         IEEE pagination may differ.
A Cortex-M4 TRM instruction count is only a cross-check, never reported as a source.

Every constant marked PLACEHOLDER is overwritten by the Phase 2 fit or the literature read.
"""
from dataclasses import dataclass

BITS = (4, 8, 16, 32)


@dataclass(frozen=True)
class CycleCalibration:
    cpm8: float = 1.0          # PLACEHOLDER, fitted in Phase 2 (cycles per MAC at int8)
    cpm32: float = 4.0         # PLACEHOLDER, fitted in Phase 2 (cycles per MAC at fp32)
    c_layer: float = 2000.0    # PLACEHOLDER, fitted in Phase 2
    c_fixed: float = 20000.0   # PLACEHOLDER, fitted in Phase 2
    k_flash: float = 0.0       # cycles per weight byte per wait state; 0 unless Phase 2 needs it
    r16: float = 1223.513 / 1034.033   # Novac 2021, see module docstring
    r4: float = 1.14           # Capotondi 2020, see module docstring
    source: str = "placeholder"

    def cpm(self, b: int) -> float:
        if b == 8:
            return self.cpm8
        if b == 32:
            return self.cpm32
        if b == 16:
            return self.cpm8 * self.r16
        if b == 4:
            return self.cpm8 * self.r4
        raise ValueError(f"unsupported bit-width {b}")


def weight_bytes_per_param(b: int) -> float:
    return b / 8.0


def inference_cycles(profile: list, b: int, ws: int, cal: CycleCalibration) -> int:
    cpm = cal.cpm(b)
    flash = cal.k_flash * ws * weight_bytes_per_param(b)
    total = cal.c_fixed
    for r in profile:
        total += r["macs"] * cpm + r["weights"] * flash + cal.c_layer
    return int(round(total))
