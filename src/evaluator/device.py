"""
STM32F446RE constants, read from the datasheet DS10693 Rev 11 (docs/reference/stm32f446re_ds10693.pdf).

Operating conditions behind every current value (datasheet section 6.3.7, p. 81): VDD = 3.3 V, TA = 25 C,
regulator ON, external clock with PLL ON, all peripherals disabled, all I/O static, CoreMark-equivalent code.
Those currents are for CoreMark-like code, not for neural-network kernels; the energy model inherits that
limit and the paper states it.

Voltage scaling (p. 81): Scale 3 up to 120 MHz, Scale 2 above 120 up to 144 MHz, Scale 1 above 144 up to
180 MHz, over-drive ON only at 180 MHz.

Typical values are tabulated at integer-mA resolution for Run mode, so each run current is good to about
+-0.5 mA (1.5 to 5 %).
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OperatingPoint:
    f_mhz: int
    vos_scale: int          # voltage scaling, 1 (fastest) to 3
    overdrive: bool
    i_run_ma: float         # Run mode, flash with ART accelerator and prefetch, typical
    i_sleep_ma: float       # Sleep mode, flash on, typical
    flash_ws: int           # flash wait states at VDD 2.7-3.6 V

    @property
    def f_hz(self) -> float:
        return self.f_mhz * 1e6


@dataclass(frozen=True)
class Device:
    name: str
    flash_bytes: int
    sram_bytes: int
    vdd: float
    points: tuple
    i_stop_ma: float


def flash_wait_states(f_mhz: float) -> int:
    """Table 17 (p. 77): at VDD 2.7-3.6 V flash runs with no wait state up to 30 MHz and gains one wait
    state per further 30 MHz (180 MHz needs 5). Thanks to the ART accelerator these do not slow execution
    from flash (Table 17, footnote 2)."""
    return math.ceil(f_mhz / 30.0) - 1


STM32F446RE = Device(
    name="STM32F446RE",
    flash_bytes=512 * 1024,     # "512 Kbytes of flash memory", features list (p. 1)
    sram_bytes=128 * 1024,      # "128 Kbytes of SRAM", features list (p. 1)
    vdd=3.3,                    # typical conditions, section 6.3.7 (p. 81)
    points=(
        # Run: Table 24 (p. 83), "External clock, PLL ON, all peripherals disabled", ART enabled with prefetch.
        # Sleep: Table 26 (p. 86), "all peripherals disabled", "Flash memory on".
        OperatingPoint(30,  3, False, 10.0, 4.43,  flash_wait_states(30)),    # Run row 30 MHz;  Sleep row 30 MHz
        OperatingPoint(60,  3, False, 18.0, 5.28,  flash_wait_states(60)),    # Run row 60 MHz;  Sleep row 60 MHz
        OperatingPoint(120, 3, False, 32.0, 7.35,  flash_wait_states(120)),   # Run row 120 MHz; Sleep row 120 MHz
        OperatingPoint(150, 1, False, 41.0, 9.53,  flash_wait_states(150)),   # Run row 150 MHz; Sleep row 150 MHz
        OperatingPoint(180, 1, True,  46.0, 11.36, flash_wait_states(180)),   # Run row 180 MHz; Sleep row 180 MHz
    ),
    # Table 27 (p. 88): Stop mode, main regulator, flash in Stop mode, all oscillators off, typical 25 C.
    i_stop_ma=0.234,
)
