"""
Energy estimation proxy for embedded inference.
Formula: E (mJ) = FLOPs × V² × C × dvfs_factor × 1000
Physical basis: CMOS dynamic power P = α·C·V²·f → E = P·t = α·C·V²·FLOPs
Source: Weste & Harris, CMOS VLSI Design, 4th ed., Ch. 5
"""
import os, sys, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import SIM_VOLTAGE, SIM_CAPACITANCE, LOG_FORMAT
logger = logging.getLogger(__name__)
VOLTAGE = SIM_VOLTAGE # 3.3 V
CAPACITANCE = SIM_CAPACITANCE # 1e-9 F
def estimate_energy_mj(flops: int, dvfs_factor: float,
 voltage: float = VOLTAGE, capacitance: float =
CAPACITANCE) -> float:
 """
 Estimate inference energy in millijoules.
  Args:
 flops (int): MACs from torchinfo.total_mult_adds.
 dvfs_factor (float): CPU clock scaling factor [0.5, 1.0].
 From scheduler config — lower = less energy.
 Returns:
 float: Energy in mJ, 4 decimal places.
 """
 if not (0.0 < dvfs_factor <= 1.0):
  raise ValueError(f"dvfs_factor must be in (0,1]. Got: {dvfs_factor}")
 if flops <= 0:
  raise ValueError(f"flops must be positive. Got: {flops}")
 energy_mj = flops * (voltage ** 2) * capacitance * dvfs_factor * 1000.0
 logger.debug(f"Energy: flops={flops:,}, dvfs={dvfs_factor:.2f} →{energy_mj:.6f} mJ")
 return energy_mj
if __name__ == "__main__":
 logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
 BASELINE_FLOPS = 10_340_170 # Replace with actual value from get_model_stats()
 print("\n=== Energy Proxy Validation ===\n")
 print(f"FLOPs: {BASELINE_FLOPS:,} | V={VOLTAGE}V | C={CAPACITANCE}F\n")
 print(f"{'DVFS':^12} {'Energy (mJ)':^14} {'vs 1.0x':^12}")
 print("-" * 40)
 ref = estimate_energy_mj(BASELINE_FLOPS, 1.0)
 for dvfs in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]:
  e = estimate_energy_mj(BASELINE_FLOPS, dvfs)
  print(f"{dvfs:<12.2f} {e:<14.4f} {e/ref:<12.4f}")

 # Linearity check: dvfs=0.5 must be exactly 50% of dvfs=1.0
 assert abs(estimate_energy_mj(BASELINE_FLOPS, 0.5) / ref - 0.5) < 1e-6
 print("\nLinearity check: PASSED ✓ ")