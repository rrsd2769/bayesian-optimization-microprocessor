"""
Energy over a horizon from datasheet currents:

    E_total = V_DD * I_run(f) * t_active + V_DD * I_idle * t_idle
    E_per_inference = E_total / N_inf

I_idle is the Sleep-mode current at the same frequency (Table 26) or the Stop-mode current (Table 27),
chosen by the scenario. Active time covers inference, periodic tasks, context switches and tick interrupts;
the workload simulator (Phase 3) reports the active cycles. Units: V * mA * s = mJ, returned in microjoules.
"""
from src.evaluator.device import Device, OperatingPoint

IDLE_MODES = ("sleep", "stop")


def idle_current_ma(device: Device, point: OperatingPoint, idle_mode: str) -> float:
    if idle_mode == "sleep":
        return point.i_sleep_ma
    if idle_mode == "stop":
        return device.i_stop_ma
    raise ValueError(f"idle_mode must be one of {IDLE_MODES}, got {idle_mode}")


def total_energy_uj(device: Device, point: OperatingPoint, active_cycles: float, horizon_s: float,
                    idle_mode: str = "sleep") -> float:
    t_active = active_cycles / point.f_hz
    if t_active > horizon_s * (1 + 1e-9):
        raise ValueError(f"active time {t_active:.6f} s exceeds horizon {horizon_s:.6f} s")
    t_idle = max(horizon_s - t_active, 0.0)
    mj = device.vdd * (point.i_run_ma * t_active + idle_current_ma(device, point, idle_mode) * t_idle)
    return mj * 1000.0


def energy_per_inference_uj(device: Device, point: OperatingPoint, active_cycles: float, horizon_s: float,
                            n_inferences: int, idle_mode: str = "sleep") -> float:
    if n_inferences <= 0:
        raise ValueError("n_inferences must be positive")
    return total_energy_uj(device, point, active_cycles, horizon_s, idle_mode) / n_inferences
