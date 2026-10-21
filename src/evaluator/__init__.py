"""
Evaluator module. Its whole interface is
    evaluate(design, scenario) -> Outcome, plus Design, Scenario and Outcome.
Everything else in this package is internal; nothing outside src/evaluator/ imports from its files.
The remaining names below are the vocabulary of a Scenario (so callers can build or vary one) and the two
Phase 2 calibration entry points the pipeline stages call.
"""
from src.evaluator.accuracy import TABLE_SEED, AccuracyTable
from src.evaluator.calibrate import fit_cycles as fit_cycle_calibration
from src.evaluator.calibrate import fit_memory as fit_memory_calibration
from src.evaluator.calibrate import to_json as calibration_to_json
from src.evaluator.cloud import import_cloud_csv, sha256_file
from src.evaluator.core import Design, Outcome, evaluate, predict_benchmark
from src.evaluator.cycles import CycleCalibration
from src.evaluator.device import STM32F446RE
from src.evaluator.memory import MemoryCalibration
from src.evaluator.rtos_sim import PI_LEVELS, PeriodicTask, Workload
from src.evaluator.scenario import Limits, Scenario

__all__ = ["evaluate", "predict_benchmark", "Design", "Outcome", "Scenario", "Limits", "AccuracyTable", "CycleCalibration",
           "TABLE_SEED", "MemoryCalibration", "Workload", "PeriodicTask", "PI_LEVELS", "STM32F446RE",
           "fit_cycle_calibration", "fit_memory_calibration", "calibration_to_json", "import_cloud_csv", "sha256_file"]
