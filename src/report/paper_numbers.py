"""
Every number the paper quotes, read through the results store (and the public evaluator and search vocabulary for the
design-space and scenario constants) and saved as one nested dictionary, the `paper_numbers` artifact.
scripts/check_paper_numbers.py confirms that each number in the .tex files can be found here, up to rounding.
"""
import importlib.metadata as metadata

import numpy as np

from src import results
from src.evaluator import Scenario
from src.network.arch import BASE_CHANNELS, HIDDEN, Arch
from src.network.quantize import QAT_LR
from src.network.quantize import CALIB_COUNT
from src.search.space import SPACE


def _space_and_scenario() -> dict:
    sc = Scenario.nominal()
    d, w, lim = sc.device, sc.workload, sc.limits
    return dict(
        space={p.name: (list(p.values) if p.kind == "choice" else list(p.bounds)) for p in SPACE},
        n_designs_grid=12000,
        base_channels=list(BASE_CHANNELS), hidden_units=HIDDEN,
        device=dict(flash_kb=d.flash_bytes // 1024, sram_kb=d.sram_bytes // 1024, vdd=d.vdd, i_stop_ma=d.i_stop_ma,
                    f_mhz=[p.f_mhz for p in d.points], i_run_ma=[p.i_run_ma for p in d.points],
                    i_sleep_ma=[p.i_sleep_ma for p in d.points], flash_ws=[p.flash_ws for p in d.points]),
        workload=dict(tasks=[dict(name=t.name, period_ms=t.period_ms, cycles=t.cycles, priority=t.priority) for t in w.periodic],
                      inference_priority=dict(w.inference_priority), n_requests=w.n_requests,
                      mean_interarrival_ms=w.mean_interarrival_ms, jitter_pct=w.jitter * 100,
                      switch_cycles=w.switch_cycles, tick_cycles=w.tick_cycles, max_horizon_s=w.max_horizon_s,
                      tick_ms=1),
        limits=dict(latency_ms=lim.latency_ms, miss_ratio=lim.miss_ratio, miss_ratio_pct=lim.miss_ratio * 100,
                    accuracy_floor=lim.accuracy_floor, flash_kb=lim.flash_bytes // 1024, sram_kb=lim.sram_bytes // 1024),
        calibration_images=CALIB_COUNT,
    )


def _training() -> dict:
    from config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY
    return dict(float_epochs=results.load("train_bench")["epochs"] if results.exists("train_bench") else NUM_EPOCHS,
                batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                qat_epochs=results.load("qat_pilot")["qat_epochs"], qat_pilot_epochs=results.load("qat_pilot")["pilot_epochs"],
                qat_learning_rate=QAT_LR, train_images=45000, validation_images=5000, test_images=10000, dropout=0.3, crop_padding=4)


def _accuracy() -> dict:
    a = results.load("accuracy_validation")
    stds = {b: [v[b]["std"] * 100 for v in a["seed_noise"].values() if b in v] for b in ("32", "16", "8", "4")}
    drop = np.array(list(a["drop_32_to_4_points"].values()))
    meas = results.load("accuracy_measurements")
    table = meas[meas["seed"] == a["table_seed"]]
    return dict(n_architectures=a["n_architectures"], n_trainings=int(meas[["arch", "seed"]].drop_duplicates().shape[0]),
                n_seed_noise_architectures=len(a["seed_noise"]),
                seed_std_points={b: dict(mean=float(np.mean(s)), max=float(np.max(s))) for b, s in stds.items()},
                drop_32_to_4_points=dict(mean=float(drop.mean()), min=float(drop.min()), max=float(drop.max())),
                n_drop_above_zero=a["n_with_drop_above_zero"],
                val_acc_range=dict(min=float(table["val_acc"].min()), max=float(table["val_acc"].max())))


QUOTED = dict(    # values read from the cited sources, not computed here
    ungurean_switch_us=4.13, ungurean_signal_us=1.82, ungurean_net_us=2.31, qat_plateau_points=0.2,
    edge_ai_core="ST Edge AI Core 4.0.1", sensitivity_scales=[0.5, 2], cmix_overhead_pct=14, hours_per_seed=2.3, ax_step_integer_s=[5, 15], ax_step_categorical_s=[12, 30], tick_hz=1000, rtos_f407_mhz=168)


def _cycles_per_mac() -> dict:
    cloud = results.load("cloud_calibration")
    cloud = cloud[cloud["precision"] == "int8"]
    rate = {}
    for _, r in cloud.iterrows():
        layers, width = r["arch"][1:].split("_w")
        macs = sum(x["macs"] for x in Arch(int(layers), float(width)).profile())
        rate[r["arch"]] = float(r["cycles"]) / macs
    lo, hi = min(rate, key=rate.get), max(rate, key=rate.get)
    return dict(min=rate[lo], max=rate[hi], smallest_model=rate["L2_w0.25"])


def _calibration() -> dict:
    cyc, mem = results.load("cycle_calibration"), results.load("memory_calibration")
    cb = results.load("cloud_board")
    out = dict(board_clock_mhz=cb["clock_mhz"], board_ram_kb=cb["internal_ram_kb"], board_flash_kb=cb["internal_flash_kb"],
               cycles={k.replace("cycles_", ""): v for k, v in cyc["report"].items()},
               cpm8=cyc["cal"]["cpm8"], cpm32=cyc["cal"]["cpm32"], r16=cyc["cal"]["r16"], r4=cyc["cal"]["r4"],
               r16_pct=(cyc["cal"]["r16"] - 1) * 100, r4_pct=(cyc["cal"]["r4"] - 1) * 100,
               memory={k: v for k, v in mem["report"].items()}, rtos_reserve_kb=mem["cal"]["rtos_reserve"] // 1024, cycles_per_mac_int8=_cycles_per_mac(),
               n_benchmarks=int(len(results.load("cloud_calibration"))))
    return out


def _front() -> dict:
    s = results.load("reference_front_summary")
    f = results.load("reference_front")
    sens = results.load("sensitivity")
    return dict(grid_size=s["grid_size"], n_acceptable=s["n_acceptable"], acceptable_pct=s["n_acceptable"] / s["grid_size"] * 100,
                n_front=s["n_front"], hv=s["hypervolume"],
                tradeoff=dict(accuracy_points=float((f["accuracy"].max() - f["accuracy"].min()) * 100),
                              energy_increase_pct=float((f["energy_uj"].max() / f["energy_uj"].min() - 1) * 100),
                              evaluations_pct_of_grid=100 / s["grid_size"] * 100, n90_pct_of_grid=28 / s["grid_size"] * 100), composition=s["front_composition"], rejected=s["rejected"],
                corner=results.load("hv_reference"),
                designs=[{k: (float(v) if k not in ("pi",) else v) for k, v in r.items()
                          if k in ("L", "w", "b", "tau", "pi", "f_idx", "accuracy", "energy_uj", "latency_ms")}
                         for r in f.to_dict("records")],
                front_accuracy=dict(min=float(f["accuracy"].min()), max=float(f["accuracy"].max())),
                front_energy_uj=dict(min=float(f["energy_uj"].min()), max=float(f["energy_uj"].max())),
                sensitivity={r["case"]: dict(n_acceptable=int(r["n_acceptable"]), n_front=int(r["n_front"]),
                                             hv_A=float(r["hv_A"]), hv_B=float(r["hv_B"])) for _, r in sens.iterrows()},
                n_sensitivity_cases=int(len(sens)))


def _comparison() -> dict:
    s = results.load("comparison_summary")
    per = {}
    for m, d in s["per_method"].items():
        per[m] = {k: v for k, v in d.items() if k != "rejected"}
        per[m]["rejected"] = d["rejected"]
    ref = results.load("reference_front_summary")["hypervolume"]["A"]
    q = [results.load("run", method="qnehvi", seed=sd)["hv_A"].iloc[-1] / ref * 100 for sd in s["common_seeds"]]
    for m, d in per.items():
        d["unacceptable_pct"] = d["rejected"]["unacceptable"] / d["rejected"]["total_evaluations"] * 100
    return dict(proposal_time=_proposal_time(s["common_seeds"]), n_seeds=len(s["common_seeds"]), evaluations=100, per_method=per, tests=s["tests"],
                qnehvi_seeds_at_100=int(sum(v >= 99.999 for v in q)), qnehvi_lowest_final_pct=float(min(q)),
                qnehvi_second_lowest_final_pct=float(sorted(q)[1]))


def _proposal_time(seeds) -> dict:
    out = {}
    for m in ("qnehvi", "qnparego", "sequential", "motpe", "nsga2", "random"):
        t = [float(results.load("run", method=m, seed=sd)["ask_s"].sum()) for sd in seeds]
        out[m] = dict(median_min=float(np.median(t)) / 60, median_s=float(np.median(t)), max_s=float(np.max(t)),
                      per_evaluation_s=float(np.median(t)) / 100)
    return out


def _hardware() -> dict:
    hw, ta = results.load("hw_validation"), results.load("test_accuracy")
    under = sum(1 for m in hw["held_out"]["models"] if m["cycles"]["error_pct"] < 0)
    return dict(n_cycles_underestimated=under, held_out=hw["held_out"]["summary"], in_sample=hw["in_sample"]["summary"],
                n_held_out=hw["held_out"]["n"], n_in_sample=hw["in_sample"]["n"],
                test_accuracy={d["arch"] + f"_b{d['b']}": dict(validation=d["predicted_val_accuracy"], test=d["test_accuracy"],
                                                                gap_points=(d["test_accuracy"] - d["predicted_val_accuracy"]) * 100)
                               for d in ta["designs"]},
                mean_gap_points=ta["mean_gap_points"], n_test=ta["n_test"])


def collect() -> dict:
    versions = {n: metadata.version(n) for n in ("ax-platform", "botorch", "pymoo", "optuna")}
    return dict(quoted=QUOTED, versions=versions, setup=_space_and_scenario(), training=_training(), accuracy=_accuracy(), calibration=_calibration(),
                front=_front(), comparison=_comparison(), hardware=_hardware())
