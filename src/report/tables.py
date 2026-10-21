"""
LaTeX tables from the results store, in the paper's style: "Table N: ..." caption, full borders, shaded header row.
Each function returns the table environment as text; `make_all` writes one .tex file per table.

    tab_space        design space
    tab_constants    workload, device and limit constants
    tab_accuracy     accuracy table summary (seed noise, 32-to-4-bit drop)
    tab_methods      method comparison at 100 evaluations
    tab_hardware     held-out hardware validation

Small magnitudes are written as d x 10^e, the rest as plain decimals.
"""
import math
from pathlib import Path

import numpy as np

from src import results
from src.evaluator import Scenario
from src.report.figures import METHOD_LABELS, ORDER
from src.search.space import SPACE


def num(x: float, digits: int = 2) -> str:
    """Plain decimal, or d x 10^e below 0.1 in magnitude."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if math.isinf(x):
        return "never"
    if x == 0:
        return "0"
    if x != 0 and abs(x) < 0.1:
        mant, exp = f"{x:.{digits}e}".split("e")
        return f"${mant}\\times10^{{{int(exp)}}}$"
    return f"{x:.{digits}f}"


def _table(label: str, caption: str, spec: str, header: list, rows: list, note: str = "", wide: bool = False) -> str:
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{tab:{label}}}", "\\footnotesize" if not wide else "\\small", "\\resizebox{\\columnwidth}{!}{%" if not wide else "",
             f"\\begin{{tabular}}{{{spec}}}", "\\hline", "\\rowcolor{tblhead}",
             " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\", "\\hline"]
    for r in rows:
        lines += [" & ".join(str(c) for c in r) + " \\\\", "\\hline"]
    lines.append("\\end{tabular}" + ("" if wide else "}"))
    if note:
        lines += ["", "\\vspace{2pt}", f"{{\\footnotesize {note}}}"]
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines) + "\n"


def tab_space() -> str:
    names = {"num_layers": ("Convolutional layers", "$L$"), "width_mult": ("Width multiplier", "$w$"),
             "quant_bits": ("Weight bit-width", "$b$"), "time_slice": ("Time slice (ms)", "$\\tau$"),
             "priority": ("Inference priority", "$\\pi$"), "freq_idx": ("Clock (MHz)", "$f$")}
    pts = Scenario.nominal().device.points
    rows = []
    for p in SPACE:
        label, sym = names[p.name]
        if p.name == "priority":
            vals = "below, equal, above"
        elif p.name == "freq_idx":
            vals = ", ".join(str(q.f_mhz) for q in pts)
        elif p.kind == "range":
            vals = f"[{p.bounds[0]:g}, {p.bounds[1]:g}]"
        else:
            vals = ", ".join(f"{v:g}" for v in p.values)
        rows.append([label, sym, vals, "range" if p.kind == "range" else f"{len(p.values)} values"])
    return _table("space", "Design Space", "|l|c|l|c|", ["Parameter", "Symbol", "Values", "Type"], rows)


def tab_constants() -> str:
    sc = Scenario.nominal()
    w, d, lim = sc.workload, sc.device, sc.limits
    rows = [["Device", f"{d.name}, {d.flash_bytes // 1024} KB flash, {d.sram_bytes // 1024} KB SRAM, {d.vdd:g} V"]]
    for t in w.periodic:
        rows.append([f"Task {t.name}", f"period {t.period_ms} ms, {t.cycles:,}".replace(f"{t.period_ms} ms, ", f"{t.period_ms} ms;; ").replace(",", "{,}").replace(";;", ",") + f" cycles, priority {t.priority}"])
    rows += [["Inference requests", f"{w.n_requests}, Poisson, mean interarrival {w.mean_interarrival_ms:g} ms, jitter $\\pm${w.jitter * 100:g}\\%"],
             ["Inference priority levels", ", ".join(f"{k} = {v}" for k, v in w.inference_priority)],
             ["Context switch, tick", f"{w.switch_cycles} cycles, {w.tick_cycles} cycles"],
             ["Idle mode", sc.idle_mode],
             ["Limits", f"p95 latency $\\leq$ {lim.latency_ms:g} ms, miss ratio $\\leq$ {lim.miss_ratio:g}, accuracy $\\geq$ {lim.accuracy_floor:g}"]]
    return _table("constants", "Workload and Device Constants", "|l|l|", ["Item", "Value"], rows)


def tab_accuracy() -> str:
    a = results.load("accuracy_validation")
    stds = {b: [v[b]["std"] * 100 for v in a["seed_noise"].values() if b in v] for b in ("32", "16", "8", "4")}
    drop = np.array(list(a["drop_32_to_4_points"].values()))
    rows = [[f"$b = {b}$", len(s), num(float(np.mean(s))), num(float(np.max(s)))] for b, s in stds.items()]
    note = (f"Seed std over {len(a['seed_noise'])} architectures trained with 3 seeds. Mean drop from 32 to 4 bits "
            f"{drop.mean():.2f} points (range {drop.min():.2f} to {drop.max():.2f}) over {a['n_architectures']} architectures.")
    return _table("accuracy", "Accuracy Seed Noise (Points)", "|l|c|c|c|",
                  ["Bit-width", "Architectures", "Mean std", "Max std"], rows, note)


def _med(stat: dict, digits: int = 1) -> str:
    if stat["median"] is None:
        return "--"
    return num(stat["median"], digits)


def _evals(stat: dict) -> str:
    if math.isinf(stat["median"]):
        return f"never"
    return f"{stat['median']:.0f}"


def tab_methods() -> str:
    s = results.load("comparison_summary")
    tests = s["tests"]["wilcoxon_qnehvi_vs_A"]
    rows = []
    for m in ORDER:
        pm = s["per_method"][m]
        if m in tests:
            p, delta = num(tests[m]["p_holm"], 2), num(tests[m]["cliffs_delta"], 2)
        else:
            p, delta = "--", "--"
        rows.append([METHOD_LABELS[m], f"{_med(pm['hv_pct_A_at_100'])} ({pm['hv_pct_A_at_100']['q1']:.1f}--{pm['hv_pct_A_at_100']['q3']:.1f})",
                     _med(pm["hv_pct_B_at_100"]), _evals(pm["evals_to_90_A"]), num(pm["igd_at_100"]["median"], 3), p, delta])
    note = ("Median (IQR) over 20 seeds at 100 evaluations. HV is the percentage of the reference-front hypervolume at "
            "corner A, and at corner B in the next column. $p$ is the one-sided Wilcoxon test of qNEHVI against the method "
            "with Holm correction, and $\\delta$ is Cliff's delta.")
    return _table("methods", "Method Comparison", "|l|c|c|c|c|c|c|",
                  ["Method", "HV$_A$ (\\%)", "HV$_B$ (\\%)", "Evals. to 90\\%", "IGD", "$p$", "$\\delta$"], rows, note, wide=True)


def tab_hardware() -> str:
    hw = results.load("hw_validation")
    rows = []
    for part, label in (("held_out", "Held out"), ("in_sample", "Calibration")):
        sm = hw[part]["summary"]
        rows.append([label, hw[part]["n"]] + [f"{sm[q]['mae_pct']:.2f} ({sm[q]['max_pct']:.2f})" for q in ("cycles", "flash_bytes", "ram_bytes")])
    note = "Mean absolute error in percent (maximum in brackets) of predicted against measured, STM32F469I-DISCO at 180 MHz."
    return _table("hardware", "Hardware Validation Error (\\%)", "|l|c|c|c|c|", ["Models", "n", "Cycles", "Flash", "RAM"], rows, note)


def tab_front() -> str:
    f = results.load("reference_front").sort_values("energy_uj")
    ta = {d["arch"] + f"_b{d['b']}": d for d in results.load("test_accuracy")["designs"]}
    mhz = [p.f_mhz for p in Scenario.nominal().device.points]
    rows = []
    for _, r in f.iterrows():
        t = ta[f"L{int(r['L'])}_w{r['w']:g}_b{int(r['b'])}"]
        rows.append([int(r["L"]), f"{r['w']:g}", int(r["b"]), f"{r['tau']:.1f}", r["pi"], mhz[int(r["f_idx"])],
                     f"{r['accuracy']:.4f}", f"{t['test_accuracy']:.4f}", f"{r['energy_uj']:,.0f}".replace(",", "{,}"),
                     f"{r['latency_ms']:.1f}"])
    return _table("front", "Reference Front", "|c|c|c|c|l|c|c|c|c|c|",
                  ["$L$", "$w$", "$b$", "$\\tau$", "$\\pi$", "$f$", "Val.", "Test", "$E$", "$\\ell_{95}$"], rows,
                  "$\\tau$ in ms, $f$ in MHz, $E$ in $\\mu$J per inference, $\\ell_{95}$ in ms; accuracy is validation (table) and test.")


SENS_CASES = (("nominal", "Nominal"), ("lat50_miss1", "Latency 50 ms"), ("lat200_miss1", "Latency 200 ms"),
              ("lat100_miss0", "Miss limit 0\\%"), ("lat100_miss5", "Miss limit 5\\%"), ("latency_mean", "Mean latency"),
              ("idle_stop", "Stop-mode idle"), ("kernel_x0.5", "Switch and tick $\\times0.5$"),
              ("kernel_x2", "Switch and tick $\\times2$"), ("r4_x0.5", "Int4 overhead $\\times0.5$"),
              ("taskset_u20", "Task set, 20\\% load"), ("taskset_u50", "Task set, 50\\% load"))


def tab_sensitivity() -> str:
    sens = results.load("sensitivity").set_index("case")
    mhz = [p.f_mhz for p in Scenario.nominal().device.points]
    rows = []
    for key, label in SENS_CASES:
        r = sens.loc[key]
        clocks = ", ".join(str(mhz[int(i)]) for i in str(r["front_f"]).split())
        rows.append([label, int(r["n_acceptable"]), int(r["n_front"]), str(r["front_b"]).replace(" ", ", "),
                     str(r["front_L"]).replace(" ", ", "), clocks])
    return _table("sensitivity", "Reference Front Under Changed Scenarios", "|l|c|c|c|c|l|",
                  ["Case", "Acceptable", "Front", "$b$", "$L$", "$f$ (MHz)"], rows)


def tab_calibration() -> str:
    cyc, mem = results.load("cycle_calibration")["report"], results.load("memory_calibration")["report"]
    names = (("Cycles, int8", cyc["cycles_int8"]), ("Cycles, fp32", cyc["cycles_fp32"]),
             ("Kernel flash, int8", mem["kflash8"]), ("Kernel flash, fp32", mem["kflash32"]),
             ("Kernel RAM, int8", mem["kram8"]), ("Kernel RAM, fp32", mem["kram32"]),
             ("Activations, int8", mem["act8"]), ("Activations, fp32", mem["act32"]))
    rows = [[n, r["n"], f"{r['r2']:.3f}", f"{r['loo_mean_rel_err'] * 100:.1f}", f"{r['loo_max_rel_err'] * 100:.1f}"] for n, r in names]
    return _table("calibration", "Calibration Quality", "|l|c|c|c|c|",
                  ["Quantity", "n", "$R^2$", "LOO mean (\\%)", "LOO max (\\%)"], rows,
                  "Leave-one-architecture-out (LOO) relative error of the fits to the Cloud benchmarks.")


TABLES = {"table_space": tab_space, "table_constants": tab_constants, "table_accuracy": tab_accuracy,
          "table_methods": tab_methods, "table_hardware": tab_hardware,
          "table_calibration": tab_calibration, "table_front": tab_front, "table_sensitivity": tab_sensitivity}


def make_all(out: Path) -> list:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in TABLES.items():
        p = out / f"{name}.tex"
        p.write_text(fn())
        paths.append(p)
    return paths
