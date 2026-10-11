"""
The single entry point.

    python -m src.pipeline <stage> [options]
    python -m src.pipeline status

A stage is a few lines that combine the deep modules: build a job list, call
run_jobs, save through results. Stage logic that grows beyond that belongs in a
module. Each phase registers its own stages here.
"""
import argparse
import json
import logging
import sys

from src import results
from src.jobs import Job, run_jobs

STAGES = {}   # name -> dict(fn, help, artifacts, add_args)


def stage(name, help, artifacts, add_args=None):
    def register(fn):
        STAGES[name] = dict(fn=fn, help=help, artifacts=artifacts, add_args=add_args)
        return fn
    return register


# ── Phase 0 ──────────────────────────────────────────────────────────────────

@stage("bench", "time worst-case training and choose the epoch count", ["train_bench"],
       lambda p: p.add_argument("--epochs-timed", type=int, default=3))
def bench(args):
    from config import DEVICE
    from src.network.bench import run_bench

    def work(job, tmp, deadline):
        tmp.write_text(json.dumps(run_bench(DEVICE, args.epochs_timed, deadline), indent=2, sort_keys=True))

    return run_jobs("bench", [Job("bench", results.path("train_bench"))], work, args.minutes)


def _train_args(p):
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--width", type=float, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=None, help="default: the benchmark's choice, else config.NUM_EPOCHS")


@stage("train", "train one architecture (resumable per epoch)", ["train_run"], _train_args)
def train_one(args):
    from config import CHECKPOINT_DIR, DEVICE, NUM_EPOCHS
    from src.network.arch import Arch
    from src.network.train import load_data, train

    arch = Arch(args.layers, args.width)
    epochs = args.epochs or (results.load("train_bench")["epochs"] if results.exists("train_bench") else NUM_EPOCHS)
    job = Job(f"{arch.name}_seed{args.seed}", results.path("train_run", arch=arch.name, seed=args.seed))

    def work(job, tmp, deadline):
        res = train(arch, args.seed, epochs, load_data(DEVICE), DEVICE,
                    CHECKPOINT_DIR / f"{arch.name}_seed{args.seed}.pt", deadline)
        tmp.write_text(json.dumps(dict(arch=arch.name, seed=args.seed, epochs=epochs, device=DEVICE,
                                       best_val_acc=res.best_val_acc, best_epoch=res.best_epoch,
                                       final_val_acc=res.final_val_acc, wall_s=res.wall_s), indent=2, sort_keys=True))

    return run_jobs("train", [job], work, args.minutes)


# ── Phase 2 ──────────────────────────────────────────────────────────────────

CALIB_LAYERS = (2, 3, 4, 5)
CALIB_WIDTHS = (0.25, 0.625, 1.0)
def calib_archs():
    from src.network.arch import Arch
    return [Arch(L, w) for L in CALIB_LAYERS for w in CALIB_WIDTHS]


@stage("export-calib", "write fp32 and int8 ONNX models of the 12 calibration architectures", ["calib_model"])
def export_calib(args):
    from src.network.export import calibration_images
    cache = {}

    def work(job, tmp, deadline):
        arch, prec = job.meta
        if "calib" not in cache:
            cache["calib"] = calibration_images()
        arch.onnx(32 if prec == "fp32" else 8, tmp, cache["calib"])

    jobs = [Job(f"{a.name}_{p}", results.path("calib_model", arch=a.name, prec=p), meta=(a, p))
            for a in calib_archs() for p in ("fp32", "int8")]
    return run_jobs("export-calib", jobs, work, args.minutes)


@stage("cloud-calib", "manual: benchmark the models on the Cloud, then check coverage", ["cloud_board"])
def cloud_calib(args):
    expected = [(a.name, p) for a in calib_archs() for p in ("fp32", "int8")]
    if not results.exists("cloud_calibration"):
        print(f"Benchmark each model below on the board named in {results.path('cloud_board')}, tick the rows on the "
              "Results step, press 'Export as CSV', then run:\n  python -m src.pipeline import-cloud --csv <file> [<file> ...]")
        for name, prec in expected:
            print(f"  {results.path('calib_model', arch=name, prec=prec)}")
        return
    df = results.load("cloud_calibration")
    todo = [e for e in expected if e not in set(zip(df["arch"], df["precision"]))]
    print(f"benchmarked {len(df)} of {len(expected)} models | still to do: {len(todo)}")
    for name, prec in todo:
        print(f"  {name} {prec}")


@stage("import-cloud", "read Cloud 'Export as CSV' files into cloud_calibration", ["cloud_calibration"],
       lambda p: p.add_argument("--csv", nargs="+", required=True))
def import_cloud(args):
    from src.evaluator import import_cloud_csv, sha256_file
    hashes = {sha256_file(results.path("calib_model", arch=a.name, prec=p)): (a.name, p)
              for a in calib_archs() for p in ("fp32", "int8")}
    existing = results.load("cloud_calibration") if results.exists("cloud_calibration") else None
    merged, problems = import_cloud_csv(args.csv, hashes, existing)
    results.save("cloud_calibration", merged)
    for msg in problems:
        print("problem:", msg)
    print(f"cloud_calibration now has {len(merged)} of {len(hashes)} models")


@stage("calibrate", "fit the cycle and memory models to the Cloud benchmarks", ["cycle_calibration", "memory_calibration"])
def calibrate(args):
    from src.evaluator import calibration_to_json as to_json
    from src.evaluator import fit_cycle_calibration as fit_cycles, fit_memory_calibration as fit_memory
    cloud = results.load("cloud_calibration")
    for (cal, report), name in ((fit_cycles(cloud), "cycle_calibration"), (fit_memory(cloud), "memory_calibration")):
        results.save(name, to_json(cal, report))
        for target, m in report.items():
            print(f"{target:<12} n={m['n']} R2={m['r2']:.4f} fit max err {m['max_rel_err']:.1%} "
                  f"LOO mean {m['loo_mean_rel_err']:.1%} max {m['loo_max_rel_err']:.1%}")


# ── Phase 4 ──────────────────────────────────────────────────────────────────

ACC_GRID_LAYERS = (2, 3, 4, 5)
ACC_GRID_WIDTHS = (0.25, 0.4375, 0.625, 0.8125, 1.0)
SEED_NOISE_ARCHS = ((2, 0.25), (3, 0.625), (5, 1.0))
QAT_PILOT_ARCHS = ((2, 0.25), (5, 1.0))
QAT_PILOT_EPOCHS = 10


def float_epochs():
    from config import NUM_EPOCHS
    return results.load("train_bench")["epochs"] if results.exists("train_bench") else NUM_EPOCHS


@stage("qat-pilot", "4-bit QAT for up to 10 epochs on two architectures; choose the QAT epoch count",
       ["qat_pilot_run", "qat_pilot"])
def qat_pilot(args):
    from config import CHECKPOINT_DIR, DEVICE
    from src.network.arch import Arch
    from src.network.measure import qat_curve
    from src.network.quantize import pick_qat_epochs
    from src.network.train import load_data
    cache = {}
    epochs = float_epochs()

    def work(job, tmp, deadline):
        if "data" not in cache:
            cache["data"] = load_data(DEVICE)
        tmp.write_text(json.dumps(qat_curve(job.meta, 1, epochs, QAT_PILOT_EPOCHS, cache["data"], DEVICE,
                                            CHECKPOINT_DIR, deadline), indent=2, sort_keys=True))

    jobs = [Job(Arch(L, w).name, results.path("qat_pilot_run", arch=Arch(L, w).name), meta=Arch(L, w))
            for L, w in QAT_PILOT_ARCHS]
    summary = run_jobs("qat-pilot", jobs, work, args.minutes)
    if summary.finished:
        runs = [results.load("qat_pilot_run", arch=j.key) for j in jobs]
        chosen = pick_qat_epochs([r["val_accs"] for r in runs])
        results.save("qat_pilot", dict(qat_epochs=chosen, pilot_epochs=QAT_PILOT_EPOCHS, runs=runs))
        for r in runs:
            print(f"{r['arch']}: float {r['float_val_acc']:.4f} | 4-bit before QAT {r['ptq4_val_acc']:.4f} | QAT per epoch "
                  + " ".join(f"{a:.4f}" for a in r["val_accs"]))
        print(f"chosen QAT epochs: {chosen}" + (" (pilot did not converge within 10)" if chosen == QAT_PILOT_EPOCHS else ""))
    return summary


def accuracy_jobs():
    from src.network.arch import Arch
    grid = [(Arch(L, w), 1) for L in ACC_GRID_LAYERS for w in ACC_GRID_WIDTHS]
    repeats = [(Arch(L, w), s) for L, w in SEED_NOISE_ARCHS for s in (2, 3)]
    return [Job(f"{a.name}_seed{s}", results.path("accuracy_run", arch=a.name, seed=s), meta=(a, s)) for a, s in grid + repeats]


def merge_accuracy(jobs):
    import pandas as pd
    parts = [pd.read_csv(j.output) for j in jobs if j.output.exists()]
    if parts:
        results.save("accuracy_measurements", pd.concat(parts, ignore_index=True))
    return len(parts)


@stage("accuracy", "train the 26 networks and measure validation accuracy at b = 32, 16, 8, 4",
       ["accuracy_run", "accuracy_measurements"])
def accuracy(args):
    from config import CHECKPOINT_DIR, DEVICE
    from src.network.measure import measure_arch
    from src.network.train import load_data
    if not results.exists("qat_pilot"):
        raise SystemExit("run the qat-pilot stage first: it sets the number of QAT epochs")
    epochs, qat_epochs = float_epochs(), results.load("qat_pilot")["qat_epochs"]
    cache = {}

    def work(job, tmp, deadline):
        if "data" not in cache:
            cache["data"] = load_data(DEVICE)
        arch, seed = job.meta
        measure_arch(arch, seed, epochs, qat_epochs, cache["data"], DEVICE, CHECKPOINT_DIR, deadline).to_csv(tmp, index=False)

    jobs = accuracy_jobs()
    summary = run_jobs("accuracy", jobs, work, args.minutes)
    print(f"accuracy_measurements: {merge_accuracy(jobs)} of {len(jobs)} trainings merged")
    return summary


@stage("validate-accuracy", "seed noise and the 32-to-4-bit accuracy drop", ["accuracy_validation"])
def validate_accuracy(args):
    from src.report.accuracy_validation import validate
    out = validate(results.load("accuracy_measurements"))
    results.save("accuracy_validation", out)
    for arch, per_b in out["seed_noise"].items():
        print(f"{arch}: seed std (points) " + " ".join(f"b{b}={v['std'] * 100:.2f}" for b, v in sorted(per_b.items(), key=lambda kv: int(kv[0]))))
    print(f"architectures with 4-bit accuracy below 32-bit: {out['n_with_drop_above_zero']} of {out['n_architectures']}")


# ── Phase 5 ──────────────────────────────────────────────────────────────────

def _read_blocks():
    import pandas as pd
    from src.search.reference_front import blocks
    return pd.concat([results.load("front_block", L=L, b=b) for L, b in blocks()], ignore_index=True)


@stage("front", "enumerate the 12,000-design grid, save the Pareto front and the hypervolume corners",
       ["front_block", "hv_reference", "reference_front", "reference_front_summary"])
def front(args):
    from src.evaluator import Scenario
    from src.search import reference_front as rf
    sc = Scenario.nominal()

    def work(job, tmp, deadline):
        L, b = job.meta
        rf.evaluate_block(L, b, sc).to_csv(tmp, index=False)

    jobs = [Job(f"L{L}_b{b}", results.path("front_block", L=L, b=b), meta=(L, b)) for L, b in rf.blocks()]
    summary = run_jobs("front", jobs, work, args.minutes)
    if not summary.finished:
        return summary
    df = _read_blocks()
    if not results.exists("hv_reference"):     # chosen once, never changed afterwards
        ok = df[df["acceptable"]]
        results.save("hv_reference", rf.hv_corners(float(ok["energy_uj"].max())))
    corners = results.load("hv_reference")
    results.save("reference_front", rf.front_of(df))
    out = rf.summarise(df, corners)
    out["wall_s"] = float(df.groupby(["L", "b"])["block_wall_s"].first().sum())
    results.save("reference_front_summary", out)
    print(json.dumps({k: out[k] for k in ("grid_size", "n_acceptable", "n_front", "rejected", "hypervolume", "front_composition")}, indent=1))
    print("hv corners:", corners)
    return summary


def sensitivity_cases(base):
    """Every sensitivity case as a Scenario derived from the nominal one."""
    cases = {"nominal": base}
    for lat in (50, 100, 200):
        for miss in (0.0, 0.01, 0.05):
            if (lat, miss) != (100, 0.01):
                cases[f"lat{lat}_miss{round(miss * 100)}"] = base.with_limits(latency_ms=float(lat), miss_ratio=miss)
    cases["idle_stop"] = base.but(idle_mode="stop")
    cases["latency_mean"] = base.with_limits(latency_stat="mean")
    for k in (0.5, 2.0):
        cases[f"kernel_x{k:g}"] = base.with_kernel_scale(k)
        cases[f"r16_x{k:g}"] = base.with_overhead_scale(r16=k)
        cases[f"r4_x{k:g}"] = base.with_overhead_scale(r4=k)
    for u in (0.2, 0.5):     # Kim 2021 harmonicT2, 2 tasks, utilization at 180 MHz
        cases[f"taskset_u{round(u * 100)}"] = base.with_generated_tasks(u)
    return cases


@stage("sensitivity", "recompute the front under limits, idle mode, kernel costs and literature overheads",
       ["sensitivity_case", "sensitivity"])
def sensitivity(args):
    import pandas as pd
    from src.evaluator import Scenario
    from src.search import reference_front as rf
    if not results.exists("hv_reference"):
        raise SystemExit("run the front stage first: it fixes the hypervolume corners")
    corners = results.load("hv_reference")
    cases = sensitivity_cases(Scenario.nominal())

    def work(job, tmp, deadline):
        df = pd.concat([rf.evaluate_block(L, b, cases[job.key]) for L, b in rf.blocks()], ignore_index=True)
        out = rf.summarise(df, corners)
        out["case"] = job.key
        tmp.write_text(json.dumps(out, indent=2, sort_keys=True))

    jobs = [Job(name, results.path("sensitivity_case", case=name)) for name in cases]
    summary = run_jobs("sensitivity", jobs, work, args.minutes)
    rows = []
    for j in jobs:
        if j.output.exists():
            r = json.loads(j.output.read_text())
            comp = r["front_composition"]
            rows.append(dict(case=r["case"], n_acceptable=r["n_acceptable"], n_front=r["n_front"],
                             hv_A=r["hypervolume"]["A"], hv_B=r["hypervolume"]["B"],
                             front_b=" ".join(comp["b"]), front_L=" ".join(comp["L"]), front_f=" ".join(comp["f_idx"]),
                             front_pi=" ".join(comp["pi"])))
    if rows:
        results.save("sensitivity", pd.DataFrame(rows))
        print(pd.DataFrame(rows).to_string(index=False))
    return summary


# ── Phase 6 ──────────────────────────────────────────────────────────────────

# Minutes of chunk time a run needs before it may start (a run cannot be resumed, so none is started that the
# deadline would cut off). Set from the pilot.
RUN_MINUTES = {"qnehvi": 40.0, "qnparego": 40.0, "sequential": 40.0, "nsga2": 1.0, "motpe": 2.0, "random": 1.0}


def parse_seeds(text):
    """'1-20' or '3,5,9' -> [ints]."""
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def _compare_args(p):
    from src.search.harness import METHODS
    p.add_argument("--method", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--seeds", default="1-20", help="e.g. 1-20 or 1,2,5")
    p.add_argument("--budget", type=int, default=100)


@stage("compare", "run the search methods: one CSV per method and seed with a running hypervolume", ["run"], _compare_args)
def compare(args):
    from src.jobs import DeadlineReached
    from src.search.harness import run_search

    def work(job, tmp, deadline):
        method, seed = job.meta
        if deadline.remaining_s() < RUN_MINUTES[method] * 60:
            raise DeadlineReached()
        run_search(method, seed, args.budget).frame.to_csv(tmp, index=False)

    seeds = parse_seeds(args.seeds)
    jobs = [Job(f"{m}/seed{s}", results.path("run", method=m, seed=s), meta=(m, s)) for s in seeds for m in args.method]
    return run_jobs("compare", jobs, work, args.minutes)


@stage("status", "list every stage, its artifacts and whether they exist", [])
def status(args):
    for name, s in STAGES.items():
        if name == "status":
            continue
        print(f"{name:<10} {s['help']}")
        for art in s["artifacts"]:
            n = results.count(art)
            print(f"    {art:<22} {'done' if n else 'missing'}" + (f" ({n} files)" if n > 1 else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.pipeline")
    parser.add_argument("--minutes", type=float, default=55.0, help="chunk time budget")
    sub = parser.add_subparsers(dest="stage", required=True)
    for name, s in STAGES.items():
        sp = sub.add_parser(name, help=s["help"])
        if s["add_args"]:
            s["add_args"](sp)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    STAGES[args.stage]["fn"](args)


if __name__ == "__main__":
    main()
