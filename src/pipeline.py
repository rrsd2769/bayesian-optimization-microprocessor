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
    from src.evaluator.cloud import import_cloud_csv, sha256_file
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
    from src.evaluator.calibrate import fit_cycles, fit_memory, to_json
    cloud = results.load("cloud_calibration")
    for (cal, report), name in ((fit_cycles(cloud), "cycle_calibration"), (fit_memory(cloud), "memory_calibration")):
        results.save(name, to_json(cal, report))
        for target, m in report.items():
            print(f"{target:<12} n={m['n']} R2={m['r2']:.4f} fit max err {m['max_rel_err']:.1%} "
                  f"LOO mean {m['loo_mean_rel_err']:.1%} max {m['loo_max_rel_err']:.1%}")


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
