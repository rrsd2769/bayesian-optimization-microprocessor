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
