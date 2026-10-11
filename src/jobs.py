"""
Job runner: resume, deadline, skip-done and atomic writes for every chunked stage.

Interface: run_jobs(stage, jobs, work, minutes=55) -> RunSummary

A Job is a key plus the file it produces. A job is done when its output exists,
so rerunning a stage skips finished jobs. `work(job, tmp, deadline)` writes its
result to `tmp`; the runner renames `tmp` to the job's output only when `work`
returns normally, so a killed or interrupted run never leaves a half-written
file. `work` may call `deadline.check()` between finer units (epochs); that
raises DeadlineReached, which stops the stage cleanly after the unit of work
has saved whatever resume state it keeps.
"""
import os, time, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

NICE_LEVEL = 10
TORCH_THREADS = 4


class DeadlineReached(Exception):
    """Raised by Deadline.check() when the chunk time budget is used up."""


class Deadline:
    def __init__(self, minutes: float, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._start = clock()
        self._end = self._start + minutes * 60.0

    def remaining_s(self) -> float:
        return self._end - self._clock()

    def expired(self) -> bool:
        return self.remaining_s() <= 0.0

    def elapsed_s(self) -> float:
        return self._clock() - self._start

    def check(self) -> None:
        if self.expired():
            raise DeadlineReached()


@dataclass(frozen=True)
class Job:
    key: str
    output: Path


@dataclass
class RunSummary:
    stage: str
    done_now: list
    skipped: list
    remaining: list
    elapsed_s: float

    @property
    def finished(self) -> bool:
        return not self.remaining


def apply_compute_limits() -> None:
    """Lower priority and cap torch threads. Idempotent; called on entry to run_jobs."""
    current = os.nice(0)
    if current < NICE_LEVEL:
        os.nice(NICE_LEVEL - current)
    try:
        import torch
        torch.set_num_threads(TORCH_THREADS)
    except ImportError:
        pass


def run_jobs(stage: str, jobs: Iterable[Job],
             work: Callable[[Job, Path, Deadline], None],
             minutes: float = 55.0,
             clock: Callable[[], float] = time.monotonic,
             limits: bool = True) -> RunSummary:
    jobs = list(jobs)
    keys = [j.key for j in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{stage}: duplicate job keys")
    if limits:
        apply_compute_limits()
    deadline = Deadline(minutes, clock)
    done_now = []
    pending = [j for j in jobs if not j.output.exists()]
    skipped = [j.key for j in jobs if j.output.exists()]
    stopped = False
    try:
        for job in pending:
            if deadline.expired():
                stopped = True
                break
            job.output.parent.mkdir(parents=True, exist_ok=True)
            tmp = job.output.with_name(job.output.name + ".tmp")
            tmp.unlink(missing_ok=True)
            try:
                work(job, tmp, deadline)
            except DeadlineReached:
                tmp.unlink(missing_ok=True)
                stopped = True
                break
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            if not tmp.exists():
                raise RuntimeError(f"{stage}/{job.key}: work() did not write {tmp.name}")
            os.replace(tmp, job.output)
            done_now.append(job.key)
    finally:
        finished = set(done_now) | set(skipped)
        remaining = [k for k in keys if k not in finished]
        summary = RunSummary(stage, done_now, skipped, remaining, deadline.elapsed_s())
        print(f"[{stage}] elapsed {summary.elapsed_s:.0f}s | finished now {len(done_now)} | "
              f"already done {len(skipped)} | remaining {len(remaining)}"
              + (" | stopped at deadline" if stopped else ""), flush=True)
    return summary
