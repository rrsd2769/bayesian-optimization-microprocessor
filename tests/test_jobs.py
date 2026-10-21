"""Job runner: skip-done, deadline, atomic writes, no-op rerun (fake clock, temp dir)."""
import pytest

from src.jobs import Deadline, DeadlineReached, Job, run_jobs


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_jobs(tmp_path, n):
    return [Job(f"j{i}", tmp_path / f"out{i}.txt") for i in range(n)]


def writer(clock=None, cost_s=0.0, log=None):
    def work(job, tmp, deadline):
        if log is not None:
            log.append(job.key)
        if clock is not None:
            clock.t += cost_s
        tmp.write_text(job.key)
    return work


def test_all_jobs_run_and_outputs_written(tmp_path):
    jobs = make_jobs(tmp_path, 3)
    s = run_jobs("t", jobs, writer(), limits=False)
    assert s.finished and s.done_now == ["j0", "j1", "j2"]
    assert [j.output.read_text() for j in jobs] == ["j0", "j1", "j2"]
    assert not list(tmp_path.glob("*.tmp"))


def test_rerun_of_finished_stage_is_a_noop(tmp_path):
    jobs = make_jobs(tmp_path, 3)
    run_jobs("t", jobs, writer(), limits=False)
    log = []
    s = run_jobs("t", jobs, writer(log=log), limits=False)
    assert log == [] and s.done_now == [] and s.skipped == ["j0", "j1", "j2"] and s.finished


def test_deadline_stops_between_jobs_and_next_chunk_continues(tmp_path):
    clock = FakeClock()
    jobs = make_jobs(tmp_path, 5)
    s = run_jobs("t", jobs, writer(clock, cost_s=20 * 60), minutes=55, clock=clock, limits=False)
    assert s.done_now == ["j0", "j1", "j2"]        # 3 x 20 min passes 55 min; j3 never starts
    assert s.remaining == ["j3", "j4"] and not s.finished
    clock2 = FakeClock()
    s2 = run_jobs("t", jobs, writer(clock2, 1), minutes=55, clock=clock2, limits=False)
    assert s2.skipped == ["j0", "j1", "j2"] and s2.done_now == ["j3", "j4"] and s2.finished


def test_deadline_raised_inside_work_stops_cleanly_without_output(tmp_path):
    jobs = make_jobs(tmp_path, 2)

    def work(job, tmp, deadline):
        tmp.write_text("partial")
        raise DeadlineReached()

    s = run_jobs("t", jobs, work, limits=False)
    assert s.remaining == ["j0", "j1"] and not any(j.output.exists() for j in jobs)
    assert not list(tmp_path.glob("*.tmp"))


def test_job_killed_mid_write_leaves_no_output(tmp_path):
    jobs = make_jobs(tmp_path, 2)

    def work(job, tmp, deadline):
        tmp.write_text("partial")
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run_jobs("t", jobs, work, limits=False)
    assert not any(j.output.exists() for j in jobs) and not list(tmp_path.glob("*.tmp"))


def test_work_that_writes_nothing_is_an_error(tmp_path):
    with pytest.raises(RuntimeError):
        run_jobs("t", make_jobs(tmp_path, 1), lambda job, tmp, d: None, limits=False)


def test_duplicate_keys_rejected(tmp_path):
    jobs = [Job("a", tmp_path / "x"), Job("a", tmp_path / "y")]
    with pytest.raises(ValueError):
        run_jobs("t", jobs, writer(), limits=False)


def test_deadline_object():
    clock = FakeClock()
    d = Deadline(1, clock)
    d.check()
    clock.t = 61
    assert d.expired()
    with pytest.raises(DeadlineReached):
        d.check()
