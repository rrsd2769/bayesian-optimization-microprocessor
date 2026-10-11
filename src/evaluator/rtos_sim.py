"""
Deterministic event-driven simulator of FreeRTOS-style scheduling on one core:
fixed-priority preemptive, round robin among equal priorities, 1 kHz tick.

Time is an integer cycle count at the chosen clock. One tick is f_hz / 1000 cycles, so periods, slices and
arrivals are whole milliseconds. Everything that releases a task or reschedules happens in the tick handler:

    tick       costs tick_cycles (taken from whatever is running), then releases due jobs, then decides:
               a ready job of higher priority preempts the running one; a ready job of equal priority takes over
               once the running one has used its time slice tau (whole ticks, at least 1)
    completion the next ready job (highest priority, FIFO within a priority) is dispatched at once
    dispatch   costs switch_cycles and counts as one context switch (also from the idle task)

A preempted job returns to the front of its priority queue, a job whose slice expired to the back. Higher number =
higher priority. Inference requests arrive as a Poisson process in continuous time; the job is released at the next
tick, but its latency is measured from the true arrival. The random draws are fixed by the workload seed, so the
same requests (arrival times and +-jitter on cycles) are replayed for every design: designs differ in nothing but
cycles, clock, tau and priority.

Horizon: time 0 to completion of the last inference request. A periodic job misses when it finishes after its
deadline (= next release) or has not finished by it; only jobs whose deadline lies within the horizon count.
If the simulation reaches `max_horizon_s` before the last request completes, it stops and reports completed=False
with unfinished requests counted as latency-so-far, so overload remains ordered rather than infinite.

Kernel costs:
    switch_cycles = 390   DERIVED from Ungurean 2020 (Symmetry 12(4):592, 10.3390/sym12040592), FreeRTOS 10.2.0 on
                          an STM32F407IG (Cortex-M4, 168 MHz, Keil O3, oscilloscope on a GPIO pin). Their low-to-high
                          priority switch triggered by an event takes 4.13 us (Fig. 5a) and the signal primitive alone
                          1.82 us (Fig. 11a), so the switch is 2.31 us = 388 cycles; for a mailbox 3.65 - 1.37 =
                          2.28 us = 383 cycles (Figs 7a, 13a); for a semaphore 5.26 - 2.16 = 3.10 us = 521 cycles
                          (Figs 6a, 12a). The subtraction is ours, not the paper's, and what remains still contains
                          the return from the blocking call, so it is an upper bound on a bare switch. Used as 390.
                          Cycles are assumed not to change with the clock (same core; not tested in the paper).
    tick_cycles   = 100   ASSUMPTION, no source: the tick-interrupt cost is not reported by Ungurean 2020 or by the
                          Oliveira and Lima 2023 preprint (which gives only total scheduling overhead as a share of
                          time). The jitter Ungurean reports on high-to-low switches (0.65 to 1.22 us for mailbox and
                          event, 110 to 205 cycles) is partly tick-interrupt interference and suggests the right order
                          of magnitude only.
The sensitivity runs scale both by 0.5 and 2 (`Workload.with_kernel_scale`).
"""
from collections import deque
from dataclasses import dataclass, replace
import numpy as np

PI_LEVELS = ("below", "equal", "above")


@dataclass(frozen=True)
class PeriodicTask:
    name: str
    period_ms: int
    cycles: int
    priority: int


@dataclass(frozen=True)
class Workload:
    periodic: tuple = (PeriodicTask("sensor", 10, 90_000, 3), PeriodicTask("comms", 50, 360_000, 2))
    inference_priority: tuple = (("below", 1), ("equal", 2), ("above", 4))
    n_requests: int = 50
    mean_interarrival_ms: float = 200.0
    jitter: float = 0.05
    seed: int = 20261011
    switch_cycles: int = 390      # derived from Ungurean 2020, see module docstring
    tick_cycles: int = 100        # ASSUMPTION, no source
    max_horizon_s: float = 120.0

    def priority_of(self, pi: str) -> int:
        return dict(self.inference_priority)[pi]

    def with_kernel_scale(self, k: float) -> "Workload":
        return replace(self, switch_cycles=round(self.switch_cycles * k), tick_cycles=round(self.tick_cycles * k))


@dataclass(frozen=True)
class SimResult:
    mean_latency_ms: float
    p95_latency_ms: float
    miss_ratio: float
    active_cycles: int
    horizon_s: float
    n_switches: int
    completed: bool


def _requests(w: Workload):
    rng = np.random.default_rng(w.seed)
    arrivals_ms = np.cumsum(rng.exponential(w.mean_interarrival_ms, w.n_requests))
    factors = rng.uniform(1 - w.jitter, 1 + w.jitter, w.n_requests)
    return arrivals_ms, factors


def simulate(inference_cycles: float, f_hz: int, tau_ms: float, pi: str, workload: Workload) -> SimResult:
    if pi not in PI_LEVELS:
        raise ValueError(f"pi must be one of {PI_LEVELS}, got {pi}")
    w = workload
    tick = f_hz // 1000
    tau_ticks = max(1, int(np.ceil(tau_ms)))
    sw, tk = w.switch_cycles, w.tick_cycles
    arrivals_ms, factors = _requests(w)
    inf_prio = w.priority_of(pi)
    inf_cycles = [max(1, int(round(inference_cycles * x))) for x in factors]
    rel_tick = [int(np.ceil(a)) for a in arrivals_ms]            # tick index at which each request is released
    arr_cycle = [a * tick for a in arrivals_ms]                  # true arrival, in cycles (float)

    # job = [remaining, priority, kind, index, deadline_cycle]; kind 0 periodic, 1 inference
    ready = {}
    def queue(p):
        q = ready.get(p)
        if q is None:
            q = ready[p] = deque()
        return q
    prios = sorted({t.priority for t in w.periodic} | {inf_prio}, reverse=True)

    t = 0                   # clock, cycles
    active = 0
    n_switch = 0
    running = None
    slice_used = 0
    k = 0                   # tick index
    next_req = 0
    done = 0
    backlog = deque()       # released requests waiting for the inference task
    inf_active = False      # the inference task has a request running or queued
    finish = [None] * w.n_requests
    periodic_jobs = 0
    missed = 0              # finished late
    pending_periodic = {}   # id(job) -> deadline, for jobs not yet finished
    cap = int(w.max_horizon_s * f_hz)
    periods = [(p.period_ms, p.cycles, p.priority) for p in w.periodic]
    t_tick = 0
    horizon = None
    deadlines_done = []     # (deadline, finished_cycle) for periodic jobs

    def top_ready():
        for p in prios:
            if ready.get(p):
                return p
        return None

    while True:
        # run until the running job completes or the next tick, whichever comes first
        if running is not None and t + running[0] <= t_tick:
            t += running[0]
            active += running[0]
            job, running = running, None
            if job[2] == 1:
                finish[job[3]] = t
                done += 1
                if done == w.n_requests:
                    horizon = t
                    break
                if backlog:                      # the same task takes its next request
                    nxt = backlog.popleft()
                    p = top_ready()
                    if p is None or p <= inf_prio:
                        running = nxt
                        continue
                    queue(inf_prio).appendleft(nxt)
                else:
                    inf_active = False
            else:
                deadlines_done.append((job[4], t))
            p = top_ready()
            if p is not None:
                running = ready[p].popleft()
                t += sw; active += sw; n_switch += 1; slice_used = 0
            continue
        if running is not None and t < t_tick:
            run = t_tick - t
            running[0] -= run
            t += run
            active += run
        # tick
        if t < t_tick:
            t = t_tick                                           # idle until the tick
        t += tk; active += tk
        if t > cap:
            break
        for period_ms, cyc, prio in periods:
            if k % period_ms == 0:
                queue(prio).append([cyc, prio, 0, 0, (k + period_ms) * tick])
        while next_req < w.n_requests and rel_tick[next_req] <= k:
            req = [inf_cycles[next_req], inf_prio, 1, next_req, 0]
            if inf_active:
                backlog.append(req)
            else:
                queue(inf_prio).append(req)
                inf_active = True
            next_req += 1
        if running is not None:
            slice_used += 1
        p = top_ready()
        if p is not None:
            if running is None:
                running = ready[p].popleft()
                t += sw; active += sw; n_switch += 1; slice_used = 0
            elif p > running[1]:
                queue(running[1]).appendleft(running)
                running = ready[p].popleft()
                t += sw; active += sw; n_switch += 1; slice_used = 0
            elif p == running[1] and slice_used >= tau_ticks:
                queue(p).append(running)
                running = ready[p].popleft()
                t += sw; active += sw; n_switch += 1; slice_used = 0
        k += 1
        t_tick += tick

    completed = horizon is not None
    end = horizon if completed else t
    lat = []        # requests that arrived before the end; unfinished ones count their latency so far
    for i in range(w.n_requests):
        if arr_cycle[i] <= end:
            fin = finish[i] if finish[i] is not None else end
            lat.append((fin - arr_cycle[i]) / f_hz * 1000.0)
    # periodic misses over jobs whose deadline lies within the horizon: finished late, or not finished by the deadline
    total = missed = 0
    for dl, fin in deadlines_done:
        if dl <= end:
            total += 1
            missed += fin > dl
    unfinished = [j[4] for q in ready.values() for j in q if j[2] == 0] + ([running[4]] if running and running[2] == 0 else [])
    for dl in unfinished:
        if dl <= end:
            total += 1
            missed += 1
    return SimResult(mean_latency_ms=float(np.mean(lat)), p95_latency_ms=float(np.percentile(lat, 95)),
                     miss_ratio=missed / total if total else 0.0, active_cycles=int(active),
                     horizon_s=end / f_hz, n_switches=n_switch, completed=completed)
