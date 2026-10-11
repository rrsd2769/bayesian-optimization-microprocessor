"""The search harness and the method adapters, through run_search / ask / tell."""
import numpy as np
import pytest

from src.evaluator import Design, Scenario
from src.search.harness import COLUMNS, METHODS, run_search
from src.search.methods._encoding import design_from_indices, indices_of

@pytest.fixture(autouse=True)
def real_results_store():
    """Other tests point the store at a temp dir; the Ax adapters read the real hv_reference."""
    from config import RESULTS_DIR
    from src import results
    results.set_root(RESULTS_DIR)


CORNERS = dict(A=dict(accuracy=0.70, energy_uj=17000.0), B=dict(accuracy=0.75, energy_uj=23000.0))
SC = Scenario.nominal()

GOOD = Design(4, 0.25, 8, 10.278, "equal", 2)         # acceptable at nominal (on the reference front)
TOO_BIG = Design(5, 1.0, 32, 10.0, "equal", 4)       # flash violated: short-circuited by the evaluator
LOW_ACC = Design(2, 0.25, 4, 10.0, "equal", 4)


class Fixed:
    """A fake adapter proposing a fixed list, two designs per ask."""
    def __init__(self, designs):
        self.queue, self.told = list(designs), []

    def ask(self):
        batch, self.queue = self.queue[:2], self.queue[2:]
        return batch

    def tell(self, designs, outcomes):
        self.told.extend(zip(designs, outcomes))


def test_harness_stops_at_exactly_the_budget_and_counts_short_circuited():
    adapter = Fixed([GOOD, TOO_BIG, LOW_ACC, TOO_BIG, GOOD])
    trace = run_search(adapter, seed=1, budget=5, scenario=SC, corners=CORNERS)
    df = trace.frame
    assert len(df) == 5 and list(df["eval"]) == [1, 2, 3, 4, 5]
    assert df["violated"].str.contains("flash").sum() == 2          # short-circuited designs count
    assert len(adapter.told) == 5


def test_harness_truncates_a_batch_at_the_budget():
    trace = run_search(Fixed([GOOD] * 6), seed=1, budget=3, scenario=SC, corners=CORNERS)
    assert len(trace.frame) == 3


def test_harness_columns_and_nondecreasing_hv():
    trace = run_search(Fixed([LOW_ACC, GOOD, TOO_BIG, GOOD]), seed=1, budget=4, scenario=SC, corners=CORNERS)
    df = trace.frame
    assert list(df.columns) == COLUMNS
    for k in ("hv_A", "hv_B"):
        assert (np.diff(df[k]) >= 0).all()
    assert df["hv_A"].iloc[0] == 0.0 and df["hv_A"].iloc[1] > 0.0     # the unacceptable design adds nothing


def test_running_dry_adapter_is_an_error():
    with pytest.raises(RuntimeError):
        run_search(Fixed([GOOD]), seed=1, budget=4, scenario=SC, corners=CORNERS)


def test_unknown_method():
    with pytest.raises(ValueError):
        run_search("simulated-annealing", seed=1, budget=1, scenario=SC, corners=CORNERS)


def test_encoding_round_trip():
    for d in (GOOD, TOO_BIG, LOW_ACC):
        assert design_from_indices(indices_of(d)) == d


@pytest.mark.parametrize("method", [m for m in METHODS if m in ("random", "nsga2", "motpe")])
def test_cheap_adapters_smoke_and_determinism(method):
    a = run_search(method, seed=3, budget=12, scenario=SC, corners=CORNERS).frame
    b = run_search(method, seed=3, budget=12, scenario=SC, corners=CORNERS).frame
    assert len(a) == 12
    cols = [c for c in COLUMNS if c not in ("ask_s", "method")]
    assert a[cols].equals(b[cols])
    assert (np.diff(a["hv_A"]) >= 0).all()


@pytest.mark.parametrize("method", ["qnehvi", "qnparego"])
def test_ax_adapters_smoke_through_the_gp(method):
    df = run_search(method, seed=2, budget=14, scenario=SC, corners=CORNERS).frame     # 12 Sobol + 2 GP steps
    assert len(df) == 14 and (np.diff(df["hv_A"]) >= 0).all()


def test_sequential_switches_stage(monkeypatch):
    from src.search.methods import sequential
    monkeypatch.setattr(sequential, "STAGE_BUDGET", 4)
    df = run_search("sequential", seed=2, budget=8, scenario=SC, corners=CORNERS).frame
    assert len(df) == 8
    first, second = df.iloc[:4], df.iloc[4:]
    assert (first["time_slice_ms"] == 10.0).all() and (first["freq_idx"] == 4).all() and (first["priority"] == 1).all()
    nets = set(zip(first["num_layers"], first["width_mult"], first["quant_bits"]))
    assert set(zip(second["num_layers"], second["width_mult"], second["quant_bits"])) <= nets
