"""Results store: save/load round-trip, missing-artifact error, parameter checking."""
import pandas as pd
import pytest

from src import results
from src.results import REGISTRY, Artifact, ResultsStore


@pytest.fixture
def store(tmp_path):
    return ResultsStore(tmp_path)


def test_json_round_trip(store):
    store.save("train_bench", {"epochs": 40, "sec_per_epoch": 12.5})
    assert store.load("train_bench") == {"epochs": 40, "sec_per_epoch": 12.5}


def test_parametrised_artifact_round_trip_and_count(store):
    store.save("train_run", {"v": 1}, arch="L2_w0.25", seed=1)
    store.save("train_run", {"v": 2}, arch="L3_w1", seed=2)
    assert store.load("train_run", arch="L3_w1", seed=2) == {"v": 2}
    assert store.exists("train_run", arch="L2_w0.25", seed=1)
    assert not store.exists("train_run", arch="L5_w1", seed=1)
    assert store.count("train_run") == 2


def test_csv_round_trip(store, monkeypatch):
    monkeypatch.setitem(REGISTRY, "tmp_csv", Artifact("tmp_csv", "t.csv", "csv", "demo"))
    df = pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]})
    store.save("tmp_csv", df)
    pd.testing.assert_frame_equal(store.load("tmp_csv"), df)


def test_missing_artifact_names_its_stage(store):
    with pytest.raises(FileNotFoundError, match="run stage 'bench' first"):
        store.load("train_bench")


def test_wrong_params_and_unknown_name_rejected(store):
    with pytest.raises(TypeError):
        store.path("train_run", arch="L2_w0.25")
    with pytest.raises(KeyError):
        store.path("nope")


def test_no_partial_file_after_save(store):
    store.save("train_bench", {"a": 1})
    assert not list(store.root.rglob("*.tmp"))


def test_module_level_functions_use_set_root(tmp_path):
    results.set_root(tmp_path)
    results.save("train_bench", {"x": 1})
    assert results.load("train_bench") == {"x": 1} and (tmp_path / "train_bench.json").exists()
