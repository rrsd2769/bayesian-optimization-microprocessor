"""Pipeline: every stage's artifacts are registered, and `status` runs."""
from src import pipeline, results


def test_stage_artifacts_are_registered_and_produced_by_that_stage():
    for name, s in pipeline.STAGES.items():
        for art in s["artifacts"]:
            assert results.REGISTRY[art].stage == name


def test_status_runs_on_an_empty_store(tmp_path, capsys):
    results.set_root(tmp_path)
    pipeline.main(["status"])
    out = capsys.readouterr().out
    assert "bench" in out and "train_bench" in out and "missing" in out
