"""Seed noise and 32-to-4-bit drop from a hand-made measurement table."""
import pandas as pd
import pytest

from src.network.quantize import pick_qat_epochs
from src.report.accuracy_validation import validate


def test_validate_seed_noise_and_drop():
    rows = [("L2_w0.25", s, b, v, 50, 1.0) for s, b, v in
            [(1, 32, .80), (2, 32, .82), (3, 32, .78), (1, 4, .75), (2, 4, .74), (3, 4, .76)]]
    rows += [("L3_w1", 1, 32, .9, 50, 1.0), ("L3_w1", 1, 4, .92, 54, 1.0)]
    out = validate(pd.DataFrame(rows, columns=["arch", "seed", "b", "val_acc", "epochs", "wall_s"]))
    assert out["seed_noise"]["L2_w0.25"]["32"]["std"] == pytest.approx(0.02)
    assert "L3_w1" not in out["seed_noise"]
    assert out["drop_32_to_4_points"]["L2_w0.25"] == pytest.approx(5.0)
    assert out["drop_32_to_4_points"]["L3_w1"] == pytest.approx(-2.0)
    assert out["n_with_drop_above_zero"] == 1


def test_pick_qat_epochs_needs_every_later_epoch_within_0p2_points_for_both_curves():
    a = [.70, .75, .76, .761, .760, .7605]
    b = [.60, .70, .74, .75, .751, .7505]
    assert pick_qat_epochs([a, b]) == 4         # a: gain after 3 is 0.1 pt; b: after 4 it is 0.1 pt
    assert pick_qat_epochs([a]) == 3
    assert pick_qat_epochs([[.5, .6, .7, .8]]) == 4     # never converges: returns K


def test_job_list_is_26_trainings():
    from src.pipeline import accuracy_jobs
    jobs = accuracy_jobs()
    assert len(jobs) == 26 and len({j.key for j in jobs}) == 26
