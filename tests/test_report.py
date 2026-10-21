"""Report helpers: median seed, front extraction, number formatting, and every table and figure from the real store."""
import numpy as np
import pandas as pd
import pytest

from src import results
from src.report import figures, tables


@pytest.fixture(autouse=True)
def real_store():
    from config import RESULTS_DIR
    results.set_root(RESULTS_DIR)


def test_median_seed_ties_go_to_the_lowest_seed():
    assert figures.median_seed({3: 100.0, 1: 100.0, 2: 100.0, 4: 10.0}) == 1


def test_median_seed_picks_the_closest_to_the_median():
    assert figures.median_seed({1: 10.0, 2: 50.0, 3: 90.0}) == 2


def test_front_points_keep_only_acceptable_non_dominated():
    run = pd.DataFrame({"accuracy": [0.8, 0.7, 0.9, 0.85], "energy_uj": [10.0, 5.0, 30.0, 40.0],
                        "acceptable": [True, True, True, False]})
    pts = figures.front_points(run)
    assert pts.tolist() == [[0.7, 5.0], [0.8, 10.0], [0.9, 30.0]]


def test_num_uses_scientific_form_for_small_values():
    assert tables.num(0.0166, 2) == "$1.66\\times10^{-2}$"
    assert tables.num(12.345, 1) == "12.3"
    assert tables.num(float("inf")) == "never"


def test_every_figure_and_table_is_generated_from_the_store(tmp_path):
    figs = figures.make_all(tmp_path / "f")
    tabs = tables.make_all(tmp_path / "t")
    assert len(figs) == 3 and len(tabs) == 8
    assert all(p.stat().st_size > 1000 for p in figs)
    for p in tabs:
        text = p.read_text()
        assert text.count("\\begin{table") == 1 and "\\rowcolor{tblhead}" in text and "nan" not in text
