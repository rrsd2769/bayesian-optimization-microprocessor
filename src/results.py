"""
Results store: every artifact the pipeline produces, by name.

Interface:
    results.load(name, **params)        -> DataFrame (csv) or dict (json)
    results.save(name, data, **params)
    results.exists(name, **params) / results.path(name, **params)

An artifact is a name, a path template under the results root, a format (json, csv or file) and the
pipeline stage that produces it. Nothing outside this module builds paths into
results/. Artifacts are registered in the phase that first writes them.
Tests point the store at a temp dir with `results.set_root(tmp_path)`.
"""
import os, re, json
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Artifact:
    name: str
    template: str        # relative to the results root, may contain {param} fields
    fmt: str             # "json" or "csv"
    stage: str           # pipeline stage that produces it

    @property
    def params(self) -> tuple:
        return tuple(f for _, f, _, _ in Formatter().parse(self.template) if f)


REGISTRY = {a.name: a for a in [
    # Phase 0
    Artifact("train_bench", "train_bench.json", "json", "bench"),
    Artifact("train_run", "train/{arch}_seed{seed}.json", "json", "train"),
    # Phase 2
    Artifact("calib_model", "exports/calib/{arch}_{prec}.onnx", "file", "export-calib"),
    Artifact("cloud_board", "cloud_board.json", "json", "cloud-calib"),
    Artifact("cloud_calibration", "cloud_calibration.csv", "csv", "import-cloud"),
    Artifact("cycle_calibration", "cycle_calibration.json", "json", "calibrate"),
    Artifact("memory_calibration", "memory_calibration.json", "json", "calibrate"),
    # Phase 4
    Artifact("qat_pilot_run", "qat_pilot/{arch}.json", "json", "qat-pilot"),
    Artifact("qat_pilot", "qat_pilot.json", "json", "qat-pilot"),
    Artifact("accuracy_run", "accuracy/{arch}_seed{seed}.csv", "csv", "accuracy"),
    Artifact("accuracy_measurements", "accuracy_measurements.csv", "csv", "accuracy"),
    Artifact("accuracy_validation", "accuracy_validation.json", "json", "validate-accuracy"),
    # Phase 5
    Artifact("front_block", "front/L{L}_b{b}.csv", "csv", "front"),
    Artifact("hv_reference", "hv_reference.json", "json", "front"),
    Artifact("reference_front", "reference_pareto_front.csv", "csv", "front"),
    Artifact("reference_front_summary", "reference_front_summary.json", "json", "front"),
    Artifact("sensitivity_case", "sensitivity/{case}.json", "json", "sensitivity"),
    Artifact("sensitivity", "sensitivity.csv", "csv", "sensitivity"),
    # Phase 6
    Artifact("run", "runs/{method}/seed{seed}.csv", "csv", "compare"),
    Artifact("comparison_summary", "comparison_summary.json", "json", "stats"),
]}


class ResultsStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _artifact(self, name: str, params: dict) -> Artifact:
        if name not in REGISTRY:
            raise KeyError(f"unknown artifact '{name}'; known: {sorted(REGISTRY)}")
        art = REGISTRY[name]
        if set(params) != set(art.params):
            raise TypeError(f"artifact '{name}' takes parameters {art.params}, got {tuple(params)}")
        return art

    def path(self, name: str, **params) -> Path:
        art = self._artifact(name, params)
        return self.root / art.template.format(**params)

    def exists(self, name: str, **params) -> bool:
        return self.path(name, **params).exists()

    def load(self, name: str, **params) -> Any:
        art = self._artifact(name, params)
        p = self.path(name, **params)
        if art.fmt == "file":
            raise TypeError(f"artifact '{name}' is a file; use path()")
        if not p.exists():
            raise FileNotFoundError(f"artifact '{name}' is missing: run stage '{art.stage}' first")
        if art.fmt == "json":
            return json.loads(p.read_text())
        return pd.read_csv(p)

    def save(self, name: str, data: Any, **params) -> Path:
        art = self._artifact(name, params)
        p = self.path(name, **params)
        if art.fmt == "file":
            raise TypeError(f"artifact '{name}' is a file written by its stage; use path()")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        if art.fmt == "json":
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        else:
            data.to_csv(tmp, index=False)
        os.replace(tmp, p)
        return p

    def count(self, name: str) -> int:
        """Number of files present for an artifact (parametrised ones are globbed)."""
        pattern = re.sub(r"\{[^}]+\}", "*", REGISTRY[name].template)
        return len(list(self.root.glob(pattern)))


_store = None


def set_root(root: Path) -> None:
    global _store
    _store = ResultsStore(root)


def store() -> ResultsStore:
    global _store
    if _store is None:
        from config import RESULTS_DIR
        _store = ResultsStore(RESULTS_DIR)
    return _store


def path(name, **params): return store().path(name, **params)
def exists(name, **params): return store().exists(name, **params)
def load(name, **params): return store().load(name, **params)
def save(name, data, **params): return store().save(name, data, **params)
def count(name): return store().count(name)
