"""
Import of ST Edge AI Developer Cloud benchmark exports ("Export as CSV" on the Results step).

A row is matched to one of our exported models by the sha256 of the uploaded file (the Cloud's own
model_name column is empty), so there is no hand transcription. The board clock is derived as
cycles / duration, because the frequency column is empty in the export.

Cloud columns used: cycles, duration_ms, total_flash_bytes, total_ram_bytes, weights_bytes,
activations_bytes, kernel_flash_bytes, kernel_ram_bytes. Observed on the first model (L2_w0.25 int8):
the benchmark build reports 8 B more flash and 512 B more RAM than the Optimize-step report
(23221 vs 23213, 9368 vs 8856); weights_bytes equals weights + int32 biases exactly, with no scale bytes.
"""
import hashlib
from pathlib import Path

import pandas as pd

KEEP = ["cycles", "duration_ms", "total_flash_bytes", "total_ram_bytes", "weights_bytes", "activations_bytes",
        "kernel_flash_bytes", "kernel_ram_bytes"]
OUT_COLUMNS = ["arch", "precision", "cycles", "duration_ms", "frequency_mhz", "total_flash_bytes",
               "total_ram_bytes", "weights_bytes", "activations_bytes", "kernel_flash_bytes", "kernel_ram_bytes",
               "benchmark_id", "created_at"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_cloud_csv(csv_paths: list, model_hashes: dict, existing: pd.DataFrame = None):
    """model_hashes: {sha256: (arch_name, precision)}. Returns (merged DataFrame, list of problems)."""
    problems, rows = [], []
    for path in csv_paths:
        raw = pd.read_csv(path)
        for _, r in raw.iterrows():
            key = model_hashes.get(str(r["sha256"]))
            if key is None:
                problems.append(f"{Path(path).name}: row {r['benchmarkId']} matches none of our exported models")
                continue
            if r["state"] != "done":
                problems.append(f"{key[0]} {key[1]}: benchmark state is '{r['state']}', not done")
                continue
            cycles, ms = float(r["cycles"]), float(r["duration_ms"])
            row = dict(arch=key[0], precision=key[1], frequency_mhz=round(cycles / ms / 1000.0, 1),
                       benchmark_id=r["benchmarkId"], created_at=r["createdAt"])
            for c in KEEP:
                row[c] = float(r[c])
            rows.append(row)
    new = pd.DataFrame(rows, columns=OUT_COLUMNS)
    merged = new if existing is None or existing.empty else pd.concat([existing, new], ignore_index=True)
    merged = merged.drop_duplicates(["arch", "precision"], keep="last")      # a rerun replaces the older row
    return merged.sort_values(["arch", "precision"]).reset_index(drop=True)[OUT_COLUMNS], problems
