"""Cloud CSV import: match by model hash, derive the clock, merge reruns, report problems."""
import pandas as pd
import pytest

from src.evaluator.cloud import OUT_COLUMNS, import_cloud_csv

HEADER = ("benchmarkId,board,createdAt,sha256,md5,state,cycles,duration_ms,device,frequency_MHz,model_name,"
          "model_size,rom_n_macc,val_error,allocate-inputs,allocate-outputs,compression,optimization,"
          "layersInExternalFlash,layersInInternalFlash,useExternalFlash,useExternalRam,total_flash_bytes,"
          "total_ram_bytes,internal_flash,external_flash,internal_ram,external_ram,activations_bytes,"
          "weights_bytes,extra_flash_bytes,kernel_flash_bytes,extra_ram_bytes,kernel_ram_bytes,io_bytes")


def row(bid, sha, state="done", cycles=2305914, ms=12.810633385553956):
    return (f"{bid},STM32F469I-DISCO,created-{bid},{sha},md5,{state},\"{cycles}\",\"{ms}\",\"undefined\",,,,,,,,,balanced,"
            f"\"\",\"\",false,false,23221,9368,23221,0,9368,0,6800,5344,,17877,,2568,0/0")


def write(tmp_path, name, *rows):
    p = tmp_path / name
    p.write_text("\n".join([HEADER, *rows]) + "\n")
    return p


HASHES = {"aaa": ("L2_w0.25", "int8"), "bbb": ("L3_w1", "fp32")}


def test_row_is_matched_by_hash_and_clock_is_derived(tmp_path):
    df, problems = import_cloud_csv([write(tmp_path, "a.csv", row("r1", "aaa"))], HASHES)
    assert problems == [] and list(df.columns) == OUT_COLUMNS
    r = df.iloc[0]
    assert (r["arch"], r["precision"], r["cycles"], r["weights_bytes"], r["total_flash_bytes"]) == \
        ("L2_w0.25", "int8", 2305914.0, 5344.0, 23221.0)
    assert r["frequency_mhz"] == 180.0


def test_unmatched_and_failed_rows_are_reported_not_imported(tmp_path):
    df, problems = import_cloud_csv([write(tmp_path, "a.csv", row("r1", "zzz"), row("r2", "bbb", state="error"))], HASHES)
    assert df.empty and len(problems) == 2
    assert "matches none" in problems[0] and "not done" in problems[1]


def test_rerun_replaces_older_row_and_other_models_are_kept(tmp_path):
    first, _ = import_cloud_csv([write(tmp_path, "a.csv", row("r1", "aaa", cycles=1000000), row("r2", "bbb"))], HASHES)
    second, _ = import_cloud_csv([write(tmp_path, "b.csv", row("r3", "aaa", cycles=2000000))], HASHES, existing=first)
    assert len(second) == 2
    assert second[second["arch"] == "L2_w0.25"].iloc[0]["cycles"] == 2000000.0
    assert second[second["arch"] == "L3_w1"].iloc[0]["benchmark_id"] == "r2"


def test_several_files_in_one_call(tmp_path):
    df, _ = import_cloud_csv([write(tmp_path, "a.csv", row("r1", "aaa")), write(tmp_path, "b.csv", row("r2", "bbb"))], HASHES)
    assert set(zip(df["arch"], df["precision"])) == set(HASHES.values())
