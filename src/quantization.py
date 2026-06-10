"""
INT8 dynamic quantization pipeline.
quant_bits mapping:
 32 → no quantization (full float32)
 16 → float16 (half precision)
 8 → INT8 dynamic quantization (torch.quantization.quantize_dynamic)
 4 → simulated 4-bit (INT8 × 0.5 size approximation — stated as such in paper)
"""
import os, sys, io, csv, logging, copy
from typing import Tuple
os.environ["PYTHONWARNINGS"] = "ignore"
import torch
import warnings
import torch.nn as nn
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
if "qnnpack" in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = "qnnpack"
#print("Quantized engine:", torch.backends.quantized.engine)
#print("Supported engines:", torch.backends.quantized.supported_engines)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import RESULTS_DIR, CHECKPOINT_DIR, LOG_FORMAT
logger = logging.getLogger(__name__)
def apply_quantization(model: nn.Module, quant_bits: int) -> Tuple[nn.Module,
float]:
 """
 Apply quantization and return (quantized_model, size_kb).
 All quantized models run on CPU.
 """
 model = copy.deepcopy(model).eval()
 if quant_bits == 32:
  size_kb = _param_size_kb(model, bits=32)
  return model, size_kb
 elif quant_bits == 16:
  model = model.half().float() # .float() for eval compatibility
  size_kb = _param_size_kb(model, bits=16)
  return model, size_kb
 elif quant_bits == 8:
  q_model = torch.quantization.quantize_dynamic(
   model, {nn.Linear}, dtype=torch.qint8)
  size_kb = _param_size_kb(q_model,bits=8)
  return q_model, size_kb
 elif quant_bits == 4:
  # Approximation: INT8 with halved size estimate
  q_model = torch.quantization.quantize_dynamic(
   model, {nn.Linear}, dtype=torch.qint8)
  size_kb = _param_size_kb(q_model,bits=4)
  return q_model, round(size_kb, 2)
 else:
  raise ValueError(f"quant_bits must be 4, 8, 16, or 32. Got:{quant_bits}")
def _param_size_kb(model: nn.Module, bits: int) -> float:
 params = sum(p.numel() for p in model.parameters())
 return round((params * bits / 8) / 1024.0, 2)
def _serialized_size_kb(model: nn.Module) -> float:
 buf = io.BytesIO()
 torch.save(model.state_dict(), buf)
 return round(buf.tell() / 1024.0, 2)
def measure_quantization_impact(model: nn.Module, test_loader: DataLoader,
 quant_bits: int) -> dict:
 from src.train import evaluate_accuracy
 baseline_kb = _param_size_kb(model, 32)
 q_model, q_kb = apply_quantization(model, quant_bits)
 acc = evaluate_accuracy(q_model, test_loader, device="cpu")
 return {"quant_bits": quant_bits, "accuracy": round(acc, 4),
 "memory_kb": q_kb, "compression_ratio": round(baseline_kb/q_kb, 2)}
if __name__ == "__main__":
 logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
 from src.train import get_dataloaders
 from src.model import build_model
 BASELINE_CONFIG = {"num_layers": 3, "width_mult": 1.0, "quant_bits": 32,
 "time_slice": 20.0, "priority": 1, "dvfs_factor": 1.0}
 logger.info("Loading baseline checkpoint...")
 ckpt = torch.load(CHECKPOINT_DIR / "baseline_model.pth",
map_location="cpu")
 model = build_model(BASELINE_CONFIG)
 model.load_state_dict(ckpt["model_state"])
 model.eval()
 _, test_loader = get_dataloaders()

 results = []
 for bits in [32, 16, 8, 4]:
  r = measure_quantization_impact(model, test_loader, bits)
  results.append(r)
 print("\n=== Quantization Trade-off Experiment ===")
 print("Independent variable: quant_bits ∈ {32, 16, 8, 4}") 
 print("Dependent variables: accuracy, memory_kb, compression_ratio")
 print("Control: same trained model, same test set\n")
 print(f"{'Bits':<8} {'Accuracy':<12} {'Size(KB)':<12} {'Compression'}")
 print("-" * 46)

 for r in results:
  print(
        f"{r['quant_bits']:<8}"
        f"{r['accuracy']:<12.4f}"
        f"{r['memory_kb']:<12.1f}"
        f"{r['compression_ratio']:.2f}×"
  )


#print(f"{bits:<8} {r['accuracy']:<12.4f} {r['memory_kb']:<12.1f}{r['compression_ratio']:.2f}×")
 # Append rows 2-4 to baseline_measurements.csv
 csv_path = RESULTS_DIR / "baseline_measurements.csv"
 fieldnames =["experiment","num_layers","width_mult","quant_bits","time_slice",

"priority","dvfs_factor","accuracy","memory_kb","flops","total_params",
 "energy_mj","latency_ms"]
 with open(csv_path, "a", newline="") as f:
  w = csv.DictWriter(f, fieldnames=fieldnames)
  for r in results[1:]: # Skip quant_bits=32 — already in Day 1 row
   row = {k: "" for k in fieldnames}
   row.update({"experiment": f"quantization_bits_{r['quant_bits']}",
 "num_layers": 3, "width_mult": 1.0,
 "quant_bits": r["quant_bits"], "accuracy":
r["accuracy"],
 "memory_kb": r["memory_kb"]})
   w.writerow(row)
 print(f"\nResults appended → {csv_path}")