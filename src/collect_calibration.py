"""
Collect additional accuracy calibration points across the num_layers x
width_mult architecture space (not just quant_bits), to calibrate the
IDW accuracy proxy used in objective.py.

The original CALIBRATION_POINTS only varied quant_bits at a single fixed
architecture (num_layers=3, width_mult=1.0), so the proxy had no real
data on how architecture size affects accuracy — it returned ~constant
accuracy regardless of num_layers/width_mult, which let BO shrink to the
smallest architecture for free.

This script trains NEW_ARCHS at full NUM_EPOCHS (checkpoints are NOT
saved — only accuracy is kept), then runs the quant_bits={32,16,8,4}
sweep on each, giving real cross-terms between architecture size and
quantization instead of assuming they're independent.

Run: python -m src.collect_calibration
Output:
  - Appends rows to results/baseline_measurements.csv
  - Prints a CALIBRATION_POINTS-formatted snippet to paste into objective.py
"""
import os, sys, csv, copy, logging, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import RESULTS_DIR, LOGS_DIR, LOG_FORMAT, NUM_EPOCHS, DEVICE
from src.train import get_dataloaders, train_model, evaluate_accuracy
from src.model import build_model, get_model_stats
from src.quantization import apply_quantization

logging.basicConfig(
    level=logging.INFO, format=LOG_FORMAT,
    force=True,  # src.train's import already calls basicConfig() first
    handlers=[
        logging.FileHandler(LOGS_DIR / "collect_calibration.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Corners + center of the (num_layers, width_mult) space.
# (3, 1.0) is the existing baseline (checkpoints/baseline_model.pth) and
# is NOT retrained here — its accuracy is already in CALIBRATION_POINTS.
NEW_ARCHS = [
    {"num_layers": 2, "width_mult": 0.25},
    {"num_layers": 2, "width_mult": 1.00},
    {"num_layers": 4, "width_mult": 0.25},
    {"num_layers": 4, "width_mult": 1.00},
    {"num_layers": 3, "width_mult": 0.625},
]

QUANT_BITS = [32, 16, 8, 4]

FIELDNAMES = ["experiment", "num_layers", "width_mult", "quant_bits",
              "time_slice", "priority", "dvfs_factor", "accuracy",
              "memory_kb", "flops", "total_params", "energy_mj", "latency_ms"]


def collect():
    train_loader, test_loader = get_dataloaders()
    csv_path = RESULTS_DIR / "baseline_measurements.csv"

    all_points = []  # (num_layers, width_mult, quant_bits, accuracy)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for arch in NEW_ARCHS:
            t0 = time.perf_counter()
            logger.info("=" * 60)
            logger.info(f"Training architecture: {arch}")
            logger.info("=" * 60)
            model, best_acc = train_model(
                arch, train_loader, test_loader,
                save_path=None, device=DEVICE, num_epochs=NUM_EPOCHS,
            )
            model = model.to("cpu").eval()
            stats = get_model_stats(model)
            elapsed = time.perf_counter() - t0
            logger.info(f"Trained {arch} in {elapsed/60:.1f}min | fp32 acc={best_acc:.4f}")

            for bits in QUANT_BITS:
                if bits == 32:
                    acc = best_acc
                    mem_kb = stats["model_size_kb"]
                else:
                    q_model, mem_kb = apply_quantization(copy.deepcopy(model), bits)
                    acc = evaluate_accuracy(q_model, test_loader, device="cpu")
                acc = round(acc, 4)
                all_points.append((arch["num_layers"], arch["width_mult"], bits, acc))
                logger.info(f"  bits={bits:<3} acc={acc:.4f} mem={mem_kb:.1f}KB")

                row = {k: "" for k in FIELDNAMES}
                row.update({
                    "experiment": f"calib_L{arch['num_layers']}_W{arch['width_mult']}_B{bits}",
                    "num_layers": arch["num_layers"],
                    "width_mult": arch["width_mult"],
                    "quant_bits": bits,
                    "accuracy": acc,
                    "memory_kb": mem_kb,
                    "flops": stats["flops"] if bits == 32 else "",
                    "total_params": stats["total_params"] if bits == 32 else "",
                })
                writer.writerow(row)
                f.flush()

    print("\n" + "=" * 70)
    print("CALIBRATION_POINTS snippet — paste into src/objective.py")
    print("=" * 70)
    for nl, wm, qb, acc in all_points:
        print(f" ({nl}, {wm}, {qb}, {acc}),")
    print("=" * 70)
    logger.info(f"Calibration collection complete. Results appended to {csv_path}")
    return all_points


if __name__ == "__main__":
    collect()
