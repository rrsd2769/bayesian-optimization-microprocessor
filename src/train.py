"""
Training pipeline for ConfigurableCNN on CIFAR-10.
Provides: get_dataloaders(), train_model(), evaluate_accuracy()
"""
import os, sys, time, logging, csv
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
 RANDOM_SEED, DEVICE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
 NUM_EPOCHS, CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR, DATA_DIR,
 NUM_WORKERS, PIN_MEMORY, LOG_FORMAT
)
from src.model import build_model, get_model_stats
logging.basicConfig(
 level=logging.INFO, format=LOG_FORMAT,
 handlers=[
 logging.FileHandler(LOGS_DIR / "training.log"),
 logging.StreamHandler(sys.stdout),
 ]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int = RANDOM_SEED) -> None:
 import random, numpy as np
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
def get_dataloaders(batch_size: int = BATCH_SIZE) -> Tuple[DataLoader,
DataLoader]:
 """
 Returns CIFAR-10 train/test DataLoaders.
 Normalization: CIFAR-10 channel means/stds (NOT ImageNet values).
 These exact values must be used for ALL experiments.
 """
 MEAN = (0.4914, 0.4822, 0.4465)
 STD = (0.2023, 0.1994, 0.2010)
 train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
 T.ToTensor(), T.Normalize(MEAN, STD)])
 test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
 train_ds = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True,download=True, transform=train_tf)
 test_ds = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False,download=True, transform=test_tf)
 train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
 test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
 logger.info(f"DataLoaders ready — train: {len(train_ds)}, test:{len(test_ds)}")
 return train_loader, test_loader
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str =DEVICE) -> float:
 model.eval(); correct = total = 0
 with torch.no_grad():
  for images, labels in loader:
   images, labels = images.to(device), labels.to(device)
   _, predicted = model(images).max(1)
   correct += predicted.eq(labels).sum().item()
   total += labels.size(0)
 return correct / total

def train_model(config: dict, train_loader: DataLoader, test_loader: DataLoader,
 save_path: Path = None, device: str = DEVICE,
 num_epochs: int = NUM_EPOCHS) -> Tuple[nn.Module, float]:
 """Full training loop with CosineAnnealingLR and best-checkpoint saving."""
 set_seed()
 print("DEVICE =", device)
 model = build_model(config).to(device)
 #stats = get_model_stats(model)
 stats = {"total_params": 0, "model_size_kb": 0, "flops": 0}
 criterion = nn.CrossEntropyLoss()
 optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
weight_decay=WEIGHT_DECAY)
 # CosineAnnealingLR: smooth convergence, better final acc than StepLR
 scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer,
T_max=num_epochs)
 logger.info(f"Training | config={config} | stats={stats} | device={device}")
 best_acc = 0.0
 for epoch in range(1, num_epochs + 1):
  model.train()
  running_loss = correct = total = 0
  t0 = time.time()
  for images, labels in train_loader:
   images, labels = images.to(device), labels.to(device)
   optimizer.zero_grad()
   loss = criterion(model(images), labels)
   loss.backward(); optimizer.step()
   running_loss += loss.item() * images.size(0)
   _, pred = model(images).max(1)
   correct += pred.eq(labels).sum().item()
   total += labels.size(0)
  scheduler_lr.step()
  train_acc = correct / total
  test_acc = evaluate_accuracy(model, test_loader, device)
  logger.info(f"Epoch [{epoch:02d}/{num_epochs}] "
   f"loss={running_loss/total:.4f} | train={train_acc:.4f} | "
   f"test={test_acc:.4f} | {time.time()-t0:.1f}s")
  if test_acc > best_acc:
   best_acc = test_acc
   if save_path:
    torch.save({"epoch": epoch, "config": config,
 "model_state": model.state_dict(),
 "optimizer_state": optimizer.state_dict(),
 "best_acc": best_acc}, save_path)
   logger.info(f"Checkpoint → {save_path} (acc={best_acc:.4f})")
 logger.info(f"Training complete. Best test accuracy: {best_acc:.4f}")
 return model, best_acc
if __name__ == "__main__":
 BASELINE_CONFIG = {"num_layers": 3, "width_mult": 1.0, "quant_bits": 32,
 "time_slice": 20.0, "priority": 1, "dvfs_factor": 1.0}
 logger.info("=" * 60)
 logger.info("BASELINE TRAINING RUN")
 logger.info(f"Config: {BASELINE_CONFIG}")
 logger.info("=" * 60)
 train_loader, test_loader = get_dataloaders()
 save_path = CHECKPOINT_DIR / "baseline_model.pth"
 model, acc = train_model(BASELINE_CONFIG, train_loader, test_loader,
save_path)
 stats = get_model_stats(model)
 # First row of results/baseline_measurements.csv
 row = {"experiment": "baseline_no_optimization",
 "num_layers": 3, "width_mult": 1.0, "quant_bits": 32,
 "time_slice": 20.0, "priority": 1, "dvfs_factor": 1.0,
 "accuracy": round(acc, 4), "memory_kb": stats["model_size_kb"],
 "flops": stats["flops"], "total_params": stats["total_params"],
 "energy_mj": None, "latency_ms": None}
 csv_path = RESULTS_DIR / "baseline_measurements.csv"
 fieldnames = list(row.keys())
 with open(csv_path, "w", newline="") as f:
  w = csv.DictWriter(f, fieldnames=fieldnames)
  w.writeheader(); w.writerow(row)
 logger.info(f"Baseline measurements → {csv_path}")
 logger.info(f"Final: accuracy={acc:.4f}, size={stats['model_size_kb']}KB,FLOPs={stats['flops']:,}")