"""
Configurable CNN for CIFAR-10.
Parameterized by num_layers {2,3,4} and width_mult [0.25, 1.0]. ∈ ∈
These are the architecture variables Bayesian Optimization searches over.
"""
import os, sys, logging
import torch
import torch.nn as nn
from torchinfo import summary
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import NUM_CLASSES, IMAGE_SIZE, RANDOM_SEED
logger = logging.getLogger(__name__)
BASE_CHANNELS = [32, 64, 128, 256] # Per-layer channels before width_multscaling



class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → MaxPool2d"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
    def forward(self, x):
        return self.block(x)
class ConfigurableCNN(nn.Module):
 """
 CNN whose depth and width are controlled by config parameters.
 Spatial flow (CIFAR-10 = 32×32):
 2 layers: 32→16→8 → AdaptiveAvgPool(1)
 3 layers: 32→16→8→4 → AdaptiveAvgPool(1)
 4 layers: 32→16→8→4→2 → AdaptiveAvgPool(1)
 AdaptiveAvgPool makes the classifier shape-agnostic — no manual
 calculation needed when depth changes.
 """
 def __init__(self, num_layers: int = 3, width_mult: float = 1.0,
 num_classes: int = NUM_CLASSES):
    super().__init__()
    if num_layers not in {2, 3, 4}:
        raise ValueError(f"num_layers must be 2, 3, or 4. Got:{num_layers}")
    if not (0.25 <= width_mult <= 1.0):
        raise ValueError(f"width_mult must be in [0.25, 1.0]. Got:{width_mult}")
    channels = [max(4, int(c * width_mult)) for c in BASE_CHANNELS]
    blocks, in_ch = [], 3
    for i in range(num_layers):
        out_ch = channels[i]
        blocks.append(ConvBlock(in_ch, out_ch))
        in_ch = out_ch
    self.features = nn.Sequential(*blocks)
    self.pool = nn.AdaptiveAvgPool2d((1, 1))
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_ch, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
 def forward(self, x):
    return self.classifier(self.pool(self.features(x)))


def build_model(config: dict) -> ConfigurableCNN:
    """
    Factory function — builds model from the config dict used throughout the
    project.
    Called by train.py, objective.py, and the BO loop.
    """
    torch.manual_seed(RANDOM_SEED)
    model = ConfigurableCNN(
    num_layers=int(config["num_layers"]),
    width_mult=float(config["width_mult"]),
 )
    logger.info(f"Built model: layers={config['num_layers']},width={config['width_mult']:.2f}")
    return model
def get_model_stats(model: ConfigurableCNN) -> dict:
 """
 Returns stats used in paper tables:
 total_params, model_size_kb, flops (MACs)
 model_size_kb = params × 4 bytes / 1024 (float32 baseline)
 """
 stats = summary(model, input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE), verbose=0)
 total_params = stats.total_params
 model_size_kb = round((total_params * 4) / 1024.0, 2)
 flops = stats.total_mult_adds
 result = {"total_params": total_params, "model_size_kb": model_size_kb,
"flops": flops}
 logger.info(f"Model stats: {result}")
 return result
if __name__ == "__main__":
 logging.basicConfig(level=logging.INFO)
 test_configs = [
    {"num_layers": 2, "width_mult": 0.25},
    {"num_layers": 3, "width_mult": 0.50},
    {"num_layers": 3, "width_mult": 1.00},
    {"num_layers": 4, "width_mult": 0.75},
 ]
 print("\n=== Model Smoke Test ===\n")
 for cfg in test_configs:
    m = build_model(cfg)
    s = get_model_stats(m)
    x = torch.randn(1, 3, 32, 32)
    out = m(x)
    assert out.shape == (1, 10)
    print(f"layers={cfg['num_layers']}, width={cfg['width_mult']:.2f} → "
        f"params={s['total_params']:,}, size={s['model_size_kb']:.1f}KB, "
        f"FLOPs={s['flops']:,}, output={tuple(out.shape)} ✓") 
    print("\nAll configurations: VERIFIED")