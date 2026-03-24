import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import get_dataloaders
import torch

train_loader, val_loader, test_loader = get_dataloaders(
    train_manifest="data/processed/test/test_manifest.json",
    val_manifest="data/processed/train/train_manifest.json",
    test_manifest="data/processed/val/val_manifest.json",
    batch_size=4,
    limit=20       # only load 20 samples for speed
)

batch = next(iter(train_loader))
print("test code print")
assert batch["mixture"].shape == (4, 480000)
assert batch["source"].shape  == (4, 3, 480000)
assert batch["labels"].shape  == (4, 3000, 3)
assert batch["mask"].shape    == (4, 480000)
assert batch["mask"].dtype    == torch.bool