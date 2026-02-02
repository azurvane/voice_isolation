import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.dataset_generator import generate_full_dataset

# Generate small test dataset
data_dir = project_root / "data" / "raw"
output_dir = project_root / "data" / "processed"

generate_full_dataset(
    data_dir=data_dir,
    output_dir=output_dir,
    train_size=10,    # Small for testing
    val_size=2,
    test_size=2
)

print("\n✅ Test dataset created!")
print("Check data/processed/ for train/val/test folders")
