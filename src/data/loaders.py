from torch.utils.data import DataLoader
from pathlib import Path

from .dataset import EENDSSDataset


def get_dataloaders(
    train_manifest: str | Path,
    val_manifest: str | Path,
    test_manifest: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    limit: int | None = None,
    **dataset_kwargs          # forwarded to EENDSSDataset (sample_rate, max_samples, etc.)
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_kwargs.pop("limit", None)
    
    splits = [
        {"manifest": train_manifest, "shuffle": True,   "drop_last": True},
        {"manifest": val_manifest,   "shuffle": False,  "drop_last": False},
        {"manifest": test_manifest,  "shuffle": False,  "drop_last": False},
    ]
    
    loaders = []
    
    for split in splits:
        dataset = EENDSSDataset(
            manifest_path=split["manifest"],
            limit=limit,
            **dataset_kwargs
        )
        
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split["shuffle"],
            num_workers=num_workers,
            drop_last=split["drop_last"]
        )
        
        loaders.append(loader)
    
    return tuple(loaders)