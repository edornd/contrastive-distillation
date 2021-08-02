from pathlib import Path
from typing import Callable

from saticl.datasets.base import DatasetBase
from saticl.datasets.isprs import PotsdamDataset, VaihingenDataset

available_datasets = {"potsdam": PotsdamDataset, "vaihingen": VaihingenDataset}


def create_dataset(name: str,
                   path: Path,
                   subset: str,
                   transform: Callable = None,
                   channels: int = 3,
                   **kwargs) -> DatasetBase:
    try:
        dataset = available_datasets[name](path=path, subset=subset, transform=transform, channels=channels, **kwargs)
        return dataset
    except KeyError:
        raise NotImplementedError(f"Dataset '{name}' not implemented")
