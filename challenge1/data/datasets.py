from __future__ import annotations

from bisect import bisect_right
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset


class LabeledWindowDataset(Dataset):
    def __init__(self, base_dataset: Dataset, label: int):
        self.base_dataset = base_dataset
        self.label = label

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        X = self.base_dataset[index][0]
        return X, torch.tensor(self.label, dtype=torch.long)


class MultiTaskConcatDataset(Dataset):
    def __init__(self, datasets: Iterable[Dataset]):
        self.datasets = [dataset for dataset in datasets if len(dataset) > 0]
        self.cumulative_sizes = []
        running = 0
        for dataset in self.datasets:
            running += len(dataset)
            self.cumulative_sizes.append(running)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, index: int):
        dataset_idx = bisect_right(self.cumulative_sizes, index)
        prev_size = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        sample_idx = index - prev_size
        return self.datasets[dataset_idx][sample_idx]


def make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
