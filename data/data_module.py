from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from data.dataset import BrainTumorDataset


class BrainTumorDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module that serves as the single entry point for data."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        prefetch_factor: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        include_empty_masks: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.val_split = max(0.0, float(val_split))
        self.test_split = max(0.0, float(test_split))
        self.seed = seed
        self.transform = None  # Add your transforms here if needed
        self.include_empty_masks = bool(include_empty_masks)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit", "validate", "test"):
            return

        dataset = BrainTumorDataset(
            self.data_dir,
            transform=self.transform,
            include_empty_masks=self.include_empty_masks,
        )
        total_samples = len(dataset)
        if total_samples == 0:
            raise RuntimeError("BrainTumorDataset is empty. Please verify the dataset path and contents.")

        val_ratio = max(0.0, float(self.val_split))
        test_ratio = max(0.0, float(self.test_split))
        train_ratio = max(0.0, 1.0 - val_ratio - test_ratio)

        ratio_sum = train_ratio + val_ratio + test_ratio
        if ratio_sum == 0:
            train_ratio = 1.0
            ratio_sum = 1.0

        train_ratio /= ratio_sum
        val_ratio /= ratio_sum
        test_ratio /= ratio_sum

        ratios = [train_ratio, val_ratio, test_ratio]
        lengths = [int(r * total_samples) for r in ratios]
        remainder = total_samples - sum(lengths)

        if remainder > 0:
            order = sorted(range(len(ratios)), key=lambda idx: ratios[idx], reverse=True)
            for idx in order:
                if remainder == 0:
                    break
                lengths[idx] += 1
                remainder -= 1

        if lengths[0] == 0 and total_samples > 0:
            largest_idx = max(range(len(lengths)), key=lambda idx: lengths[idx])
            if lengths[largest_idx] > 0:
                lengths[largest_idx] -= 1
                lengths[0] += 1

        generator = torch.Generator().manual_seed(self.seed)
        subsets = list(random_split(dataset, lengths, generator=generator))

        self.train_dataset, self.val_dataset, self.test_dataset = subsets
        if lengths[1] == 0:
            self.val_dataset = self.train_dataset
        if lengths[2] == 0:
            self.test_dataset = self.val_dataset

    def _dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Dataset has not been set up. Call `.setup()` before requesting dataloaders.")

        persistent = self.persistent_workers and self.num_workers > 0
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=persistent,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset, shuffle=False)
