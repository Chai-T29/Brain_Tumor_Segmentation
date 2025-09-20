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
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.val_split = max(0.0, float(val_split))
        self.test_split = max(0.0, float(test_split))
        self.seed = seed
        self.transform = None  # Add your transforms here if needed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit", "validate", "test"):
            return

        dataset = BrainTumorDataset(self.data_dir, transform=self.transform)
        total_samples = len(dataset)
        if total_samples == 0:
            raise RuntimeError("BrainTumorDataset is empty. Please verify the dataset path and contents.")

        base_train = 1
        base_val = 1 if self.val_split > 0 else 0
        base_test = 1 if self.test_split > 0 else 0
        remaining = total_samples - (base_train + base_val + base_test)

        val_ratio = self.val_split
        test_ratio = self.test_split
        train_ratio = max(0.0, 1.0 - val_ratio - test_ratio)
        ratio_sum = train_ratio + val_ratio + test_ratio
        if ratio_sum <= 0:
            ratio_sum = 1.0
        train_ratio /= ratio_sum
        val_ratio /= ratio_sum
        test_ratio /= ratio_sum

        extra_train = int(round(remaining * train_ratio))
        extra_val = int(round(remaining * val_ratio))
        extra_test = remaining - extra_train - extra_val

        train_size = base_train + max(0, extra_train)
        val_size = base_val + max(0, extra_val)
        test_size = base_test + max(0, extra_test)

        while train_size + val_size + test_size < total_samples:
            train_size += 1

        while train_size + val_size + test_size > total_samples:
            if train_size > base_train:
                train_size -= 1
            elif val_size > base_val:
                val_size -= 1
            elif test_size > base_test:
                test_size -= 1
            else:
                break

        if test_size < 1 and self.test_split > 0:
            test_size = 1
            if train_size > base_train:
                train_size -= 1
            elif val_size > base_val:
                val_size = max(0, val_size - 1)

        if val_size < 1 and self.val_split > 0:
            val_size = 1
            if train_size > base_train:
                train_size -= 1
            elif test_size > base_test:
                test_size = max(0, test_size - 1)

        total_assigned = train_size + val_size + test_size
        if total_assigned != total_samples:
            diff = total_samples - total_assigned
            train_size = max(1, train_size + diff)

        if test_size < 0:
            test_size = 0

        remaining_for_test = total_samples - train_size - val_size
        if remaining_for_test < 0:
            train_size = max(1, train_size + remaining_for_test)
            remaining_for_test = total_samples - train_size - val_size
        test_size = max(0, remaining_for_test)

        generator = torch.Generator().manual_seed(self.seed)
        if val_size > 0 and test_size > 0:
            remaining = total_samples - train_size - val_size
            if remaining < 1:
                adjust = 1 - remaining
                if train_size > base_train:
                    train_size = max(base_train, train_size - adjust)
                val_size = min(val_size, total_samples - train_size - 1)
                remaining = max(1, total_samples - train_size - val_size)
            splits = [train_size, val_size, remaining]
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset, splits, generator=generator
            )
        elif val_size > 0:
            remaining = max(1, total_samples - train_size)
            splits = [train_size, remaining]
            self.train_dataset, self.val_dataset = random_split(dataset, splits, generator=generator)
            self.test_dataset = self.val_dataset
        elif test_size > 0:
            remaining = max(1, total_samples - train_size)
            splits = [train_size, remaining]
            self.train_dataset, self.test_dataset = random_split(dataset, splits, generator=generator)
            self.val_dataset = self.test_dataset
        else:
            self.train_dataset = dataset
            self.val_dataset = dataset
            self.test_dataset = dataset

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
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset, shuffle=False)
