import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from data.dataset import BrainTumorDataset
import os

class BrainTumorDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int = 32,
            num_workers: int = 4,
            persistent_workers: bool = True
        ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.transform = None # Add your transforms here

    def setup(self, stage: str = None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            full_dataset = BrainTumorDataset(self.data_dir, transform=self.transform)
            dataset_size = len(full_dataset)
            train_size = int(dataset_size * 0.7)
            val_size = dataset_size - train_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = BrainTumorDataset(self.data_dir, transform=self.transform) # Assumes test data is the same as train/val for this example

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )
