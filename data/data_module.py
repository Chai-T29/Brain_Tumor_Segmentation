import os
import random
import torch
from torch.utils.data import DataLoader, random_split
from typing import Optional
import pytorch_lightning as pl
from data.dataset import BrainTumorDataset

# New imports for one-time decompression + progress bar
import numpy as np
import nibabel as nib
try:
    from tqdm.auto import tqdm
except Exception:  # minimal fallback if tqdm is unavailable
    def tqdm(x, **kwargs):
        return x


class NormalizeSlice:
    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]  # [1,H,W]
        nz = (image != 0)
        if nz.any():
            mean = image[nz].mean()
            std = image[nz].std().clamp(min=1e-6)
        else:
            mean = image.mean()
            std = image.std().clamp(min=1e-6)
        image = (image - mean) / std
        image = image.clamp_(-3, 3)
        return {"image": image, "mask": mask}


class BrainTumorDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module that serves as the single entry point for data.

    Adds a one-time decompression phase in `setup` that converts `.nii.gz` volumes
    into NumPy memory-mapped arrays (.npy). Subsequent indexing uses memory maps
    directly for fast random access without re-reading and decompressing gz files.
    """

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
        self.transform = NormalizeSlice()
        self.include_empty_masks = bool(include_empty_masks)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit", "validate", "test"):
            return

        # ----------------------------
        # Step 1: one-time decompression into memory-mapped arrays
        # ----------------------------
        cache_dir = os.path.join(self.data_dir, ".cache_mm")
        os.makedirs(cache_dir, exist_ok=True)

        # find all (image, mask) pairs
        pairs = []  # (image_path, mask_path, key)
        for patient_dir in sorted(os.listdir(self.data_dir)):
            patient_path = os.path.join(self.data_dir, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            for timepoint_dir in sorted(os.listdir(patient_path)):
                timepoint_path = os.path.join(patient_path, timepoint_dir)
                if not os.path.isdir(timepoint_path):
                    continue

                image_path, mask_path = None, None
                for f in os.listdir(timepoint_path):
                    if f.endswith("_brain_t1c.nii.gz"):
                        image_path = os.path.join(timepoint_path, f)
                    elif f.endswith("_tumorMask.nii.gz"):
                        mask_path = os.path.join(timepoint_path, f)

                if image_path and mask_path:
                    key = f"{patient_dir}__{timepoint_dir}"
                    pairs.append((image_path, mask_path, key))

        # create / reuse memmaps with a progress bar
        groups = []  # each entry: (image_mm_path, mask_mm_path, [slice_idxs])
        for image_path, mask_path, key in tqdm(pairs, desc="Preparing memmaps"):
            img_mm_path = os.path.join(cache_dir, f"{key}_image.npy")
            msk_mm_path = os.path.join(cache_dir, f"{key}_mask.npy")

            need_img = not os.path.exists(img_mm_path)
            need_msk = not os.path.exists(msk_mm_path)

            if need_img or need_msk:
                # load volumes only if needed, then write to .npy memmap
                if need_img:
                    img = nib.load(image_path).get_fdata().astype(np.float32)
                    mm = np.lib.format.open_memmap(
                        img_mm_path, mode="w+", dtype="float32", shape=img.shape
                    )
                    mm[...] = img
                    del mm
                    del img
                if need_msk:
                    msk = nib.load(mask_path).get_fdata().astype(np.float32)
                    mm = np.lib.format.open_memmap(
                        msk_mm_path, mode="w+", dtype="float32", shape=msk.shape
                    )
                    mm[...] = msk
                    del mm
                    del msk

            # determine slice indices using the memmapped mask
            mask_mm = np.load(msk_mm_path, mmap_mode="r")
            slice_indices = []
            for i in range(mask_mm.shape[2]):
                has_tumor = float(mask_mm[:, :, i].sum()) > 0.0
                if has_tumor or self.include_empty_masks:
                    slice_indices.append(i)
            if slice_indices:
                groups.append((img_mm_path, msk_mm_path, slice_indices))

        if not groups:
            raise RuntimeError("No patient/timepoint groups found in dataset.")

        # ----------------------------
        # Step 2: split groups (not slices)
        # ----------------------------
        rng = random.Random(self.seed)
        rng.shuffle(groups)
        n_total = len(groups)
        n_val = int(self.val_split * n_total)
        n_test = int(self.test_split * n_total)
        val_groups = groups[:n_val]
        test_groups = groups[n_val : n_val + n_test]
        train_groups = groups[n_val + n_test :]

        def expand(groups_list):
            samples = []
            for img_mm, msk_mm, idxs in groups_list:
                samples.extend([(img_mm, msk_mm, i) for i in idxs])
            return samples

        train_samples = expand(train_groups)
        val_samples = expand(val_groups)
        test_samples = expand(test_groups)

        # ----------------------------
        # Step 3: build datasets (will read slices from memmaps)
        # ----------------------------
        self.train_dataset = BrainTumorDataset.from_samples(
            train_samples, transform=self.transform, resize_shape=(224, 224)
        )
        self.val_dataset = BrainTumorDataset.from_samples(
            val_samples, transform=self.transform, resize_shape=(224, 224)
        )
        self.test_dataset = BrainTumorDataset.from_samples(
            test_samples, transform=self.transform, resize_shape=(224, 224)
        )

        # Fallbacks if splits are empty
        if not val_samples:
            self.val_dataset = self.train_dataset
        if not test_samples:
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
