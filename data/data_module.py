import os
import random
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import BrainTumorDataset
from rl.encoder import build_encoder

try:
    from tqdm.auto import tqdm
except Exception:  # minimal fallback if tqdm is unavailable
    def tqdm(x, **kwargs):
        return x


class NormalizeSlice:
    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]  # [1,H,W]
        meta = sample.get("meta")
        embedding = sample.get("embedding")
        nz = (image != 0)
        if nz.any():
            mean = image[nz].mean()
            std = image[nz].std().clamp(min=1e-6)
        else:
            mean = image.mean()
            std = image.std().clamp(min=1e-6)
        image = (image - mean) / std
        image = image.clamp_(-6, 6)
        result = {"image": image, "mask": mask}
        if embedding is not None:
            result["embedding"] = embedding
        if meta is not None:
            result["meta"] = meta
        return result


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
        encoder_config: Optional[Dict] = None,
        embedding_batch_size: int = 128,
        embedding_device: Optional[str] = None,
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
        self.encoder_config_dict = dict(encoder_config or {})
        self.embedding_batch_size = max(1, int(embedding_batch_size))
        self.embedding_device = embedding_device

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._cache_prepared = False
        self._groups: List[Dict[str, Any]] = []
        self.embedding_dim: Optional[int] = None
        self.embedding_model_name: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Cache preparation helpers
    # ------------------------------------------------------------------ #

    def _prepare_cache(self) -> None:
        if self._cache_prepared:
            return

        if not self.encoder_config_dict:
            raise RuntimeError(
                "An encoder configuration must be provided to precompute embeddings."
            )

        cache_dir = os.path.join(self.data_dir, "decompressed")
        os.makedirs(cache_dir, exist_ok=True)

        # find all (image, mask) pairs
        pairs: List[tuple[str, str, str, str, str]] = []
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
                    pairs.append((image_path, mask_path, key, patient_dir, timepoint_dir))

        groups: List[Dict[str, any]] = []
        for image_path, mask_path, key, patient_dir, timepoint_dir in tqdm(
            pairs, desc="Preparing memmaps"
        ):
            img_mm_path = os.path.join(cache_dir, f"{key}_image_224.npy")
            msk_mm_path = os.path.join(cache_dir, f"{key}_mask_224.npy")

            need_img = not os.path.exists(img_mm_path)
            need_msk = not os.path.exists(msk_mm_path)

            if need_img or need_msk:
                if need_img:
                    img = nib.load(image_path).get_fdata().astype(np.float32)
                    D = img.shape[2]
                    mm = np.lib.format.open_memmap(
                        img_mm_path, mode="w+", dtype="float32", shape=(224, 224, D)
                    )
                    for i in range(D):
                        t = (
                            torch.from_numpy(img[:, :, i])
                            .float()
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                        r = F.interpolate(
                            t, size=(224, 224), mode="bilinear", align_corners=False
                        )
                        mm[:, :, i] = r.squeeze(0).squeeze(0).numpy()
                    del mm
                    del img
                if need_msk:
                    msk = nib.load(mask_path).get_fdata().astype(np.float32)
                    D = msk.shape[2]
                    mm = np.lib.format.open_memmap(
                        msk_mm_path, mode="w+", dtype="float32", shape=(224, 224, D)
                    )
                    for i in range(D):
                        t = (
                            torch.from_numpy(msk[:, :, i])
                            .float()
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                        r = F.interpolate(t, size=(224, 224), mode="nearest")
                        mm[:, :, i] = r.squeeze(0).squeeze(0).numpy()
                    del mm
                    del msk

            mask_mm = np.load(msk_mm_path, mmap_mode="r")
            slice_indices = []
            for i in range(mask_mm.shape[2]):
                has_tumor = float(mask_mm[:, :, i].sum()) > 0.0
                if has_tumor or self.include_empty_masks:
                    slice_indices.append(i)
            if slice_indices:
                groups.append(
                    {
                        "image_mm": img_mm_path,
                        "mask_mm": msk_mm_path,
                        "slice_indices": slice_indices,
                        "key": key,
                        "patient_id": patient_dir,
                        "timepoint_id": timepoint_dir,
                        "source_image": image_path,
                        "source_mask": mask_path,
                    }
                )

        if not groups:
            raise RuntimeError("No patient/timepoint groups found in dataset.")

        encoder, encoder_config = build_encoder(self.encoder_config_dict)
        encoder.eval()
        device_str = self.embedding_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
        device = torch.device(device_str)
        encoder.to(device)

        embedding_dir = os.path.join(
            self.data_dir, "embeddings", encoder_config.name
        )
        os.makedirs(embedding_dir, exist_ok=True)

        embedding_dim = int(encoder.embedding_dim)
        for group in tqdm(groups, desc="Computing embeddings"):
            img_mm_path = group["image_mm"]
            embedding_mm_path = os.path.join(
                embedding_dir, f"{group['key']}_embeddings.npy"
            )
            group["embedding_mm"] = embedding_mm_path
            if os.path.exists(embedding_mm_path):
                existing = np.load(embedding_mm_path, mmap_mode="r")
                if existing.shape[1] == embedding_dim:
                    del existing
                    continue
                del existing
                os.remove(embedding_mm_path)

            img_vol = np.load(img_mm_path, mmap_mode="r")
            depth = img_vol.shape[2]
            emb_mm = np.lib.format.open_memmap(
                embedding_mm_path,
                mode="w+",
                dtype="float32",
                shape=(depth, embedding_dim),
            )

            for start in range(0, depth, self.embedding_batch_size):
                end = min(depth, start + self.embedding_batch_size)
                batch_np = np.ascontiguousarray(
                    np.moveaxis(img_vol[:, :, start:end], -1, 0)
                )
                batch = torch.from_numpy(batch_np).unsqueeze(1).to(device=device)
                with torch.no_grad():
                    batch_emb = encoder.embed_without_noise(batch)
                emb_mm[start:end, :] = batch_emb.cpu().numpy()
            del emb_mm

        self.embedding_dim = embedding_dim
        self.embedding_model_name = encoder_config.name
        self._groups = groups
        self._cache_prepared = True

        del encoder

    # ------------------------------------------------------------------ #

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit", "validate", "test"):
            return

        self._prepare_cache()

        if self.train_dataset is not None and stage in (None, "fit"):
            return
        if self.val_dataset is not None and stage in ("validate",):
            return
        if self.test_dataset is not None and stage in ("test",):
            return

        rng = random.Random(self.seed)
        groups = list(self._groups)
        rng.shuffle(groups)
        n_total = len(groups)
        n_val = int(self.val_split * n_total)
        n_test = int(self.test_split * n_total)
        val_groups = groups[:n_val]
        test_groups = groups[n_val : n_val + n_test]
        train_groups = groups[n_val + n_test :]

        def expand(group_records: List[Dict[str, Any]]):
            samples = []
            for record in group_records:
                for idx in record["slice_indices"]:
                    meta = {
                        "group_key": record["key"],
                        "patient_id": record["patient_id"],
                        "timepoint_id": record["timepoint_id"],
                        "slice_index": int(idx),
                        "source_image": record["source_image"],
                        "source_mask": record["source_mask"],
                        "image_mm_path": record["image_mm"],
                        "mask_mm_path": record["mask_mm"],
                        "embedding_mm_path": record["embedding_mm"],
                    }
                    samples.append(
                        {
                            "image_path": record["image_mm"],
                            "mask_path": record["mask_mm"],
                            "embedding_path": record["embedding_mm"],
                            "slice_index": int(idx),
                            "meta": meta,
                        }
                    )
            return samples

        train_samples = expand(train_groups)
        val_samples = expand(val_groups)
        test_samples = expand(test_groups)

        dataset_kwargs = dict(transform=self.transform, resize_shape=None)
        self.train_dataset = BrainTumorDataset.from_samples(train_samples, **dataset_kwargs)
        self.val_dataset = BrainTumorDataset.from_samples(val_samples, **dataset_kwargs)
        self.test_dataset = BrainTumorDataset.from_samples(test_samples, **dataset_kwargs)

        if not val_samples:
            self.val_dataset = self.train_dataset
        if not test_samples:
            self.test_dataset = self.val_dataset

    def _dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Dataset has not been set up. Call `.setup()` before requesting dataloaders.")

        persistent = self.persistent_workers and self.num_workers > 0

        loader_kwargs = dict(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=persistent,
            pin_memory=self.pin_memory,
        )

        if self.num_workers > 0 and self.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(**loader_kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset, shuffle=False)
