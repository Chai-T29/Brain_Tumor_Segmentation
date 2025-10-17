import math
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from rl.environment import EnvironmentConfig

class BrainTumorDataset(Dataset):
    """Brain Tumor Segmentation Dataset.

    Supports two modes:
    1) Legacy: samples contain paths to `.nii.gz` volumes; slices are read via nibabel.
    2) Memmap: samples contain paths to `.npy` memory-mapped arrays; slices are read via np.load(..., mmap_mode='r').
    """

    def __init__(
        self,
        data_dir,
        transform=None,
        include_empty_masks: bool = False,
        resize_shape: Tuple[int, int] | None = (224, 224),
        environment_config: Union[EnvironmentConfig, Dict[str, Any], None] = None,
    ):
        """
        Args:
            data_dir (string): Directory with all the patient folders.
            transform (callable, optional): Optional transform to be applied on a sample.
            include_empty_masks (bool): Whether to keep slices where the mask sums to zero.
            resize_shape (tuple): (H, W) to resize slices to; set None to disable.
            environment_config: Optional environment configuration used to precompute
                ground-truth polygon parameters for guided exploration.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.include_empty_masks = include_empty_masks
        self.resize_shape = resize_shape
        self.samples = []
        self._mm_cache = {}  # lazy-open cache for memmaps
        self._env_config = self._resolve_env_config(environment_config)
        self._target_cache: Optional[List[np.ndarray]] = None

        if data_dir is None:
            # Constructed via `from_samples`.
            if self._env_config is not None:
                self._precompute_ground_truth()
            return

        # Legacy enumeration over NIfTI files (kept for backward compatibility)
        for patient_dir in sorted(os.listdir(data_dir)):
            patient_path = os.path.join(data_dir, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            for timepoint_dir in sorted(os.listdir(patient_path)):
                timepoint_path = os.path.join(patient_path, timepoint_dir)
                if not os.path.isdir(timepoint_path):
                    continue

                image_path = None
                mask_path = None
                for f in os.listdir(timepoint_path):
                    if f.endswith('_brain_t1c.nii.gz'):
                        image_path = os.path.join(timepoint_path, f)
                    elif f.endswith('_tumorMask.nii.gz'):
                        mask_path = os.path.join(timepoint_path, f)

                if image_path and mask_path:
                    mask_nii = nib.load(mask_path)
                    mask_data = mask_nii.get_fdata()
                    for i in range(mask_data.shape[2]):
                        has_tumor = np.sum(mask_data[:, :, i]) > 0
                        if has_tumor or self.include_empty_masks:
                            self.samples.append((image_path, mask_path, i))

        if self._env_config is not None:
            self._precompute_ground_truth()

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _resolve_env_config(
        environment_config: Union[EnvironmentConfig, Dict[str, Any], None]
    ) -> Optional[EnvironmentConfig]:
        if environment_config is None:
            return None
        if isinstance(environment_config, EnvironmentConfig):
            return EnvironmentConfig(**asdict(environment_config))
        if isinstance(environment_config, dict):
            return EnvironmentConfig(**environment_config)
        raise TypeError("environment_config must be a dict, EnvironmentConfig, or None.")

    def _get_memmap(self, path: str):
        arr = self._mm_cache.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode='r')
            self._mm_cache[path] = arr
        return arr

    def _resolve_entry(self, entry: Any) -> Tuple[str, str, int, Optional[str], Dict[str, Any]]:
        if isinstance(entry, dict):
            image_path = entry.get("image_path")
            mask_path = entry.get("mask_path")
            embedding_path = entry.get("embedding_path")
            slice_idx = int(entry.get("slice_index", 0))
            meta = dict(entry.get("meta", {}))
        else:
            embedding_path = None
            if len(entry) == 3:
                image_path, mask_path, slice_idx = entry
                meta = {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "slice_index": int(slice_idx),
                }
            elif len(entry) == 4:
                image_path, mask_path, slice_idx, embedding_path = entry
                meta = {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "slice_index": int(slice_idx),
                    "embedding_path": embedding_path,
                }
            else:
                image_path, mask_path, slice_idx, embedding_path, meta = entry
                if not isinstance(meta, dict):
                    meta = {
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "slice_index": int(slice_idx),
                    }
                else:
                    meta = dict(meta)

        if image_path is None or mask_path is None:
            raise ValueError("Sample entry is missing image or mask path.")
        return image_path, mask_path, int(slice_idx), embedding_path, meta

    def _load_volume_slice(self, path: str, slice_idx: int) -> np.ndarray:
        if path.endswith('.npy'):
            vol = self._get_memmap(path)
            slice_arr = vol[:, :, slice_idx]
        else:
            nii = nib.load(path)
            data = nii.get_fdata()
            slice_arr = data[:, :, slice_idx]
        return np.asarray(slice_arr, dtype=np.float32)

    def _precompute_ground_truth(self) -> None:
        if self.samples is None or self._env_config is None:
            self._target_cache = None
            return

        targets: List[np.ndarray] = []
        for entry in self.samples:
            _, mask_path, slice_idx, _, _ = self._resolve_entry(entry)
            mask_slice = self._load_volume_slice(mask_path, slice_idx)
            if self.resize_shape is not None:
                mask_tensor = torch.from_numpy(mask_slice).float().unsqueeze(0).unsqueeze(0)
                mask_tensor = F.interpolate(mask_tensor, size=self.resize_shape, mode="nearest")
                mask_np = mask_tensor.squeeze(0).squeeze(0).numpy()
            else:
                mask_np = mask_slice
            target = self._compute_ground_truth_polygon(mask_np.astype(np.float32))
            targets.append(target)
        self._target_cache = targets

    def _compute_ground_truth_polygon(self, mask_slice: np.ndarray) -> np.ndarray:
        env_cfg = self._env_config
        if env_cfg is None:
            raise RuntimeError("Environment configuration is required for ground-truth computation.")

        num_lines = int(env_cfg.num_sides)
        if num_lines <= 0:
            raise ValueError("Environment configuration must define a positive number of sides.")

        mask_binary = mask_slice > 0.5
        if not np.any(mask_binary):
            distances = np.full(num_lines, env_cfg.line_min_distance, dtype=np.float32)
            offsets = np.zeros(num_lines, dtype=np.float32)
            return np.concatenate([distances, offsets]).astype(np.float32)

        coords_y, coords_x = np.nonzero(mask_binary)
        points = np.stack([coords_x.astype(np.float32), coords_y.astype(np.float32)], axis=1)

        hull = self._convex_hull(points)
        if hull.shape[0] < 3:
            hull = points

        height, width = mask_slice.shape
        center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32)
        relative_points = hull - center

        max_distance = min(width, height) / 2.0 - float(env_cfg.line_max_distance_margin)
        max_distance = max(max_distance, float(env_cfg.line_min_distance) + 1.0)

        base_angles = np.linspace(0.0, 2.0 * math.pi, num_lines, endpoint=False).astype(np.float32)
        max_offset_rad = math.radians(float(env_cfg.line_max_angle_offset_deg))

        hull_normals = self._compute_hull_normals(hull)
        hull_angles = np.arctan2(hull_normals[:, 1], hull_normals[:, 0]) if hull_normals.size else np.array([], dtype=np.float32)

        distances = np.zeros(num_lines, dtype=np.float32)
        offsets_deg = np.zeros(num_lines, dtype=np.float32)

        for idx, base_angle in enumerate(base_angles):
            theta = base_angle
            if hull_angles.size > 0:
                diffs = self._wrap_angle_rad(hull_angles - base_angle)
                best_idx = int(np.argmin(np.abs(diffs)))
                theta = base_angle + float(np.clip(diffs[best_idx], -max_offset_rad, max_offset_rad))

            normal_vec = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
            projections = relative_points @ normal_vec
            if projections.size == 0:
                distance = float(env_cfg.line_min_distance)
            else:
                distance = float(np.max(projections))
            if not np.isfinite(distance):
                distance = float(env_cfg.line_min_distance)
            distance = float(np.clip(distance, env_cfg.line_min_distance, max_distance))
            distances[idx] = distance

            offset_rad = self._wrap_angle_rad(theta - base_angle)
            offset_rad = float(np.clip(offset_rad, -max_offset_rad, max_offset_rad))
            offsets_deg[idx] = math.degrees(offset_rad)

        return np.concatenate([distances, offsets_deg]).astype(np.float32)

    @staticmethod
    def _convex_hull(points: np.ndarray) -> np.ndarray:
        if points.shape[0] <= 1:
            return points.copy()

        sorted_idx = np.lexsort((points[:, 1], points[:, 0]))
        sorted_points = points[sorted_idx]

        lower: List[np.ndarray] = []
        for p in sorted_points:
            while len(lower) >= 2 and BrainTumorDataset._cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper: List[np.ndarray] = []
        for p in reversed(sorted_points):
            while len(upper) >= 2 and BrainTumorDataset._cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull_points: List[np.ndarray] = lower[:-1] + upper[:-1]
        if not hull_points:
            hull_points = lower or upper or [sorted_points[0]]
        return np.stack(hull_points, axis=0).astype(np.float32)

    @staticmethod
    def _compute_hull_normals(hull: np.ndarray) -> np.ndarray:
        if hull.shape[0] < 2:
            return np.array([[1.0, 0.0]], dtype=np.float32)

        normals: List[np.ndarray] = []
        count = hull.shape[0]
        for i in range(count):
            p0 = hull[i]
            p1 = hull[(i + 1) % count]
            edge = p1 - p0
            normal = np.array([edge[1], -edge[0]], dtype=np.float32)
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normals.append(normal / norm)
        if not normals:
            normals.append(np.array([1.0, 0.0], dtype=np.float32))
        return np.stack(normals, axis=0)

    @staticmethod
    def _wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        entry = self.samples[idx]
        image_path, mask_path, slice_idx, embedding_path, meta = self._resolve_entry(entry)

        image_slice = self._load_volume_slice(image_path, slice_idx)
        mask_slice = self._load_volume_slice(mask_path, slice_idx)

        image = torch.from_numpy(image_slice.copy()).float().unsqueeze(0)
        mask = torch.from_numpy(mask_slice.copy()).float().unsqueeze(0)

        if self.resize_shape is not None:
            image = F.interpolate(
                image.unsqueeze(0),
                size=self.resize_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask = F.interpolate(
                mask.unsqueeze(0), size=self.resize_shape, mode="nearest"
            ).squeeze(0)

        meta.setdefault("image_path", image_path)
        meta.setdefault("mask_path", mask_path)
        meta["slice_index"] = int(slice_idx)

        sample: Dict[str, Any] = {"image": image, "mask": mask, "meta": meta}

        if embedding_path:
            embedding_vol = self._get_memmap(embedding_path)
            embedding_vec = embedding_vol[slice_idx]
            sample["embedding"] = torch.from_numpy(
                np.asarray(embedding_vec).copy()
            ).float()
            sample["meta"]["embedding_path"] = embedding_path

        if self._target_cache is not None:
            target_state = self._target_cache[idx]
            sample["target_polygon_state"] = torch.from_numpy(target_state.copy())

        if self.transform:
            sample = self.transform(sample)
        return sample

    @classmethod
    def from_samples(
        cls,
        samples,
        transform=None,
        include_empty_masks: bool = False,
        resize_shape: Tuple[int, int] | None = (224, 224),
        environment_config: Union[EnvironmentConfig, Dict[str, Any], None] = None,
    ):
        obj = cls.__new__(cls)
        obj.data_dir = None
        obj.transform = transform
        obj.include_empty_masks = include_empty_masks
        obj.samples = samples  # list of tuples or dictionaries describing slice metadata
        obj.resize_shape = resize_shape
        obj._mm_cache = {}
        obj._env_config = cls._resolve_env_config(environment_config)
        obj._target_cache = None
        if obj._env_config is not None:
            obj._precompute_ground_truth()
        return obj
