import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional, List, Tuple
import pytorch_lightning as pl
import nibabel as nib
from pathlib import Path
import hashlib
import pickle
from tqdm import tqdm
import torch.nn.functional as F

class NormalizeSlice:
    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
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

class PreprocessedBrainTumorDataModule(pl.LightningDataModule):
    """Data module with preprocessing, lazy loading, and progress bars."""

    def __init__(
        self,
        data_dir: str,
        cache_dir: str = "preprocessed_data",
        batch_size: int = 16,
        num_workers: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        include_empty_masks: bool = False,
        target_size: Tuple[int, int] = (224, 224),
        force_reprocess: bool = False,
        max_cached_volumes: int = 10,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.include_empty_masks = include_empty_masks
        self.target_size = target_size
        self.force_reprocess = force_reprocess
        self.max_cached_volumes = max_cached_volumes
        self.transform = NormalizeSlice()

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_data_hash(self) -> str:
        """Generate hash of data directory for cache validation."""
        hash_inputs = []
        
        # Include directory structure and file names
        for root, dirs, files in os.walk(self.data_dir):
            for file in sorted(files):
                if file.endswith(('.nii.gz', '.nii')):
                    file_path = os.path.join(root, file)
                    try:
                        stat = os.stat(file_path)
                        hash_inputs.append(f"{file_path}_{stat.st_mtime}_{stat.st_size}")
                    except OSError:
                        continue
        
        # Include preprocessing parameters
        hash_inputs.extend([
            str(self.target_size),
            str(self.include_empty_masks),
            str(self.val_split),
            str(self.test_split),
            str(self.seed)
        ])
        
        return hashlib.md5('\n'.join(hash_inputs).encode()).hexdigest()

    def _should_reprocess(self) -> bool:
        """Check if preprocessing is needed."""
        if self.force_reprocess:
            print("Force reprocessing enabled")
            return True
            
        metadata_file = self.cache_dir / "metadata.pkl"
        if not metadata_file.exists():
            print("No metadata found, preprocessing required")
            return True
            
        try:
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            current_hash = self._get_data_hash()
            stored_hash = metadata.get('data_hash')
            
            if stored_hash != current_hash:
                print("Data has changed, reprocessing required")
                return True
            else:
                print("Using cached preprocessed data")
                return False
        except Exception as e:
            print(f"Error reading metadata: {e}, reprocessing required")
            return True

    def _preprocess_volume(self, image_path: Path, mask_path: Path, output_dir: Path) -> List[int]:
        """Preprocess a single volume and return valid slice indices."""
        try:
            # Load volumes
            image_nii = nib.load(image_path)
            mask_nii = nib.load(mask_path)
            image_data = image_nii.get_fdata().astype(np.float32)
            mask_data = mask_nii.get_fdata().astype(np.float32)
            
            # Determine valid slices
            valid_slices = []
            _, _, n_slices = image_data.shape
            
            for slice_idx in range(n_slices):
                mask_slice = mask_data[:, :, slice_idx]
                has_tumor = np.sum(mask_slice) > 0
                if has_tumor or self.include_empty_masks:
                    valid_slices.append(slice_idx)
            
            if not valid_slices:
                return []
            
            # Prepare output arrays
            n_valid = len(valid_slices)
            H, W = self.target_size
            
            # Pre-allocate memory-mapped arrays
            image_file = output_dir / "images.npy"
            mask_file = output_dir / "masks.npy"
            
            # Create memory-mapped arrays
            image_mmap = np.memmap(image_file, dtype=np.float32, mode='w+', 
                                  shape=(n_valid, 1, H, W))
            mask_mmap = np.memmap(mask_file, dtype=np.float32, mode='w+', 
                                 shape=(n_valid, 1, H, W))
            
            # Process and store each valid slice with interpolation
            for i, slice_idx in enumerate(valid_slices):
                # Extract slices
                image_slice = image_data[:, :, slice_idx]
                mask_slice = mask_data[:, :, slice_idx]
                
                # Convert to tensors for interpolation
                image_tensor = torch.from_numpy(image_slice).float().unsqueeze(0).unsqueeze(0)
                mask_tensor = torch.from_numpy(mask_slice).float().unsqueeze(0).unsqueeze(0)
                
                # Resize with interpolation
                image_resized = F.interpolate(
                    image_tensor, size=self.target_size, mode='bilinear', align_corners=False
                ).squeeze(0).numpy()
                
                mask_resized = F.interpolate(
                    mask_tensor, size=self.target_size, mode='nearest'
                ).squeeze(0).numpy()
                
                # Store in memory-mapped arrays
                image_mmap[i] = image_resized
                mask_mmap[i] = mask_resized
            
            # Ensure data is written to disk
            del image_mmap, mask_mmap
            
            # Save slice indices
            indices_file = output_dir / "slice_indices.npy"
            np.save(indices_file, np.array(valid_slices))
            
            return valid_slices
            
        except Exception as e:
            print(f"Error processing volume {image_path}: {e}")
            return []

    def _preprocess_all_data(self) -> Tuple[List, List, List]:
        """Preprocess all data with progress bars and return train/val/test groups."""
        print("🔍 Scanning dataset...")
        
        # Collect all volumes
        all_volumes = []
        for patient_dir in sorted(self.data_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
                
            for timepoint_dir in sorted(patient_dir.iterdir()):
                if not timepoint_dir.is_dir():
                    continue
                
                image_path = mask_path = None
                for file in timepoint_dir.iterdir():
                    if file.name.endswith('_brain_t1c.nii.gz'):
                        image_path = file
                    elif file.name.endswith('_tumorMask.nii.gz'):
                        mask_path = file
                
                if image_path and mask_path:
                    volume_id = f"{patient_dir.name}_{timepoint_dir.name}"
                    all_volumes.append((volume_id, image_path, mask_path))
        
        print(f"📊 Found {len(all_volumes)} volumes")
        
        # Create train/val/test splits
        rng = random.Random(self.seed)
        rng.shuffle(all_volumes)
        
        n_total = len(all_volumes)
        n_val = int(self.val_split * n_total)
        n_test = int(self.test_split * n_total)
        
        val_volumes = all_volumes[:n_val]
        test_volumes = all_volumes[n_val:n_val + n_test]
        train_volumes = all_volumes[n_val + n_test:]
        
        print(f"📈 Split: Train={len(train_volumes)}, Val={len(val_volumes)}, Test={len(test_volumes)}")
        
        # Process each split
        splits = [
            ("train", train_volumes),
            ("val", val_volumes), 
            ("test", test_volumes)
        ]
        
        split_data = {}
        total_processed_slices = 0
        
        for split_name, volumes in splits:
            if not volumes:
                split_data[split_name] = []
                continue
                
            split_dir = self.cache_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            split_samples = []
            split_slices = 0
            
            # Progress bar for this split
            pbar = tqdm(volumes, desc=f"🔄 Processing {split_name}", 
                       unit="volume", colour="blue")
            
            for volume_id, image_path, mask_path in pbar:
                volume_dir = split_dir / volume_id
                volume_dir.mkdir(exist_ok=True)
                
                # Check if already processed (and not forcing reprocess)
                if (volume_dir / "images.npy").exists() and not self.force_reprocess:
                    # Load existing slice indices
                    try:
                        indices = np.load(volume_dir / "slice_indices.npy")
                        valid_slices = indices.tolist()
                        pbar.set_postfix({"status": "cached", "slices": len(valid_slices)})
                    except:
                        # File exists but corrupted, reprocess
                        valid_slices = self._preprocess_volume(image_path, mask_path, volume_dir)
                        pbar.set_postfix({"status": "reprocessed", "slices": len(valid_slices)})
                else:
                    # Process volume
                    valid_slices = self._preprocess_volume(image_path, mask_path, volume_dir)
                    pbar.set_postfix({"status": "processed", "slices": len(valid_slices)})
                
                # Add samples for this volume
                for i, slice_idx in enumerate(valid_slices):
                    split_samples.append((volume_id, i, slice_idx))
                
                split_slices += len(valid_slices)
            
            pbar.close()
            print(f"✅ {split_name}: {len(volumes)} volumes → {split_slices} slices")
            split_data[split_name] = split_samples
            total_processed_slices += split_slices
        
        print(f"🎉 Preprocessing complete! Total slices: {total_processed_slices}")
        
        # Save metadata
        metadata = {
            'data_hash': self._get_data_hash(),
            'target_size': self.target_size,
            'include_empty_masks': self.include_empty_masks,
            'splits': {k: len(v) for k, v in split_data.items()},
            'total_slices': total_processed_slices
        }
        
        with open(self.cache_dir / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"💾 Metadata saved to {self.cache_dir / 'metadata.pkl'}")
        
        return split_data["train"], split_data["val"], split_data["test"]

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit", "validate", "test"):
            return
        
        print("🚀 Setting up data module...")
        
        # Check if preprocessing is needed
        if self._should_reprocess():
            train_samples, val_samples, test_samples = self._preprocess_all_data()
        else:
            # Load existing metadata and reconstruct sample lists
            with open(self.cache_dir / "metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
                
            print(f"📋 Loading cached data: {metadata.get('total_slices', '?')} total slices")
            
            train_samples = self._load_existing_samples("train")
            val_samples = self._load_existing_samples("val")  
            test_samples = self._load_existing_samples("test")
        
        # Create datasets with lazy loading
        self.train_dataset = LazyMemoryMappedDataset(
            self.cache_dir / "train", train_samples, self.transform, self.max_cached_volumes
        )
        self.val_dataset = LazyMemoryMappedDataset(
            self.cache_dir / "val", val_samples, self.transform, self.max_cached_volumes
        )
        self.test_dataset = LazyMemoryMappedDataset(
            self.cache_dir / "test", test_samples, self.transform, self.max_cached_volumes
        )
        
        # Handle empty splits
        if len(val_samples) == 0:
            print("⚠️  No validation samples, using train dataset for validation")
            self.val_dataset = self.train_dataset
        if len(test_samples) == 0:
            print("⚠️  No test samples, using validation dataset for testing")
            self.test_dataset = self.val_dataset
            
        print(f"✅ Data module ready: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    def _load_existing_samples(self, split_name: str) -> List:
        """Load sample list for existing preprocessed split."""
        split_dir = self.cache_dir / split_name
        samples = []
        
        if not split_dir.exists():
            return samples
        
        for volume_dir in sorted(split_dir.iterdir()):
            if not volume_dir.is_dir():
                continue
                
            indices_file = volume_dir / "slice_indices.npy"
            if indices_file.exists():
                try:
                    indices = np.load(indices_file)
                    volume_id = volume_dir.name
                    for i, slice_idx in enumerate(indices):
                        samples.append((volume_id, i, int(slice_idx)))
                except:
                    print(f"⚠️  Corrupted indices file: {indices_file}")
                    continue
        
        return samples

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )


class LazyMemoryMappedDataset:
    """Dataset with lazy loading and LRU caching of memory-mapped arrays."""
    
    def __init__(self, cache_dir: Path, samples: List, transform=None, max_open_volumes: int = 10):
        self.cache_dir = cache_dir
        self.samples = samples  # List of (volume_id, slice_index, original_slice_idx)
        self.transform = transform
        self.max_open_volumes = max_open_volumes
        
        # Thread-safe LRU cache
        from collections import OrderedDict
        import threading
        self._volume_cache: OrderedDict = OrderedDict()
        self._cache_lock = threading.Lock()
        
        # Pre-compute volume metadata without opening files
        self._volume_metadata = self._scan_volume_metadata()
    
    def _scan_volume_metadata(self) -> dict:
        """Scan all volume directories to get metadata without opening files."""
        metadata = {}
        
        for volume_dir in self.cache_dir.iterdir():
            if not volume_dir.is_dir():
                continue
                
            volume_id = volume_dir.name
            image_file = volume_dir / "images.npy"
            mask_file = volume_dir / "masks.npy"
            indices_file = volume_dir / "slice_indices.npy"
            
            if all(f.exists() for f in [image_file, mask_file, indices_file]):
                try:
                    # Get number of slices from indices file
                    indices = np.load(indices_file)
                    n_slices = len(indices)
                    
                    metadata[volume_id] = {
                        'image_file': image_file,
                        'mask_file': mask_file,
                        'indices_file': indices_file,
                        'n_slices': n_slices,
                        'slice_shape': (224, 224)  # Assuming target_size from preprocessing
                    }
                except:
                    print(f"⚠️  Could not read metadata for volume {volume_id}")
                    continue
        
        return metadata
    
    def _open_volume(self, volume_id: str) -> Tuple[np.memmap, np.memmap]:
        """Open memory-mapped arrays for a specific volume."""
        if volume_id not in self._volume_metadata:
            raise ValueError(f"Volume {volume_id} not found in metadata")
        
        meta = self._volume_metadata[volume_id]
        
        # Open memory-mapped files (read-only)
        images = np.memmap(
            meta['image_file'], 
            dtype=np.float32, 
            mode='r'
        ).reshape(meta['n_slices'], 1, *meta['slice_shape'])
        
        masks = np.memmap(
            meta['mask_file'], 
            dtype=np.float32, 
            mode='r'
        ).reshape(meta['n_slices'], 1, *meta['slice_shape'])
        
        return images, masks
    
    def _get_volume_with_caching(self, volume_id: str) -> Tuple[np.memmap, np.memmap]:
        """Get volume with LRU caching and automatic cleanup."""
        with self._cache_lock:
            # Check if already in cache
            if volume_id in self._volume_cache:
                # Move to end (most recently used)
                volume_data = self._volume_cache.pop(volume_id)
                self._volume_cache[volume_id] = volume_data
                return volume_data
            
            # Need to open new volume
            # First, check if cache is full
            if len(self._volume_cache) >= self.max_open_volumes:
                # Remove least recently used volume
                old_volume_id, old_data = self._volume_cache.popitem(last=False)
                # Memory maps are automatically closed when references are deleted
                del old_data
            
            # Open new volume
            try:
                volume_data = self._open_volume(volume_id)
                self._volume_cache[volume_id] = volume_data
                return volume_data
            except Exception as e:
                raise RuntimeError(f"Failed to open volume {volume_id}: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        volume_id, slice_idx, original_slice_idx = self.samples[idx]
        
        # Get volume data (with lazy loading and caching)
        images, masks = self._get_volume_with_caching(volume_id)
        
        # Extract the specific slice - IMPORTANT: copy the data
        image = torch.from_numpy(images[slice_idx].copy())
        mask = torch.from_numpy(masks[slice_idx].copy())
        
        sample = {"image": image, "mask": mask}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_cache_info(self) -> dict:
        """Get information about cache usage."""
        with self._cache_lock:
            return {
                'cached_volumes': len(self._volume_cache),
                'max_volumes': self.max_open_volumes,
                'available_volumes': len(self._volume_metadata),
                'cached_volume_ids': list(self._volume_cache.keys())
            }