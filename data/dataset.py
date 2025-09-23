import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import numpy as np
import torch.nn.functional as F

class BrainTumorDataset(Dataset):
    """Brain Tumor Segmentation Dataset for .nii.gz files."""

    def __init__(self, data_dir, transform=None, include_empty_masks=False, resize_shape=None):
        """
        Args:
            data_dir (string): Directory with all the patient folders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            include_empty_masks (bool): Whether to keep slices where the mask
                sums to zero. Defaults to ``False`` which filters out tumour-
                free slices.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.include_empty_masks = include_empty_masks
        self.resize_shape = resize_shape
        self.samples = []

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, mask_path, slice_idx = self.samples[idx]

        image_nii = nib.load(image_path)
        image_data = image_nii.get_fdata()
        image_slice = image_data[:, :, slice_idx]

        mask_nii = nib.load(mask_path)
        mask_data = mask_nii.get_fdata()
        mask_slice = mask_data[:, :, slice_idx]

        image = torch.from_numpy(image_slice).float().unsqueeze(0)  # (1,H,W)
        mask = torch.from_numpy(mask_slice).float().unsqueeze(0)

        if self.resize_shape is not None:
            image = F.interpolate(image.unsqueeze(0), size=self.resize_shape, mode="bilinear", align_corners=False).squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0), size=self.resize_shape, mode="nearest").squeeze(0)

        sample = {"image": image, "mask": mask}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    @classmethod
    def from_samples(cls, samples, transform=None, include_empty_masks=False):
        obj = cls.__new__(cls)
        obj.data_dir = None
        obj.transform = transform
        obj.include_empty_masks = include_empty_masks
        obj.samples = samples
        return obj
