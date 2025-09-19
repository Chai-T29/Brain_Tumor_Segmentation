import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import numpy as np

class BrainTumorDataset(Dataset):
    """Brain Tumor Segmentation Dataset for .nii.gz files."""

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the patient folders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
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
                        if np.sum(mask_data[:, :, i]) > 0: # Only include slices with a tumor
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

        # Convert to float tensors
        image = torch.from_numpy(image_slice).float().unsqueeze(0) # Add channel dimension
        mask = torch.from_numpy(mask_slice).float().unsqueeze(0)

        # The U-Net I created expects 3 input channels.
        # The T1c image is single channel. I will stack it 3 times to simulate a 3-channel image.
        image = image.repeat(3, 1, 1)


        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample