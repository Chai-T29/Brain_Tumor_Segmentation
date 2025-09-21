import numpy as np
import nibabel as nib
import torch

from data.dataset import BrainTumorDataset
from data.data_module import BrainTumorDataModule


def _create_mock_dataset(root_path):
    patient_dir = root_path / "patient_0001" / "timepoint_01"
    patient_dir.mkdir(parents=True)

    image = np.zeros((4, 4, 2), dtype=np.float32)
    mask = np.zeros((4, 4, 2), dtype=np.float32)
    mask[:, :, 1] = 1.0

    image_path = patient_dir / "patient_0001_timepoint_01_brain_t1c.nii.gz"
    mask_path = patient_dir / "patient_0001_timepoint_01_tumorMask.nii.gz"

    nib.save(nib.Nifti1Image(image, affine=np.eye(4)), str(image_path))
    nib.save(nib.Nifti1Image(mask, affine=np.eye(4)), str(mask_path))


def test_dataset_can_include_empty_slices(tmp_path):
    _create_mock_dataset(tmp_path)

    dataset_without_empty = BrainTumorDataset(str(tmp_path), include_empty_masks=False)
    dataset_with_empty = BrainTumorDataset(str(tmp_path), include_empty_masks=True)

    assert len(dataset_without_empty) == 1
    assert len(dataset_with_empty) == 2

    mask_sums = {torch.sum(sample["mask"]).item() for sample in dataset_with_empty}
    assert 0.0 in mask_sums


def test_datamodule_exposes_negative_samples_when_enabled(tmp_path):
    _create_mock_dataset(tmp_path)

    data_module = BrainTumorDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        val_split=0.0,
        test_split=0.0,
        include_empty_masks=True,
    )

    data_module.setup()

    train_loader = data_module.train_dataloader()
    batches = list(train_loader)
    assert len(batches) >= 1

    mask_sums = {torch.sum(batch["mask"]).item() for batch in batches}
    assert 0.0 in mask_sums
