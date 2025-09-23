import nibabel as nib
from data.dataset import BrainTumorDataset

dat = BrainTumorDataset("MU-Glioma-Post/")

for i in range(dat.__len__()):
    shap = dat.__getitem__(i)["image"].shape
    if shap != (1, 240, 240):
        print(shap)