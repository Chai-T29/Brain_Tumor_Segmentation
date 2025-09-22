import pytorch_lightning as pl
from model_deeplearning.network import UNet
from data.data_module import BrainTumorDataModule

def main():
    # Initialize the DataModule
    # Make sure to point data_dir to the correct path to your dataset
    data_module = BrainTumorDataModule(data_dir="MU-Glioma-Post/", batch_size=16)

    # Initialize the Model
    model = UNet(in_channels=3, out_channels=1)

    # Initialize the Trainer
    # Adjust trainer settings as needed (e.g., gpus, max_epochs)
    trainer = pl.Trainer(max_epochs=10, accelerator='auto')

    # Start training
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()
