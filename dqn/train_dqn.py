import pytorch_lightning as pl
from dqn.lightning_model import DQNLightning
from pytorch_lightning.callbacks import ModelCheckpoint

def main():
    """Main function to train the DQN agent using PyTorch Lightning."""
    # --- Hyperparameters ---
    DATA_DIR = "MU-Glioma-Post/"
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY = 1000
    MEMORY_SIZE = 10000
    TARGET_UPDATE = 10
    MAX_STEPS = 100
    MAX_EPOCHS = 500 # Now represents number of episodes

    # --- Initialization ---
    model = DQNLightning(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        eps_start=EPSILON_START,
        eps_end=EPSILON_END,
        eps_decay=EPSILON_DECAY,
        memory_size=MEMORY_SIZE,
        target_update=TARGET_UPDATE,
        max_steps=MAX_STEPS
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='lightning_logs/checkpoints',
        filename='dqn-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model)

if __name__ == "__main__":
    main()