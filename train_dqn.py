import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dqn.lightning_model import DQNLightning


def main():
    """Main function to train the DQN agent using PyTorch Lightning."""
    DATA_DIR = "MU-Glioma-Post/"
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 2000
    MEMORY_SIZE = 50000
    TARGET_UPDATE = 10
    MAX_STEPS_PER_EPISODE = 100
    MAX_EPISODES = 200
    TRAIN_SAMPLE_SIZE = 512
    VAL_SAMPLE_SIZE = 256
    VAL_EPISODES = 8
    LR_GAMMA = 0.995
    TEST_GIF_LIMIT = 100

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
        max_steps=MAX_STEPS_PER_EPISODE,
        train_sample_size=TRAIN_SAMPLE_SIZE,
        val_sample_size=VAL_SAMPLE_SIZE,
        val_episodes=VAL_EPISODES,
        lr_gamma=LR_GAMMA,
        test_gif_limit=TEST_GIF_LIMIT,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/avg_reward",
        dirpath="lightning_logs/checkpoints",
        filename="dqn-epoch{epoch:02d}-reward{val_avg_reward:.2f}",
        save_top_k=3,
        mode="max",
    )

    early_stopping = EarlyStopping(
        monitor="val/avg_reward",
        min_delta=1e-3,
        patience=20,
        mode="max",
        verbose=True,
    )

    logger = TensorBoardLogger(save_dir="lightning_logs", name="dqn_agent")

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        max_epochs=MAX_EPISODES,
        check_val_every_n_epoch=model.hparams.val_interval,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=1,
        enable_model_summary=True,
        deterministic=True,
    )

    trainer.fit(model)

    trainer.validate(model, ckpt_path="best")
    trainer.test(model, ckpt_path="best")


if __name__ == "__main__":
    main()
