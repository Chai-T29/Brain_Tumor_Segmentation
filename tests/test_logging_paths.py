from pathlib import Path

from pytorch_lightning.loggers import TensorBoardLogger

from test_dqn import find_best_checkpoint


def test_tensorboard_logger_respects_explicit_version(tmp_path):
    """Specifying a logger version should influence both logging and checkpoint paths."""

    log_dir = tmp_path / "lightning_logs"
    logger_name = "dqn_agent"
    version = 7
    checkpoint_subdir = "checkpoints"

    logger = TensorBoardLogger(
        save_dir=str(log_dir),
        name=logger_name,
        version=version,
    )

    expected_log_dir = log_dir / logger_name / f"version_{version}"
    assert Path(logger.log_dir) == expected_log_dir

    checkpoint_dir = expected_log_dir / checkpoint_subdir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "dummy.ckpt"
    checkpoint_path.write_text("checkpoint content", encoding="utf-8")

    found_checkpoint = find_best_checkpoint(
        log_dir=log_dir,
        logger_name=logger_name,
        checkpoint_subdir=checkpoint_subdir,
        versions=[expected_log_dir],
    )

    assert found_checkpoint == checkpoint_path
