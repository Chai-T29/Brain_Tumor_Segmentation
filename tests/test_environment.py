import pytest
import torch

from dqn.environment import TumorLocalizationEnv


def _no_tumor_batch(batch_size=1):
    images = torch.zeros(batch_size, 3, 84, 84)
    masks = torch.zeros(batch_size, 1, 84, 84)
    return images, masks


def test_no_tumor_stop_action_reward():
    env = TumorLocalizationEnv(max_steps=5, iou_threshold=0.5)
    images, masks = _no_tumor_batch()
    env.reset(images, masks)

    _, rewards, done, _ = env.step(torch.tensor([env._STOP_ACTION]))

    assert rewards.item() == pytest.approx(2.0)
    assert done.item()


def test_no_tumor_continue_action_penalty():
    env = TumorLocalizationEnv(max_steps=5, iou_threshold=0.5)
    images, masks = _no_tumor_batch()
    env.reset(images, masks)

    _, rewards, done, _ = env.step(torch.tensor([0]))

    assert rewards.item() == pytest.approx(-0.5)
    assert not done.item()
