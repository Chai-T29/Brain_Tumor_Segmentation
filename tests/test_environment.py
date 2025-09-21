import pytest
import torch

from dqn.environment import TumorLocalizationEnv


def _no_tumor_batch(batch_size=1):
    images = torch.zeros(batch_size, 3, 84, 84)
    masks = torch.zeros(batch_size, 1, 84, 84)
    return images, masks


def test_initial_bbox_covers_resized_frame():
    resize_shape = (84, 84)
    env = TumorLocalizationEnv(resize_shape=resize_shape)
    height, width = 128, 96
    batch_size = 3
    images = torch.zeros(batch_size, 3, height, width)
    masks = torch.zeros(batch_size, 1, height, width)

    (resized_images, bboxes) = env.reset(images, masks)

    expected_xy = torch.zeros(batch_size, 2, dtype=torch.float32)
    expected_wh = torch.tensor([list(resize_shape[::-1])] * batch_size, dtype=torch.float32)

    assert resized_images.shape[-2:] == resize_shape
    assert torch.allclose(bboxes[:, :2], expected_xy)
    assert torch.allclose(bboxes[:, 2:], expected_wh)


def test_bbox_scaling_with_non_square_resize():
    resize_shape = (100, 150)
    env = TumorLocalizationEnv(resize_shape=resize_shape)
    height, width = 200, 300
    images = torch.zeros(1, 3, height, width)
    masks = torch.zeros(1, 1, height, width)

    (_, bboxes) = env.reset(images, masks)

    expected = torch.tensor([[0.0, 0.0, resize_shape[1], resize_shape[0]]], dtype=torch.float32)
    assert torch.allclose(bboxes, expected)


def test_initial_bbox_overlaps_tumor_mask():
    env = TumorLocalizationEnv()
    images = torch.zeros(1, 3, 84, 84)
    masks = torch.zeros(1, 1, 84, 84)
    masks[:, :, 10:20, 15:25] = 1.0

    env.reset(images, masks)

    assert env.last_iou is not None
    assert env.last_iou.shape == (1,)
    assert env.last_iou.item() > 0.0


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


def test_gt_margin_initialisation_allows_movement():
    env = TumorLocalizationEnv(initial_mode="gt_margin", initial_margin=5.0, step_size=5.0)
    images = torch.zeros(1, 3, 84, 84)
    masks = torch.zeros(1, 1, 84, 84)
    masks[:, :, 30:40, 30:40] = 1.0

    _, bboxes = env.reset(images, masks)
    initial_bbox = bboxes.clone()

    (_, updated_bboxes), _, _, _ = env.step(torch.tensor([3]))

    assert updated_bboxes[0, 0] > initial_bbox[0, 0]
