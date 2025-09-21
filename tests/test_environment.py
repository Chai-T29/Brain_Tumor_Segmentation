import torch

from dqn.environment import TumorLocalizationEnv


def test_reset_initialises_bbox_covering_entire_image():
    env = TumorLocalizationEnv()
    batch_size = 3
    height, width = 96, 128
    images = torch.zeros(batch_size, 1, height, width)
    masks = torch.zeros(batch_size, 1, height, width)

    _, initial_bboxes = env.reset(images, masks)

    expected = torch.tensor([0.0, 0.0, float(width), float(height)], dtype=torch.float32)

    assert initial_bboxes.shape == (batch_size, 4)
    assert torch.allclose(initial_bboxes, expected.unsqueeze(0).expand_as(initial_bboxes))
