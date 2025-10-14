import torch

from rl.environment import EnvironmentConfig, PolygonLocalizationEnv


def _build_env(**overrides):
    cfg = EnvironmentConfig(
        num_sides=32,
        max_steps=5,
        iou_threshold=0.0,
        initial_radius=8.0,
        radial_step_scale=2.0,
        rotation_step_scale_deg=5.0,
        length_step_scale=2.0,
        stop_action_threshold=0.0,
        reward_success=3.0,
        reward_no_tumor=2.0,
        reward_false_stop=-1.0,
        time_penalty=0.0,
        hold_penalty=0.0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return PolygonLocalizationEnv(cfg)


def test_reset_returns_expected_shape():
    env = _build_env()
    images = torch.zeros(2, 1, 32, 32)
    masks = torch.zeros(2, 1, 32, 32)
    state = env.reset(images, masks)
    assert state.shape == (2, env.config.num_sides * 3)


def test_stop_action_success_reward():
    env = _build_env(iou_threshold=0.0, stop_action_threshold=-0.5)
    images = torch.zeros(1, 1, 32, 32)
    masks = torch.zeros(1, 1, 32, 32)
    masks[:, :, 10:22, 10:22] = 1.0
    env.reset(images, masks)

    actions = torch.zeros(1, env.action_dim)
    actions[:, -1] = 1.0  # stop immediately
    _, reward, done, info = env.step(actions)

    assert done.item() is True
    assert info["success"].item() is True
    assert torch.isclose(reward, torch.tensor(env.config.reward_success)).all()


def test_no_tumor_stop_reward_matches_config():
    env = _build_env(stop_action_threshold=-0.5)
    images = torch.zeros(1, 1, 32, 32)
    masks = torch.zeros(1, 1, 32, 32)
    env.reset(images, masks)

    actions = torch.zeros(1, env.action_dim)
    actions[:, -1] = 1.0
    _, reward, done, info = env.step(actions)

    assert done.item() is True
    assert info["success"].item() is False
    assert torch.isclose(reward, torch.tensor(env.config.reward_no_tumor)).all()


def test_actions_keep_vertices_within_bounds():
    env = _build_env(radial_step_scale=50.0)
    images = torch.zeros(1, 1, 32, 32)
    masks = torch.zeros(1, 1, 32, 32)
    env.reset(images, masks)

    actions = torch.zeros(1, env.action_dim)
    actions[:, :-1] = 1.0  # push outward aggressively
    state, _, _, _ = env.step(actions)

    vertices = env.vertices
    assert vertices[..., 0].min().item() >= 0.0
    assert vertices[..., 0].max().item() <= 31.0
    assert vertices[..., 1].min().item() >= 0.0
    assert vertices[..., 1].max().item() <= 31.0
    assert state.shape == (1, env.config.num_sides * 3)
