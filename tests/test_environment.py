import torch

from rl.environment import EnvironmentConfig, PolygonLocalizationEnv


def _build_env(**overrides):
    cfg = EnvironmentConfig(
        num_sides=32,
        max_steps=5,
        iou_low_threshold=0.0,
        iou_high_threshold=0.0,
        initial_radius=8.0,
        line_distance_step_scale=2.0,
        line_angle_step_scale_deg=5.0,
        line_max_angle_offset_deg=45.0,
        line_min_distance=0.0,
        line_max_distance_margin=1.0,
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
    assert state.shape == (2, env.state_dim)


def test_manual_stop_success_reward():
    env = _build_env(iou_low_threshold=0.0, iou_high_threshold=0.0)
    images = torch.zeros(1, 1, 32, 32)
    masks = torch.zeros(1, 1, 32, 32)
    masks[:, :, 10:22, 10:22] = 1.0
    env.reset(images, masks)

    actions = torch.zeros(1, env.action_dim)
    actions[..., -1] = 1.0  # trigger manual stop
    _, reward, done, info = env.step(actions)

    assert done.item() is True
    assert info["manual_stop"].item() is True
    assert info["success"].item() is True
    assert torch.isclose(reward, torch.tensor(env.config.reward_success)).all()


def test_manual_stop_no_tumor_reward_matches_config():
    env = _build_env()
    images = torch.zeros(1, 1, 32, 32)
    masks = torch.zeros(1, 1, 32, 32)
    env.reset(images, masks)

    actions = torch.zeros(1, env.action_dim)
    actions[..., -1] = 1.0  # trigger manual stop
    _, reward, done, info = env.step(actions)

    assert done.item() is True
    assert info["manual_stop"].item() is True
    assert info["success"].item() is False
    assert torch.isclose(reward, torch.tensor(env.config.reward_no_tumor)).all()


def test_no_manual_stop_continues_episode():
    env = _build_env()
    images = torch.zeros(1, 1, 32, 32)
    masks = torch.zeros(1, 1, 32, 32)
    env.reset(images, masks)

    actions = torch.zeros(1, env.action_dim)
    _, reward, done, info = env.step(actions)

    assert done.item() is False
    assert info["manual_stop"].item() is False
    # Reward should reflect ongoing step (here zero because no change and no penalties)
    assert torch.isclose(reward, torch.tensor(0.0)).all()


def test_actions_keep_vertices_within_bounds():
    env = _build_env(line_distance_step_scale=50.0)
    images = torch.zeros(1, 1, 32, 32)
    masks = torch.zeros(1, 1, 32, 32)
    env.reset(images, masks)

    actions = torch.zeros(1, env.action_dim)
    actions[:] = 1.0  # push outward aggressively
    state, _, _, _ = env.step(actions)

    vertices = env.vertices
    assert vertices[..., 0].min().item() >= 0.0
    assert vertices[..., 0].max().item() <= 31.0
    assert vertices[..., 1].min().item() >= 0.0
    assert vertices[..., 1].max().item() <= 31.0
    assert state.shape == (1, env.state_dim)
