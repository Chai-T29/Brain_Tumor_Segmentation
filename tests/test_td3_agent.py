import torch
import numpy as np

from rl.agent import TD3Agent, TD3Config, NoiseScheduleConfig
from rl.lightning_module import TD3Lightning
from rl.replay_buffer import ReplayBuffer, Transition


def test_td3_agent_action_bounds():
    config = TD3Config(
        gamma=0.99,
        tau=0.005,
        n_step=3,
        exploration_noise=NoiseScheduleConfig(sigma_init=0.2, sigma_final=0.05, steps=10),
        target_policy_noise_std=0.2,
        target_policy_noise_clip=0.5,
        embedding_noise_std=0.0,
        embedding_projected_dim=4,
    )
    config.guidance_mode = "critic_guidance"
    agent = TD3Agent(
        embedding_shape=(1, 2, 2),
        polygon_dim=6,
        action_dim=7,
        config=config,
        device=torch.device("cpu"),
    )
    embedding = torch.randn(3, *agent.embedding_shape)
    polygon_state = torch.randn(3, 6)

    action = agent.act(embedding, polygon_state, deterministic=False)
    assert action.shape == (3, 7)
    assert torch.all(action <= 1.0 + 1e-6)
    assert torch.all(action >= -1.0 - 1e-6)

    deterministic_action = agent.act(embedding, polygon_state, deterministic=True, apply_embedding_noise=False)
    assert torch.all(deterministic_action <= 1.0 + 1e-6)
    assert torch.all(deterministic_action >= -1.0 - 1e-6)

    agent.set_warmup_steps(0)
    agent._interaction_count = 0
    sigma_start = agent._current_exploration_sigma()
    agent._interaction_count = 5
    sigma_mid = agent._current_exploration_sigma()
    agent._interaction_count = 15
    sigma_end = agent._current_exploration_sigma()

    assert sigma_start > sigma_mid >= sigma_end


def test_replay_buffer_sample_shapes():
    buffer = ReplayBuffer(
        capacity=10,
        embedding_shape=(1, 2, 2),
        polygon_dim=6,
        action_dim=7,
        alpha=0.6,
        beta_start=0.4,
        beta_steps=1000,
        eps=1e-6,
    )

    for _ in range(6):
        transition = Transition(
            embedding=torch.randn(1, 2, 2),
            polygon_state=torch.randn(6),
            action=torch.tanh(torch.randn(7)),
            reward=torch.tensor([0.5]),
            discount=torch.tensor([0.99]),
            next_polygon_state=torch.randn(6),
            done=torch.tensor([0.0]),
        )
        buffer.add(transition)

    batch, indices, weights = buffer.sample(batch_size=4)
    assert batch["embedding"].shape == (4, 1, 2, 2)
    assert batch["polygon"].shape == (4, 6)
    assert batch["action"].shape == (4, 7)
    assert batch["reward"].shape == (4, 1)
    assert batch["discount"].shape == (4, 1)
    assert batch["next_polygon"].shape == (4, 6)
    assert batch["done"].shape == (4, 1)
    assert indices.shape[0] == 4
    assert weights.shape[0] == 4


def test_replay_buffer_pointer_mode(tmp_path):
    buffer = ReplayBuffer(
        capacity=4,
        embedding_shape=(1, 2, 2),
        polygon_dim=6,
        action_dim=7,
        alpha=0.6,
        beta_start=0.4,
        beta_steps=1000,
        eps=1e-6,
        use_embedding_pointers=True,
    )

    ptr_path = tmp_path / "embeddings.npy"
    data = np.random.randn(5, 1, 2, 2).astype(np.float32)
    np.save(ptr_path, data)

    pointer = {"path": str(ptr_path), "slice_index": 0}

    transition = Transition(
        embedding=pointer,
        polygon_state=torch.zeros(6),
        action=torch.zeros(7),
        reward=torch.tensor([0.0]),
        discount=torch.tensor([0.99]),
        next_polygon_state=torch.zeros(6),
        done=torch.tensor([0.0]),
    )
    buffer.add(transition)

    batch, _, _ = buffer.sample(batch_size=1)
    assert batch["embedding"].shape == (1, 1, 2, 2)


def test_warmup_keeps_sigma_constant():
    config = TD3Config(
        exploration_noise=NoiseScheduleConfig(sigma_init=0.5, sigma_final=0.1, steps=10),
        embedding_noise_std=0.0,
        embedding_projected_dim=2,
    )
    agent = TD3Agent(embedding_shape=(1, 3, 3), polygon_dim=2, action_dim=1, config=config)
    agent.set_warmup_steps(20)

    agent._interaction_count = 0
    start_sigma = agent._current_exploration_sigma()
    agent._interaction_count = 10
    mid_sigma = agent._current_exploration_sigma()

    assert start_sigma == mid_sigma == config.exploration_noise.sigma_init


def test_guidance_targets_match_action_layout():
    env_cfg = {
        "num_sides": 2,
        "line_distance_step_scale": 2.0,
        "line_angle_step_scale_deg": 10.0,
        "center_step_scale": 1.0,
    }
    algo_cfg = {
        "actor_hidden_sizes": [8],
        "critic_hidden_sizes": [8],
        "exploration_noise": {"sigma_init": 0.0, "sigma_final": 0.0, "steps": 1},
        "embedding_projected_dim": 4,
    }
    training_cfg = {
        "update_batch_size": 1,
        "update_every_n_steps": 1,
        "warmup_steps": 0,
    }

    module = TD3Lightning(
        embedding_shape=(1, 1, 1),
        env_cfg=env_cfg,
        algo_cfg=algo_cfg,
        training_cfg=training_cfg,
        replay_capacity=4,
    )

    num_lines = module.environment.num_lines
    current_state = torch.zeros(1, num_lines * 2 + 2)
    target_state = current_state.clone()

    distance_scale = module.env_config.line_distance_step_scale
    angle_scale = module.env_config.line_angle_step_scale_deg

    target_state[0, 0] = distance_scale
    target_state[0, num_lines] = angle_scale

    guidance = module._compute_true_guidance_targets(current_state, target_state)

    expected_line = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=guidance.dtype)
    assert torch.allclose(guidance[0, : num_lines * 2], expected_line, atol=1e-5)
    expected_center = torch.zeros(2, dtype=guidance.dtype)
    assert torch.allclose(guidance[0, num_lines * 2 : num_lines * 2 + 2], expected_center, atol=1e-5)
    assert torch.isclose(guidance[0, -1], torch.tensor(-1.0, dtype=guidance.dtype))


def test_guidance_mode_none_requires_no_targets():
    config = TD3Config(
        guidance_mode="none",
        exploration_noise=NoiseScheduleConfig(sigma_init=0.0, sigma_final=0.0, steps=1),
        embedding_noise_std=0.0,
        embedding_projected_dim=4,
        actor_hidden_sizes=(8,),
        critic_hidden_sizes=(8,),
    )
    agent = TD3Agent(
        embedding_shape=(1, 2, 2),
        polygon_dim=6,
        action_dim=4,
        config=config,
        device=torch.device("cpu"),
    )

    embedding = torch.randn(5, *agent.embedding_shape)
    polygon_state = torch.randn(5, 6)

    assert agent.requires_guided_targets is False
    action = agent.act(embedding, polygon_state, deterministic=True)
    assert action.shape == (5, 4)
