import pytest
import torch

from rl.agent import TD3Agent, TD3Config, NoiseScheduleConfig, GuidanceScheduleConfig
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
    agent = TD3Agent(
        embedding_shape=(1, 2, 2),
        polygon_dim=6,
        action_dim=9,
        config=config,
        device=torch.device("cpu"),
    )
    embedding = torch.randn(3, agent.embedding_dim)
    polygon_state = torch.randn(3, 6)

    action = agent.act(embedding, polygon_state, deterministic=False)
    assert action.shape == (3, 9)
    assert torch.all(action <= 1.0 + 1e-6)
    assert torch.all(action >= -1.0 - 1e-6)

    action_guided, base_action = agent.act(
        embedding,
        polygon_state,
        deterministic=False,
        return_base_action=True,
    )
    assert action_guided.shape == (3, 9)
    assert base_action.shape == (3, 9)

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
        embedding_dim=4,
        polygon_dim=6,
        action_dim=9,
        alpha=0.6,
        beta_start=0.4,
        beta_steps=1000,
        eps=1e-6,
    )

    for _ in range(6):
        transition = Transition(
            embedding=torch.randn(4),
            polygon_state=torch.randn(6),
            action=torch.tanh(torch.randn(9)),
            reward=torch.tensor([0.5]),
            discount=torch.tensor([0.99]),
            next_polygon_state=torch.randn(6),
            done=torch.tensor([0.0]),
        )
        buffer.add(transition)

    batch, indices, weights = buffer.sample(batch_size=4)
    assert batch["embedding"].shape == (4, 4)
    assert batch["polygon"].shape == (4, 6)
    assert batch["action"].shape == (4, 9)
    assert batch["reward"].shape == (4, 1)
    assert batch["discount"].shape == (4, 1)
    assert batch["next_polygon"].shape == (4, 6)
    assert batch["done"].shape == (4, 1)
    assert batch["guidance_target"].shape == (4, 9)
    assert torch.allclose(batch["guidance_target"], torch.zeros_like(batch["guidance_target"]))
    assert batch["guidance_mask"].shape == (4,)
    assert batch["guidance_mask"].dtype == torch.bool
    assert not batch["guidance_mask"].any()
    assert indices.shape[0] == 4
    assert weights.shape[0] == 4


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



def test_guidance_scale_randomization_beta():
    schedule = GuidanceScheduleConfig(
        initial=0.8,
        final=0.2,
        steps=100,
        randomize=True,
        alpha=5.0,
        beta=2.0,
        blend=0.5,
    )
    config = TD3Config(
        guidance_schedule=schedule,
        embedding_projected_dim=4,
        embedding_noise_std=0.0,
        actor_hidden_sizes=(16,),
        critic_hidden_sizes=(16,),
    )
    agent = TD3Agent(
        embedding_shape=(1, 2, 2),
        polygon_dim=3,
        action_dim=5,
        config=config,
        device=torch.device('cpu'),
    )
    agent._interaction_count = 50

    torch.manual_seed(0)
    randomized = agent._current_guidance_scale()
    torch.manual_seed(0)
    deterministic = agent._current_guidance_scale(randomize=False)

    assert 0.0 <= randomized <= 1.0
    assert deterministic == pytest.approx(0.5, abs=1e-6)
    assert randomized > deterministic


def test_guidance_targets_match_action_layout():
    env_cfg = {
        "num_sides": 3,
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

    distance_actions = guidance[0, :num_lines]
    angle_actions = guidance[0, num_lines : 2 * num_lines]
    center_actions = guidance[0, 2 * num_lines : 2 * num_lines + 4]
    stop_action = guidance[0, -1]

    assert distance_actions.shape[0] == num_lines
    assert angle_actions.shape[0] == num_lines
    assert torch.isclose(distance_actions[0], torch.tensor(1.0, dtype=guidance.dtype), atol=1e-5)
    assert torch.all(distance_actions.abs() <= 1.0 + 1e-6)
    assert torch.all(angle_actions.abs() <= 1.0 + 1e-6)
    assert center_actions.shape[0] == 4
    assert torch.all(center_actions <= 1.0 + 1e-6)
    assert torch.all(center_actions >= 0.0 - 1e-6)
    assert stop_action <= 1.0 + 1e-6
    assert stop_action >= -1.0 - 1e-6
