import torch

from rl.agent import TD3Agent, TD3Config, NoiseScheduleConfig
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
    )
    agent = TD3Agent(
        embedding_dim=4,
        polygon_dim=6,
        action_dim=5,
        config=config,
        device=torch.device("cpu"),
    )
    embedding = torch.randn(3, 4)
    polygon_state = torch.randn(3, 6)

    action = agent.act(embedding, polygon_state, deterministic=False)
    assert action.shape == (3, 5)
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
    buffer = ReplayBuffer(capacity=10, embedding_dim=4, polygon_dim=6, action_dim=5)

    for _ in range(6):
        transition = Transition(
            embedding=torch.randn(4),
            polygon_state=torch.randn(6),
            action=torch.tanh(torch.randn(5)),
            reward=torch.tensor([0.5]),
            discount=torch.tensor([0.99]),
            next_polygon_state=torch.randn(6),
            done=torch.tensor([0.0]),
        )
        buffer.add(transition)

    batch = buffer.sample(batch_size=4)
    assert batch["embedding"].shape == (4, 4)
    assert batch["polygon"].shape == (4, 6)
    assert batch["action"].shape == (4, 5)
    assert batch["reward"].shape == (4, 1)
    assert batch["discount"].shape == (4, 1)
    assert batch["next_polygon"].shape == (4, 6)
    assert batch["done"].shape == (4, 1)


def test_warmup_keeps_sigma_constant():
    config = TD3Config(
        exploration_noise=NoiseScheduleConfig(sigma_init=0.5, sigma_final=0.1, steps=10),
        embedding_noise_std=0.0,
    )
    agent = TD3Agent(embedding_dim=2, polygon_dim=2, action_dim=1, config=config)
    agent.set_warmup_steps(20)

    agent._interaction_count = 0
    start_sigma = agent._current_exploration_sigma()
    agent._interaction_count = 10
    mid_sigma = agent._current_exploration_sigma()

    assert start_sigma == mid_sigma == config.exploration_noise.sigma_init
