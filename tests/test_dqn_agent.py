import random

import numpy as np
import pytest
import torch

from dqn.agent import DQNAgent
from dqn.replay_memory import Experience


class DummyNet(torch.nn.Module):
    """A lightweight network with deterministic outputs for testing."""

    def __init__(self, q_values):
        super().__init__()
        self.register_buffer("base_q_values", torch.tensor(q_values, dtype=torch.float32))
        # Parameter to ensure gradients/updates are possible during optimization.
        self.bias = torch.nn.Parameter(torch.zeros(len(q_values), dtype=torch.float32))

    def forward(self, image, bbox):
        batch_size = image.shape[0]
        return self.base_q_values.unsqueeze(0).expand(batch_size, -1) + self.bias.unsqueeze(0)


def _compute_expected_loss(agent, experiences):
    batch = Experience(*zip(*experiences))

    state_batch = batch.state
    action_batch = torch.cat(batch.action).to(agent.device)
    reward_batch = torch.cat(batch.reward).to(agent.device)

    image_batch = torch.stack([s[0] for s in state_batch]).to(agent.device)
    bbox_batch = torch.stack([s[1] for s in state_batch]).to(agent.device)

    state_action_values = agent.policy_net(image_batch, bbox_batch).gather(1, action_batch)

    non_final_mask_list = []
    non_final_next_states = []
    for next_state, done in zip(batch.next_state, batch.done):
        is_non_final = (next_state is not None) and (not done)
        non_final_mask_list.append(is_non_final)
        if is_non_final:
            non_final_next_states.append(next_state)

    non_final_mask = torch.tensor(non_final_mask_list, device=agent.device, dtype=torch.bool)

    next_state_values = torch.zeros(agent.batch_size, device=agent.device)
    if non_final_next_states:
        non_final_next_images = torch.stack([s[0] for s in non_final_next_states]).to(agent.device)
        non_final_next_bboxes = torch.stack([s[1] for s in non_final_next_states]).to(agent.device)
        next_state_values[non_final_mask] = agent.target_net(non_final_next_images, non_final_next_bboxes).max(1)[0].detach()

    expected_state_action_values = reward_batch + agent.gamma * next_state_values

    loss = torch.nn.functional.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )
    return loss, expected_state_action_values


def test_optimize_model_handles_terminal_transition():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DummyNet([1.0, 2.0]).to(device)
    target_net = DummyNet([3.0, 4.0]).to(device)
    agent = DQNAgent(
        policy_net=policy_net,
        target_net=target_net,
        num_actions=2,
        batch_size=2,
        memory_size=10,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1,
    )

    # Construct two experiences, one of which is terminal with no next state.
    image = torch.zeros(3, 84, 84)
    bbox = torch.zeros(4)
    next_image = torch.ones(3, 84, 84)
    next_bbox = torch.ones(4)

    agent.memory.push(
        (image, bbox),
        torch.tensor([[1]], dtype=torch.long, device=agent.device),
        torch.tensor([1.0], dtype=torch.float32, device=agent.device),
        (next_image, next_bbox),
        False,
    )

    agent.memory.push(
        (image + 2, bbox + 2),
        torch.tensor([[0]], dtype=torch.long, device=agent.device),
        torch.tensor([2.0], dtype=torch.float32, device=agent.device),
        None,
        True,
    )

    random.seed(0)
    np.random.seed(0)
    experiences, indices, weights = agent.memory.sample(agent.batch_size)
    expected_loss, expected_targets = _compute_expected_loss(agent, experiences)

    # Ensure terminal transitions produce targets equal to their rewards.
    sampled_batch = Experience(*zip(*experiences))
    rewards = [r.item() for r in sampled_batch.reward]
    for idx, (next_state, done) in enumerate(zip(sampled_batch.next_state, sampled_batch.done)):
        if done or next_state is None:
            assert expected_targets[idx].item() == pytest.approx(rewards[idx])

    random.seed(0)
    np.random.seed(0)
    loss_value = agent.compute_loss()

    assert loss_value is not None
    assert loss_value.item() == pytest.approx(expected_loss.item())
