from collections import defaultdict, deque

import pytest
import torch

from dqn.agent import DQNAgent


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


def _compute_expected_loss(agent, batch, weights):
    device = agent.device
    state_images = batch.state_images.to(device)
    state_bboxes = batch.state_bboxes.to(device)
    actions = batch.actions.to(device).long()
    rewards = batch.rewards.to(device).view(-1)
    next_state_images = batch.next_state_images.to(device)
    next_state_bboxes = batch.next_state_bboxes.to(device)
    dones = batch.dones.to(device).view(-1).bool()
    n_used = batch.n_used.to(device).view(-1)
    weights = weights.to(device).view(-1, 1)

    state_action_values = agent.policy_net(state_images, state_bboxes).gather(1, actions)

    next_state_values = torch.zeros_like(rewards)
    non_final_mask = ~dones
    if non_final_mask.any():
        next_images = next_state_images[non_final_mask]
        next_bboxes = next_state_bboxes[non_final_mask]
        with torch.no_grad():
            q_next_policy = agent.policy_net(next_images, next_bboxes)
            next_actions = q_next_policy.argmax(dim=1, keepdim=True)
            q_next_target = agent.target_net(next_images, next_bboxes)
            q_next_selected = q_next_target.gather(1, next_actions).squeeze(1)
        next_state_values[non_final_mask] = q_next_selected

    gamma_power = agent.gamma ** n_used
    targets = rewards + gamma_power * next_state_values

    loss_elements = torch.nn.functional.smooth_l1_loss(
        state_action_values,
        targets.unsqueeze(1),
        reduction="none",
    )
    loss = (loss_elements * weights).mean()
    return loss, targets


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
        learn_every=1,
        min_buffer_size=1,
        updates_per_step=1,
    )
    agent.n_step = 1
    agent.n_step_buffers = defaultdict(lambda: deque(maxlen=agent.n_step))
    agent.sync_memory_device()

    image = torch.zeros(1, 84, 84, device=device)
    bbox = torch.zeros(4, device=device)
    next_image = torch.ones(1, 84, 84, device=device)
    next_bbox = torch.ones(4, device=device)

    state_images = torch.stack([image, image + 2], dim=0)
    state_bboxes = torch.stack([bbox, bbox + 2], dim=0)
    actions = torch.tensor([1, 0], dtype=torch.long, device=device)
    rewards = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
    next_state_images = torch.stack([next_image, next_image * 0], dim=0)
    next_state_bboxes = torch.stack([next_bbox, next_bbox * 0], dim=0)
    dones = torch.tensor([False, True], dtype=torch.bool, device=device)

    agent.push_experience_batch(
        state_images,
        state_bboxes,
        actions,
        rewards,
        next_state_images,
        next_state_bboxes,
        dones,
    )

    assert len(agent.memory) == 2

    torch.manual_seed(0)
    beta = agent.beta
    batch, _, weights = agent.memory.sample(agent.batch_size, beta=beta)
    expected_loss, expected_targets = _compute_expected_loss(agent, batch, weights)

    # Ensure terminal transitions produce targets equal to their rewards.
    for idx in range(expected_targets.size(0)):
        if batch.dones[idx].item():
            assert expected_targets[idx].item() == pytest.approx(batch.rewards[idx].item())

    agent.beta = beta
    torch.manual_seed(0)
    loss_value = agent.compute_loss()

    assert loss_value is not None
    assert loss_value.item() == pytest.approx(expected_loss.item())
