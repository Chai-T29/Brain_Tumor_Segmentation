import torch

from rl.n_step import NStepAccumulator, StepTuple


def test_flush_preserves_bootstrap_for_truncated_sequences():
    n_step = 3
    gamma = 0.9
    accumulator = NStepAccumulator(n_step=n_step, gamma=gamma, num_envs=1)

    for i in range(2):
        step = StepTuple(
            embedding=torch.tensor([i], dtype=torch.float32),
            polygon=torch.tensor([i + 10], dtype=torch.float32),
            action=torch.tensor([i + 20], dtype=torch.float32),
            reward=torch.tensor([1.0], dtype=torch.float32),
            next_polygon=torch.tensor([i + 30], dtype=torch.float32),
            done=torch.tensor([0.0], dtype=torch.float32),
        )
        accumulator.push(0, step)

    transitions = accumulator.flush(0)
    assert len(transitions) == 2

    _, _, _, reward_acc, next_polygon, done_flag, discount, guidance_target = transitions[0]
    expected_reward = torch.tensor([1.0 + gamma * 1.0], dtype=torch.float32)
    assert torch.allclose(reward_acc, expected_reward)
    assert next_polygon is not None
    assert done_flag.item() == 0.0
    expected_discount = torch.tensor(gamma ** 2, dtype=torch.float32)
    assert torch.allclose(discount, expected_discount)

    _, _, _, _, _, done_flag_last, discount_last, _ = transitions[1]
    assert done_flag_last.item() == 0.0
    assert torch.allclose(discount_last, torch.tensor(gamma, dtype=torch.float32))
