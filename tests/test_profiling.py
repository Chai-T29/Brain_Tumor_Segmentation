import time

from rl.profiling import measure_bottlenecks


def test_measure_bottlenecks_identifies_actor_critic_dominance():
    def encoder_stub():
        time.sleep(0.001)

    def actor_stub():
        time.sleep(0.003)

    def other_stub():
        time.sleep(0.0005)

    result = measure_bottlenecks(encoder_stub, actor_stub, other_stub)

    assert set(result.keys()) == {"encoder_time", "actor_critic_time", "other_time", "bottleneck"}
    assert result["encoder_time"] >= 0.0
    assert result["actor_critic_time"] >= 0.0
    assert result["other_time"] >= 0.0
    assert result["bottleneck"] == "actor_critic_time"

if __name__ == "__main__":
    test_measure_bottlenecks_identifies_actor_critic_dominance()