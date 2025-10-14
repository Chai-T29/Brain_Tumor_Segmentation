from __future__ import annotations

import time
from typing import Callable, Dict, Optional


def measure_bottlenecks(
    encoder_step: Callable[[], None],
    actor_critic_step: Callable[[], None],
    other_step: Optional[Callable[[], None]] = None,
) -> Dict[str, float]:
    """Profile high-level components of a training iteration.

    Returns a dictionary containing wall-clock seconds spent in each component
    and a ``bottleneck`` key indicating which component dominated runtime.
    """

    def _measure(fn: Callable[[], None]) -> float:
        start = time.perf_counter()
        fn()
        return time.perf_counter() - start

    encoder_time = _measure(encoder_step)
    actor_time = _measure(actor_critic_step)
    other_time = _measure(other_step) if other_step is not None else 0.0

    timings = {
        "encoder_time": encoder_time,
        "actor_critic_time": actor_time,
        "other_time": other_time,
    }
    bottleneck = max(timings, key=timings.get)
    timings["bottleneck"] = bottleneck
    return timings
