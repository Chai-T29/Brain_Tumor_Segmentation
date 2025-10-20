# Guidance Updates Deep Dive

This note explains two waves of changes to the guidance system that drives the TD3 agent. The first wave laid the groundwork for curriculum-like guidance using schedules, supervised actor loss, and richer replay data. The second wave adds the ability to shake up that schedule with controlled randomness so the agent keeps seeing difficult situations. The goal is to show how each code change connects to the training loop rather than listing edits in isolation.

## Wave 1: Structured Guidance (commit: `Add guidance schedule support and dataset pickling`)

### 1. Cleaning up configuration files
Earlier YAML files mixed quoted strings and scientific notation formats, which made diffs noisy and sometimes confused Lightning when parsing. We normalized entries such as `'MU-Glioma-Post/'` to `MU-Glioma-Post/` and `1.0e-6` to `1.0e-06`. That makes it easier to generate configs programmatically and removes parsing surprises. More importantly, the configs now expose new fields under `algorithm`:

- `guided_actor_loss` and `guided_actor_loss_weight` toggle a supervised loss that nudges the actor toward labeled targets when they exist.
- `guidance_schedule` defines how strong that guidance should start and how fast it should fade.
- `actor_hidden_sizes` and `critic_hidden_sizes` are spelled out explicitly so experiments can easily swap network widths.

We also reduced the dataloader footprint (fewer workers, lower prefetch) and pushed embeddings to CPU to cut down on out-of-memory errors encountered during guided runs on smaller GPUs.

### 2. Making the dataset picklable
`BrainTumorDataset` now defines `__getstate__` and `__setstate__`. When Lightning forks worker processes, it pickles the dataset instance. Without these methods the memory-mapped cache (`_mm_cache`) could not be serialized cleanly, leading to crashes. The new implementation strips the cache before pickling and recreates it on the other side. That change is isolated to `data/dataset.py` but it unblocks multi-worker loading, which is essential once guidance enables larger batches.

### 3. Threading guidance through the RL pipeline
This was the biggest slice of work:

1. `GuidanceScheduleConfig` was introduced and exported from `rl/__init__.py` so configs can reference it directly.
2. `TD3Agent` reads that schedule and keeps track of interaction counts. When you call `act`, it now returns either the final action or a tuple of `(guided_action, base_action)` when `return_base_action=True`. That tuple is important for comparing what the policy wanted to do versus what guidance made it do.
3. During acting, the agent optionally runs a guided exploration pass. If `true_guided_exploration` is enabled, it blends the actor output with a target polygon that the environment provides. If `guided_exploration` is enabled, it uses the critic gradient to steer noisy actions toward higher-value regions.
4. The Lightning module computes those target polygons by comparing the current polygon to the ground truth (`_compute_true_guidance_targets`). It stores the targets in the `NStepAccumulator`, which now carries an extra `guidance_target` field. When enough steps accumulate, that target flows into the `Transition` dataclass and finally into the replay buffer arrays.
5. The replay buffer saves both the targets and a boolean mask that tells the actor loss which samples actually have usable supervision (for example, actions that depend on the target polygon versus those that do not).
6. When the agent updates, it checks `guidance_mask`. If any entries are true, it computes a mean squared error between actor outputs and guidance targets, scales that loss by `guided_actor_loss_weight`, and optionally shrinks it as the guidance schedule decays. This way early training is heavily supervised, while later training shifts to pure TD3 behavior.

By passing guidance targets all the way from environment to replay buffer, the agent can mix reinforcement and supervised learning seamlessly.

### 4. Guarding the feature with tests
Three tests validate the new behavior:

- `tests/test_dataset.py::test_dataset_picklable` stress-tests the new pickling path by round-tripping the dataset.
- `tests/test_n_step.py` ensures the n-step accumulator returns the new `guidance_target` alongside the usual fields without breaking discount logic.
- `tests/test_td3_agent.py` verifies that replay samples now include `guidance_target` and `guidance_mask`, that `return_base_action=True` behaves as expected, and that guidance targets line up with the action layout (distance, angle, center, stop components).

In short, the first wave brought deterministic guidance into the loop and ensured every layer of the stack understood the additional data.

## Wave 2: Randomized Guidance (current changes)

The deterministic schedule above works, but once the guidance fades the agent may struggle because most replay entries came from 'good' actions. The new changes let you inject randomness into the guidance scale to maintain a healthy mix of easy and hard experiences.

### 1. Extending the configuration surface
`GuidanceScheduleConfig` now contains additional fields:

- `randomize`: master switch controlling whether randomness is active.
- `distribution`: what distribution to sample from (`beta` or `uniform`, with `beta` as default).
- `alpha` and `beta`: parameters for the Beta distribution; larger `alpha` relative to `beta` biases samples toward 1.0, and vice versa.
- `blend`: linear interpolation factor between the scheduled value and the sampled value. `blend=0.5` means the final guidance scale is the average of schedule and sample. Values closer to 1 keep you near the schedule.
- `min_scale` and `max_scale`: clamps that keep guidance within safe bounds, so you can avoid ever hitting zero if that destabilizes learning.

`base_config.yaml` leaves `randomize` off so the default behavior is unchanged. `config.yaml` turns it on with a Beta(5, 2) distribution, which produces values skewed toward the high end but still variable.

### 2. Updating the agent logic
Inside `TD3Agent` we split the old `_current_guidance_scale` helper into two steps:

1. `_scheduled_guidance_scale` computes the deterministic value by walking along the linear schedule and clamping it to `[min_scale, max_scale]`.
2. `_apply_guidance_randomization` draws a sample, blends it with the schedule, re-applies the clamp, and returns the result.

`_current_guidance_scale(randomize=True)` stitches those together. The `act` method calls it without arguments, so it receives randomized values when enabled. The actor-loss path calls `_current_guidance_scale(randomize=False)` so the supervised weighting keeps following the clean schedule instead of the random draw. We also store the last guidance scale in `self._last_guidance_scale` and expose it via the `metrics` dict, which means you can plot the actual draw frequency in TensorBoard.

### 3. Testing the randomization path
`tests/test_td3_agent.py::test_guidance_scale_randomization_beta` seeds PyTorch, samples the randomized scale, then re-seeds and samples the deterministic schedule. The assertion confirms the randomized value stays between 0 and 1, matches the schedule when randomization is disabled, and actually deviates (in the seeded case the random value is larger, demonstrating the blend effect).

### 4. Putting it all together during training
When you launch training with `config.yaml` now, each interaction step proceeds as follows:

1. The agent reads the deterministic schedule (e.g., 0.65 halfway through a 0.7 -> 0.0 schedule).
2. It draws a Beta sample (say 0.9) and blends them: `0.5 * 0.65 + 0.5 * 0.9 = 0.775`.
3. That value scales true-guided exploration and critic-guided exploration for the actions executed in the environment.
4. The same step logs `guidance_scale=0.775` so you can inspect the distribution later.
5. When the actor updates, it still uses the 0.65 deterministic value to scale the supervised loss, ensuring the random spike does not mute learning.

As the schedule decays, the random samples produce a mix of near-zero and mid-range guidance levels, which encourages the agent to explore states it previously avoided while still leveraging the deterministic curriculum.

## How to experiment safely
1. **Reproduce a deterministic baseline**: run training with `randomize: false` to confirm the code still behaves like the first wave.
2. **Enable randomization gradually**: start with `blend=0.8` or a Beta distribution that looks more uniform so randomness is gentle. Watch the logged `guidance_scale` histograms.
3. **Adjust `alpha`/`beta`**: if you want more low-guidance experience early on, set `alpha < beta`. If you prefer to keep guidance high but not constant, choose larger `alpha` with moderate `blend`.
4. **Monitor replay quality**: compare success rates or Q-value estimates before and after enabling randomness. If the agent destabilizes, double-check `min_scale`/`max_scale` to keep values away from extremes.

With these tools you can modulate how much the agent relies on guidance versus raw exploration, and you have logging, configuration, and tests to support the workflow.
