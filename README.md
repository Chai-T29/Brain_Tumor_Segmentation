# Brain Tumor Localization with DrQ-TD3

This project reframes 2D brain tumor localization as a continuous-control reinforcement learning problem. Instead of iteratively tweaking axis-aligned boxes, the agent manipulates a high-resolution polygon that conforms to the tumor boundary. An EfficientNet encoder produces compact embeddings for each 224×224 slice, and a TD3 learner augmented with DrQ-style stochasticity optimizes the policy directly in embedding space.

## Key Capabilities

- **Frozen EfficientNet Encoder:** A pretrained EfficientNet (configurable) transforms each slice into a fixed embedding vector. Light Gaussian noise is injected into embeddings at runtime to emulate DrQ-v2 regularization without re-encoding pixels.
- **Polygonal Environment:** The environment models a configurable `N`-sided polygon (default 32) whose alternating sides are actively controlled. Each controlled side receives three continuous commands—radial motion, rotation, and length adjustment—plus a stop signal shared across the polygon. IoU with the tumor mask is computed by rasterising the polygon, and rewards are tied to IoU deltas with the original terminal bonuses.
- **TD3 with n-step Targets:** Actor and twin critics share the EfficientNet embeddings and polygon state. Targets incorporate configurable n-step returns, Polyak averaging, target policy smoothing, and delayed policy updates.
- **Flexible Data Pipeline:** A Lightning `DataModule` performs one-time memmap caching of slices, optional inclusion of tumor-free samples, and returns per-sample metadata so raw images can be fetched on demand for visualization.
- **Config-Driven Training:** All tunable hyperparameters (encoder, environment geometry, RL algorithm, replay buffer, update cadence, logging) live in `config.yaml`, keeping experiments reproducible.
- **Validation & GIF-producing Test Loops:** Deterministic validation/test rollouts reuse the same simulator, and the test loop records configurable GIFs that overlay the polygon trajectory on MRI slices for qualitative analysis.

## Project Layout

```
Brain_Tumor_Segmentation/
├── config.yaml                  # Central configuration for data, encoder, environment, and training
├── data/
│   ├── data_module.py           # Lightning DataModule with memmap caching and normalisation
│   └── dataset.py               # Dataset returning slice tensors + metadata
├── rl/
│   ├── agent.py                 # TD3 agent (actor/critics, optimisation logic)
│   ├── encoder.py               # EfficientNet wrapper with configurable noise
│   ├── environment.py           # Polygon-based localisation environment
│   ├── lightning_module.py      # Lightning wrapper orchestrating collection and optimisation
│   ├── n_step.py                # n-step transition accumulator per environment
│   ├── networks.py              # Actor and critic network definitions
│   └── replay_buffer.py         # CPU replay buffer storing embeddings and polygon states
├── train_dqn.py                 # Entry point for training (now DrQ-TD3)
├── test_dqn.py                  # Deterministic evaluation on the test split
└── tests/                       # Updated unit tests
```

## Getting Started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the dataset**

   Download the MU-Glioma-Post collection from TCIA and place it under `MU-Glioma-Post/`, preserving the patient/timepoint folder structure. The data module handles slice extraction and caching.

3. **Configure the run**

   Edit `config.yaml` to adjust:

   - `encoder`: EfficientNet variant, noise scale, trainable flag.
   - `environment`: polygon side count, step magnitudes, IoU threshold, reward shaping.
   - `algorithm`: TD3 hyperparameters, n-step horizon, replay capacity.
   - `training`: batch sizes, update cadence, warm-up steps, precision, logging frequency.

## Training

Launch training with:

```bash
python train_dqn.py
```

PyTorch Lightning handles checkpointing (`train/final_iou` as the monitor) and TensorBoard logging. The replay buffer remains on CPU, while the actor/critics run on the selected accelerator. Multiple critic updates per collection batch are triggered according to `update_every_n_steps`.

## Evaluation

Evaluate the latest checkpoint on the test split:

```bash
python test_dqn.py
```

The script loads the most recent checkpoint, rolls out deterministic policies (no exploration or embedding noise), reports success rate, mean IoU, and average steps per slice, and stores GIFs (up to `logging.test_gif_limit`) in the configured directory.

## Configuration Highlights

- **Polygon geometry:** `environment.num_sides` (default 32) determines control dimensionality (`(num_sides/2)*3 + 1`). Scales for radial, rotational, and length adjustments keep actions interpretable.
- **n-step targets:** `algorithm.n_step` (default 3) matches the replay accumulator. Discounting uses `gamma ** n` for non-terminal transitions.
- **Embedding noise:** Both the encoder (`encoder.embedding_noise_std`) and agent (`algorithm.embedding_noise_std`) can inject Gaussian noise, enabling DrQ-style regularisation without image augmentations.
- **Training cadence:** `training.collect_steps_per_batch` limits how many environment steps are gathered per loader batch, while `update_every_n_steps` and `update_batch_size` govern the number of critic updates run afterwards.
- **Optimiser control:** Separate learning rates (`algorithm.actor_lr`, `algorithm.critic_lr`) and network widths (`algorithm.actor_hidden_sizes`, `algorithm.critic_hidden_sizes`) are exposed, so policy and critic capacity can be tuned alongside their optimisers.

## Authors

1. Chaitanya Tatipigari
2. Suraj Godithi

## References

1. Yarats, D., Kostrikov, I., & Fergus, R. “Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels.” *ICLR*, 2021.
2. Fujimoto, S., van Hoof, H., & Meger, D. “Addressing Function Approximation Error in Actor-Critic Methods.” *ICML*, 2018.
3. Tan, M. & Le, Q. V. “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.” *ICML*, 2019.
4. Yaseen, D. et al. “University of Missouri Post-operative Glioma Dataset (MU-Glioma-Post).” *The Cancer Imaging Archive*, 2025.
