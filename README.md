# Brain Tumor Localization with Deep Q-Networks (DQN)

This project focuses on localizing brain tumors in MRI scans using a Deep Q-Network (DQN). The DQN frames the localization problem as a reinforcement learning task, where an intelligent agent learns to iteratively adjust a bounding box on 2D MRI slices to accurately pinpoint tumor locations.

## Project Goals

-   **DQN Localization:** To train an intelligent agent that can localize brain tumors by iteratively adjusting a bounding box on 2D MRI slices, framed as a reinforcement learning problem.
-   **Accurate Tumor Pinpointing:** To achieve precise localization of tumor regions within MRI scans.

## Features

### Deep Q-Network (DQN)
-   **Reinforcement Learning Agent:** Implements a DQN agent capable of learning optimal strategies for tumor localization.
-   **Custom Environment:** Features a custom Gym-like environment (`TumorLocalizationEnv`) where the agent interacts with MRI slices.
    -   **State:** Comprises the 2D MRI slice and the current bounding box coordinates.
    -   **Action Space:** Allows the agent to move the bounding box (up, down, left, right), resize it (expand/shrink horizontally, expand/shrink vertically), or stop.
    -   **Slice-aware Ground Truth:** Ground-truth bounding boxes are computed directly from each 2D tumor mask, ensuring that no bounding box is produced when a slice is tumor-free and preventing false positives inherited from neighbouring slices.
    -   **Reward System:** The reward signal follows modern localization research practices—scaled IoU improvement encourages precise refinement, decisive stopping on tumor-free slices is rewarded, and terminal decisions are strongly reinforced or penalized depending on whether the IoU exceeds the success threshold.
-   **Replay Memory:** Incorporates a replay memory for experience storage and sampling, crucial for stable DQN training.
-   **PyTorch Lightning Integration:** The DQN agent and its training loop are encapsulated within a `LightningModule` for organized and efficient training.
-   **Checkpointing:** Automatically saves model checkpoints based on validation loss, allowing for easy resumption of training or evaluation of the best-performing models.
-   **Visualization:** Generates GIF visualizations of the agent's localization process during testing, showing the bounding box adjustments over time.

### Reward Shaping Strategy

-   **IoU-driven Refinement:** Each active step is rewarded in proportion to the improvement in IoU (scaled by a factor of 2) so that the agent focuses on steady, high-quality localization progress.
-   **Tumor-aware Termination:** Stopping on a tumor-free slice yields a strong positive reward, while continuing to act accrues a mild penalty to encourage quick rejection of negatives.
-   **Precision Stopping:** When a tumor is present, the agent receives a large terminal reward for stopping once the IoU surpasses the success threshold and is penalized for stopping early, reinforcing state-of-the-art localization practices that prioritise confident, high-overlap detections.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Brain_Tumor_Segmentation.git
    cd Brain_Tumor_Segmentation
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The project utilizes the **University of Missouri Post-operative Glioma Dataset (MU-Glioma-Post)**, which can be accessed from The Cancer Imaging Archive (TCIA) at [https://www.cancerimagingarchive.net/collection/mu-glioma-post/](https://www.cancerimagingarchive.net/collection/mu-glioma-post/).

The project expects MRI data in NIfTI (`.nii.gz`) format, organized in a specific directory structure. After downloading, the `MU-Glioma-Post` directory should be placed in the project root, and it should contain your patient data as follows:

```
MU-Glioma-Post/
├── PatientID_0001/
│   ├── Timepoint_1/
│   │   ├── PatientID_0001_Timepoint_1_brain_t1c.nii.gz  (Image)
│   │   ├── PatientID_0001_Timepoint_1_tumorMask.nii.gz (Mask)
│   │   └── ... (other MRI sequences)
│   └── Timepoint_2/
│       ├── PatientID_0001_Timepoint_2_brain_t1c.nii.gz
│       ├── PatientID_0001_Timepoint_2_tumorMask.nii.gz
│       └── ...
└── PatientID_0002/
    └── ...
```

-   **Image Files:** Should contain `_brain_t1c.nii.gz` in their filename.
-   **Mask Files:** Should contain `_tumorMask.nii.gz` in their filename.

## Usage

### DQN Training

To train the Deep Q-Network agent for tumor localization:

```bash
python -m dqn.train_dqn
```
You can adjust hyperparameters like `MAX_EPOCHS` (number of episodes), `BATCH_SIZE`, `LEARNING_RATE`, etc., in `dqn/train_dqn.py`.

### DQN Testing and Visualization

To test the trained DQN agent and generate a GIF visualization of its performance:

```bash
python -m dqn.test_dqn
```
This script will automatically find the best-performing checkpoint based on validation loss and save a GIF named `dqn_test_episode.gif` in the project root, showcasing the agent's bounding box adjustments on a randomly selected validation image.

## Project Structure

```
Brain_Tumor_Segmentation/
├── data/
│   └── dataset.py             # Custom PyTorch Dataset for NIfTI data
├── dqn/
│   ├── agent.py               # Implements the DQN agent logic
│   ├── environment.py         # Custom Gym-like environment for tumor localization
│   ├── lightning_model.py     # PyTorch Lightning module for DQN training
│   ├── model.py               # Q-Network architecture for DQN
│   ├── replay_memory.py       # Replay memory for DQN experience storage
│   └── train_dqn.py           # Script to train the DQN agent
│   └── test_dqn.py            # Script to test the DQN agent and generate video
├── lightning_logs/            # Directory for PyTorch Lightning logs and checkpoints
├── MU-Glioma-Post/            # Directory for your dataset (as described above)
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## Authors

-   Suraj Godithi
-   Chaitanya Tatipigari

## Key Technologies Used

-   **Python**
-   **PyTorch**
-   **PyTorch Lightning**
-   **NumPy**
-   **Pandas**
-   **NiBabel:** For handling NIfTI image files.
-   **Gymnasium:** For creating the reinforcement learning environment.
-   **Imageio:** For generating GIF visualizations.
-   **Matplotlib:** For plotting and rendering visualizations.
-   **Tqdm:** For progress bars.

## Future Work

-   **Advanced DQN Techniques:** Explore Multiple Agents, Prioritized Experience Replay, or Dueling DQN for improved performance.
-   **3D Localization:** Extend the DQN to operate on 3D MRI volumes directly.
-   **More Robust Evaluation Metrics:** Implement additional metrics for both segmentation and localization.

