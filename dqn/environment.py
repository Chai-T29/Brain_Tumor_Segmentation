import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torchvision.transforms import transforms

def calculate_iou(box1, box2):
    """Calculates Intersection over Union for two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_inter = max(x1, x2)
    y_inter = max(y1, y2)
    x_inter_end = min(x1 + w1, x2 + w2)
    y_inter_end = min(y1 + h1, y2 + h2)

    inter_area = max(0, x_inter_end - x_inter) * max(0, y_inter_end - y_inter)

    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou

class TumorLocalizationEnv(gym.Env):
    """A custom Gym environment for tumor localization."""
    def __init__(self, dataset, max_steps=100, iou_threshold=0.8):
        super(TumorLocalizationEnv, self).__init__()

        self.dataset = dataset
        self.max_steps = max_steps
        self.iou_threshold = iou_threshold

        self.action_space = spaces.Discrete(9) # up, down, left, right, expand_h, shrink_h, expand_v, shrink_v, stop
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=1, shape=(3, 84, 84), dtype=np.float32),
            spaces.Box(low=0, high=240, shape=(4,), dtype=np.float32) # x, y, w, h
        ))
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
        ])

    def reset(self):
        self.current_sample_idx = np.random.randint(0, len(self.dataset))
        sample = self.dataset[self.current_sample_idx]
        self.image = sample['image']
        self.mask = sample['mask']

        self.gt_bbox = self._get_bbox_from_mask(self.mask)

        img_height, img_width = self.image.shape[1:]
        w = np.random.randint(img_width // 4, img_width // 2)
        h = np.random.randint(img_height // 4, img_height // 2)
        x = np.random.randint(0, img_width - w)
        y = np.random.randint(0, img_height - h)
        self.current_bbox = (x, y, w, h)

        self.current_step = 0
        self.last_iou = calculate_iou(self.current_bbox, self.gt_bbox)

        return self._get_state()

    def step(self, action):
        self.current_step += 1
        self._apply_action(action)

        iou = calculate_iou(self.current_bbox, self.gt_bbox)
        reward = iou - self.last_iou
        self.last_iou = iou

        done = False
        if action == 8: # Stop action
            done = True
            reward += 1.0 if iou > self.iou_threshold else -1.0

        if self.current_step >= self.max_steps or iou > self.iou_threshold:
            done = True
            if iou > self.iou_threshold:
                reward += 1.0

        return self._get_state(), reward, done, {'iou': iou}

    def _get_state(self):
        resized_image = self.transform(self.image)
        bbox_tensor = torch.tensor(self.current_bbox, dtype=torch.float32)
        return (resized_image, bbox_tensor)

    def _get_bbox_from_mask(self, mask):
        mask_np = mask.squeeze(0).numpy()
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        if not np.any(rows) or not np.any(cols):
            return (0, 0, 0, 0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return (xmin, ymin, xmax - xmin, ymax - ymin)

    def _apply_action(self, action):
        x, y, w, h = self.current_bbox
        img_height, img_width = self.image.shape[1:]
        step_size = 10
        scale_factor = 1.1

        if action == 0: # Move up
            y = max(0, y - step_size)
        elif action == 1: # Move down
            y = min(img_height - h, y + step_size)
        elif action == 2: # Move left
            x = max(0, x - step_size)
        elif action == 3: # Move right
            x = min(img_width - w, x + step_size)
        elif action == 4: # Expand horizontally
            w = min(img_width, int(w * scale_factor))
        elif action == 5: # Shrink horizontally
            w = max(10, int(w / scale_factor))
        elif action == 6: # Expand vertically
            h = min(img_height, int(h * scale_factor))
        elif action == 7: # Shrink vertically
            h = max(10, int(h / scale_factor))
        # action 8 is stop

        self.current_bbox = (x, y, w, h)

    def render(self, mode='human'):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        fig, ax = plt.subplots(1)
        image_to_render = self.image.permute(1, 2, 0).cpu().numpy()
        min_val = image_to_render.min()
        max_val = image_to_render.max()
        if max_val > min_val:
            image_to_render = (image_to_render - min_val) / (max_val - min_val)
        ax.imshow(image_to_render)

        # Draw ground truth bbox
        x, y, w, h = self.gt_bbox
        gt_rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none', label='Ground Truth')
        ax.add_patch(gt_rect)

        # Draw agent's bbox
        x, y, w, h = self.current_bbox
        agent_rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none', label='Agent')
        ax.add_patch(agent_rect)
        
        plt.legend()

        if mode == 'human':
            plt.show()
            plt.close(fig)
        elif mode == 'rgb_array':
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            data = np.asarray(buf)
            plt.close(fig)
            return data[:, :, :3]

    def close(self):
        pass