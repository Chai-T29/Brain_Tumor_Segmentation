from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


class TumorLocalizationEnv:
    """Vectorized environment for batched tumor localization."""

    _STOP_ACTION = 8

    def __init__(
        self,
        max_steps: int = 100,
        iou_threshold: float = 0.8,
        resize_shape: Tuple[int, int] = (84, 84),
        step_size: float = 10.0,
        scale_factor: float = 1.1,
        min_bbox_size: float = 10.0,
    ) -> None:
        self.max_steps = max(1, int(max_steps))
        self.iou_threshold = float(iou_threshold)
        self.resize_shape = resize_shape
        self.step_size = float(step_size)
        self.scale_factor = float(scale_factor)
        self.min_bbox_size = float(min_bbox_size)

        self.images: Optional[torch.Tensor] = None
        self.masks: Optional[torch.Tensor] = None
        self.gt_bboxes: Optional[torch.Tensor] = None
        self.current_bboxes: Optional[torch.Tensor] = None
        self.last_iou: Optional[torch.Tensor] = None
        self.current_step: Optional[torch.Tensor] = None
        self.active_mask: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        if self.images is None:
            return torch.device("cpu")
        return self.images.device

    def reset(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialise a new batch of environments for the provided data."""

        if images.dim() != 4 or masks.dim() != 4:
            raise ValueError("Images and masks must be batched tensors with shape [B, C, H, W].")
        if images.size(0) != masks.size(0):
            raise ValueError("Images and masks must have the same batch size.")

        self.images = images.detach()
        self.masks = masks.detach()
        batch_size = images.size(0)
        height = images.size(-2)
        width = images.size(-1)

        self.gt_bboxes = self._get_bbox_from_mask(self.masks)
        self.current_bboxes = self._initialise_bboxes(batch_size, height, width)
        self.last_iou = self._calculate_iou(self.current_bboxes, self.gt_bboxes)
        self.current_step = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        self.active_mask = torch.ones(batch_size, device=self.device, dtype=torch.bool)

        return self._get_state()

    def step(self, actions: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.images is None or self.masks is None:
            raise RuntimeError("Environment must be reset before stepping.")
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)
        if actions.dim() != 1:
            raise ValueError("Actions must be a 1D tensor of shape [batch_size].")
        if actions.size(0) != self.images.size(0):
            raise ValueError("Action batch size must match the environment batch size.")

        actions = actions.to(self.device)
        prev_active = self.active_mask.clone()
        actions = torch.where(prev_active, actions, torch.full_like(actions, self._STOP_ACTION))

        self.current_step = self.current_step + prev_active.long()
        self.current_bboxes = self._apply_action(actions, prev_active)

        current_iou = self._calculate_iou(self.current_bboxes, self.gt_bboxes)
        rewards = torch.where(prev_active, current_iou - self.last_iou, torch.zeros_like(current_iou))

        stop_mask = (actions == self._STOP_ACTION) & prev_active
        success_mask = (current_iou > self.iou_threshold) & prev_active
        timeout_mask = (self.current_step >= self.max_steps) & prev_active

        rewards = rewards + torch.where(stop_mask, torch.where(success_mask, torch.ones_like(rewards), -torch.ones_like(rewards)), torch.zeros_like(rewards))
        rewards = rewards + torch.where(success_mask & ~stop_mask, torch.ones_like(rewards), torch.zeros_like(rewards))

        done = stop_mask | success_mask | timeout_mask
        self.active_mask = prev_active & ~done
        self.last_iou = torch.where(prev_active, current_iou, self.last_iou)

        info = {"iou": current_iou.detach()}
        next_state = self._get_state()
        return next_state, rewards.detach(), done.detach(), info

    def render(self, index: int = 0, mode: str = "human"):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        if self.images is None or self.current_bboxes is None or self.gt_bboxes is None:
            raise RuntimeError("Environment must be reset before rendering.")
        if not 0 <= index < self.images.size(0):
            raise IndexError("Render index out of range for current batch.")

        image = self.images[index].detach().cpu()
        image_np = image.permute(1, 2, 0).numpy()
        min_val = float(image_np.min())
        max_val = float(image_np.max())
        if max_val > min_val:
            image_np = (image_np - min_val) / (max_val - min_val)

        fig, ax = plt.subplots(1)
        ax.imshow(image_np, cmap="gray")

        gt_x, gt_y, gt_w, gt_h = self.gt_bboxes[index].detach().cpu().tolist()
        agent_x, agent_y, agent_w, agent_h = self.current_bboxes[index].detach().cpu().tolist()

        gt_rect = patches.Rectangle((gt_x, gt_y), gt_w, gt_h, linewidth=2, edgecolor="g", facecolor="none", label="Ground Truth")
        agent_rect = patches.Rectangle((agent_x, agent_y), agent_w, agent_h, linewidth=2, edgecolor="r", facecolor="none", label="Agent")
        ax.add_patch(gt_rect)
        ax.add_patch(agent_rect)
        ax.legend()

        if mode == "human":
            plt.show()
            plt.close(fig)
            return None

        if mode == "rgb_array":
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            data = np.asarray(buf)
            plt.close(fig)
            return data[:, :, :3]

        raise ValueError(f"Unsupported render mode: {mode}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.images is None or self.current_bboxes is None:
            raise RuntimeError("Environment must be reset before retrieving state.")
        resized_images = F.interpolate(self.images, size=self.resize_shape, mode="bilinear", align_corners=False)
        return resized_images.detach(), self.current_bboxes.detach()

    def _initialise_bboxes(self, batch_size: int, height: int, width: int) -> torch.Tensor:
        min_w = max(1, width // 4)
        max_w = max(min_w + 1, width // 2)
        min_h = max(1, height // 4)
        max_h = max(min_h + 1, height // 2)

        widths = torch.randint(min_w, max_w, (batch_size,), device=self.device, dtype=torch.long).to(torch.float32)
        heights = torch.randint(min_h, max_h, (batch_size,), device=self.device, dtype=torch.long).to(torch.float32)
        widths = torch.clamp(widths, min=self.min_bbox_size, max=float(width))
        heights = torch.clamp(heights, min=self.min_bbox_size, max=float(height))

        max_x = torch.clamp(torch.tensor(width, device=self.device, dtype=torch.float32) - widths, min=0.0)
        max_y = torch.clamp(torch.tensor(height, device=self.device, dtype=torch.float32) - heights, min=0.0)
        xs = torch.rand(batch_size, device=self.device) * max_x
        ys = torch.rand(batch_size, device=self.device) * max_y

        return torch.stack((xs, ys, widths, heights), dim=1)

    def _calculate_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        if boxes1.size() != boxes2.size():
            raise ValueError("Boxes must have the same shape for IoU calculation.")

        x1, y1, w1, h1 = boxes1.unbind(dim=1)
        x2, y2, w2, h2 = boxes2.unbind(dim=1)

        x1_end = x1 + w1
        y1_end = y1 + h1
        x2_end = x2 + w2
        y2_end = y2 + h2

        inter_w = torch.clamp(torch.min(x1_end, x2_end) - torch.max(x1, x2), min=0.0)
        inter_h = torch.clamp(torch.min(y1_end, y2_end) - torch.max(y1, y2), min=0.0)
        inter_area = inter_w * inter_h

        area1 = torch.clamp(w1, min=0.0) * torch.clamp(h1, min=0.0)
        area2 = torch.clamp(w2, min=0.0) * torch.clamp(h2, min=0.0)
        union = area1 + area2 - inter_area

        iou = torch.where(union > 0, inter_area / union, torch.zeros_like(union))
        return iou

    def _apply_action(self, actions: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        if self.current_bboxes is None or self.images is None:
            raise RuntimeError("Environment must be reset before applying actions.")

        x, y, w, h = self.current_bboxes.unbind(dim=1)
        width_limit = torch.tensor(self.images.size(-1), device=self.device, dtype=torch.float32)
        height_limit = torch.tensor(self.images.size(-2), device=self.device, dtype=torch.float32)

        step = torch.full_like(x, self.step_size)
        scale = torch.full_like(x, self.scale_factor)
        min_size = torch.full_like(x, self.min_bbox_size)

        move_up = (actions == 0) & active_mask
        move_down = (actions == 1) & active_mask
        move_left = (actions == 2) & active_mask
        move_right = (actions == 3) & active_mask
        expand_w = (actions == 4) & active_mask
        shrink_w = (actions == 5) & active_mask
        expand_h = (actions == 6) & active_mask
        shrink_h = (actions == 7) & active_mask

        y_candidate = torch.clamp(y - step, min=0.0)
        y = torch.where(move_up, y_candidate, y)

        y_candidate = torch.clamp(y + step, max=torch.clamp(height_limit - h, min=0.0))
        y = torch.where(move_down, y_candidate, y)

        x_candidate = torch.clamp(x - step, min=0.0)
        x = torch.where(move_left, x_candidate, x)

        x_candidate = torch.clamp(x + step, max=torch.clamp(width_limit - w, min=0.0))
        x = torch.where(move_right, x_candidate, x)

        w_candidate = torch.clamp(w * scale, max=width_limit)
        w = torch.where(expand_w, w_candidate, w)

        w_candidate = torch.clamp(w / scale, min=min_size)
        w = torch.where(shrink_w, w_candidate, w)

        h_candidate = torch.clamp(h * scale, max=height_limit)
        h = torch.where(expand_h, h_candidate, h)

        h_candidate = torch.clamp(h / scale, min=min_size)
        h = torch.where(shrink_h, h_candidate, h)

        max_x = torch.clamp(width_limit - w, min=0.0)
        max_y = torch.clamp(height_limit - h, min=0.0)

        zero_x = torch.zeros_like(x)
        zero_y = torch.zeros_like(y)

        x = torch.clamp(x, min=zero_x, max=max_x)
        y = torch.clamp(y, min=zero_y, max=max_y)
        
        return torch.stack((x, y, w, h), dim=1)

    def _get_bbox_from_mask(self, masks: torch.Tensor) -> torch.Tensor:
        batch_size = masks.size(0)
        boxes = torch.zeros(batch_size, 4, device=self.device, dtype=torch.float32)
        mask_binary = masks > 0
        for index in range(batch_size):
            coords = torch.nonzero(mask_binary[index], as_tuple=False)
            if coords.numel() == 0:
                continue
            ys = coords[:, -2].float()
            xs = coords[:, -1].float()
            x_min = xs.min()
            x_max = xs.max()
            y_min = ys.min()
            y_max = ys.max()
            width = torch.clamp(x_max - x_min + 1.0, min=self.min_bbox_size)
            height = torch.clamp(y_max - y_min + 1.0, min=self.min_bbox_size)
            boxes[index] = torch.tensor([x_min, y_min, width, height], device=self.device, dtype=torch.float32)
        return boxes

