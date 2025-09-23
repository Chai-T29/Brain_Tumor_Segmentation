from typing import ClassVar, Dict, Optional, Tuple
import torch
import torch.nn.functional as F


class TumorLocalizationEnv:
    """Vectorized environment for batched tumor localization."""

    REWARD_CLIP_RANGE: ClassVar[Tuple[float, float]] = (-6.0, 6.0)
    STOP_REWARD_SUCCESS: ClassVar[float] = 4.0
    STOP_REWARD_NO_TUMOR: ClassVar[float] = 2.0
    STOP_REWARD_FALSE: ClassVar[float] = -3.0
    TIME_PENALTY: ClassVar[float] = 0.01
    HOLD_PENALTY: ClassVar[float] = 0.5

    _STOP_ACTION = 8

    def __init__(
        self,
        max_steps: int = 100,
        iou_threshold: float = 0.8,
        resize_shape: Tuple[int, int] = (84, 84),
        step_size: float = 10.0,
        scale_factor: float = 1.1,
        min_bbox_size: float = 10.0,
        initial_mode: str = "full_image",
        initial_margin: float = 8.0,
    ) -> None:
        self.max_steps = max(1, int(max_steps))
        self.iou_threshold = float(iou_threshold)
        self.resize_shape = resize_shape
        self.step_size = float(step_size)
        self.scale_factor = float(scale_factor)
        self.min_bbox_size = float(min_bbox_size)
        self.initial_mode = initial_mode
        if self.initial_mode not in {"full_image", "gt_margin"}:
            raise ValueError("initial_mode must be either 'full_image' or 'gt_margin'.")
        self.initial_margin = float(initial_margin)
        if self.initial_margin < 0.0:
            raise ValueError("initial_margin must be non-negative.")

        self.images: Optional[torch.Tensor] = None
        self.masks: Optional[torch.Tensor] = None
        self.gt_bboxes: Optional[torch.Tensor] = None
        self.current_bboxes: Optional[torch.Tensor] = None
        self.current_bboxes_unscaled: Optional[torch.Tensor] = None
        self.last_iou: Optional[torch.Tensor] = None
        self.current_step: Optional[torch.Tensor] = None
        self.active_mask: Optional[torch.Tensor] = None
        self.has_tumor: Optional[torch.Tensor] = None
        self._threshold_reached: Optional[torch.Tensor] = None
        self._original_height: Optional[float] = None
        self._original_width: Optional[float] = None
        self._scale_x: Optional[float] = None
        self._scale_y: Optional[float] = None
        self._resized_images: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        if self.images is None:
            return torch.device("cpu")
        return self.images.device

    def reset(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if images.dim() != 4 or masks.dim() != 4:
            raise ValueError("Images and masks must be batched tensors with shape [B, C, H, W].")
        if images.size(0) != masks.size(0):
            raise ValueError("Images and masks must have the same batch size.")

        self.images = images.detach()
        self.masks = masks.detach()
        self._resized_images = F.interpolate(
            self.images,
            size=self.resize_shape,
            mode="bilinear",
            align_corners=False,
        ).detach()
        batch_size = images.size(0)
        height = images.size(-2)
        width = images.size(-1)

        self.gt_bboxes, self.has_tumor = self._get_bbox_from_mask(self.masks)

        self._original_height = float(height)
        self._original_width = float(width)
        self._scale_y = self.resize_shape[0] / self._original_height
        self._scale_x = self.resize_shape[1] / self._original_width

        self.current_bboxes_unscaled = self._initialise_bboxes(height, width)
        self.current_bboxes = self._scale_bboxes_to_resize(self.current_bboxes_unscaled)
        self.last_iou = self._calculate_iou(self.current_bboxes_unscaled, self.gt_bboxes)

        self.current_step = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        self.active_mask = torch.ones(batch_size, device=self.device, dtype=torch.bool)
        self._threshold_reached = torch.zeros(batch_size, device=self.device, dtype=torch.bool)

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

        if self._threshold_reached is None:
            raise RuntimeError("Threshold tracking not initialised. Call reset before stepping.")
        prev_threshold_reached = torch.where(
            prev_active,
            self._threshold_reached,
            torch.zeros_like(self._threshold_reached),
        )

        self.current_step = self.current_step + prev_active.long()
        timeout_mask = (self.current_step >= self.max_steps) & prev_active
        self.current_bboxes_unscaled, self.current_bboxes = self._apply_action(actions, prev_active)

        current_iou = self._calculate_iou(self.current_bboxes_unscaled, self.gt_bboxes)
        rewards, stop_mask, success_mask, threshold_hit = self._compute_rewards(
            actions,
            prev_active,
            current_iou,
            prev_threshold_reached,
            timeout_mask,
        )

        done = stop_mask | timeout_mask
        self.active_mask = prev_active & ~done
        self.last_iou = torch.where(prev_active, current_iou, self.last_iou)
        updated_threshold = prev_threshold_reached | threshold_hit
        self._threshold_reached = torch.where(prev_active, updated_threshold, self._threshold_reached)

        info = {"iou": current_iou.detach(), "success": success_mask.detach()}
        next_state = self._get_state()
        return next_state, rewards.detach(), done.detach(), info

    def render(self, index: int = 0, mode: str = "human"):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        if (
            self.images is None
            or self.current_bboxes_unscaled is None
            or self.gt_bboxes is None
        ):
            raise RuntimeError("Environment must be reset before rendering.")
        if not 0 <= index < self.images.size(0):
            raise IndexError("Render index out of range for current batch.")

        image = self.images[index].detach().cpu()
        image_np = image.permute(1, 2, 0).numpy()
        if image_np.shape[-1] == 1:
            image_np = image_np[..., 0]
        min_val = float(image_np.min())
        max_val = float(image_np.max())
        if max_val > min_val:
            image_np = (image_np - min_val) / (max_val - min_val)

        fig, ax = plt.subplots(1)
        ax.imshow(image_np, cmap="gray")

        gt_x, gt_y, gt_w, gt_h = self.gt_bboxes[index].detach().cpu().tolist()
        agent_x, agent_y, agent_w, agent_h = (
            self.current_bboxes_unscaled[index].detach().cpu().tolist()
        )

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
        if self._resized_images is None or self.current_bboxes is None:
            raise RuntimeError("Environment must be reset before retrieving state.")
        return self._resized_images.detach(), self.current_bboxes.detach()

    def _initialise_bboxes(self, height: int, width: int) -> torch.Tensor:
        if self.gt_bboxes is None or self.has_tumor is None:
            raise RuntimeError("Ground-truth bounding boxes must be computed before initialising.")

        batch_size = self.gt_bboxes.size(0)
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer when initialising bounding boxes.")

        if self.initial_mode == "full_image":
            widths = torch.full(
                (batch_size,),
                float(width),
                device=self.device,
                dtype=torch.float32,
            )
            heights = torch.full(
                (batch_size,),
                float(height),
                device=self.device,
                dtype=torch.float32,
            )

            xs = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
            ys = torch.zeros_like(xs)

            return torch.stack((xs, ys, widths, heights), dim=1)

        boxes = torch.zeros(batch_size, 4, device=self.device, dtype=torch.float32)
        width_limit = float(width)
        height_limit = float(height)

        for index in range(batch_size):
            if not self.has_tumor[index]:
                boxes[index] = torch.tensor(
                    [0.0, 0.0, width_limit, height_limit],
                    device=self.device,
                    dtype=torch.float32,
                )
                continue

            gt_x, gt_y, gt_w, gt_h = self.gt_bboxes[index].detach().cpu().tolist()
            start_x, end_x = self._expand_interval(gt_x, gt_w, width_limit)
            start_y, end_y = self._expand_interval(gt_y, gt_h, height_limit)

            width_val = max(end_x - start_x, self.min_bbox_size)
            height_val = max(end_y - start_y, self.min_bbox_size)

            end_x = min(start_x + width_val, width_limit)
            end_y = min(start_y + height_val, height_limit)
            start_x = max(end_x - width_val, 0.0)
            start_y = max(end_y - height_val, 0.0)

            boxes[index] = torch.tensor(
                [start_x, start_y, end_x - start_x, end_y - start_y],
                device=self.device,
                dtype=torch.float32,
            )

        return boxes

    def _expand_interval(self, start: float, length: float, limit: float) -> Tuple[float, float]:
        margin = self.initial_margin
        expanded_start = max(start - margin, 0.0)
        expanded_end = min(start + length + margin, limit)

        # Ensure the interval still contains the tumour entirely.
        expanded_start = min(expanded_start, start)
        expanded_end = max(expanded_end, start + length)
        expanded_start = max(expanded_start, 0.0)
        expanded_end = min(expanded_end, limit)

        # If expansion reached the borders but there is room inside the image, shift to
        # create some slack so that immediate moves can have an effect.
        available_left = expanded_start
        available_right = limit - expanded_end
        extra_left = start - expanded_start
        extra_right = expanded_end - (start + length)

        if expanded_start <= 0.0 and available_right > 0.0:
            shift = min(self.step_size, available_right, extra_left)
            if shift > 0.0:
                expanded_start += shift
                expanded_end = min(expanded_end + shift, limit)

        if expanded_end >= limit and available_left > 0.0:
            shift = min(self.step_size, available_left, extra_right)
            if shift > 0.0:
                expanded_start = max(expanded_start - shift, 0.0)
                expanded_end -= shift

        return expanded_start, expanded_end

    def _scale_bboxes_to_resize(self, boxes: torch.Tensor) -> torch.Tensor:
        if self._scale_x is None or self._scale_y is None:
            raise RuntimeError("Scale factors are not initialised.")

        x, y, w, h = boxes.unbind(dim=1)
        scaled_x = x * self._scale_x
        scaled_y = y * self._scale_y
        scaled_w = w * self._scale_x
        scaled_h = h * self._scale_y

        return torch.stack((scaled_x, scaled_y, scaled_w, scaled_h), dim=1)

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

    def _apply_action(
        self, actions: torch.Tensor, active_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply actions to bounding boxes. Actions are:
            0: move right
            1: move left
            2: move up
            3: move down
            4: scale up (expand box symmetrically about center)
            5: scale down (shrink box symmetrically about center)
            6: shrink top/bottom (reduce height symmetrically, increase width to keep area bigger/wider)
            7: shrink left/right (reduce width symmetrically, increase height to make box taller)
            8: stop
        """
        if self.current_bboxes_unscaled is None or self.images is None:
            raise RuntimeError("Environment must be reset before applying actions.")

        if self._scale_x is None or self._scale_y is None:
            raise RuntimeError("Scale factors are not initialised.")

        x, y, w, h = self.current_bboxes_unscaled.unbind(dim=1)
        width_limit = torch.tensor(self.images.size(-1), device=self.device, dtype=torch.float32)
        height_limit = torch.tensor(self.images.size(-2), device=self.device, dtype=torch.float32)

        step = torch.full_like(x, self.step_size)
        scale = torch.full_like(x, self.scale_factor)
        min_size = torch.full_like(x, self.min_bbox_size)

        move_right = (actions == 0) & active_mask
        move_left = (actions == 1) & active_mask
        move_up = (actions == 2) & active_mask
        move_down = (actions == 3) & active_mask
        scale_up = (actions == 4) & active_mask
        scale_down = (actions == 5) & active_mask
        shrink_tb = (actions == 6) & active_mask
        shrink_lr = (actions == 7) & active_mask

        # Move actions: shift box center, then recompute x/y
        cx = x + w / 2
        cy = y + h / 2
        cx = torch.where(move_right, torch.minimum(cx + step, width_limit - w / 2), cx)
        cx = torch.where(move_left, torch.maximum(cx - step, w / 2), cx)
        cy = torch.where(move_up, torch.maximum(cy - step, h / 2), cy)
        cy = torch.where(move_down, torch.minimum(cy + step, height_limit - h / 2), cy)

        # Scale up/down: expand/shrink w and h symmetrically about center
        w_new = w
        h_new = h
        # scale up (expand)
        w_candidate = torch.clamp(w * scale, max=width_limit)
        h_candidate = torch.clamp(h * scale, max=height_limit)
        w_new = torch.where(scale_up, w_candidate, w_new)
        h_new = torch.where(scale_up, h_candidate, h_new)
        # scale down (shrink)
        w_candidate = torch.clamp(w / scale, min=min_size)
        h_candidate = torch.clamp(h / scale, min=min_size)
        w_new = torch.where(scale_down, w_candidate, w_new)
        h_new = torch.where(scale_down, h_candidate, h_new)

        # Shrink top/bottom (reduce height, increase width to keep area bigger/wider)
        # Let area = w * h; reduce h, increase w by same factor (but not exceeding limits)
        # For simplicity, multiply h by 1/scale, w by scale (like a "widen" op)
        w_candidate = torch.clamp(w * scale, max=width_limit)
        h_candidate = torch.clamp(h / scale, min=min_size)
        w_new = torch.where(shrink_tb, w_candidate, w_new)
        h_new = torch.where(shrink_tb, h_candidate, h_new)

        # Shrink left/right (reduce width, increase height to make box taller)
        w_candidate = torch.clamp(w / scale, min=min_size)
        h_candidate = torch.clamp(h * scale, max=height_limit)
        w_new = torch.where(shrink_lr, w_candidate, w_new)
        h_new = torch.where(shrink_lr, h_candidate, h_new)

        # Clamp so box fits inside image
        w_new = torch.clamp(w_new, min=min_size, max=width_limit)
        h_new = torch.clamp(h_new, min=min_size, max=height_limit)
        # Clamp center so box stays inside image
        cx = torch.clamp(cx, min=w_new/2, max=width_limit - w_new/2)
        cy = torch.clamp(cy, min=h_new/2, max=height_limit - h_new/2)
        # Recompute x/y from center
        x_new = cx - w_new / 2
        y_new = cy - h_new / 2
        x_new = torch.clamp(
            x_new,
            min=torch.zeros_like(x_new),
            max=width_limit - w_new,
        )
        y_new = torch.clamp(
            y_new,
            min=torch.zeros_like(y_new),
            max=height_limit - h_new,
        )

        updated_unscaled = torch.stack((x_new, y_new, w_new, h_new), dim=1)
        updated_scaled = self._scale_bboxes_to_resize(updated_unscaled)
        return updated_unscaled, updated_scaled

    def _compute_rewards(
        self,
        actions: torch.Tensor,
        prev_active: torch.Tensor,
        current_iou: torch.Tensor,
        prev_threshold_reached: torch.Tensor,
        timeout_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute shaped rewards for the tumor localization agent.
        Incorporates step-based shaping (IoU delta + absolute IoU), strong STOP signals,
        small universal time penalty, and DQN-friendly reward clipping.
        """
        if self.last_iou is None:
            raise RuntimeError("Environment must be reset before computing rewards.")
        if self.has_tumor is None:
            raise RuntimeError("Ground-truth tumor presence information missing.")

        # --- Shaping terms ---
        # 1) Delta-IoU shaping encourages moves that increase IoU step-to-step.
        # 2) Absolute IoU bonus stabilizes around high-IoU regions (maintain good boxes).
        #    These two together approximate potential-based shaping with \Psi(s) \propto IoU(s).
        delta_iou = torch.where(prev_active, current_iou - self.last_iou, torch.zeros_like(current_iou))
        alpha = 2.5  # weight for delta-IoU
        beta = 0.5   # weight for absolute IoU
        rewards = alpha * delta_iou + beta * torch.where(prev_active, current_iou, torch.zeros_like(current_iou))

        # --- STOP action logic ---
        stop_mask = (actions == self._STOP_ACTION) & prev_active
        tumor_present = self.has_tumor & prev_active
        no_tumor = (~self.has_tumor) & prev_active
        threshold_hit = (current_iou >= self.iou_threshold) & tumor_present
        success_mask = stop_mask & threshold_hit

        # Rewards/penalties for STOP decisions
        r_stop_success = self.STOP_REWARD_SUCCESS  # correct STOP when IoU >= threshold
        r_stop_none = self.STOP_REWARD_NO_TUMOR    # correct STOP when no tumor present
        r_stop_false = self.STOP_REWARD_FALSE      # premature/incorrect STOP when tumor present but IoU < threshold

        rewards = rewards + torch.where(success_mask, torch.full_like(rewards, r_stop_success), torch.zeros_like(rewards))
        rewards = rewards + torch.where(stop_mask & no_tumor, torch.full_like(rewards, r_stop_none), torch.zeros_like(rewards))
        rewards = rewards + torch.where(stop_mask & tumor_present & ~threshold_hit, torch.full_like(rewards, r_stop_false), torch.zeros_like(rewards))

        # --- Non-STOP penalties ---
        # Small universal time penalty: encourages shorter trajectories and decisive moves.
        time_penalty = self.TIME_PENALTY
        apply_time_penalty = prev_active & ~stop_mask & ~(no_tumor | success_mask)
        rewards = rewards - torch.where(apply_time_penalty, torch.full_like(rewards, time_penalty), torch.zeros_like(rewards))

        # If agent should STOP (no tumor OR already successful) but keeps moving, add extra penalty.
        hold_penalty = self.HOLD_PENALTY
        rewards = rewards - torch.where((~stop_mask) & no_tumor, torch.full_like(rewards, hold_penalty), torch.zeros_like(rewards))
        threshold_hold_mask = prev_threshold_reached & (~stop_mask) & (~timeout_mask) & prev_active
        rewards = rewards - torch.where(
            threshold_hold_mask,
            torch.full_like(rewards, hold_penalty),
            torch.zeros_like(rewards),
        )

        # Only assign rewards on previously-active envs; keep zeros for finished ones.
        rewards = torch.where(prev_active, rewards, torch.zeros_like(rewards))

        # Optional clipping to stabilize DQN targets.
        min_clip, max_clip = self.REWARD_CLIP_RANGE
        rewards = torch.clamp(rewards, min=min_clip, max=max_clip)
        threshold_hit = torch.where(prev_active, threshold_hit, torch.zeros_like(prev_active, dtype=torch.bool))
        return rewards, stop_mask, success_mask, threshold_hit
    
    # def _compute_rewards(
    #     self,
    #     actions: torch.Tensor,
    #     prev_active: torch.Tensor,
    #     current_iou: torch.Tensor,
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     if self.last_iou is None:
    #         raise RuntimeError("Environment must be reset before computing rewards.")
    #     if self.has_tumor is None:
    #         raise RuntimeError("Ground-truth tumor presence information missing.")

    #     delta_iou = torch.where(prev_active, current_iou - self.last_iou, torch.zeros_like(current_iou))
    #     rewards = delta_iou * 2.0

    #     stop_mask = (actions == self._STOP_ACTION) & prev_active
    #     tumor_present = self.has_tumor & prev_active
    #     no_tumor = (~self.has_tumor) & prev_active

    #     success_mask = (current_iou >= self.iou_threshold) & tumor_present

    #     rewards = rewards + torch.where(stop_mask & no_tumor, torch.full_like(rewards, 2.0), torch.zeros_like(rewards))
    #     rewards = rewards + torch.where(stop_mask & success_mask, torch.full_like(rewards, 3.0), torch.zeros_like(rewards))
    #     rewards = rewards + torch.where(stop_mask & tumor_present & ~success_mask, -torch.full_like(rewards, 2.5), torch.zeros_like(rewards))

    #     rewards = rewards + torch.where((~stop_mask) & no_tumor, -torch.full_like(rewards, 0.5), torch.zeros_like(rewards))
    #     rewards = rewards + torch.where((~stop_mask) & success_mask, -torch.full_like(rewards, 0.5), torch.zeros_like(rewards))

    #     rewards = torch.where(prev_active, rewards, torch.zeros_like(rewards))

    #     return rewards, stop_mask, success_mask


    def _get_bbox_from_mask(self, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = masks.size(0)
        boxes = torch.zeros(batch_size, 4, device=self.device, dtype=torch.float32)
        has_tumor = torch.zeros(batch_size, device=self.device, dtype=torch.bool)

        mask_binary = masks > 0
        if mask_binary.dim() != 4:
            raise ValueError("Masks must be 4D tensors with shape [B, C, H, W].")

        per_slice_mask = mask_binary.any(dim=1)

        for index in range(batch_size):
            coords = torch.nonzero(per_slice_mask[index], as_tuple=False)
            if coords.numel() == 0:
                continue

            ys = coords[:, -2].float()
            xs = coords[:, -1].float()

            y_min = ys.min()
            y_max = ys.max()
            x_min = xs.min()
            x_max = xs.max()

            width = torch.clamp(x_max - x_min + 1.0, min=self.min_bbox_size)
            height = torch.clamp(y_max - y_min + 1.0, min=self.min_bbox_size)

            boxes[index] = torch.tensor([x_min, y_min, width, height], device=self.device, dtype=torch.float32)
            has_tumor[index] = True

        return boxes, has_tumor
    
    # ------------------------------------------------------------------
    # New vectorized helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _candidate_boxes_all_actions(self) -> torch.Tensor:
        """Return candidate boxes for all 8 actions, shape (B, 8, 4), with symmetric (centered) transforms."""
        assert self.current_bboxes_unscaled is not None and self.images is not None
        x, y, w, h = self.current_bboxes_unscaled.unbind(dim=1)
        width_limit = torch.as_tensor(self.images.size(-1), device=self.device, dtype=torch.float32)
        height_limit = torch.as_tensor(self.images.size(-2), device=self.device, dtype=torch.float32)
        step = torch.full_like(x, self.step_size)
        scale = torch.full_like(x, self.scale_factor)
        min_sz = torch.full_like(x, self.min_bbox_size)

        # Compute current center
        cx = x + w / 2
        cy = y + h / 2

        # 0: move right
        cx0 = torch.minimum(cx + step, width_limit - w/2)
        cy0 = cy
        w0 = w
        h0 = h
        x0 = torch.clamp(
            cx0 - w0/2,
            min=torch.zeros_like(cx0),
            max=width_limit - w0,
        )
        y0 = torch.clamp(
            cy0 - h0/2,
            min=torch.zeros_like(cy0),
            max=height_limit - h0,
        )

        # 1: move left
        cx1 = torch.maximum(cx - step, w/2)
        cy1 = cy
        w1 = w
        h1 = h
        x1 = torch.clamp(
            cx1 - w1/2,
            min=torch.zeros_like(cx1),
            max=width_limit - w1,
        )
        y1 = torch.clamp(
            cy1 - h1/2,
            min=torch.zeros_like(cy1),
            max=height_limit - h1,
        )

        # 2: move up
        cx2 = cx
        cy2 = torch.maximum(cy - step, h/2)
        w2 = w
        h2 = h
        x2 = torch.clamp(
            cx2 - w2/2,
            min=torch.zeros_like(cx2),
            max=width_limit - w2,
        )
        y2 = torch.clamp(
            cy2 - h2/2,
            min=torch.zeros_like(cy2),
            max=height_limit - h2,
        )

        # 3: move down
        cx3 = cx
        cy3 = torch.minimum(cy + step, height_limit - h/2)
        w3 = w
        h3 = h
        x3 = torch.clamp(
            cx3 - w3/2,
            min=torch.zeros_like(cx3),
            max=width_limit - w3,
        )
        y3 = torch.clamp(
            cy3 - h3/2,
            min=torch.zeros_like(cy3),
            max=height_limit - h3,
        )

        # 4: scale up (expand symmetrically)
        w4 = torch.clamp(w * scale, max=width_limit)
        h4 = torch.clamp(h * scale, max=height_limit)
        cx4 = torch.clamp(cx, min=w4/2, max=width_limit - w4/2)
        cy4 = torch.clamp(cy, min=h4/2, max=height_limit - h4/2)
        x4 = torch.clamp(
            cx4 - w4/2,
            min=torch.zeros_like(cx4),
            max=width_limit - w4,
        )
        y4 = torch.clamp(
            cy4 - h4/2,
            min=torch.zeros_like(cy4),
            max=height_limit - h4,
        )

        # 5: scale down (shrink symmetrically)
        w5 = torch.clamp(w / scale, min=min_sz)
        h5 = torch.clamp(h / scale, min=min_sz)
        cx5 = torch.clamp(cx, min=w5/2, max=width_limit - w5/2)
        cy5 = torch.clamp(cy, min=h5/2, max=height_limit - h5/2)
        x5 = torch.clamp(
            cx5 - w5/2,
            min=torch.zeros_like(cx5),
            max=width_limit - w5,
        )
        y5 = torch.clamp(
            cy5 - h5/2,
            min=torch.zeros_like(cy5),
            max=height_limit - h5,
        )

        # 6: shrink top/bottom (reduce height, increase width)
        w6 = torch.clamp(w * scale, max=width_limit)
        h6 = torch.clamp(h / scale, min=min_sz)
        cx6 = torch.clamp(cx, min=w6/2, max=width_limit - w6/2)
        cy6 = torch.clamp(cy, min=h6/2, max=height_limit - h6/2)
        x6 = torch.clamp(
            cx6 - w6/2,
            min=torch.zeros_like(cx6),
            max=width_limit - w6,
        )
        y6 = torch.clamp(
            cy6 - h6/2,
            min=torch.zeros_like(cy6),
            max=height_limit - h6,
        )

        # 7: shrink left/right (reduce width, increase height)
        w7 = torch.clamp(w / scale, min=min_sz)
        h7 = torch.clamp(h * scale, max=height_limit)
        cx7 = torch.clamp(cx, min=w7/2, max=width_limit - w7/2)
        cy7 = torch.clamp(cy, min=h7/2, max=height_limit - h7/2)
        x7 = torch.clamp(
            cx7 - w7/2,
            min=torch.zeros_like(cx7),
            max=width_limit - w7,
        )
        y7 = torch.clamp(
            cy7 - h7/2,
            min=torch.zeros_like(cy7),
            max=height_limit - h7,
        )

        c0 = torch.stack([x0, y0, w0, h0], dim=1)
        c1 = torch.stack([x1, y1, w1, h1], dim=1)
        c2 = torch.stack([x2, y2, w2, h2], dim=1)
        c3 = torch.stack([x3, y3, w3, h3], dim=1)
        c4 = torch.stack([x4, y4, w4, h4], dim=1)
        c5 = torch.stack([x5, y5, w5, h5], dim=1)
        c6 = torch.stack([x6, y6, w6, h6], dim=1)
        c7 = torch.stack([x7, y7, w7, h7], dim=1)
        return torch.stack([c0, c1, c2, c3, c4, c5, c6, c7], dim=1)  # (B, 8, 4)

    @torch.no_grad()
    def _iou_candidates(self, cand: torch.Tensor) -> torch.Tensor:
        """Compute IoU for all candidate boxes vs GT. cand: (B, 8, 4) -> (B, 8)."""
        gt = self.gt_bboxes
        x1, y1, w1, h1 = cand.unbind(dim=2)
        x2, y2, w2, h2 = gt[:, 0:1], gt[:, 1:2], gt[:, 2:3], gt[:, 3:4]

        x1e = x1 + w1; y1e = y1 + h1
        x2e = x2 + w2; y2e = y2 + h2

        inter_w = torch.clamp(torch.minimum(x1e, x2e) - torch.maximum(x1, x2), min=0.0)
        inter_h = torch.clamp(torch.minimum(y1e, y2e) - torch.maximum(y1, y2), min=0.0)
        inter = inter_w * inter_h

        area1 = torch.clamp(w1, min=0.0) * torch.clamp(h1, min=0.0)
        area2 = torch.clamp(w2, min=0.0) * torch.clamp(h2, min=0.0)
        union = area1 + area2 - inter
        return torch.where(union > 0, inter / union, torch.zeros_like(union))

    @torch.no_grad()
    def positive_actions_mask(self, eps: float = 1e-6) -> torch.Tensor:
        """Return boolean mask (B, 8) of which actions improve IoU."""
        cand = self._candidate_boxes_all_actions()
        iou_new = self._iou_candidates(cand)
        cur = self.last_iou.unsqueeze(1)
        pos = (iou_new - cur) > eps
        if self.has_tumor is not None:
            pos = pos & self.has_tumor.view(-1, 1)
        if self.active_mask is not None:
            pos = pos & self.active_mask.view(-1, 1)
        return pos

    @torch.no_grad()
    def best_action_by_iou(self, include_stop: bool = True) -> torch.Tensor:
        """Return argmax IoU action per env. If include_stop=True, STOP (8) is allowed."""
        cand = self._candidate_boxes_all_actions()
        if include_stop:
            cur = self.current_bboxes_unscaled.unsqueeze(1)
            cand = torch.cat([cand, cur], dim=1)
        ious = self._iou_candidates(cand)
        return ious.argmax(dim=1)