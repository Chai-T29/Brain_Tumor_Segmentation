from typing import ClassVar, Dict, Optional, Tuple
import torch


class TumorLocalizationEnv:
    """Vectorized environment for batched tumor localization with center-anchored, symmetric actions."""

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
        step_size: float = 10.0,
        scale_factor: float = 1.1,
        min_bbox_size: float = 10.0,
        initial_mode: str = "random_corners",
        initial_margin: float = 8.0,
        reward_clip_range: Tuple[float, float] = (-6.0, 6.0),
        delta_iou: float = 2.5,
        current_iou: float = 0.5,
        stop_reward_success: float = 4.0,
        stop_reward_no_tumor: float = 2.0,
        stop_reward_false: float = -3.0,
        time_penalty: float = 0.01,
        hold_penalty: float = 0.5,
    ) -> None:
        self.max_steps = max(1, int(max_steps))
        self.iou_threshold = float(iou_threshold)
        self.step_size = float(step_size)
        self.scale_factor = float(scale_factor)
        self.min_bbox_size = float(min_bbox_size)
        self.initial_mode = initial_mode
        if self.initial_mode not in {"random_corners", "full_image", "gt_margin"}:
            raise ValueError("initial_mode must be 'random_corners', 'full_image' or 'gt_margin'.")
        self.initial_margin = float(initial_margin)
        if self.initial_margin < 0.0:
            raise ValueError("initial_margin must be non-negative.")

        # Allow overriding of class-level reward/penalty constants per instance
        self.DELTA_IOU = delta_iou
        self.CURRENT_IOU = current_iou
        self.REWARD_CLIP_RANGE = (float(reward_clip_range[0]), float(reward_clip_range[1]))
        self.STOP_REWARD_SUCCESS = float(stop_reward_success)
        self.STOP_REWARD_NO_TUMOR = float(stop_reward_no_tumor)
        self.STOP_REWARD_FALSE = float(stop_reward_false)
        self.TIME_PENALTY = float(time_penalty)
        self.HOLD_PENALTY = float(hold_penalty)

        self.images: Optional[torch.Tensor] = None
        self.masks: Optional[torch.Tensor] = None
        self.gt_bboxes: Optional[torch.Tensor] = None
        self.current_bboxes: Optional[torch.Tensor] = None
        self.last_iou: Optional[torch.Tensor] = None
        self.current_step: Optional[torch.Tensor] = None
        self.active_mask: Optional[torch.Tensor] = None
        self.has_tumor: Optional[torch.Tensor] = None
        self._threshold_reached: Optional[torch.Tensor] = None
        self.episode_id: int = 0

    @property
    def device(self) -> torch.device:
        if self.images is None:
            return torch.device("cpu")
        return self.images.device

    def reset(self, images: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if images.dim() != 4 or masks.dim() != 4:
            raise ValueError("Images and masks must be [B,C,H,W].")
        if images.size(0) != masks.size(0):
            raise ValueError("Batch size mismatch.")

        self.images = images.detach()
        self.masks = masks.detach()
        batch_size, _, height, width = images.shape
        # Bump episode id each time we reset to mark a new episode
        self.episode_id = getattr(self, "episode_id", 0) + 1

        self.gt_bboxes, self.has_tumor = self._get_bbox_from_mask(self.masks)
        self.current_bboxes = self._initialise_bboxes(height, width)
        self.last_iou = self._calculate_iou(self.current_bboxes, self.gt_bboxes)

        self.current_step = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        self.active_mask = torch.ones(batch_size, device=self.device, dtype=torch.bool)
        self._threshold_reached = torch.zeros(batch_size, device=self.device, dtype=torch.bool)

        return self._get_state()

    def step(self, actions: torch.Tensor):
        if self.images is None:
            raise RuntimeError("Call reset() first.")
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)
        if actions.dim() != 1:
            raise ValueError("Actions must be 1D [B].")

        actions = actions.to(self.device)
        prev_active = self.active_mask.clone()
        actions = torch.where(prev_active, actions, torch.full_like(actions, self._STOP_ACTION))

        prev_threshold_reached = torch.where(
            prev_active,
            self._threshold_reached,
            torch.zeros_like(self._threshold_reached),
        )

        self.current_step = self.current_step + prev_active.long()
        timeout_mask = (self.current_step >= self.max_steps) & prev_active
        self.current_bboxes = self._apply_action(actions, prev_active)

        current_iou = self._calculate_iou(self.current_bboxes, self.gt_bboxes)
        rewards, stop_mask, success_mask, threshold_hit = self._compute_rewards(
            actions, prev_active, current_iou, prev_threshold_reached, timeout_mask
        )

        done = stop_mask | timeout_mask
        self.active_mask = prev_active & ~done
        self.last_iou = torch.where(prev_active, current_iou, self.last_iou)
        updated_threshold = prev_threshold_reached | threshold_hit
        self._threshold_reached = torch.where(prev_active, updated_threshold, self._threshold_reached)

        info = {"iou": current_iou.detach(), "success": success_mask.detach()}
        return self._get_state(), rewards.detach(), done.detach(), info

    def _get_state(self):
        return self.images.detach(), self.current_bboxes.detach()

    def _initialise_bboxes(self, height: int, width: int) -> torch.Tensor:
        batch_size = self.gt_bboxes.size(0)
        boxes = torch.zeros(batch_size, 4, device=self.device, dtype=torch.float32)

        if self.initial_mode == "random_corners":
            # Initialize a box that covers 3/4 of the image in both width and height,
            # and place it at a random corner for each element in the batch.
            width_t  = torch.full((batch_size,), float(width),  device=self.device)
            height_t = torch.full((batch_size,), float(height), device=self.device)

            w = torch.clamp(0.75 * width_t, min=self.min_bbox_size)
            w = torch.minimum(w, width_t)
            h = torch.clamp(0.75 * height_t, min=self.min_bbox_size)
            h = torch.minimum(h, height_t)

            corners = torch.randint(0, 4, (batch_size,), device=self.device)  # 0: TL, 1: TR, 2: BL, 3: BR
            x = torch.where((corners == 0) | (corners == 2), torch.zeros_like(w), width_t - w)
            y = torch.where((corners == 0) | (corners == 1), torch.zeros_like(h), height_t - h)

            boxes = torch.stack([x, y, w, h], dim=1)
            return boxes

        if self.initial_mode == "full_image":
            boxes[:] = torch.tensor([0.0, 0.0, float(width), float(height)], device=self.device)
            return boxes

        # "gt_margin"
        for i in range(batch_size):
            if not self.has_tumor[i]:
                boxes[i] = torch.tensor([0.0, 0.0, float(width), float(height)], device=self.device)
                continue
            gt_x, gt_y, gt_w, gt_h = self.gt_bboxes[i].tolist()
            start_x = max(gt_x - self.initial_margin, 0.0)
            start_y = max(gt_y - self.initial_margin, 0.0)
            end_x = min(gt_x + gt_w + self.initial_margin, width)
            end_y = min(gt_y + gt_h + self.initial_margin, height)
            boxes[i] = torch.tensor([start_x, start_y, end_x - start_x, end_y - start_y], device=self.device)
        return boxes

    def _calculate_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        x1, y1, w1, h1 = boxes1.unbind(1)
        x2, y2, w2, h2 = boxes2.unbind(1)
        x1e, y1e, x2e, y2e = x1 + w1, y1 + h1, x2 + w2, y2 + h2
        inter_w = torch.clamp(torch.min(x1e, x2e) - torch.max(x1, x2), min=0.0)
        inter_h = torch.clamp(torch.min(y1e, y2e) - torch.max(y1, y2), min=0.0)
        inter = inter_w * inter_h
        area1 = torch.clamp(w1, min=0.0) * torch.clamp(h1, min=0.0)
        area2 = torch.clamp(w2, min=0.0) * torch.clamp(h2, min=0.0)
        union = area1 + area2 - inter
        return torch.where(union > 0, inter / union, torch.zeros_like(union))
    
    def _transform_boxes(self, boxes: torch.Tensor, actions: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """Pure functional version of the action transform.
        Given current boxes, actions and active mask, return transformed boxes without mutating state.
        """
        x, y, w, h = boxes.unbind(1)
        width_limit = torch.tensor(self.images.size(-1), device=self.device)
        height_limit = torch.tensor(self.images.size(-2), device=self.device)
        step = torch.full_like(x, self.step_size)
        scale = torch.full_like(x, self.scale_factor)
        min_size = torch.full_like(x, self.min_bbox_size)

        cx = x + w / 2
        cy = y + h / 2

        # moves (center-based)
        cx = torch.where((actions == 0) & active_mask, torch.minimum(cx + step, width_limit - w / 2), cx)  # right
        cx = torch.where((actions == 1) & active_mask, torch.maximum(cx - step, w / 2), cx)                # left
        cy = torch.where((actions == 2) & active_mask, torch.maximum(cy - step, h / 2), cy)                # up
        cy = torch.where((actions == 3) & active_mask, torch.minimum(cy + step, height_limit - h / 2), cy) # down

        w_new, h_new = w, h
        # scale up
        w_new = torch.where((actions == 4) & active_mask, torch.clamp(w * scale, max=width_limit), w_new)
        h_new = torch.where((actions == 4) & active_mask, torch.clamp(h * scale, max=height_limit), h_new)
        # scale down
        w_new = torch.where((actions == 5) & active_mask, torch.clamp(w / scale, min=min_size), w_new)
        h_new = torch.where((actions == 5) & active_mask, torch.clamp(h / scale, min=min_size), h_new)
        # shrink top/bottom (reduce height, increase width)
        w_new = torch.where((actions == 6) & active_mask, torch.clamp(w * scale, max=width_limit), w_new)
        h_new = torch.where((actions == 6) & active_mask, torch.clamp(h / scale, min=min_size), h_new)
        # shrink left/right (reduce width, increase height)
        w_new = torch.where((actions == 7) & active_mask, torch.clamp(w / scale, min=min_size), w_new)
        h_new = torch.where((actions == 7) & active_mask, torch.clamp(h * scale, max=height_limit), h_new)

        # final clamps
        w_new = torch.clamp(w_new, min=min_size, max=width_limit)
        h_new = torch.clamp(h_new, min=min_size, max=height_limit)
        cx = torch.clamp(cx, min=w_new / 2, max=width_limit - w_new / 2)
        cy = torch.clamp(cy, min=h_new / 2, max=height_limit - h_new / 2)

        x_new = cx - w_new / 2
        y_new = cy - h_new / 2
        return torch.stack((x_new, y_new, w_new, h_new), dim=1)

    def _apply_action(self, actions: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        return self._transform_boxes(self.current_bboxes, actions, active_mask)

    def _compute_rewards(self, actions, prev_active, current_iou, prev_threshold_reached, timeout_mask):
        if self.last_iou is None:
            raise RuntimeError("reset before rewards.")
        delta_iou = torch.where(prev_active, current_iou - self.last_iou, torch.zeros_like(current_iou))
        rewards = self.DELTA_IOU * delta_iou + self.CURRENT_IOU * torch.where(prev_active, current_iou, torch.zeros_like(current_iou))

        stop_mask = (actions == self._STOP_ACTION) & prev_active
        tumor_present = self.has_tumor & prev_active
        no_tumor = (~self.has_tumor) & prev_active
        threshold_hit = (current_iou >= self.iou_threshold) & tumor_present
        success_mask = stop_mask & threshold_hit

        # stop when tumor has been located
        rewards += torch.where(success_mask, torch.full_like(rewards, self.STOP_REWARD_SUCCESS), 0)

        # stop when no tumor is present
        rewards += torch.where(stop_mask & no_tumor, torch.full_like(rewards, self.STOP_REWARD_NO_TUMOR), 0)

        # penalty if stopped before hitting threshold
        # this can cause model to never stop
        rewards -= torch.where(stop_mask & tumor_present & ~threshold_hit, torch.full_like(rewards, self.STOP_REWARD_FALSE), 0)

        # penalty for every step taken
        time_penalty_mask = prev_active & ~stop_mask & ~(no_tumor | success_mask)
        rewards -= torch.where(time_penalty_mask, torch.full_like(rewards, self.TIME_PENALTY), 0)

        # penalty for holding when no tumor is present
        rewards -= torch.where((~stop_mask) & no_tumor, torch.full_like(rewards, self.HOLD_PENALTY), 0)

        # penalizing if not stopped after hitting threshold
        # can cause preemptive stopping due to fear of penalty
        rewards -= torch.where(prev_threshold_reached & (~stop_mask) & (~timeout_mask) & prev_active,
                               torch.full_like(rewards, self.HOLD_PENALTY), 0)

        rewards = torch.where(prev_active, rewards, torch.zeros_like(rewards))
        return torch.clamp(rewards, *self.REWARD_CLIP_RANGE), stop_mask, success_mask, threshold_hit

    def _get_bbox_from_mask(self, masks: torch.Tensor):
        batch_size = masks.size(0)
        boxes = torch.zeros(batch_size, 4, device=self.device, dtype=torch.float32)
        has_tumor = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        mask_binary = masks > 0.5   # robust threshold; matches nearest-neighbor resized masks
        for i in range(batch_size):
            coords = torch.nonzero(mask_binary[i], as_tuple=False)
            if coords.numel() == 0:
                continue
            ys, xs = coords[:, -2].float(), coords[:, -1].float()
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            w = torch.clamp(x_max - x_min + 1, min=self.min_bbox_size)
            h = torch.clamp(y_max - y_min + 1, min=self.min_bbox_size)
            boxes[i] = torch.tensor([x_min, y_min, w, h], device=self.device)
            has_tumor[i] = True
        return boxes, has_tumor

    # ------------------------------------------------------------------
    # Oracle-style helpers (operate in the same coordinate system as images)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _candidate_boxes_all_actions(self) -> torch.Tensor:
        """Return candidate boxes for all 8 non-stop actions, shape (B, 8, 4),
        using the exact same transform as `_apply_action`.
        """
        assert self.current_bboxes is not None and self.images is not None
        B = self.current_bboxes.size(0)
        active_all = torch.ones(B, dtype=torch.bool, device=self.device)
        cands = []
        for act in range(8):
            actions = torch.full((B,), act, dtype=torch.long, device=self.device)
            cands.append(self._transform_boxes(self.current_bboxes, actions, active_all))
        return torch.stack(cands, dim=1)  # (B, 8, 4)
    
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
    def positive_actions_mask(self, eps: float = 1e-3) -> torch.Tensor:
        """Return boolean mask (B, 8) of which actions improve IoU by > eps."""
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
    def best_action_by_iou(self, include_stop: bool = True, eps: float = 1e-6) -> torch.Tensor:
        """Return the best lookahead action by IoU.
        If the current IoU meets/exceeds the threshold and `include_stop` is True, return STOP.
        Otherwise return the argmax among the 8 non-stop candidates.
        """
        assert self.last_iou is not None and self.current_bboxes is not None
        cand = self._candidate_boxes_all_actions()          # (B, 8, 4)
        iou_new = self._iou_candidates(cand)                # (B, 8)
        cur = self.last_iou                                 # (B,)
        _, max_idx = iou_new.max(dim=1)                     # (B,)
        if include_stop:
            tumor_present = self.has_tumor if self.has_tumor is not None else torch.ones_like(cur, dtype=torch.bool)
            at_threshold = (cur >= self.iou_threshold) & tumor_present
            stop_idx = torch.full_like(max_idx, self._STOP_ACTION)
            return torch.where(at_threshold, stop_idx, max_idx)
        return max_idx
    

    def render(self, index: int = 0, mode: str = "human"):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        if (
            self.images is None
            or self.current_bboxes is None
            or self.gt_bboxes is None
        ):
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
        agent_x, agent_y, agent_w, agent_h = (
            self.current_bboxes[index].detach().cpu().tolist()
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