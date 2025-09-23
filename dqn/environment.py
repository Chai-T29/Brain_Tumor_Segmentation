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
        initial_mode: str = "full_image",
        initial_margin: float = 8.0,
    ) -> None:
        self.max_steps = max(1, int(max_steps))
        self.iou_threshold = float(iou_threshold)
        self.step_size = float(step_size)
        self.scale_factor = float(scale_factor)
        self.min_bbox_size = float(min_bbox_size)
        self.initial_mode = initial_mode
        if self.initial_mode not in {"full_image", "gt_margin"}:
            raise ValueError("initial_mode must be 'full_image' or 'gt_margin'.")
        self.initial_margin = float(initial_margin)
        if self.initial_margin < 0.0:
            raise ValueError("initial_margin must be non-negative.")

        self.images: Optional[torch.Tensor] = None
        self.masks: Optional[torch.Tensor] = None
        self.gt_bboxes: Optional[torch.Tensor] = None
        self.current_bboxes: Optional[torch.Tensor] = None
        self.last_iou: Optional[torch.Tensor] = None
        self.current_step: Optional[torch.Tensor] = None
        self.active_mask: Optional[torch.Tensor] = None
        self.has_tumor: Optional[torch.Tensor] = None
        self._threshold_reached: Optional[torch.Tensor] = None

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

        if self.initial_mode == "full_image":
            boxes[:] = torch.tensor([0.0, 0.0, float(width), float(height)], device=self.device)
            return boxes

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

    def _apply_action(self, actions: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        x, y, w, h = self.current_bboxes.unbind(1)
        width_limit = torch.tensor(self.images.size(-1), device=self.device)
        height_limit = torch.tensor(self.images.size(-2), device=self.device)
        step = torch.full_like(x, self.step_size)
        scale = torch.full_like(x, self.scale_factor)
        min_size = torch.full_like(x, self.min_bbox_size)

        cx = x + w / 2
        cy = y + h / 2

        # moves
        cx = torch.where((actions == 0) & active_mask, torch.minimum(cx + step, width_limit - w/2), cx)
        cx = torch.where((actions == 1) & active_mask, torch.maximum(cx - step, w/2), cx)
        cy = torch.where((actions == 2) & active_mask, torch.maximum(cy - step, h/2), cy)
        cy = torch.where((actions == 3) & active_mask, torch.minimum(cy + step, height_limit - h/2), cy)

        w_new, h_new = w, h
        # scale up
        w_new = torch.where((actions == 4) & active_mask, torch.clamp(w * scale, max=width_limit), w_new)
        h_new = torch.where((actions == 4) & active_mask, torch.clamp(h * scale, max=height_limit), h_new)
        # scale down
        w_new = torch.where((actions == 5) & active_mask, torch.clamp(w / scale, min=min_size), w_new)
        h_new = torch.where((actions == 5) & active_mask, torch.clamp(h / scale, min=min_size), h_new)
        # shrink top/bottom
        w_new = torch.where((actions == 6) & active_mask, torch.clamp(w * scale, max=width_limit), w_new)
        h_new = torch.where((actions == 6) & active_mask, torch.clamp(h / scale, min=min_size), h_new)
        # shrink left/right
        w_new = torch.where((actions == 7) & active_mask, torch.clamp(w / scale, min=min_size), w_new)
        h_new = torch.where((actions == 7) & active_mask, torch.clamp(h * scale, max=height_limit), h_new)

        w_new = torch.clamp(w_new, min=min_size, max=width_limit)
        h_new = torch.clamp(h_new, min=min_size, max=height_limit)
        cx = torch.clamp(cx, min=w_new/2, max=width_limit - w_new/2)
        cy = torch.clamp(cy, min=h_new/2, max=height_limit - h_new/2)

        x_new = cx - w_new / 2
        y_new = cy - h_new / 2
        return torch.stack((x_new, y_new, w_new, h_new), dim=1)

    def _compute_rewards(self, actions, prev_active, current_iou, prev_threshold_reached, timeout_mask):
        if self.last_iou is None:
            raise RuntimeError("reset before rewards.")
        delta_iou = torch.where(prev_active, current_iou - self.last_iou, torch.zeros_like(current_iou))
        rewards = 2.5 * delta_iou + 0.5 * torch.where(prev_active, current_iou, torch.zeros_like(current_iou))

        stop_mask = (actions == self._STOP_ACTION) & prev_active
        tumor_present = self.has_tumor & prev_active
        no_tumor = (~self.has_tumor) & prev_active
        threshold_hit = (current_iou >= self.iou_threshold) & tumor_present
        success_mask = stop_mask & threshold_hit

        rewards += torch.where(success_mask, torch.full_like(rewards, self.STOP_REWARD_SUCCESS), 0)
        rewards += torch.where(stop_mask & no_tumor, torch.full_like(rewards, self.STOP_REWARD_NO_TUMOR), 0)
        rewards += torch.where(stop_mask & tumor_present & ~threshold_hit, torch.full_like(rewards, self.STOP_REWARD_FALSE), 0)

        time_penalty_mask = prev_active & ~stop_mask & ~(no_tumor | success_mask)
        rewards -= torch.where(time_penalty_mask, torch.full_like(rewards, self.TIME_PENALTY), 0)
        rewards -= torch.where((~stop_mask) & no_tumor, torch.full_like(rewards, self.HOLD_PENALTY), 0)
        rewards -= torch.where(prev_threshold_reached & (~stop_mask) & (~timeout_mask) & prev_active,
                               torch.full_like(rewards, self.HOLD_PENALTY), 0)

        rewards = torch.where(prev_active, rewards, torch.zeros_like(rewards))
        return torch.clamp(rewards, *self.REWARD_CLIP_RANGE), stop_mask, success_mask, threshold_hit

    def _get_bbox_from_mask(self, masks: torch.Tensor):
        batch_size = masks.size(0)
        boxes = torch.zeros(batch_size, 4, device=self.device, dtype=torch.float32)
        has_tumor = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        mask_binary = masks > 0
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