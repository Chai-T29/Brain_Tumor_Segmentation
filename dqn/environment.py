from typing import ClassVar, Dict, Optional, Tuple
import torch
import numpy as np


class TumorLocalizationEnv:
    """Vectorized environment for batched tumor localization with center-anchored, symmetric actions."""

    REWARD_CLIP_RANGE: ClassVar[Tuple[float, float]] = (-6.0, 6.0)
    STOP_REWARD_SUCCESS: ClassVar[float] = 4.0
    STOP_REWARD_NO_TUMOR: ClassVar[float] = 2.0
    STOP_REWARD_FALSE: ClassVar[float] = -3.0
    TIME_PENALTY: ClassVar[float] = 0.01
    HOLD_PENALTY: ClassVar[float] = 0.5
    CENTERING_REWARD_WEIGHT: ClassVar[float] = 0.5  # Weight for centering reward

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
        self.last_centering_score: Optional[torch.Tensor] = None
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
        self.last_centering_score = self._calculate_centering_score(self.current_bboxes, self.gt_bboxes)

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
        current_centering_score = self._calculate_centering_score(self.current_bboxes, self.gt_bboxes)
        
        rewards, stop_mask, success_mask, threshold_hit = self._compute_rewards(
            actions, prev_active, current_iou, current_centering_score, 
            prev_threshold_reached, timeout_mask
        )

        done = stop_mask | timeout_mask
        self.active_mask = prev_active & ~done
        self.last_iou = torch.where(prev_active, current_iou, self.last_iou)
        self.last_centering_score = torch.where(prev_active, current_centering_score, self.last_centering_score)
        updated_threshold = prev_threshold_reached | threshold_hit
        self._threshold_reached = torch.where(prev_active, updated_threshold, self._threshold_reached)

        info = {"iou": current_iou.detach(), "success": success_mask.detach(), 
                "centering_score": current_centering_score.detach()}
        return self._get_state(), rewards.detach(), done.detach(), info

    def _get_state(self):
        return self.images.detach(), self.current_bboxes.detach()

    def _calculate_centering_score(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """Calculate how well the predicted box center aligns with the ground truth box center.
        
        Args:
            pred_boxes: Predicted bounding boxes [B, 4] in format [x, y, w, h]
            gt_boxes: Ground truth bounding boxes [B, 4] in format [x, y, w, h]
            
        Returns:
            Centering scores [B] where 1.0 is perfectly centered, 0.0 is maximally off-center
        """
        # Get centers of predicted and ground truth boxes
        pred_cx = pred_boxes[:, 0] + pred_boxes[:, 2] / 2  # x + w/2
        pred_cy = pred_boxes[:, 1] + pred_boxes[:, 3] / 2  # y + h/2
        
        gt_cx = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
        gt_cy = gt_boxes[:, 1] + gt_boxes[:, 3] / 2
        
        # Calculate distance between centers
        center_dist = torch.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2)
        
        # Normalize by image diagonal (assuming square images for simplicity)
        # This makes the score resolution-independent
        image_diagonal = torch.sqrt(torch.tensor(self.images.size(-1) ** 2 + self.images.size(-2) ** 2, 
                                                device=self.device, dtype=torch.float32))
        
        normalized_dist = center_dist / image_diagonal
        
        # Convert to score: 1.0 when perfectly centered, approaching 0 as distance increases
        centering_score = torch.exp(-5 * normalized_dist)  # Exponential decay
        
        # Only apply centering score where there's actually a tumor
        centering_score = torch.where(self.has_tumor, centering_score, 
                                    torch.ones_like(centering_score))
        
        return centering_score

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

    def _compute_rewards(self, actions, prev_active, current_iou, current_centering_score, 
                        prev_threshold_reached, timeout_mask):
        if self.last_iou is None or self.last_centering_score is None:
            raise RuntimeError("reset before rewards.")
            
        delta_iou = torch.where(prev_active, current_iou - self.last_iou, torch.zeros_like(current_iou))
        delta_centering = torch.where(prev_active, current_centering_score - self.last_centering_score, 
                                     torch.zeros_like(current_centering_score))
        
        # Main reward: IoU improvement + current IoU + centering improvement
        rewards = (2.5 * delta_iou + 
                  0.5 * torch.where(prev_active, current_iou, torch.zeros_like(current_iou)) +
                  self.CENTERING_REWARD_WEIGHT * delta_centering)

        stop_mask = (actions == self._STOP_ACTION) & prev_active
        tumor_present = self.has_tumor & prev_active
        no_tumor = (~self.has_tumor) & prev_active
        threshold_hit = (current_iou >= self.iou_threshold) & tumor_present
        success_mask = stop_mask & threshold_hit

        # Stopping rewards
        rewards += torch.where(success_mask, torch.full_like(rewards, self.STOP_REWARD_SUCCESS), 0)
        rewards += torch.where(stop_mask & no_tumor, torch.full_like(rewards, self.STOP_REWARD_NO_TUMOR), 0)
        rewards += torch.where(stop_mask & tumor_present & ~threshold_hit, torch.full_like(rewards, self.STOP_REWARD_FALSE), 0)

        # Time and holding penalties
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

    # ------------------------------------------------------------------
    # Oracle-style helpers (operate in the same coordinate system as images)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _candidate_boxes_all_actions(self) -> torch.Tensor:
        """Return candidate boxes for all 8 non-stop actions, shape (B, 8, 4).
        Actions are center-anchored and symmetric:
          0: right, 1: left, 2: up, 3: down,
          4: scale up, 5: scale down,
          6: shrink top/bottom (wider), 7: shrink left/right (taller)
        """
        assert self.current_bboxes is not None and self.images is not None
        x, y, w, h = self.current_bboxes.unbind(dim=1)

        width_limit  = torch.as_tensor(self.images.size(-1), device=self.device, dtype=torch.float32)
        height_limit = torch.as_tensor(self.images.size(-2), device=self.device, dtype=torch.float32)
        step   = torch.full_like(x, self.step_size)
        scale  = torch.full_like(x, self.scale_factor)
        min_sz = torch.full_like(x, self.min_bbox_size)

        # Current center
        cx = x + w / 2
        cy = y + h / 2

        # 0: move right
        cx0 = torch.minimum(cx + step, width_limit - w / 2);  cy0 = cy;  w0 = w;  h0 = h
        x0 = torch.clamp(cx0 - w0 / 2, min=torch.zeros_like(cx0), max=width_limit - w0)
        y0 = torch.clamp(cy0 - h0 / 2, min=torch.zeros_like(cy0), max=height_limit - h0)

        # 1: move left
        cx1 = torch.maximum(cx - step, w / 2);  cy1 = cy;  w1 = w;  h1 = h
        x1 = torch.clamp(cx1 - w1 / 2, min=torch.zeros_like(cx1), max=width_limit - w1)
        y1 = torch.clamp(cy1 - h1 / 2, min=torch.zeros_like(cy1), max=height_limit - h1)

        # 2: move up
        cx2 = cx;  cy2 = torch.maximum(cy - step, h / 2);  w2 = w;  h2 = h
        x2 = torch.clamp(cx2 - w2 / 2, min=torch.zeros_like(cx2), max=width_limit - w2)
        y2 = torch.clamp(cy2 - h2 / 2, min=torch.zeros_like(cy2), max=height_limit - h2)

        # 3: move down
        cx3 = cx;  cy3 = torch.minimum(cy + step, height_limit - h / 2);  w3 = w;  h3 = h
        x3 = torch.clamp(cx3 - w3 / 2, min=torch.zeros_like(cx3), max=width_limit - w3)
        y3 = torch.clamp(cy3 - h3 / 2, min=torch.zeros_like(cy3), max=height_limit - h3)

        # 4: scale up (expand symmetrically)
        w4 = torch.clamp(w * scale, max=width_limit)
        h4 = torch.clamp(h * scale, max=height_limit)
        cx4 = torch.clamp(cx, min=w4 / 2, max=width_limit - w4 / 2)
        cy4 = torch.clamp(cy, min=h4 / 2, max=height_limit - h4 / 2)
        x4 = torch.clamp(cx4 - w4 / 2, min=torch.zeros_like(cx4), max=width_limit - w4)
        y4 = torch.clamp(cy4 - h4 / 2, min=torch.zeros_like(cy4), max=height_limit - h4)

        # 5: scale down (shrink symmetrically)
        w5 = torch.clamp(w / scale, min=min_sz)
        h5 = torch.clamp(h / scale, min=min_sz)
        cx5 = torch.clamp(cx, min=w5 / 2, max=width_limit - w5 / 2)
        cy5 = torch.clamp(cy, min=h5 / 2, max=height_limit - h5 / 2)
        x5 = torch.clamp(cx5 - w5 / 2, min=torch.zeros_like(cx5), max=width_limit - w5)
        y5 = torch.clamp(cy5 - h5 / 2, min=torch.zeros_like(cy5), max=height_limit - h5)

        # 6: shrink top/bottom (reduce height, increase width)
        w6 = torch.clamp(w * scale, max=width_limit)
        h6 = torch.clamp(h / scale, min=min_sz)
        cx6 = torch.clamp(cx, min=w6 / 2, max=width_limit - w6 / 2)
        cy6 = torch.clamp(cy, min=h6 / 2, max=height_limit - h6 / 2)
        x6 = torch.clamp(cx6 - w6 / 2, min=torch.zeros_like(cx6), max=width_limit - w6)
        y6 = torch.clamp(cy6 - h6 / 2, min=torch.zeros_like(cy6), max=height_limit - h6)

        # 7: shrink left/right (reduce width, increase height)
        w7 = torch.clamp(w / scale, min=min_sz)
        h7 = torch.clamp(h * scale, max=height_limit)
        cx7 = torch.clamp(cx, min=w7 / 2, max=width_limit - w7 / 2)
        cy7 = torch.clamp(cy, min=h7 / 2, max=height_limit - h7 / 2)
        x7 = torch.clamp(cx7 - w7 / 2, min=torch.zeros_like(cx7), max=width_limit - w7)
        y7 = torch.clamp(cy7 - h7 / 2, min=torch.zeros_like(cy7), max=height_limit - h7)

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
            cur = self.current_bboxes.unsqueeze(1)  # STOP corresponds to "keep current box"
            cand = torch.cat([cand, cur], dim=1)    # now shape (B, 9, 4); index 8 == STOP
        ious = self._iou_candidates(cand)
        return ious.argmax(dim=1)

    def render(self, index: int = 0, mode: str = "rgb_array"):
        """Simple rendering function for visualization (placeholder implementation)."""
        if mode == "rgb_array":
            # Convert grayscale image to RGB for visualization
            image = self.images[index, 0].cpu().numpy()  # (H, W)
            # Normalize to 0-255
            image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype('uint8')
            # Convert to RGB
            rgb_image = np.stack([image, image, image], axis=-1)  # (H, W, 3)
            
            # Draw current bounding box in red
            bbox = self.current_bboxes[index].cpu().numpy()
            x, y, w, h = bbox.astype(int)
            
            # Simple box drawing (just outline)
            rgb_image[y:y+2, x:x+int(w)] = [255, 0, 0]  # top
            rgb_image[y+int(h)-2:y+int(h), x:x+int(w)] = [255, 0, 0]  # bottom
            rgb_image[y:y+int(h), x:x+2] = [255, 0, 0]  # left
            rgb_image[y:y+int(h), x+int(w)-2:x+int(w)] = [255, 0, 0]  # right
            
            # Draw ground truth bounding box in green if tumor exists
            if self.has_tumor[index]:
                gt_bbox = self.gt_bboxes[index].cpu().numpy()
                gt_x, gt_y, gt_w, gt_h = gt_bbox.astype(int)
                
                rgb_image[gt_y:gt_y+2, gt_x:gt_x+int(gt_w)] = [0, 255, 0]  # top
                rgb_image[gt_y+int(gt_h)-2:gt_y+int(gt_h), gt_x:gt_x+int(gt_w)] = [0, 255, 0]  # bottom
                rgb_image[gt_y:gt_y+int(gt_h), gt_x:gt_x+2] = [0, 255, 0]  # left
                rgb_image[gt_y:gt_y+int(gt_h), gt_x+int(gt_w)-2:gt_x+int(gt_w)] = [0, 255, 0]  # right
                
            return rgb_image
        else:
            raise NotImplementedError(f"Render mode '{mode}' not supported")