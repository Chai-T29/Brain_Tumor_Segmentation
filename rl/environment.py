from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from matplotlib.path import Path


@dataclass
class EnvironmentConfig:
    num_sides: int = 32
    max_steps: int = 50
    iou_threshold: float = 0.7
    initial_radius: float = 90.0
    radial_step_scale: float = 8.0
    rotation_step_scale_deg: float = 15.0
    length_step_scale: float = 6.0
    stop_action_threshold: float = 0.7
    reward_success: float = 4.0
    reward_no_tumor: float = 2.0
    reward_false_stop: float = -3.0
    time_penalty: float = 0.02
    hold_penalty: float = 0.5


class PolygonLocalizationEnv:
    """Environment for continuous control of a polygonal tumor localization agent."""

    def __init__(self, config: EnvironmentConfig) -> None:
        if config.num_sides < 6 or config.num_sides % 2 != 0:
            raise ValueError("num_sides must be an even integer >= 6.")
        self.config = config
        self.num_sides = config.num_sides
        self.num_controlled_sides = self.num_sides // 2
        self.action_dim = self.num_controlled_sides * 3 + 1  # (radial, rotation, length) × controlled sides + stop

        # State buffers initialised in reset.
        self.images: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.vertices: torch.Tensor | None = None  # [B, num_sides, 2]
        self.last_iou: torch.Tensor | None = None
        self.step_count: torch.Tensor | None = None
        self.active_mask: torch.Tensor | None = None
        self.has_tumor: torch.Tensor | None = None

        self.height: int | None = None
        self.width: int | None = None
        self._grid_coords: np.ndarray | None = None

    def reset(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4 or images.size(1) != 1:
            raise ValueError("Images must be [B, 1, H, W].")
        if masks.dim() != 4 or masks.size(1) != 1:
            raise ValueError("Masks must be [B, 1, H, W].")
        if images.shape != masks.shape:
            raise ValueError("Images and masks must share the same shape.")

        batch_size, _, height, width = images.shape
        self.images = images.clone()
        self.masks = masks.clone()
        self.height = int(height)
        self.width = int(width)
        self._grid_coords = self._build_grid_coords(self.height, self.width)

        self.vertices = self._initialise_vertices(batch_size, height, width)
        self.has_tumor = (masks.view(batch_size, -1).sum(dim=1) > 0).to(torch.bool)
        self.last_iou = self._calculate_iou(self.vertices)
        self.step_count = torch.zeros(batch_size, dtype=torch.long)
        self.active_mask = torch.ones(batch_size, dtype=torch.bool)

        return self._polygon_state()

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.vertices is None or self.last_iou is None or self.active_mask is None:
            raise RuntimeError("Environment must be reset before stepping.")

        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if actions.size(-1) != self.action_dim:
            raise ValueError(f"Expected action dimension {self.action_dim}, got {actions.size(-1)}.")

        actions = actions.to(torch.float32)
        batch_size = actions.size(0)
        if batch_size != self.vertices.size(0):
            raise ValueError("Action batch does not match environment batch size.")

        active = self.active_mask.clone()
        if active.any():
            self._apply_side_actions(actions[:, :-1], active=active)

        stop_values = actions[:, -1]
        stop_threshold = self.config.stop_action_threshold
        stop_mask = (stop_values > stop_threshold) & active

        self.step_count = self.step_count + active.long()
        timeout_mask = (self.step_count >= self.config.max_steps) & active

        current_iou = self._calculate_iou(self.vertices)
        rewards, success_mask = self._compute_rewards(
            current_iou=current_iou,
            active_mask=active,
            stop_mask=stop_mask,
        )

        done = stop_mask | timeout_mask
        self.active_mask = active & ~done
        self.last_iou = torch.where(active, current_iou, self.last_iou)

        next_state = self._polygon_state()
        info = {"iou": current_iou.detach(), "success": success_mask.detach()}
        return next_state, rewards.detach(), done.detach(), info

    def _compute_rewards(
        self,
        current_iou: torch.Tensor,
        active_mask: torch.Tensor,
        stop_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.last_iou is None or self.has_tumor is None:
            raise RuntimeError("Environment must be reset before computing rewards.")

        delta_iou = current_iou - self.last_iou
        rewards = torch.where(active_mask, delta_iou, torch.zeros_like(delta_iou))

        tumor_active = self.has_tumor & active_mask
        no_tumor_active = (~self.has_tumor) & active_mask

        success_mask = stop_mask & tumor_active & (current_iou >= self.config.iou_threshold)
        no_tumor_stop = stop_mask & no_tumor_active
        false_stop = stop_mask & tumor_active & (current_iou < self.config.iou_threshold)

        rewards = rewards + torch.where(success_mask, torch.full_like(rewards, self.config.reward_success), 0.0)
        rewards = rewards + torch.where(no_tumor_stop, torch.full_like(rewards, self.config.reward_no_tumor), 0.0)
        rewards = rewards + torch.where(false_stop, torch.full_like(rewards, self.config.reward_false_stop), 0.0)

        ongoing_mask = active_mask & (~stop_mask)
        rewards = rewards - torch.where(ongoing_mask & tumor_active, torch.full_like(rewards, self.config.time_penalty), 0.0)
        rewards = rewards - torch.where(ongoing_mask & no_tumor_active, torch.full_like(rewards, self.config.hold_penalty), 0.0)

        return rewards, success_mask

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_grid_coords(self, height: int, width: int) -> np.ndarray:
        ys, xs = np.mgrid[0:height, 0:width]
        coords = np.stack([xs, ys], axis=-1).reshape(-1, 2)
        return coords.astype(np.float32)

    def _initialise_vertices(self, batch_size: int, height: int, width: int) -> torch.Tensor:
        radius = self.config.initial_radius
        max_radius = 0.45 * float(min(height, width))
        if radius <= 0 or radius > max_radius:
            radius = max_radius

        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0

        angles = torch.linspace(0, 2 * math.pi, steps=self.num_sides + 1)[:-1]
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)

        vx = center_x + radius * cos_vals
        vy = center_y + radius * sin_vals
        vertices = torch.stack([vx, vy], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1)
        return vertices.to(torch.float32)

    def _polygon_state(self) -> torch.Tensor:
        if self.vertices is None:
            raise RuntimeError("Environment not initialised.")
        v = self.vertices
        v_next = torch.roll(v, shifts=-1, dims=1)
        midpoints = (v + v_next) / 2.0
        edges = v_next - v
        lengths = torch.linalg.norm(edges, dim=-1, keepdim=True)
        orientations = torch.atan2(edges[..., 1], edges[..., 0]).unsqueeze(-1) * (180.0 / math.pi)
        features = torch.cat([midpoints, orientations, lengths], dim=-1)
        return features.view(v.size(0), -1)

    def _apply_side_actions(self, side_actions: torch.Tensor, active: torch.Tensor) -> None:
        if self.vertices is None:
            raise RuntimeError("Environment not initialised.")
        vertices = self.vertices
        batch_size = vertices.size(0)
        side_actions = side_actions.view(batch_size, self.num_controlled_sides, 3)
        center = vertices.mean(dim=1, keepdim=True)

        active_rows = active.nonzero(as_tuple=False).squeeze(1)
        if active_rows.numel() == 0:
            return

        for control_idx in range(self.num_controlled_sides):
            v0_idx = (2 * control_idx) % self.num_sides
            v1_idx = (v0_idx + 1) % self.num_sides

            radial_delta = side_actions[:, control_idx, 0] * self.config.radial_step_scale
            rotation_delta = side_actions[:, control_idx, 1] * self.config.rotation_step_scale_deg
            length_delta = side_actions[:, control_idx, 2] * self.config.length_step_scale

            # Radial move
            for vidx in (v0_idx, v1_idx):
                vec = vertices[:, vidx, :] - center.squeeze(1)
                norm = vec.norm(dim=1).clamp(min=1e-6)
                direction = vec / norm.unsqueeze(1)
                update = direction * radial_delta.unsqueeze(1)
                vertices[active_rows, vidx, :] += update[active_rows]

            # Rotation around the edge midpoint
            midpoint = (vertices[:, v0_idx, :] + vertices[:, v1_idx, :]) / 2.0
            theta = rotation_delta * (math.pi / 180.0)
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            for vidx in (v0_idx, v1_idx):
                rel = vertices[:, vidx, :] - midpoint
                x_new = rel[:, 0] * cos_theta - rel[:, 1] * sin_theta
                y_new = rel[:, 0] * sin_theta + rel[:, 1] * cos_theta
                rotated = torch.stack([x_new, y_new], dim=1)
                vertices[active_rows, vidx, :] = midpoint[active_rows] + rotated[active_rows]

            # Length adjustment along the edge direction
            edge_vec = vertices[:, v1_idx, :] - vertices[:, v0_idx, :]
            edge_norm = edge_vec.norm(dim=1).clamp(min=1e-6)
            edge_dir = edge_vec / edge_norm.unsqueeze(1)
            delta = edge_dir * length_delta.unsqueeze(1)
            vertices[active_rows, v0_idx, :] -= delta[active_rows]
            vertices[active_rows, v1_idx, :] += delta[active_rows]

        # Clamp to image bounds
        if self.width is None or self.height is None:
            raise RuntimeError("Image dimensions unknown.")
        vertices[..., 0].clamp_(0.0, float(self.width - 1))
        vertices[..., 1].clamp_(0.0, float(self.height - 1))
        self.vertices = vertices

    def _calculate_iou(self, vertices: torch.Tensor) -> torch.Tensor:
        if self.masks is None or self._grid_coords is None or self.height is None or self.width is None:
            raise RuntimeError("Environment not initialised.")

        batch_size = vertices.size(0)
        poly_masks = []
        for b in range(batch_size):
            poly = vertices[b].detach().cpu().numpy()
            path = Path(poly)
            mask_flat = path.contains_points(self._grid_coords, radius=-1e-9)
            mask = torch.from_numpy(mask_flat.reshape(self.height, self.width)).to(torch.float32)
            poly_masks.append(mask)
        poly_masks_t = torch.stack(poly_masks, dim=0)

        gt_masks = (self.masks.squeeze(1) > 0.5).to(torch.float32)
        intersection = (poly_masks_t * gt_masks).sum(dim=(1, 2))
        union = poly_masks_t.sum(dim=(1, 2)) + gt_masks.sum(dim=(1, 2)) - intersection
        iou = torch.where(union > 0, intersection / union, torch.zeros_like(union))
        return iou
