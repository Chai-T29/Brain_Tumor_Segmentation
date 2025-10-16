from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
from matplotlib import patches


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
    min_edge_length: float = 1.0


class PolygonLocalizationEnv:
    """Environment for continuous control of a polygonal tumor localization agent."""

    def __init__(self, config: EnvironmentConfig) -> None:
        if config.num_sides < 6 or config.num_sides % 2 != 0:
            raise ValueError("num_sides must be an even integer >= 6.")
        self.config = config
        self.num_sides = config.num_sides
        self.num_controlled_sides = self.num_sides // 2
        self.action_dim = self.num_controlled_sides * 3 + 1  # (radial, rotation, length) × controlled sides + stop
        base_indices = torch.arange(0, self.num_sides, 2, dtype=torch.long)
        self._control_v0_idx = base_indices
        self._control_v1_idx = (base_indices + 1) % self.num_sides
        self._edge_pairs: torch.Tensor | None = self._build_edge_pairs(self.num_sides)
        self._edge_pairs_cache: Dict[torch.device, torch.Tensor] = {}

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
        self._grid_xs: torch.Tensor | None = None
        self._grid_ys: torch.Tensor | None = None
        self._grid_axes_cache: Dict[torch.device, Tuple[torch.Tensor, torch.Tensor]] = {}

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
        self._grid_xs = torch.arange(self.width, dtype=torch.float32)
        self._grid_ys = torch.arange(self.height, dtype=torch.float32)
        self._grid_axes_cache.clear()

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
    # Geometry helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_edge_pairs(num_sides: int) -> torch.Tensor | None:
        if num_sides < 4:
            return None
        pairs = torch.triu_indices(num_sides, num_sides, offset=1)
        i_idx, j_idx = pairs[0], pairs[1]
        mask = (j_idx - i_idx) > 1
        if num_sides > 1:
            mask &= ~((i_idx == 0) & (j_idx == num_sides - 1))
        filtered_i = i_idx[mask]
        if filtered_i.numel() == 0:
            return torch.empty(0, 2, dtype=torch.long)
        filtered_j = j_idx[mask]
        return torch.stack([filtered_i, filtered_j], dim=1)

    def _is_simple_polygon(self, vertices: torch.Tensor) -> bool:
        n = vertices.size(0)
        if n < 4:
            return True

        if self._edge_pairs is None or self._edge_pairs.numel() == 0:
            return True

        device = vertices.device
        dtype = vertices.dtype

        if device not in self._edge_pairs_cache:
            self._edge_pairs_cache[device] = self._edge_pairs.to(device)
        edge_pairs = self._edge_pairs_cache[device]

        i_idx = edge_pairs[:, 0]
        j_idx = edge_pairs[:, 1]
        i_next = (i_idx + 1) % n
        j_next = (j_idx + 1) % n

        p1 = vertices.index_select(0, i_idx)
        p2 = vertices.index_select(0, i_next)
        q1 = vertices.index_select(0, j_idx)
        q2 = vertices.index_select(0, j_next)

        def orient(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])

        eps = torch.tensor(1e-6, dtype=dtype, device=device)
        o1 = orient(p1, p2, q1)
        o2 = orient(p1, p2, q2)
        o3 = orient(q1, q2, p1)
        o4 = orient(q1, q2, p2)

        def on_segment(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            return (
                (torch.minimum(a[:, 0], c[:, 0]) - eps <= b[:, 0])
                & (b[:, 0] <= torch.maximum(a[:, 0], c[:, 0]) + eps)
                & (torch.minimum(a[:, 1], c[:, 1]) - eps <= b[:, 1])
                & (b[:, 1] <= torch.maximum(a[:, 1], c[:, 1]) + eps)
            )

        colinear_intersections = (
            (torch.abs(o1) <= eps) & on_segment(p1, q1, p2)
        ) | (
            (torch.abs(o2) <= eps) & on_segment(p1, q2, p2)
        ) | (
            (torch.abs(o3) <= eps) & on_segment(q1, p1, q2)
        ) | (
            (torch.abs(o4) <= eps) & on_segment(q1, p2, q2)
        )

        general_intersections = ((o1 > 0) != (o2 > 0)) & ((o3 > 0) != (o4 > 0))

        if torch.any(colinear_intersections) or torch.any(general_intersections):
            return False
        return True

    def _try_length_step(
        self,
        row_vertices: torch.Tensor,
        v0_idx: int,
        v1_idx: int,
        step: float,
    ) -> bool:
        if abs(step) <= 1e-8:
            return True

        edge_vec = row_vertices[v1_idx] - row_vertices[v0_idx]
        length = torch.linalg.norm(edge_vec).item()
        if length < 1e-6:
            return False

        new_length = length + 2.0 * step
        if new_length < self.config.min_edge_length:
            return False

        direction = edge_vec / length
        delta_vec = direction * step

        original_v0 = row_vertices[v0_idx].clone()
        original_v1 = row_vertices[v1_idx].clone()

        row_vertices[v0_idx] = original_v0 - delta_vec
        row_vertices[v1_idx] = original_v1 + delta_vec

        if torch.linalg.norm(row_vertices[v1_idx] - row_vertices[v0_idx]).item() < self.config.min_edge_length:
            row_vertices[v0_idx] = original_v0
            row_vertices[v1_idx] = original_v1
            return False

        if not self._is_simple_polygon(row_vertices):
            row_vertices[v0_idx] = original_v0
            row_vertices[v1_idx] = original_v1
            return False

        return True

    def _try_rotation_step(
        self,
        row_vertices: torch.Tensor,
        center_ref: torch.Tensor,
        v0_idx: int,
        v1_idx: int,
        step_deg: float,
    ) -> bool:
        if abs(step_deg) <= 1e-8:
            return True

        midpoint = (row_vertices[v0_idx] + row_vertices[v1_idx]) / 2.0
        center_vec = midpoint - center_ref
        center_norm = torch.linalg.norm(center_vec).item()
        if center_norm < 1e-6:
            return False

        edge_vec = row_vertices[v1_idx] - row_vertices[v0_idx]
        current_diff = math.atan2(edge_vec[1].item(), edge_vec[0].item()) - math.atan2(
            center_vec[1].item(), center_vec[0].item()
        )
        current_diff = ((current_diff + math.pi) % (2 * math.pi)) - math.pi

        target_diff = current_diff + math.radians(step_deg)
        max_diff = math.pi / 2 - 1e-3
        if target_diff > max_diff or target_diff < -max_diff:
            return False

        cos_theta = math.cos(math.radians(step_deg))
        sin_theta = math.sin(math.radians(step_deg))
        rotation_matrix = torch.tensor(
            [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
            dtype=row_vertices.dtype,
            device=row_vertices.device,
        )

        original_v0 = row_vertices[v0_idx].clone()
        original_v1 = row_vertices[v1_idx].clone()

        for vidx in (v0_idx, v1_idx):
            rel = (row_vertices[vidx] - midpoint).unsqueeze(1)
            rotated = (rotation_matrix @ rel).squeeze(1)
            row_vertices[vidx] = midpoint + rotated

        if torch.linalg.norm(row_vertices[v1_idx] - row_vertices[v0_idx]).item() < self.config.min_edge_length:
            row_vertices[v0_idx] = original_v0
            row_vertices[v1_idx] = original_v1
            return False

        if not self._is_simple_polygon(row_vertices):
            row_vertices[v0_idx] = original_v0
            row_vertices[v1_idx] = original_v1
            return False

        return True

    def _try_radial_step(
        self,
        row_vertices: torch.Tensor,
        center_ref: torch.Tensor,
        v0_idx: int,
        v1_idx: int,
        step: float,
    ) -> bool:
        if abs(step) <= 1e-8:
            return True

        min_radius = max(1.0, self.config.min_edge_length)
        original_v0 = row_vertices[v0_idx].clone()
        original_v1 = row_vertices[v1_idx].clone()

        new_positions = {}
        for vidx in (v0_idx, v1_idx):
            vec = row_vertices[vidx] - center_ref
            norm = torch.linalg.norm(vec).item()
            if norm < 1e-6:
                row_vertices[v0_idx] = original_v0
                row_vertices[v1_idx] = original_v1
                return False
            new_norm = norm + step
            if new_norm < min_radius:
                row_vertices[v0_idx] = original_v0
                row_vertices[v1_idx] = original_v1
                return False
            new_positions[vidx] = center_ref + vec / norm * new_norm

        row_vertices[v0_idx] = new_positions[v0_idx]
        row_vertices[v1_idx] = new_positions[v1_idx]

        if torch.linalg.norm(row_vertices[v1_idx] - row_vertices[v0_idx]).item() < self.config.min_edge_length:
            row_vertices[v0_idx] = original_v0
            row_vertices[v1_idx] = original_v1
            return False

        if not self._is_simple_polygon(row_vertices):
            row_vertices[v0_idx] = original_v0
            row_vertices[v1_idx] = original_v1
            return False

        return True
    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

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
        vertices = self.vertices.clone()
        batch_size = vertices.size(0)
        side_actions = side_actions.view(batch_size, self.num_controlled_sides, 3)
        centers = self.vertices.mean(dim=1, keepdim=True).detach().clone()

        active_rows = active.nonzero(as_tuple=False).squeeze(1)
        if active_rows.numel() == 0:
            return

        scale_factors = side_actions.new_tensor(
            [
                self.config.radial_step_scale,
                self.config.rotation_step_scale_deg,
                self.config.length_step_scale,
            ]
        )

        for batch_idx in active_rows.tolist():
            row_vertices = vertices[batch_idx]
            center_ref = centers[batch_idx, 0].clone()
            params = side_actions[batch_idx] * scale_factors
            updated_vertices = self._apply_actions_single(row_vertices, center_ref, params)
            vertices[batch_idx] = updated_vertices

        if self.width is None or self.height is None:
            raise RuntimeError("Image dimensions unknown.")
        vertices[..., 0].clamp_(0.0, float(self.width - 1))
        vertices[..., 1].clamp_(0.0, float(self.height - 1))
        self.vertices = vertices

    def _apply_actions_single(
        self,
        base_vertices: torch.Tensor,
        center_ref: torch.Tensor,
        scaled_params: torch.Tensor,
    ) -> torch.Tensor:
        """Apply all side actions for a single polygon via batched updates and minimal retries."""
        if torch.all(torch.abs(scaled_params) <= 1e-8):
            return base_vertices.clone()

        candidate = self._simulate_actions(base_vertices, center_ref, scaled_params, scale=1.0)
        if candidate is not None:
            return candidate

        best_vertices = base_vertices.clone()
        low = 0.0
        high = 1.0
        tolerance = 1e-2
        max_iterations = 12

        for _ in range(max_iterations):
            mid = 0.5 * (low + high)
            if mid <= 1e-4:
                break

            candidate = self._simulate_actions(base_vertices, center_ref, scaled_params, scale=mid)
            if candidate is not None:
                best_vertices = candidate
                low = mid
            else:
                high = mid

            if (high - low) <= tolerance:
                break

        return best_vertices

    def _simulate_actions(
        self,
        base_vertices: torch.Tensor,
        center_ref: torch.Tensor,
        scaled_params: torch.Tensor,
        scale: float,
    ) -> torch.Tensor | None:
        """Return updated vertices for a given action scale, or None if constraints are violated."""
        device = base_vertices.device
        v0_idx = self._control_v0_idx.to(device)
        v1_idx = self._control_v1_idx.to(device)

        params = scaled_params * scale
        vertices = base_vertices.clone()
        if torch.isnan(vertices).any():
            return None

        radial_params = params[:, 0]
        rotation_params = params[:, 1]
        length_params = params[:, 2]

        if torch.any(torch.abs(length_params) > 1e-8):
            v0 = vertices[v0_idx]
            v1 = vertices[v1_idx]
            edge_vec = v1 - v0
            edge_len = torch.linalg.norm(edge_vec, dim=-1, keepdim=True)
            direction = edge_vec / edge_len.clamp_min(1e-6)
            length_delta = length_params.unsqueeze(-1)
            min_delta = 0.5 * (self.config.min_edge_length - edge_len)
            length_delta = torch.maximum(length_delta, min_delta)
            v0 = v0 - direction * length_delta
            v1 = v1 + direction * length_delta
            vertices[v0_idx] = v0
            vertices[v1_idx] = v1

        if torch.any(torch.abs(rotation_params) > 1e-8):
            v0 = vertices[v0_idx]
            v1 = vertices[v1_idx]
            midpoint = (v0 + v1) / 2.0
            center_vec = midpoint - center_ref.to(device).unsqueeze(0)
            center_norm = torch.linalg.norm(center_vec, dim=-1, keepdim=True)
            valid_center = (center_norm.squeeze(-1) >= 1e-6).unsqueeze(-1)

            edge_vec = v1 - v0
            edge_angle = torch.atan2(edge_vec[:, 1], edge_vec[:, 0])
            center_angle = torch.atan2(center_vec[:, 1], center_vec[:, 0])
            current_diff = edge_angle - center_angle
            current_diff = torch.remainder(current_diff + math.pi, 2 * math.pi) - math.pi

            rotation_radians = rotation_params * (math.pi / 180.0)
            target_diff = current_diff + rotation_radians
            max_diff = math.pi / 2 - 1e-3
            target_diff = torch.clamp(target_diff, -max_diff, max_diff)
            applied_rotation = target_diff - current_diff
            applied_rotation = applied_rotation.unsqueeze(-1) * valid_center

            cos_theta = torch.cos(applied_rotation)
            sin_theta = torch.sin(applied_rotation)
            rel0 = v0 - midpoint
            rel1 = v1 - midpoint
            cos_val = cos_theta.squeeze(-1)
            sin_val = sin_theta.squeeze(-1)
            rot_rel0 = torch.stack(
                [
                    rel0[:, 0] * cos_val - rel0[:, 1] * sin_val,
                    rel0[:, 0] * sin_val + rel0[:, 1] * cos_val,
                ],
                dim=-1,
            )
            rot_rel1 = torch.stack(
                [
                    rel1[:, 0] * cos_val - rel1[:, 1] * sin_val,
                    rel1[:, 0] * sin_val + rel1[:, 1] * cos_val,
                ],
                dim=-1,
            )
            rot_rel0 = torch.where(valid_center, rot_rel0, rel0)
            rot_rel1 = torch.where(valid_center, rot_rel1, rel1)
            v0 = midpoint + rot_rel0
            v1 = midpoint + rot_rel1
            vertices[v0_idx] = v0
            vertices[v1_idx] = v1

        if torch.any(torch.abs(radial_params) > 1e-8):
            v0 = vertices[v0_idx]
            v1 = vertices[v1_idx]
            center = center_ref.to(device).unsqueeze(0)
            vec0 = v0 - center
            vec1 = v1 - center
            norm0 = torch.linalg.norm(vec0, dim=-1, keepdim=True)
            norm1 = torch.linalg.norm(vec1, dim=-1, keepdim=True)
            valid0 = (norm0.squeeze(-1) >= 1e-6).unsqueeze(-1)
            valid1 = (norm1.squeeze(-1) >= 1e-6).unsqueeze(-1)

            min_radius = max(1.0, self.config.min_edge_length)
            radial_delta = radial_params.unsqueeze(-1)
            new_norm0 = norm0 + radial_delta
            new_norm1 = norm1 + radial_delta
            new_norm0 = torch.maximum(new_norm0, torch.full_like(new_norm0, min_radius))
            new_norm1 = torch.maximum(new_norm1, torch.full_like(new_norm1, min_radius))

            unit0 = vec0 / norm0.clamp_min(1e-6)
            unit1 = vec1 / norm1.clamp_min(1e-6)
            updated0 = center + unit0 * new_norm0
            updated1 = center + unit1 * new_norm1
            v0 = torch.where(valid0, updated0, v0)
            v1 = torch.where(valid1, updated1, v1)
            vertices[v0_idx] = v0
            vertices[v1_idx] = v1

        if torch.isnan(vertices).any():
            return None

        edge_vec = vertices[v1_idx] - vertices[v0_idx]
        edge_len = torch.linalg.norm(edge_vec, dim=-1)
        if torch.any(edge_len < (self.config.min_edge_length - 1e-6)):
            return None

        if not self._is_simple_polygon(vertices):
            return None

        return vertices

    def _calculate_iou(self, vertices: torch.Tensor) -> torch.Tensor:
        if self.masks is None or self.height is None or self.width is None:
            raise RuntimeError("Environment not initialised.")

        poly_masks = self._rasterize_polygons(vertices)
        gt_masks = (self.masks.squeeze(1) > 0.5).to(poly_masks.device).to(poly_masks.dtype)
        intersection = (poly_masks * gt_masks).sum(dim=(1, 2))
        union = poly_masks.sum(dim=(1, 2)) + gt_masks.sum(dim=(1, 2)) - intersection
        iou = torch.where(union > 0, intersection / union, torch.zeros_like(union))
        return iou

    def _rasterize_polygons(self, vertices: torch.Tensor) -> torch.Tensor:
        if self.height is None or self.width is None:
            raise RuntimeError("Image dimensions unknown.")
        if self._grid_xs is None or self._grid_ys is None:
            raise RuntimeError("Grid axes not initialised.")

        device = vertices.device
        dtype = vertices.dtype
        batch_size, num_vertices = vertices.shape[0], vertices.shape[1]

        xs_cache, ys_cache = self._grid_axes_cache.get(device, (None, None))
        if xs_cache is None or ys_cache is None or xs_cache.dtype != dtype:
            xs_cache = self._grid_xs.to(device=device, dtype=dtype)
            ys_cache = self._grid_ys.to(device=device, dtype=dtype)
            self._grid_axes_cache[device] = (xs_cache, ys_cache)
        xs, ys = xs_cache, ys_cache

        v1 = vertices
        v2 = torch.roll(vertices, shifts=-1, dims=1)

        x1 = v1[..., 0]
        y1 = v1[..., 1]
        x2 = v2[..., 0]
        y2 = v2[..., 1]

        denom = y2 - y1
        zero_denom = torch.abs(denom) < 1e-6
        denom_safe = torch.where(zero_denom, torch.ones_like(denom), denom)
        slopes = (x2 - x1) / denom_safe
        slopes = torch.where(zero_denom, torch.zeros_like(slopes), slopes)

        max_pairs = num_vertices // 2
        if max_pairs == 0:
            return torch.zeros(batch_size, self.height, self.width, device=device, dtype=dtype)

        row_masks: list[torch.Tensor] = []
        x_coords = xs.view(1, 1, -1)
        pair_indices = torch.arange(max_pairs, device=device)
        inf_value = torch.tensor(float("inf"), device=device, dtype=dtype)

        for y_val in ys:
            y_scalar = y_val.to(device=device, dtype=dtype)
            active = ((y1 <= y_scalar) & (y2 > y_scalar)) | ((y2 <= y_scalar) & (y1 > y_scalar))
            x_crossings = (y_scalar - y1) * slopes + x1
            x_crossings = torch.where(active, x_crossings, inf_value.expand_as(x_crossings))

            x_sorted, _ = torch.sort(x_crossings, dim=1)
            required_cols = max_pairs * 2
            if x_sorted.size(1) < required_cols:
                pad_cols = required_cols - x_sorted.size(1)
                x_sorted = F.pad(x_sorted, (0, pad_cols), value=float("inf"))
            else:
                x_sorted = x_sorted[:, :required_cols]

            x_pairs = x_sorted.view(batch_size, max_pairs, 2)
            x_start = x_pairs[:, :, 0]
            x_end = x_pairs[:, :, 1]

            valid_pairs = (active.sum(dim=1) // 2).clamp(max=max_pairs)
            pair_mask = pair_indices.view(1, -1) < valid_pairs.unsqueeze(1)

            seg_mask = pair_mask.unsqueeze(-1) & (x_coords >= x_start.unsqueeze(-1)) & (x_coords < x_end.unsqueeze(-1))
            row_mask = seg_mask.any(dim=1).to(dtype)
            row_masks.append(row_mask)

        mask_tensor = torch.stack(row_masks, dim=0).transpose(0, 1).contiguous()
        return mask_tensor

    def render(self, index: int = 0, mode: str = "rgb_array"):
        if self.images is None or self.vertices is None:
            raise RuntimeError("Environment must be reset before rendering.")
        if not 0 <= index < self.images.size(0):
            raise IndexError("Render index out of range.")

        image = self.images[index].detach().cpu().squeeze(0)
        img_np = image.numpy()
        img_min, img_max = float(img_np.min()), float(img_np.max())
        if img_max > img_min:
            img_np = (img_np - img_min) / (img_max - img_min)
        else:
            img_np = np.zeros_like(img_np)

        vertices = self.vertices[index].detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_np, cmap="gray")
        if self.masks is not None:
            mask = self.masks[index].detach().cpu().squeeze(0).numpy()
            ax.imshow(mask, cmap="Reds", alpha=0.25)

        polygon = patches.Polygon(vertices, closed=True, fill=False, edgecolor="cyan", linewidth=2.0)
        ax.add_patch(polygon)
        ax.axis("off")
        fig.tight_layout(pad=0)

        if mode == "human":
            plt.show()
            plt.close(fig)
            return None

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        return frame
