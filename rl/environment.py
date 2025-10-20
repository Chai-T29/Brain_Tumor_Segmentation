from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
from matplotlib import patches


@dataclass
class EnvironmentConfig:
    num_sides: int = 32
    max_steps: int = 50
    iou_low_threshold: float = 0.2
    iou_high_threshold: float = 0.7
    initial_radius: float = 90.0
    reward_success: float = 4.0
    reward_no_tumor: float = 2.0
    reward_false_stop: float = -3.0
    time_penalty: float = 0.02
    hold_penalty: float = 0.5
    line_distance_step_scale: float = 6.0
    line_angle_step_scale_deg: float = 5.0
    line_max_angle_offset_deg: float = 45.0
    line_min_distance: float = 0.0
    line_max_distance_margin: float = 1.0
    center_step_scale: float = 2.0


class PolygonLocalizationEnv:
    """Environment that controls supporting lines which define a convex polytope."""

    def __init__(self, config: EnvironmentConfig) -> None:
        if config.num_sides < 3:
            raise ValueError("num_sides must be >= 3.")
        self.config = config
        self.num_lines = int(config.num_sides)
        self.line_action_dim = self.num_lines * 2  # distance Δ, angle Δ
        self.center_action_dim = 2  # x and y adjustments for center
        self.action_dim = self.line_action_dim + self.center_action_dim + 1  # + stop score
        self.state_dim = self.num_lines * 2 + 2  # distances, angle offsets (degrees), center offsets

        base_angles = torch.linspace(
            0.0, 2 * math.pi, steps=self.num_lines + 1, dtype=torch.float32
        )[:-1]
        self._base_angles = base_angles
        self._angle_step = math.radians(config.line_angle_step_scale_deg)

        # State buffers initialised in reset.
        self.images: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.line_distances: torch.Tensor | None = None
        self.line_angle_offsets: torch.Tensor | None = None
        self.last_iou: torch.Tensor | None = None
        self.vertices: torch.Tensor | None = None
        self.step_count: torch.Tensor | None = None
        self.active_mask: torch.Tensor | None = None
        self.has_tumor: torch.Tensor | None = None

        self.height: int | None = None
        self.width: int | None = None
        self.base_center: torch.Tensor | None = None
        self.center_offsets: torch.Tensor | None = None
        self.center_positions: torch.Tensor | None = None
        self._pixel_coord_cache: Dict[torch.device, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._center_offset_min: torch.Tensor | None = None
        self._center_offset_max: torch.Tensor | None = None
        self._max_distance: float = 0.0

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
        base_center = torch.tensor(
            [(self.width - 1) / 2.0, (self.height - 1) / 2.0], dtype=torch.float32
        )
        self.base_center = base_center
        self.center_offsets = torch.zeros((batch_size, 2), dtype=torch.float32)
        self.center_positions = base_center.unsqueeze(0).repeat(batch_size, 1)

        self._center_offset_min = torch.tensor(
            [-base_center[0], -base_center[1]], dtype=torch.float32
        )
        self._center_offset_max = torch.tensor(
            [(self.width - 1) - base_center[0], (self.height - 1) - base_center[1]], dtype=torch.float32
        )

        self._pixel_coord_cache.clear()

        self.has_tumor = (masks.view(batch_size, -1).sum(dim=1) > 0).to(torch.bool)
        self._max_distance = float(min(self.width, self.height) / 2.0 - self.config.line_max_distance_margin)
        self._max_distance = max(self._max_distance, self.config.line_min_distance + 1.0)
        initial_distance = float(
            min(max(self.config.line_min_distance, self.config.initial_radius), self._max_distance)
        )
        self.line_distances = torch.full(
            (batch_size, self.num_lines), initial_distance, dtype=torch.float32
        )
        self.line_angle_offsets = torch.zeros(
            (batch_size, self.num_lines), dtype=torch.float32
        )
        self.step_count = torch.zeros(batch_size, dtype=torch.long)
        self.active_mask = torch.ones(batch_size, dtype=torch.bool)

        normals = self._compute_normals()
        poly_masks = self._rasterize_polytope(normals, self.line_distances)
        self.last_iou = self._calculate_iou_from_masks(poly_masks)
        self.vertices = self._compute_vertices(normals, self.line_distances)

        return self._polygon_state()

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if (
            self.line_distances is None
            or self.line_angle_offsets is None
            or self.last_iou is None
            or self.active_mask is None
        ):
            raise RuntimeError("Environment must be reset before stepping.")

        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if actions.size(-1) != self.action_dim:
            raise ValueError(f"Expected action dimension {self.action_dim}, got {actions.size(-1)}.")

        actions = actions.to(torch.float32)
        batch_size = actions.size(0)
        if batch_size != self.line_distances.size(0):
            raise ValueError("Action batch does not match environment batch size.")

        active = self.active_mask.clone()
        line_flat = actions[..., : self.line_action_dim]
        center_start = self.line_action_dim
        center_end = center_start + self.center_action_dim
        center_flat = actions[..., center_start:center_end]
        stop_scores = actions[..., center_end]
        manual_stop = (stop_scores > 0.0) & active

        line_components = line_flat.view(batch_size, self.num_lines, 2)
        update_mask = active & (~manual_stop)
        if update_mask.any():
            distance_delta = line_components[..., 0] * self.config.line_distance_step_scale
            angle_delta = line_components[..., 1] * self._angle_step
            updated_distances = self.line_distances[update_mask] + distance_delta[update_mask]
            self.line_distances[update_mask] = updated_distances.clamp(
                self.config.line_min_distance, self._max_distance
            )
            updated_angles = self.line_angle_offsets[update_mask] + angle_delta[update_mask]
            self.line_angle_offsets[update_mask] = updated_angles

            if self.center_offsets is None or self.center_positions is None:
                raise RuntimeError("Center offsets not initialised.")

            center_active = center_flat[update_mask]
            delta_x = center_active[..., 0] * self.config.center_step_scale
            delta_y = center_active[..., 1] * self.config.center_step_scale
            delta_center = torch.stack([delta_x, delta_y], dim=-1)

            offsets = self.center_offsets[update_mask] + delta_center
            if self._center_offset_min is None or self._center_offset_max is None:
                raise RuntimeError("Center offset bounds not initialised.")
            min_bounds = self._center_offset_min.to(offsets.device)
            max_bounds = self._center_offset_max.to(offsets.device)
            offsets = torch.maximum(offsets, min_bounds.unsqueeze(0))
            offsets = torch.minimum(offsets, max_bounds.unsqueeze(0))

            self.center_offsets[update_mask] = offsets

        self.step_count = self.step_count + active.long()
        timeout_mask = (self.step_count >= self.config.max_steps) & active

        if self.center_offsets is None:
            raise RuntimeError("Center offsets not initialised.")
        if self.base_center is None:
            raise RuntimeError("Base center not initialised.")
        base_center = self.base_center.to(self.center_offsets.device)
        self.center_positions = base_center.unsqueeze(0) + self.center_offsets

        normals = self._compute_normals()
        poly_masks = self._rasterize_polytope(normals, self.line_distances)
        current_iou = self._calculate_iou_from_masks(poly_masks)
        if self.last_iou is None:
            raise RuntimeError("last_iou not initialised.")
        delta_iou = current_iou - self.last_iou
        stop_mask = manual_stop
        rewards, success_mask = self._compute_rewards(
            current_iou=current_iou,
            delta_iou=delta_iou,
            active_mask=active,
            stop_mask=stop_mask,
        )

        done = stop_mask | timeout_mask
        self.active_mask = active & ~done
        self.last_iou = torch.where(active, current_iou, self.last_iou)

        self.vertices = self._compute_vertices(normals, self.line_distances)
        next_state = self._polygon_state()
        info = {
            "iou": current_iou.detach(),
            "success": success_mask.detach(),
            "delta_iou": delta_iou.detach(),
            "manual_stop": manual_stop.detach(),
        }
        return next_state, rewards.detach(), done.detach(), info

    def _compute_rewards(
        self,
        current_iou: torch.Tensor,
        delta_iou: torch.Tensor,
        active_mask: torch.Tensor,
        stop_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.last_iou is None or self.has_tumor is None:
            raise RuntimeError("Environment must be reset before computing rewards.")
        rewards = torch.where(active_mask, delta_iou, torch.zeros_like(delta_iou))

        tumor_active = self.has_tumor & active_mask
        no_tumor_active = (~self.has_tumor) & active_mask

        iou_low = torch.tensor(self.config.iou_low_threshold, dtype=current_iou.dtype, device=current_iou.device)
        iou_high = torch.tensor(self.config.iou_high_threshold, dtype=current_iou.dtype, device=current_iou.device)
        threshold_span = torch.clamp(iou_high - iou_low, min=1e-6)

        success_mask = stop_mask & tumor_active & (current_iou >= iou_high)
        partial_stop = stop_mask & tumor_active & (current_iou >= iou_low) & (current_iou < iou_high)
        no_tumor_stop = stop_mask & no_tumor_active
        false_stop = stop_mask & tumor_active & (current_iou < iou_low)

        rewards = rewards + torch.where(success_mask, torch.full_like(rewards, self.config.reward_success), 0.0)
        partial_fraction = ((current_iou - iou_low) / threshold_span).clamp(0.0, 1.0)
        partial_reward = partial_fraction * self.config.reward_success
        rewards = rewards + torch.where(partial_stop, partial_reward, torch.zeros_like(rewards))
        rewards = rewards + torch.where(no_tumor_stop, torch.full_like(rewards, self.config.reward_no_tumor), 0.0)
        rewards = rewards + torch.where(false_stop, torch.full_like(rewards, self.config.reward_false_stop), 0.0)

        ongoing_mask = active_mask & (~stop_mask)
        rewards = rewards - torch.where(ongoing_mask & tumor_active, torch.full_like(rewards, self.config.time_penalty), 0.0)
        rewards = rewards - torch.where(ongoing_mask & no_tumor_active, torch.full_like(rewards, self.config.hold_penalty), 0.0)

        return rewards, success_mask

    def _compute_normals(self) -> torch.Tensor:
        if self.line_angle_offsets is None:
            raise RuntimeError("Line offsets not initialised.")
        total_angles = self._base_angles.unsqueeze(0) + self.line_angle_offsets
        cos_vals = torch.cos(total_angles)
        sin_vals = torch.sin(total_angles)
        return torch.stack([cos_vals, sin_vals], dim=-1)

    def _polygon_state(self) -> torch.Tensor:
        if (
            self.line_angle_offsets is None
            or self.line_distances is None
            or self.center_offsets is None
        ):
            raise RuntimeError("Environment not initialised.")
        angle_degrees = torch.rad2deg(self.line_angle_offsets)
        return torch.cat([self.line_distances, angle_degrees, self.center_offsets], dim=-1)

    def _pixel_coords(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = self._pixel_coord_cache.get(device)
        if cached is None:
            x_coords = torch.arange(self.width, dtype=torch.float32, device=device).view(1, 1, 1, self.width)
            y_coords = torch.arange(self.height, dtype=torch.float32, device=device).view(1, 1, self.height, 1)
            cached = (x_coords, y_coords)
            self._pixel_coord_cache[device] = cached
        return cached

    def _rasterize_polytope(self, normals: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        pixel_x, pixel_y = self._pixel_coords(normals.device)
        if self.center_offsets is None or self.base_center is None:
            raise RuntimeError("Center state not initialised.")
        center_offsets = self.center_offsets.to(normals.device)
        base_center = self.base_center.to(normals.device)
        center_x = (base_center[0] + center_offsets[:, 0]).view(-1, 1, 1, 1)
        center_y = (base_center[1] + center_offsets[:, 1]).view(-1, 1, 1, 1)
        nx = normals[..., 0].unsqueeze(-1).unsqueeze(-1)
        ny = normals[..., 1].unsqueeze(-1).unsqueeze(-1)
        proj = nx * (pixel_x - center_x) + ny * (pixel_y - center_y)
        mask = proj <= distances[..., None, None]
        return mask.all(dim=1).to(torch.float32)

    def _calculate_iou_from_masks(self, poly_masks: torch.Tensor) -> torch.Tensor:
        if self.masks is None:
            raise RuntimeError("Masks are not available.")
        gt_masks = (self.masks.squeeze(1) > 0.5).to(poly_masks.device).to(poly_masks.dtype)
        intersection = (poly_masks * gt_masks).sum(dim=(1, 2))
        union = poly_masks.sum(dim=(1, 2)) + gt_masks.sum(dim=(1, 2)) - intersection
        return torch.where(union > 0, intersection / union, torch.zeros_like(union))

    def _compute_vertices(self, normals: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        if self.base_center is None or self.center_offsets is None or self.width is None or self.height is None:
            raise RuntimeError("Environment not initialised.")

        device = normals.device
        batch_size, num_lines, _ = normals.shape

        image_rect = torch.tensor(
            [
                [0.0, 0.0],
                [float(self.width - 1), 0.0],
                [float(self.width - 1), float(self.height - 1)],
                [0.0, float(self.height - 1)],
            ],
            dtype=torch.float32,
            device=device,
        )
        if self.base_center is None or self.center_offsets is None:
            raise RuntimeError("Center state not initialised.")
        base_center_device = self.base_center.to(device)
        center_offsets = self.center_offsets.to(device)

        polygons: list[torch.Tensor] = []
        max_vertex_count = 0
        eps = 1e-6

        for b in range(batch_size):
            poly = image_rect.clone()
            center = base_center_device + center_offsets[b]

            for i in range(num_lines):
                normal = normals[b, i]
                distance = distances[b, i]
                threshold = distance + torch.dot(normal, center)

                if poly.numel() == 0:
                    break

                new_vertices: list[torch.Tensor] = []
                prev_vertex = poly[-1]
                prev_inside = torch.dot(normal, prev_vertex) <= (threshold + eps)

                for curr_vertex in poly:
                    curr_inside = torch.dot(normal, curr_vertex) <= (threshold + eps)
                    if curr_inside != prev_inside:
                        direction = curr_vertex - prev_vertex
                        denom = torch.dot(normal, direction)
                        if abs(float(denom)) > eps:
                            t = (threshold - torch.dot(normal, prev_vertex)) / denom
                            t = torch.clamp(t, 0.0, 1.0)
                            intersection = prev_vertex + t * direction
                            new_vertices.append(intersection)
                    if curr_inside:
                        new_vertices.append(curr_vertex)
                    prev_vertex = curr_vertex
                    prev_inside = curr_inside

                if not new_vertices:
                    poly = torch.empty(0, 2, dtype=torch.float32, device=device)
                    break
                poly = torch.stack(new_vertices, dim=0)

            if poly.numel() > 0:
                poly[:, 0].clamp_(0.0, float(self.width - 1))
                poly[:, 1].clamp_(0.0, float(self.height - 1))
            else:
                base = center.to(dtype=torch.float32)
                jitter_x = torch.tensor([1e-3, 0.0], device=device, dtype=torch.float32)
                jitter_y = torch.tensor([0.0, 1e-3], device=device, dtype=torch.float32)
                poly = torch.stack(
                    [
                        base,
                        base + jitter_x,
                        base + jitter_y,
                    ],
                    dim=0,
                )

            polygons.append(poly)
            max_vertex_count = max(max_vertex_count, poly.size(0))

        padded_polys: list[torch.Tensor] = []
        for poly in polygons:
            if poly.size(0) < max_vertex_count:
                pad = poly[-1].unsqueeze(0).expand(max_vertex_count - poly.size(0), 2)
                poly = torch.cat([poly, pad], dim=0)
            padded_polys.append(poly)

        return torch.stack(padded_polys, dim=0)

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

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_np, cmap="gray")
        if self.masks is not None:
            mask = self.masks[index].detach().cpu().squeeze(0).numpy()
            ax.imshow(mask, cmap="Reds", alpha=0.25)

        verts = self.vertices[index].detach().cpu().numpy()
        if verts.shape[0] >= 3:
            polygon = patches.Polygon(
                verts,
                closed=True,
                fill=False,
                edgecolor="cyan",
                linewidth=2.0,
            )
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
