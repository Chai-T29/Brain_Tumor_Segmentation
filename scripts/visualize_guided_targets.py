from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from data.data_module import BrainTumorDataModule
from rl.environment import EnvironmentConfig, PolygonLocalizationEnv


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    vmin, vmax = float(img.min()), float(img.max())
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return (img * 255.0 + 0.5).astype(np.uint8)


def _stack_h(*imgs: np.ndarray) -> np.ndarray:
    h = min(i.shape[0] for i in imgs)
    resized = []
    for img in imgs:
        if img.shape[0] != h:
            # simple nearest resize on height
            scale = h / img.shape[0]
            w = int(round(img.shape[1] * scale))
            # crude resizing via slicing/repeat to avoid cv2/skimage
            y_idx = (np.linspace(0, img.shape[0] - 1, h)).astype(int)
            x_idx = (np.linspace(0, img.shape[1] - 1, w)).astype(int)
            img = img[y_idx][:, x_idx]
        resized.append(img)
    return np.concatenate(resized, axis=1)


def _compute_iou_for_target(env: PolygonLocalizationEnv) -> float:
    """Compute IoU for the current environment state (after target is applied)."""
    normals = env._compute_normals()
    poly_masks = env._rasterize_polytope(normals, env.line_distances)
    iou = env._calculate_iou_from_masks(poly_masks)
    return float(iou.view(-1)[0].detach().cpu().item())


def _compose_panel(image: torch.Tensor, mask: torch.Tensor, init_frame: np.ndarray, tgt_frame: np.ndarray) -> np.ndarray:
    # Compose an overlay of image+mask for context
    img_np = image.squeeze(0).cpu().numpy()
    msk_np = (mask.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
    img_gray = _to_uint8(img_np)
    overlay = np.dstack([img_gray, img_gray, img_gray])
    red = overlay.copy()
    red[..., 0] = np.clip(red[..., 0] + (msk_np * 255), 0, 255)

    # Stack: [image+mask | initial poly | target poly]
    left = red
    mid = init_frame[..., :3]
    right = tgt_frame[..., :3]
    vis = _stack_h(left, mid, right)
    return vis


def main() -> None:
    config = load_config("config.yaml")
    data_cfg = config.get("data", {})
    env_cfg = config.get("environment", {})

    # Build data module with environment config so targets are precomputed
    dm = BrainTumorDataModule(
        data_dir=data_cfg.get("data_dir", "MU-Glioma-Post/"),
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        prefetch_factor=2,
        val_split=data_cfg.get("val_split", 0.1),
        test_split=data_cfg.get("test_split", 0.1),
        seed=config.get("seed", 42),
        include_empty_masks=data_cfg.get("include_empty_masks", False),
        encoder_config=config.get("encoder", {}),
        embedding_batch_size=data_cfg.get("embedding_batch_size", 64),
        embedding_device=data_cfg.get("embedding_device"),
        environment_config=env_cfg,
    )

    # Prefer cached path to avoid recomputing embeddings
    try:
        dm.setup(stage="validate")
    except Exception:
        # Fallback: prepare cache (may be slow on first run)
        dm.setup(stage="fit")

    val_loader = dm.val_dataloader()
    out_dir = Path("lightning_logs/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = PolygonLocalizationEnv(EnvironmentConfig(**env_cfg))

    saved: int = 0
    max_save: int = 5
    total: int = 0
    limit: int = 100
    ious: List[float] = []

    for batch in val_loader:
        if total >= limit:
            break

        image = batch["image"][0]  # [1,H,W]
        mask = batch["mask"][0]
        target_state = batch.get("target_polygon_state")
        if target_state is None:
            raise RuntimeError(
                "Batch is missing 'target_polygon_state'. Ensure environment_config is passed to the dataset."
            )
        target_state = target_state[0].to(torch.float32)

        # Reset environment
        _ = env.reset(image.unsqueeze(0), mask.unsqueeze(0))
        init_frame = env.render(index=0, mode="rgb_array")
        if init_frame is None:
            raise RuntimeError("Render returned None.")

        # Apply target state (distances + angle offsets in degrees)
        num_lines = env.num_lines
        tgt_dist = target_state[:num_lines].clone()
        tgt_ang_deg = target_state[num_lines : 2 * num_lines].clone()
        tgt_center = target_state[2 * num_lines : 2 * num_lines + 2].clone()
        tgt_ang_rad = torch.deg2rad(tgt_ang_deg)

        env.line_distances[0].copy_(tgt_dist)
        env.line_angle_offsets[0].copy_(tgt_ang_rad)
        if env.center_offsets is None or env.base_center is None or env.center_positions is None:
            raise RuntimeError("Environment center state not initialised.")
        env.center_offsets[0].copy_(tgt_center)
        env.center_positions[0].copy_(env.base_center + tgt_center)

        # Recompute vertices and render target
        normals = env._compute_normals()
        env.vertices = env._compute_vertices(normals, env.line_distances)
        tgt_frame = env.render(index=0, mode="rgb_array")
        if tgt_frame is None:
            raise RuntimeError("Render returned None after applying target state.")

        # IoU for this slice
        iou_value = _compute_iou_for_target(env)
        ious.append(iou_value)
        total += 1

        # Save a few visualizations
        if saved < max_save:
            vis = _compose_panel(image, mask, init_frame, tgt_frame)
            out_path = out_dir / f"guided_targets_visual_{saved:02d}.jpg"
            try:
                import imageio.v2 as imageio

                imageio.imwrite(out_path, vis)
            except Exception:
                np.save(out_path.with_suffix(".npy"), vis)
            print(f"Saved visualization to: {out_path}")
            saved += 1

    avg_iou = float(np.mean(ious)) if ious else 0.0
    print(f"Computed guided-target IoU over {len(ious)} slices: avg IoU = {avg_iou:.4f}")


if __name__ == "__main__":
    main()
