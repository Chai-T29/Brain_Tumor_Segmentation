from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torchvision import models


@dataclass
class EncoderConfig:
    """Configuration container for the EfficientNet encoder."""

    name: str = "efficientnet_b0"
    pretrained: bool = True
    trainable: bool = False
    embedding_noise_std: float = 0.01


class EfficientNetEncoder(nn.Module):
    """Wrapper that exposes EfficientNet feature embeddings for 224×224 slices.

    The encoder repeats single-channel slices across RGB channels, feeds them
    through the selected EfficientNet backbone, and returns the final convolutional
    feature map prior to global pooling. Optionally, additive Gaussian noise can be
    injected into the feature map on every forward pass to emulate DrQ-style
    stochastic augmentation in embedding space.
    """

    _SUPPORTED_MODELS = {
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b1": models.efficientnet_b1,
        "efficientnet_b2": models.efficientnet_b2,
        "efficientnet_v2_s": models.efficientnet_v2_s,
    }

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        if config.name not in self._SUPPORTED_MODELS:
            raise ValueError(
                f"Encoder '{config.name}' is not supported. "
                f"Available: {', '.join(self._SUPPORTED_MODELS)}."
            )
        self.config = config

        builder = self._SUPPORTED_MODELS[config.name]
        weights = None
        if config.pretrained:
            weights_enum = models.get_model_weights(config.name).DEFAULT
            weights = weights_enum

        backbone = builder(weights=weights)
        self.features = backbone.features
        self.pool = backbone.avgpool

        with torch.no_grad():
            probe = torch.zeros(1, 3, 224, 224)
            feat = self.features(probe)
        self.feature_shape = (int(feat.size(1)), int(feat.size(2)), int(feat.size(3)))
        self.feature_dim = int(torch.tensor(self.feature_shape).prod().item())

        # Freeze backbone if requested.
        if not config.trainable:
            for param in self.parameters():
                param.requires_grad = False

        # Register buffers for ImageNet normalization (mean/std for RGB).
        meta = getattr(weights, "meta", {}) if weights is not None else {}
        if config.pretrained and "mean" in meta and "std" in meta:
            mean = torch.tensor(meta["mean"]).view(1, 3, 1, 1)
            std = torch.tensor(meta["std"]).view(1, 3, 1, 1)
        else:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("rgb_mean", mean, persistent=False)
        self.register_buffer("rgb_std", std, persistent=False)

    def forward(self, images: torch.Tensor, noise: bool = True) -> torch.Tensor:
        """Return feature maps for a batch of `[B, 1, 224, 224]` slices."""
        if images.dim() != 4 or images.size(1) != 1:
            raise ValueError("Expected input shape [B, 1, H, W].")

        # EfficientNet expects three-channel inputs that follow ImageNet stats.
        x = images.repeat(1, 3, 1, 1)
        x = (x - self.rgb_mean.to(x.device, x.dtype)) / self.rgb_std.to(x.device, x.dtype)

        feats = self.features(x)
        if noise and self.config.embedding_noise_std > 0:
            noise_tensor = torch.randn_like(feats) * self.config.embedding_noise_std
            feats = feats + noise_tensor
        return feats

    def embed_without_noise(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images, noise=False)


def build_encoder(config_dict: dict) -> Tuple[EfficientNetEncoder, EncoderConfig]:
    config = EncoderConfig(**config_dict)
    encoder = EfficientNetEncoder(config)
    return encoder, config
