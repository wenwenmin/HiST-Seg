"""Learnable spot encoder and differentiable high-resolution rasterizer.

Pipeline
--------
1. Direct 512-D scGPT spot embeddings are already normalized per slide by the
   dataset cache loader.
2. A shallow residual MLP learns 512 -> 128 -> 32 task-specific features.
3. Optional relative positional encoding is injected weakly. Absolute slide
   coordinates are deliberately avoided because they can become a shortcut for
   slide identity and hurt cross-slide transfer.
4. Bilinear splatting followed by normalized Gaussian convolution rasterizes the
   low-dimensional spots to the configured context grid (128 x 128 by default).
5. One or two light residual depthwise blocks aggregate local spatial context.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _inverse_sigmoid(probability: float) -> float:
    probability = min(1.0 - 1e-6, max(1e-6, float(probability)))
    return math.log(probability / (1.0 - probability))


class SpotFeatureEncoder(nn.Module):
    """Task-learned residual projection from raw scGPT features to 32-D."""

    def __init__(self, cfg):
        super().__init__()
        input_dim = int(cfg.dims.st_dim)
        hidden_dim = int(getattr(cfg.dims, "st_base_dim", 128))
        output_dim = int(getattr(cfg.dims, "st_ctx_dim", 32))
        dropout = float(getattr(cfg.model, "st_encoder_dropout", 0.10))

        if input_dim != 512:
            raise ValueError(f"SpotFeatureEncoder requires 512-D input, got {input_dim}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.shortcut = nn.Linear(input_dim, output_dim, bias=False)
        self.output_norm = nn.LayerNorm(output_dim)

        self.use_position = bool(getattr(cfg.model, "use_spot_position_encoding", True))
        self.position_bands = int(getattr(cfg.model, "spot_position_bands", 4))
        self.position_scale_max = float(
            getattr(cfg.model, "spot_position_scale_max", 0.25)
        )
        position_scale_init = float(
            getattr(cfg.model, "spot_position_scale_init", 0.05)
        )
        if not 0.0 <= position_scale_init < self.position_scale_max:
            raise ValueError(
                "spot_position_scale_init must be in [0, spot_position_scale_max)"
            )
        self.raw_position_scale = nn.Parameter(
            torch.tensor(
                _inverse_sigmoid(position_scale_init / self.position_scale_max),
                dtype=torch.float32,
            )
        )

        if self.use_position:
            position_input_dim = 2 + 4 * self.position_bands
            self.position_mlp = nn.Sequential(
                nn.Linear(position_input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )
            nn.init.zeros_(self.position_mlp[-1].weight)
            nn.init.zeros_(self.position_mlp[-1].bias)
        else:
            self.position_mlp = None

    def position_scale(self) -> torch.Tensor:
        return self.position_scale_max * torch.sigmoid(self.raw_position_scale)

    def _position_features(self, positions: torch.Tensor) -> torch.Tensor:
        centered = positions * 2.0 - 1.0
        parts = [centered]
        for band in range(self.position_bands):
            frequency = float(2 ** band) * math.pi
            parts.extend(
                [
                    torch.sin(frequency * centered),
                    torch.cos(frequency * centered),
                ]
            )
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected spot features [B,K,{self.input_dim}], got {tuple(features.shape)}"
            )
        if positions.shape != (*features.shape[:2], 2):
            raise ValueError(
                f"Position shape {tuple(positions.shape)} does not match {tuple(features.shape)}"
            )
        if valid_mask.shape != features.shape[:2]:
            raise ValueError(
                f"Mask shape {tuple(valid_mask.shape)} does not match {tuple(features.shape)}"
            )

        valid = valid_mask.to(features.dtype).unsqueeze(-1)
        normalized = self.input_norm(features)
        encoded = self.encoder(normalized) + self.shortcut(normalized)

        position_scale = self.position_scale()
        if self.position_mlp is not None:
            position_delta = self.position_mlp(self._position_features(positions))
            encoded = encoded + position_scale * position_delta

        encoded = self.output_norm(encoded) * valid
        return encoded, {"position_scale": position_scale}


class DifferentiableSpotRasterizer(nn.Module):
    """Bilinear splatting plus normalized Gaussian interpolation."""

    def __init__(self, cfg, channels: int):
        super().__init__()
        self.grid_size = int(cfg.geometry.context_grid_size)
        self.channels = int(channels)
        kernel_size = int(getattr(cfg.model, "st_raster_kernel_size", 7))
        sigma = float(getattr(cfg.model, "st_raster_sigma", 1.5))
        self.coverage_gain = float(getattr(cfg.model, "st_coverage_gain", 8.0))
        self.eps = float(getattr(cfg.model, "st_raster_eps", 1e-6))

        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("st_raster_kernel_size must be a positive odd integer")
        if sigma <= 0:
            raise ValueError("st_raster_sigma must be positive")

        coordinates = torch.arange(kernel_size, dtype=torch.float32)
        coordinates = coordinates - (kernel_size - 1) / 2.0
        one_dimensional = torch.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
        kernel = torch.outer(one_dimensional, one_dimensional)
        kernel = kernel / kernel.sum().clamp_min(1e-12)
        self.register_buffer("gaussian_kernel", kernel[None, None], persistent=False)
        self.padding = kernel_size // 2

    def _scatter(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, count, channels = features.shape
        size = self.grid_size
        flat_size = size * size

        positions = positions.clamp(0.0, 1.0)
        gx = positions[..., 0] * float(size - 1)
        gy = positions[..., 1] * float(size - 1)
        x0 = torch.floor(gx).long().clamp(0, size - 1)
        y0 = torch.floor(gy).long().clamp(0, size - 1)
        x1 = (x0 + 1).clamp(0, size - 1)
        y1 = (y0 + 1).clamp(0, size - 1)

        wx = gx - x0.to(gx.dtype)
        wy = gy - y0.to(gy.dtype)
        neighbors = (
            (x0, y0, (1.0 - wx) * (1.0 - wy)),
            (x1, y0, wx * (1.0 - wy)),
            (x0, y1, (1.0 - wx) * wy),
            (x1, y1, wx * wy),
        )

        numerator = features.new_zeros((batch, channels, flat_size))
        denominator = features.new_zeros((batch, 1, flat_size))
        source = features.transpose(1, 2).contiguous()
        valid = valid_mask.to(features.dtype)

        for x_index, y_index, weight in neighbors:
            weight = weight * valid
            flat_index = (y_index * size + x_index).unsqueeze(1)
            numerator.scatter_add_(
                2,
                flat_index.expand(-1, channels, -1),
                source * weight.unsqueeze(1),
            )
            denominator.scatter_add_(2, flat_index, weight.unsqueeze(1))

        return (
            numerator.view(batch, channels, size, size),
            denominator.view(batch, 1, size, size),
        )

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
        tissue_mask: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        numerator, denominator = self._scatter(features, positions, valid_mask)

        channel_kernel = self.gaussian_kernel.expand(self.channels, 1, -1, -1)
        numerator = F.conv2d(
            numerator,
            channel_kernel,
            padding=self.padding,
            groups=self.channels,
        )
        denominator = F.conv2d(
            denominator,
            self.gaussian_kernel,
            padding=self.padding,
        )
        raster = numerator / denominator.clamp_min(self.eps)
        coverage = 1.0 - torch.exp(-self.coverage_gain * denominator)

        if tissue_mask is not None:
            tissue = F.interpolate(
                tissue_mask.float(),
                size=(self.grid_size, self.grid_size),
                mode="nearest",
            ).clamp(0.0, 1.0)
            coverage = coverage * tissue
            raster = raster * tissue

        raster = raster * (coverage > self.eps).to(raster.dtype)
        return raster, coverage.clamp(0.0, 1.0)


class SpatialAggregationBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
        residual = self.pointwise(self.depthwise(self.norm(x)))
        strength = 0.5 * torch.tanh(self.scale)
        return (x + strength * residual * coverage) * coverage


class STSpotContextEncoder(nn.Module):
    """Complete low-memory ST path from padded 512-D spots to a 32-D grid."""

    def __init__(self, cfg):
        super().__init__()
        output_dim = int(getattr(cfg.dims, "st_ctx_dim", 32))
        dropout = float(getattr(cfg.model, "st_encoder_dropout", 0.10))
        block_count = int(getattr(cfg.model, "st_spatial_blocks", 1))
        if not 0 <= block_count <= 3:
            raise ValueError("model.st_spatial_blocks must be between 0 and 3")

        self.spot_encoder = SpotFeatureEncoder(cfg)
        self.rasterizer = DifferentiableSpotRasterizer(cfg, output_dim)
        self.spatial_blocks = nn.ModuleList(
            [
                SpatialAggregationBlock(
                    output_dim,
                    dilation=1 if index == 0 else 2,
                    dropout=dropout,
                )
                for index in range(block_count)
            ]
        )

    def forward(
        self,
        spot_features: torch.Tensor,
        spot_positions: torch.Tensor,
        spot_mask: torch.Tensor,
        context_tissue: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        encoded_spots, metadata = self.spot_encoder(
            spot_features, spot_positions, spot_mask
        )
        raster, coverage = self.rasterizer(
            encoded_spots,
            spot_positions,
            spot_mask,
            context_tissue,
        )
        for block in self.spatial_blocks:
            raster = block(raster, coverage)

        metadata = {
            **metadata,
            "encoded_spot_features": encoded_spots,
            "coverage": coverage,
        }
        return raster, coverage, metadata
