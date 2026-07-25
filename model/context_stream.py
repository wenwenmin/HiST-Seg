"""H&E context encoder and two-stage multimodal/context-detail fusion."""
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


def _logit(probability: float) -> float:
    probability = min(1.0 - 1e-6, max(1e-6, float(probability)))
    return math.log(probability / (1.0 - probability))


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
    ):
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )


class ResidualContextBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, 3, 1, dilation),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class HEContextEncoder(nn.Module):
    """Lightweight encoder for the aligned H&E context image.

    With the supplied 512/ctx4 geometry, a 512x512 context crop is encoded to a
    128x128 feature grid that is physically aligned with the ST raster.
    """

    def __init__(self, cfg):
        super().__init__()
        output_dim = int(getattr(cfg.dims, "he_context_dim", 64))
        stride = int(getattr(cfg.geometry, "context_encoder_stride", 2))
        dropout = float(getattr(cfg.model, "he_context_dropout", 0.05))
        if stride not in (1, 2, 4):
            raise ValueError("context_encoder_stride must be 1, 2, or 4")

        mid = max(24, output_dim // 2)
        layers = [ConvNormAct(3, mid, 3, 1)]
        current = mid
        remaining = stride
        while remaining > 1:
            next_channels = output_dim if remaining == 2 else min(output_dim, current * 2)
            layers.append(ConvNormAct(current, next_channels, 3, 2))
            current = next_channels
            remaining //= 2
        if current != output_dim:
            layers.append(ConvNormAct(current, output_dim, 1, 1))

        self.stem = nn.Sequential(*layers)
        self.blocks = nn.Sequential(
            ResidualContextBlock(output_dim, dilation=1, dropout=dropout),
            ResidualContextBlock(output_dim, dilation=2, dropout=dropout),
        )

    def forward(
        self,
        context_image: torch.Tensor,
        context_tissue: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.blocks(self.stem(context_image))
        if context_tissue is not None:
            tissue = F.interpolate(
                context_tissue.float(),
                size=x.shape[-2:],
                mode="nearest",
            ).clamp(0.0, 1.0)
            x = x * tissue
        return x


class PyramidContextReasoning(nn.Module):
    """Local residual reasoning plus global 1x1/2x2/4x4 context summaries."""

    def __init__(self, channels: int):
        super().__init__()
        branch_channels = max(8, channels // 4)
        self.local = nn.Sequential(
            ResidualContextBlock(channels, dilation=1),
            ResidualContextBlock(channels, dilation=2),
        )
        self.pool_sizes = (1, 2, 4)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(channels, branch_channels, 1, bias=False),
                    nn.GroupNorm(1, branch_channels),
                    nn.GELU(),
                )
                for size in self.pool_sizes
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                channels + branch_channels * len(self.pool_sizes),
                channels,
                1,
                bias=False,
            ),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            ResidualContextBlock(channels, dilation=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        target_size = x.shape[-2:]
        pooled = [
            F.interpolate(branch(x), size=target_size, mode="bilinear", align_corners=False)
            for branch in self.branches
        ]
        return self.fuse(torch.cat([local, *pooled], dim=1))


class DeepMultimodalContextFusion(nn.Module):
    """Reliability-aware deep fusion of aligned H&E and ST context grids.

    This is intentionally more expressive than the former single gate. It uses:
    modality-specific refinement, product/difference interactions, channel
    attention, spatial attention, and a joint residual. Coverage remains a
    reliability mask for ST, not the entire fusion mechanism. H&E remains the
    residual anchor to make the model robust when ST is noisy or missing.
    """

    def __init__(self, cfg):
        super().__init__()
        he_dim = int(getattr(cfg.dims, "he_context_dim", 64))
        st_dim = int(getattr(cfg.dims, "st_ctx_dim", 32))
        fusion_dim = int(getattr(cfg.dims, "context_fusion_dim", he_dim))
        hidden = int(getattr(cfg.dims, "context_attention_hidden", 64))

        self.he_project = (
            nn.Identity()
            if he_dim == fusion_dim
            else ConvNormAct(he_dim, fusion_dim, 1, 1)
        )
        self.st_project = ConvNormAct(st_dim, fusion_dim, 1, 1)
        self.he_refine = ResidualContextBlock(fusion_dim, dilation=1)
        self.st_refine = ResidualContextBlock(fusion_dim, dilation=1)

        joint_channels = fusion_dim * 4 + 1
        self.joint = nn.Sequential(
            nn.Conv2d(joint_channels, fusion_dim, 1, bias=False),
            nn.GroupNorm(_groups(fusion_dim), fusion_dim),
            nn.GELU(),
            ResidualContextBlock(fusion_dim, dilation=1),
        )

        self.channel_attention = nn.Sequential(
            nn.Conv2d(fusion_dim * 3, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, fusion_dim * 2, 1),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(7, hidden // 2 if hidden >= 2 else 1, 7, padding=3, bias=False),
            nn.GroupNorm(_groups(hidden // 2 if hidden >= 2 else 1), hidden // 2 if hidden >= 2 else 1),
            nn.GELU(),
            nn.Conv2d(hidden // 2 if hidden >= 2 else 1, 2, 1),
        )

        self.st_scale_max = float(getattr(cfg.model, "context_st_scale_max", 0.30))
        st_scale_init = float(getattr(cfg.model, "context_st_scale_init", 0.08))
        self.joint_scale_max = float(getattr(cfg.model, "context_joint_scale_max", 0.25))
        joint_scale_init = float(getattr(cfg.model, "context_joint_scale_init", 0.05))
        if not 0 <= st_scale_init < self.st_scale_max:
            raise ValueError("context_st_scale_init must be below context_st_scale_max")
        if not 0 <= joint_scale_init < self.joint_scale_max:
            raise ValueError("context_joint_scale_init must be below context_joint_scale_max")
        self.raw_st_scale = nn.Parameter(
            torch.tensor(_logit(st_scale_init / self.st_scale_max), dtype=torch.float32)
        )
        self.raw_joint_scale = nn.Parameter(
            torch.tensor(
                _logit(joint_scale_init / self.joint_scale_max), dtype=torch.float32
            )
        )

        self.reasoning = PyramidContextReasoning(fusion_dim)

    def st_scale(self) -> torch.Tensor:
        return self.st_scale_max * torch.sigmoid(self.raw_st_scale)

    def joint_scale(self) -> torch.Tensor:
        return self.joint_scale_max * torch.sigmoid(self.raw_joint_scale)

    @staticmethod
    def _summary_maps(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)

    def forward(
        self,
        he_context: torch.Tensor,
        st_context: torch.Tensor,
        coverage: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if st_context.shape[-2:] != he_context.shape[-2:]:
            st_context = F.interpolate(
                st_context,
                size=he_context.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if coverage.shape[-2:] != he_context.shape[-2:]:
            coverage = F.interpolate(
                coverage,
                size=he_context.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        coverage = coverage.clamp(0.0, 1.0)

        he = self.he_refine(self.he_project(he_context))
        st = self.st_refine(self.st_project(st_context)) * coverage
        joint = self.joint(
            torch.cat([he, st, he * st, torch.abs(he - st), coverage], dim=1)
        )

        pooled = torch.cat(
            [
                F.adaptive_avg_pool2d(he, 1),
                F.adaptive_avg_pool2d(st, 1),
                F.adaptive_avg_pool2d(joint, 1),
            ],
            dim=1,
        )
        channel_he, channel_st = torch.sigmoid(
            self.channel_attention(pooled)
        ).chunk(2, dim=1)

        he_mean, he_max = self._summary_maps(he)
        st_mean, st_max = self._summary_maps(st)
        joint_mean, joint_max = self._summary_maps(joint)
        spatial_he, spatial_st = torch.sigmoid(
            self.spatial_attention(
                torch.cat(
                    [
                        he_mean,
                        he_max,
                        st_mean,
                        st_max,
                        joint_mean,
                        joint_max,
                        coverage,
                    ],
                    dim=1,
                )
            )
        ).chunk(2, dim=1)

        effective_st_attention = channel_st * spatial_st * coverage
        he_attention = channel_he * spatial_he
        # ``st`` already contains one coverage reliability factor. Use the
        # learned channel/spatial attention here without multiplying coverage a
        # second time; ``effective_st_attention`` is retained for diagnostics.
        fused = (
            he
            + self.st_scale() * (channel_st * spatial_st) * st
            + self.joint_scale() * joint * (0.5 + 0.5 * he_attention)
        )
        fused = self.reasoning(fused)

        return fused, {
            "context_st_attention": effective_st_attention,
            "context_he_attention": he_attention,
            "coverage": coverage,
            "context_st_scale": self.st_scale(),
            "context_joint_scale": self.joint_scale(),
        }


class ContextGuidedDetailFusion(nn.Module):
    """Deep gated-attention fusion from total context into detail semantics."""

    def __init__(self, cfg):
        super().__init__()
        context_dim = int(getattr(cfg.dims, "context_fusion_dim", 64))
        detail_dim = int(cfg.dims.sem_dim)
        hidden = int(getattr(cfg.dims, "detail_context_attention_hidden", 128))
        self.footprint = int(cfg.geometry.context_footprint_size)

        self.context_project = nn.Sequential(
            nn.Conv2d(context_dim, detail_dim, 1, bias=False),
            nn.GroupNorm(_groups(detail_dim), detail_dim),
            nn.GELU(),
            nn.Conv2d(
                detail_dim,
                detail_dim,
                3,
                padding=1,
                groups=detail_dim,
                bias=False,
            ),
            nn.Conv2d(detail_dim, detail_dim, 1, bias=False),
            nn.GroupNorm(_groups(detail_dim), detail_dim),
            nn.GELU(),
        )
        self.global_project = nn.Sequential(
            nn.Conv2d(context_dim * 2, detail_dim, 1, bias=False),
            nn.GroupNorm(_groups(detail_dim), detail_dim),
            nn.GELU(),
        )
        self.detail_norm = nn.GroupNorm(_groups(detail_dim), detail_dim)
        self.context_norm = nn.GroupNorm(_groups(detail_dim), detail_dim)

        self.channel_gate = nn.Sequential(
            nn.Conv2d(detail_dim * 2, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, detail_dim, 1),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(4, 16, 7, padding=3, bias=False),
            nn.GroupNorm(_groups(16), 16),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(detail_dim * 2, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

        self.scale_max = float(getattr(cfg.model, "detail_context_scale_max", 0.50))
        scale_init = float(getattr(cfg.model, "detail_context_scale_init", 0.12))
        if not 0 <= scale_init < self.scale_max:
            raise ValueError("detail_context_scale_init must be below detail_context_scale_max")
        self.raw_scale = nn.Parameter(
            torch.tensor(_logit(scale_init / self.scale_max), dtype=torch.float32)
        )

    def scale(self) -> torch.Tensor:
        return self.scale_max * torch.sigmoid(self.raw_scale)

    @staticmethod
    def _center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
        height, width = x.shape[-2:]
        if size > height or size > width:
            raise ValueError(f"Cannot crop {size}x{size} from {height}x{width}")
        y0 = (height - size) // 2
        x0 = (width - size) // 2
        return x[..., y0 : y0 + size, x0 : x0 + size]

    @staticmethod
    def _mean_max(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)

    def forward(
        self,
        detail_semantic: torch.Tensor,
        total_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        local = self._center_crop(total_context, self.footprint)
        local = F.interpolate(
            local,
            size=detail_semantic.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        local = self.context_project(local)

        global_average = F.adaptive_avg_pool2d(total_context, 1)
        global_maximum = F.adaptive_max_pool2d(total_context, 1)
        global_context = self.global_project(
            torch.cat([global_average, global_maximum], dim=1)
        )
        context = local + global_context

        detail = self.detail_norm(detail_semantic)
        context_norm = self.context_norm(context)
        channel = torch.sigmoid(
            self.channel_gate(
                torch.cat(
                    [
                        F.adaptive_avg_pool2d(detail, 1),
                        F.adaptive_avg_pool2d(context_norm, 1),
                    ],
                    dim=1,
                )
            )
        )
        detail_mean, detail_max = self._mean_max(detail)
        context_mean, context_max = self._mean_max(context_norm)
        spatial = torch.sigmoid(
            self.spatial_gate(
                torch.cat(
                    [detail_mean, detail_max, context_mean, context_max], dim=1
                )
            )
        )
        semantic = torch.sigmoid(
            self.semantic_gate(torch.cat([detail, context_norm], dim=1))
        )
        gate = channel * spatial * semantic
        fused = detail_semantic + self.scale() * gate * context

        return fused, {
            "detail_context_gate": gate,
            "detail_context_scale": self.scale(),
            "context_local": local,
            "context_global": global_context,
        }
