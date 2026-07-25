"""PathST multimodal cancer-region segmentation network.

Default geometry
----------------
* Detail patch: 512 x 512 H&E.
* Context field: 4x wider/taller, i.e. 2048 x 2048 source pixels.
* Context cache: 1/4-resolution thumbnail, giving a 512 x 512 context crop.
* H&E and ST are aligned on a 128 x 128 context grid.

ST only enters through the semantic context path; high-resolution decoder skips
remain H&E-only so noisy molecular context cannot directly overwrite boundaries.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.context_stream import (
    ContextGuidedDetailFusion,
    DeepMultimodalContextFusion,
    HEContextEncoder,
)
from model.decoder import Decoder
from model.detail_stream import DetailStream
from model.st_spot_encoder import STSpotContextEncoder


class PathSTNet(nn.Module):
    def __init__(self, cfg, build_uni: bool = True):
        super().__init__()
        self.cfg = cfg
        self.st_dim = int(cfg.dims.st_dim)
        self.use_st = bool(getattr(cfg.model, "use_st", True))
        self.use_he_context = bool(getattr(cfg.model, "use_he_context", True))
        self.st_drop = float(getattr(cfg.model, "st_drop_rate", 0.0))
        self.use_st_aux = bool(getattr(cfg.model, "use_st_aux", True))

        if self.st_dim != 512:
            raise ValueError(f"This model requires dims.st_dim=512, got {self.st_dim}")
        self.detail = DetailStream(cfg, build_uni=build_uni)
        self.he_context_encoder = HEContextEncoder(cfg)
        self.st_encoder = STSpotContextEncoder(cfg)
        self.context_fusion = DeepMultimodalContextFusion(cfg)
        self.detail_context_fusion = ContextGuidedDetailFusion(cfg)
        self.decoder = Decoder(cfg)

        st_context_dim = int(getattr(cfg.dims, "st_ctx_dim", 32))
        self.st_aux_head = (
            nn.Conv2d(st_context_dim, 1, 1) if self.use_st_aux else None
        )

    def _apply_st_dropout(
        self,
        spot_features: torch.Tensor,
        spot_mask: torch.Tensor,
    ):
        if not self.training or self.st_drop <= 0:
            return spot_features, spot_mask
        keep = (
            torch.rand(
                (spot_features.shape[0], 1),
                device=spot_features.device,
            )
            >= self.st_drop
        ).to(spot_features.dtype)
        return spot_features * keep.unsqueeze(-1), spot_mask * keep

    def _zero_st_inputs(
        self,
        reference: torch.Tensor,
        spot_features: torch.Tensor | None,
        spot_positions: torch.Tensor | None,
        spot_mask: torch.Tensor | None,
    ):
        if spot_features is not None:
            if spot_positions is None or spot_mask is None:
                raise ValueError("spot_positions and spot_mask are required with spot_features")
            return spot_features, spot_positions, spot_mask

        batch = reference.shape[0]
        max_spots = int(getattr(self.cfg.data, "max_context_spots", 1024))
        features = reference.new_zeros((batch, max_spots, self.st_dim))
        positions = reference.new_zeros((batch, max_spots, 2))
        mask = reference.new_zeros((batch, max_spots))
        return features, positions, mask

    def forward(
        self,
        hr: torch.Tensor,
        he_context: torch.Tensor,
        spot_features: torch.Tensor | None = None,
        spot_positions: torch.Tensor | None = None,
        spot_mask: torch.Tensor | None = None,
        context_tissue: torch.Tensor | None = None,
    ):
        out_size = tuple(hr.shape[-2:])
        s2, s3, s4, detail_semantic = self.detail(hr)

        if not self.use_he_context:
            he_context = torch.zeros_like(he_context)
        he_context_feature = self.he_context_encoder(he_context, context_tissue)

        spot_features, spot_positions, spot_mask = self._zero_st_inputs(
            hr, spot_features, spot_positions, spot_mask
        )
        if not self.use_st:
            spot_features = torch.zeros_like(spot_features)
            spot_mask = torch.zeros_like(spot_mask)
        spot_features, spot_mask = self._apply_st_dropout(spot_features, spot_mask)

        st_context, coverage, st_meta = self.st_encoder(
            spot_features,
            spot_positions,
            spot_mask,
            context_tissue,
        )
        total_context, context_meta = self.context_fusion(
            he_context_feature,
            st_context,
            coverage,
        )
        fused_semantic, detail_context_meta = self.detail_context_fusion(
            detail_semantic,
            total_context,
        )

        st_aux_logits = None
        coverage_local = None
        if self.st_aux_head is not None:
            footprint = int(self.cfg.geometry.context_footprint_size)
            y0 = (st_context.shape[-2] - footprint) // 2
            x0 = (st_context.shape[-1] - footprint) // 2
            st_local = st_context[..., y0 : y0 + footprint, x0 : x0 + footprint]
            coverage_local = coverage[..., y0 : y0 + footprint, x0 : x0 + footprint]
            st_aux_logits = self.st_aux_head(st_local)

        seg_logits = self.decoder(fused_semantic, s2, s3, s4, out_size)

        # Keep legacy key names for existing metric code. ``fusion_gate`` now
        # denotes the effective ST attention inside the multimodal context
        # fusion, while the final context->detail gate is logged separately.
        return {
            "seg_logits": seg_logits,
            "st_aux_logits": st_aux_logits,
            "st_coverage": coverage_local,
            "fusion_gate": context_meta["context_st_attention"],
            "fusion_scale": context_meta["context_st_scale"],
            "context_detail_gate": detail_context_meta["detail_context_gate"],
            "context_detail_scale": detail_context_meta["detail_context_scale"],
            "context_joint_scale": context_meta["context_joint_scale"],
            "spot_position_scale": st_meta["position_scale"],
            "total_context": total_context,
            "raw_st_shortcut_scale": None,
        }

    @torch.no_grad()
    def predict(
        self,
        hr: torch.Tensor,
        he_context: torch.Tensor,
        spot_features: torch.Tensor | None = None,
        spot_positions: torch.Tensor | None = None,
        spot_mask: torch.Tensor | None = None,
        context_tissue: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.eval()
        outputs = self.forward(
            hr,
            he_context=he_context,
            spot_features=spot_features,
            spot_positions=spot_positions,
            spot_mask=spot_mask,
            context_tissue=context_tissue,
        )
        return torch.sigmoid(outputs["seg_logits"][0])

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def build_model(cfg, build_uni: bool = True):
    return PathSTNet(cfg, build_uni=build_uni)
