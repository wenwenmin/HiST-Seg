from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits, targets, masks):
        logits, targets, masks = logits.float(), targets.float(), masks.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - pt).pow(self.gamma) * bce * masks
        return loss.sum() / (masks.sum() + 1e-6)


class MaskedDiceLoss(nn.Module):
    def forward(self, logits, targets, masks):
        prob = torch.sigmoid(logits.float()) * masks.float()
        target = targets.float() * masks.float()
        inter = (prob * target).flatten(1).sum(1)
        denom = prob.flatten(1).sum(1) + target.flatten(1).sum(1)
        return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


class MaskedTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.0):
        super().__init__()
        self.alpha, self.beta, self.gamma = map(float, (alpha, beta, gamma))

    def forward(self, logits, targets, masks):
        prob = torch.sigmoid(logits.float()) * masks.float()
        target = targets.float() * masks.float()
        tp = (prob * target).flatten(1).sum(1)
        fp = (prob * (1.0 - target)).flatten(1).sum(1)
        fn = ((1.0 - prob) * target).flatten(1).sum(1)
        score = (tp + 1.0) / (tp + self.alpha * fp + self.beta * fn + 1.0)
        loss = 1.0 - score
        if abs(self.gamma - 1.0) > 1e-6:
            loss = loss.clamp_min(1e-6).pow(self.gamma)
        return loss.mean()


class BoundaryLoss(nn.Module):
    def forward(self, logits, targets, masks):
        targets, masks = targets.float(), masks.float()
        eroded = 1.0 - F.max_pool2d(1.0 - targets, 3, 1, 1)
        band = F.max_pool2d((targets - eroded).clamp_min(0.0), 3, 1, 1) * masks
        bce = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none")
        return ((bce * band).flatten(1).sum(1) / (band.flatten(1).sum(1) + 1e-6)).mean()


class SpecificityLoss(nn.Module):
    def forward(self, logits, targets, masks):
        background = (1.0 - targets.float()) * masks.float()
        bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
        return (bce * background).sum() / (background.sum() + 1e-6)


class MaskedBrierLoss(nn.Module):
    """Masked probability-calibration loss.

    Dice/Focal can still produce good ranking while shifting the sigmoid scale.
    This term directly penalizes poorly calibrated probabilities inside tissue.
    """

    def forward(self, logits, targets, masks):
        prob = torch.sigmoid(logits.float())
        targets = targets.float()
        masks = masks.float()
        loss = ((prob - targets) ** 2) * masks
        return loss.sum() / (masks.sum() + 1e-6)


def _weighted_soft_dice_per_sample(
    prob: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample weighted Dice loss and active-sample mask.

    Per-sample normalization prevents a dense ST slide from dominating a sparse
    slide merely because it has more covered pixels.
    """
    flat_weight = weight.flatten(1)
    active = flat_weight.sum(1) > 1e-6
    inter = (prob * target * weight).flatten(1).sum(1)
    denom = ((prob + target) * weight).flatten(1).sum(1)
    loss = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    return loss, active


class STAuxiliaryLoss(nn.Module):
    """Coarse ST-only supervision restricted to covered tissue."""

    def __init__(self, coverage_min: float = 0.05, bce_weight: float = 0.5):
        super().__init__()
        self.coverage_min = float(coverage_min)
        self.bce_weight = float(bce_weight)

    def forward(self, logits, target, tissue, coverage):
        size = logits.shape[-2:]
        target_ds = F.interpolate(target.float(), size=size, mode="area")
        tissue_ds = F.interpolate(tissue.float(), size=size, mode="area").clamp(0.0, 1.0)
        coverage = F.interpolate(
            coverage.float(), size=size, mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        weight = tissue_ds * coverage * (coverage >= self.coverage_min).float()
        weight_sum = weight.flatten(1).sum(1)
        active = weight_sum > 1e-6
        if not bool(active.any()):
            return logits.sum() * 0.0

        # Normalize every sample independently before averaging. Without this, a
        # dense Visium-like patch contributes far more gradient than a sparse
        # early-ST patch solely because it contains more covered pixels.
        pixel_bce = F.binary_cross_entropy_with_logits(
            logits.float(), target_ds, reduction="none"
        )
        bce_per_sample = (pixel_bce * weight).flatten(1).sum(1) / (weight_sum + 1e-6)
        dice_per_sample, dice_active = _weighted_soft_dice_per_sample(
            torch.sigmoid(logits.float()), target_ds, weight
        )
        active = active & dice_active
        combined = (
            self.bce_weight * bce_per_sample
            + (1.0 - self.bce_weight) * dice_per_sample
        )
        return combined[active].mean()


class CombinedLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        loss_cfg = cfg.loss
        self.focal = MaskedFocalLoss(loss_cfg.focal_alpha, loss_cfg.focal_gamma)
        region_name = str(getattr(loss_cfg, "region_loss", "dice")).lower()
        self.region_name = region_name
        if region_name == "tversky":
            self.region = MaskedTverskyLoss(
                loss_cfg.tversky_alpha, loss_cfg.tversky_beta, loss_cfg.tversky_gamma
            )
        else:
            self.region = MaskedDiceLoss()
        self.boundary = BoundaryLoss()
        self.specificity = SpecificityLoss()
        self.brier = MaskedBrierLoss()
        self.st_aux = STAuxiliaryLoss(
            coverage_min=float(getattr(loss_cfg, "st_aux_coverage_min", 0.05)),
            bce_weight=float(getattr(loss_cfg, "st_aux_bce_weight", 0.5)),
        )
        self.w_focal = float(loss_cfg.w_focal)
        self.w_region = float(loss_cfg.w_dice)
        self.w_boundary = float(loss_cfg.w_boundary)
        self.w_specificity = float(getattr(loss_cfg, "w_specificity", 0.0))
        self.w_brier = float(getattr(loss_cfg, "w_brier", 0.0))
        self.w_st_aux = float(getattr(loss_cfg, "w_st_aux", 0.1))
        self.ds_weights = list(loss_cfg.ds_weights)
        self.warmup = int(getattr(loss_cfg, "refine_warmup_epochs", 0))
        self.warmup_start = float(getattr(loss_cfg, "refine_warmup_start_scale", 0.0))

    def _refine_scale(self, epoch: int) -> float:
        if self.warmup <= 0:
            return 1.0
        progress = min(1.0, max(0.0, epoch / max(1, self.warmup - 1)))
        return self.warmup_start + (1.0 - self.warmup_start) * progress

    def _seg_single(
        self,
        logits,
        target,
        tissue,
        refine_scale,
        *,
        use_boundary: bool = True,
        use_specificity: bool = True,
        use_brier: bool = True,
    ):
        focal = self.focal(logits, target, tissue)
        region = self.region(logits, target, tissue)
        if use_boundary and self.w_boundary > 0:
            boundary = self.boundary(logits, target, tissue) * refine_scale
        else:
            boundary = logits.sum() * 0.0
        if use_specificity and self.w_specificity > 0:
            specificity = self.specificity(logits, target, tissue)
        else:
            specificity = logits.sum() * 0.0
        if use_brier and self.w_brier > 0:
            brier = self.brier(logits, target, tissue)
        else:
            brier = logits.sum() * 0.0
        total = (
            self.w_focal * focal
            + self.w_region * region
            + self.w_boundary * boundary
            + self.w_specificity * specificity
            + self.w_brier * brier
        )
        return total, focal, region, boundary, specificity, brier

    def forward(self, outputs: Any, target: torch.Tensor, tissue: torch.Tensor, epoch: int = 0):
        if isinstance(outputs, dict):
            seg_logits = outputs["seg_logits"]
            st_logits = outputs.get("st_aux_logits")
            coverage = outputs.get("st_coverage")
        else:
            seg_logits = outputs if isinstance(outputs, (list, tuple)) else [outputs]
            st_logits = None
            coverage = None
        refine_scale = self._refine_scale(epoch)
        total = target.new_tensor(0.0)
        logs: Dict[str, float] = {"refine_scale": refine_scale, "region": self.region_name}

        for index, logits in enumerate(seg_logits):
            weight = float(self.ds_weights[index] if index < len(self.ds_weights) else 0.25)
            # Main output receives the full objective. Deep-supervision auxiliary
            # outputs deliberately avoid boundary/specificity terms because the
            # low-resolution heads are prone to fitting noisy pixel boundaries.
            is_main = index == 0
            loss, focal, region, boundary, specificity, brier = self._seg_single(
                logits,
                target,
                tissue,
                refine_scale,
                use_boundary=is_main,
                use_specificity=is_main,
                use_brier=True,
            )
            total = total + weight * loss
            if is_main:
                logs.update(
                    focal=float(focal.detach()),
                    dice=float(region.detach()),
                    bound=float(boundary.detach()),
                    spec=float(specificity.detach()),
                    brier=float(brier.detach()),
                )

        st_value = target.new_tensor(0.0)
        if self.w_st_aux > 0 and st_logits is not None and coverage is not None:
            st_value = self.st_aux(st_logits, target, tissue, coverage)
            total = total + self.w_st_aux * st_value
        logs["st_aux"] = float(st_value.detach())
        logs["total"] = float(total.detach())
        return total, logs
