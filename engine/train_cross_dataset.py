from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PathSTDataset
from data.st_feature_utils import resolve_context_cache_dir
from model.losses import CombinedLoss
from model.pathst_net import build_model
from utils.config import ConfigNode, load_config


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.use_deterministic_algorithms(
    True,
    warn_only=True,
)
import warnings
warnings.filterwarnings("ignore")


def seed_everything(seed: int, deterministic: bool = True):
    """Seed Python/NumPy/PyTorch for one independent experimental run.

    ``deterministic=True`` improves exact reproducibility across reruns. It may
    reduce speed slightly, but does not change the model architecture or loss.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    torch.backends.cudnn.deterministic = bool(deterministic)
    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(bool(deterministic))


def cfg_get(obj: Any, name: str, default: Any):
    return getattr(obj, name, default)


def ensure_runtime_defaults(cfg):
    """Keep older YAML files executable while making the active settings explicit."""
    if not hasattr(cfg, "data"):
        cfg.data = ConfigNode()
    if not hasattr(cfg.data, "min_tissue_ratio"):
        cfg.data.min_tissue_ratio = 0.0
    if not hasattr(cfg.data, "strict_context_cache"):
        cfg.data.strict_context_cache = True
    if not hasattr(cfg.train, "checkpoint_metric"):
        cfg.train.checkpoint_metric = "macro_auprc"
    if not hasattr(cfg.train, "pr_bins"):
        cfg.train.pr_bins = 512
    if not hasattr(cfg.train, "selection_dice_weight"):
        cfg.train.selection_dice_weight = 0.45
    if not hasattr(cfg.train, "selection_fixed05_weight"):
        cfg.train.selection_fixed05_weight = 0.45
    if not hasattr(cfg.train, "selection_auprc_weight"):
        cfg.train.selection_auprc_weight = 0.10
    if not hasattr(cfg.train, "selection_brier_penalty"):
        cfg.train.selection_brier_penalty = 0.50
    if not hasattr(cfg.train, "selection_threshold_target"):
        cfg.train.selection_threshold_target = 0.50
    if not hasattr(cfg.train, "selection_threshold_distance_penalty"):
        cfg.train.selection_threshold_distance_penalty = 0.20
    if not hasattr(cfg.train, "selection_threshold_boundary_penalty"):
        cfg.train.selection_threshold_boundary_penalty = 0.08
    if not hasattr(cfg.train, "threshold_fallback_when_unstable"):
        cfg.train.threshold_fallback_when_unstable = True
    if not hasattr(cfg.train, "threshold_max_distance_for_calibration"):
        cfg.train.threshold_max_distance_for_calibration = 0.20
    if not hasattr(cfg.train, "threshold_fallback_value"):
        cfg.train.threshold_fallback_value = 0.50
    if not hasattr(cfg.train, "split_seed"):
        # Keep the spatial train/validation partition identical across model seeds.
        cfg.train.split_seed = 42
    if not hasattr(cfg.train, "deterministic"):
        cfg.train.deterministic = True

    # ---- Strong color augmentation defaults ----
    if not hasattr(cfg.train, "strong_color_aug"):
        cfg.train.strong_color_aug = ConfigNode()
    aug = cfg.train.strong_color_aug
    for key, value in {
        "enabled": False,
        "hed_p": 0.90, "hed_sigma": 0.06, "hed_bias": 0.06,
        "bc_p": 0.85, "brightness": 0.22, "contrast": 0.22,
        "hsv_p": 0.55, "saturation": 0.18, "hue": 0.035,
        "gamma_p": 0.45, "gamma": 0.20,
        "blur_p": 0.15, "noise_p": 0.15, "noise_std": 0.012,
    }.items():
        if not hasattr(aug, key):
            setattr(aug, key, value)

    # ---- Out-of-domain (cross-dataset) checkpoint selection defaults ----
    # When enabled, one *other* domain is held out of the training loss for every
    # external fold and used purely as the model-selection signal. This makes the
    # selection metric a genuine cross-domain signal instead of an in-domain one.
    if not hasattr(cfg.train, "domain_val"):
        cfg.train.domain_val = ConfigNode()
    dv = cfg.train.domain_val
    if not hasattr(dv, "enabled"):
        dv.enabled = False
    if not hasattr(dv, "slices"):
        dv.slices = ConfigNode()
    if not hasattr(dv, "selection_metric"):
        dv.selection_metric = "fixed_05_dice"
    if not hasattr(dv, "keep_indomain_monitor"):
        dv.keep_indomain_monitor = True
    return cfg


def dataset_configs(cfg, held_out: str):
    active = set(cfg.experiment.active_slices)
    result = {}
    for dataset_name, ds_cfg in cfg.datasets.items():
        slices = [str(s) for s in ds_cfg.slices if str(s) in active]
        if not slices:
            continue
        result[str(dataset_name)] = {
            "slices": slices,
            "patch_dir": str(ds_cfg.patch_dir),
            "context_dir": resolve_context_cache_dir(cfg, ds_cfg, held_out),
        }
    return result


def resolve_domain_val_slices(cfg, held_out: str) -> List[str]:
    """Return the list of slices used ONLY for cross-domain model selection.

    These slices are excluded from the training loss and from the in-domain
    validation blocks. They must not equal the held-out (test) slice.
    """
    dv = getattr(cfg.train, "domain_val", None)
    if dv is None or not bool(getattr(dv, "enabled", False)):
        return []
    slices_map = getattr(dv, "slices", None)
    if slices_map is None:
        return []
    # ConfigNode is a dict subclass -> use dict lookup so arbitrary fold names work.
    fold_slices = slices_map.get(held_out) if isinstance(slices_map, Mapping) else getattr(slices_map, held_out, None)
    if not fold_slices:
        return []
    active = set(str(s) for s in cfg.experiment.active_slices)
    resolved: List[str] = []
    for s in fold_slices:
        s = str(s)
        if s == str(held_out):
            raise ValueError(f"domain_val slice for fold {held_out} cannot be the test slice itself")
        if s not in active:
            raise ValueError(f"domain_val slice '{s}' for fold {held_out} is not in experiment.active_slices")
        if s not in resolved:
            resolved.append(s)
    return resolved


class EMA:
    """Exponential moving average over trainable parameters only."""

    def __init__(self, model: nn.Module, decay: float):
        self.base_decay = float(decay)
        self.enabled = self.base_decay > 0.0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module, step: int):
        if not self.enabled:
            return
        decay = min(self.base_decay, (1.0 + step) / (10.0 + step))
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        if not self.enabled:
            return
        state = model.state_dict()
        for name, value in self.shadow.items():
            if name in state:
                state[name].copy_(value)


def unwrap(model: nn.Module):
    return getattr(model, "_orig_mod", model)


def strip_uni_weights(state: Mapping[str, torch.Tensor]):
    output = {}
    for key, value in state.items():
        canonical = key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key
        if canonical.startswith("detail.vit."):
            continue
        output[canonical] = value
    return output


def main_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["seg_logits"][0]
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def build_param_groups(model: nn.Module, base_lr: float, st_mult: float):
    main, st = [], []
    for name, parameter in unwrap(model).named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("st_encoder.", "context_fusion.", "st_aux_head.")):
            st.append(parameter)
        else:
            main.append(parameter)
    groups = [{"params": main, "lr": base_lr, "name": "main"}]
    if st:
        groups.append({"params": st, "lr": base_lr * st_mult, "name": "st_fusion"})
    return groups


def make_scheduler(optimizer, cfg):
    warmup, total = int(cfg.train.warmup_epochs), int(cfg.train.epochs)
    eta_ratio = float(cfg.train.eta_min) / float(cfg.train.base_lr)

    def schedule(epoch: int):
        if epoch < warmup:
            return float(epoch + 1) / max(1, warmup)
        progress = min(1.0, max(0.0, (epoch - warmup) / max(1, total - warmup)))
        return eta_ratio + (1.0 - eta_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def choose_validation_blocks(dataset: PathSTDataset, val_ratio: float, block_size: int,
                             purge_margin: int, seed: int):
    by_slice = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(lambda: [0, 0])
    for index in range(len(dataset)):
        meta = dataset.get_meta(index)
        cx, cy = int(meta["x"]) + dataset.P // 2, int(meta["y"]) + dataset.P // 2
        bx, by = cx // block_size, cy // block_size
        sid = str(meta["slice_id"])
        by_slice[sid][(bx, by)].append(index)
        counts[(sid, bx, by)][0] += int(meta["cancer_pixels"])
        counts[(sid, bx, by)][1] += int(meta["tissue_pixels"])

    rng = np.random.default_rng(seed)
    selected, manifest = {}, {}
    for sid, blocks in sorted(by_slice.items()):
        candidates = []
        for bx, by in blocks:
            cancer, tissue = counts[(sid, bx, by)]
            candidates.append((bx, by, cancer / max(1, tissue), len(blocks[(bx, by)])))
        if len(candidates) < 2:
            selected[sid] = []
            manifest[sid] = {"reason": "fewer_than_2_blocks", "total_blocks": len(candidates)}
            continue
        target = min(len(candidates) - 1, max(1, int(round(len(candidates) * val_ratio))))
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: item[2])
        positions = np.linspace(0, len(candidates) - 1, target)
        indices, used = [], set()
        for position in positions:
            candidate = int(round(float(position)))
            if candidate in used:
                for delta in range(1, len(candidates)):
                    options = (candidate - delta, candidate + delta)
                    found = next((value for value in options if 0 <= value < len(candidates) and value not in used), None)
                    if found is not None:
                        candidate = found
                        break
            used.add(candidate)
            indices.append(candidate)
        selected[sid] = [(candidates[i][0], candidates[i][1]) for i in indices]
        manifest[sid] = {
            "total_blocks": len(candidates),
            "selected_blocks": [
                {"bx": candidates[i][0], "by": candidates[i][1],
                 "cancer_ratio": float(candidates[i][2]), "patches": candidates[i][3]}
                for i in indices
            ],
        }

    train_indices, val_indices, purged = [], [], []
    for index in range(len(dataset)):
        meta = dataset.get_meta(index)
        sid = str(meta["slice_id"])
        cx, cy = int(meta["x"]) + dataset.P // 2, int(meta["y"]) + dataset.P // 2
        is_val, is_purged = False, False
        for bx, by in selected.get(sid, []):
            x0, x1 = bx * block_size, (bx + 1) * block_size
            y0, y1 = by * block_size, (by + 1) * block_size
            if x0 <= cx < x1 and y0 <= cy < y1:
                is_val = True
                break
            if x0 - purge_margin <= cx < x1 + purge_margin and y0 - purge_margin <= cy < y1 + purge_margin:
                is_purged = True
        if is_val:
            val_indices.append(index)
        elif is_purged:
            purged.append(index)
        else:
            train_indices.append(index)
    manifest["summary"] = {
        "train_patches": len(train_indices), "val_patches": len(val_indices),
        "purged_patches": len(purged), "block_size": block_size,
        "purge_margin": purge_margin, "val_ratio": val_ratio,
    }
    return train_indices, val_indices, purged, manifest


def build_spatial_datasets(cfg, held_out: str, exclude_slices: Optional[Sequence[str]] = None):
    """Build training + in-domain (spatial-block) validation datasets.

    Slices in ``exclude_slices`` (the cross-domain selection slices) are removed
    from BOTH the training loss and the in-domain validation set.
    """
    exclude = set(str(s) for s in (exclude_slices or []))
    configs = dataset_configs(cfg, held_out)
    active = list(cfg.experiment.active_slices)
    train_parts, val_parts = [], []
    manifest = {
        "protocol": "external cross-dataset validation + out-of-domain model selection",
        "held_out": held_out,
        "domain_val_slices": sorted(exclude),
        "train_slices": [sid for sid in active if sid != held_out and sid not in exclude],
        "datasets": {},
    }
    for dataset_index, (name, paths) in enumerate(configs.items()):
        valid_slices = [sid for sid in paths["slices"] if sid != held_out and sid not in exclude]
        if not valid_slices:
            continue
        train_base = PathSTDataset(
            paths["patch_dir"], paths["context_dir"], valid_slices, cfg,
            is_train=True, return_meta=False, expected_fold=held_out,
        )
        val_base = PathSTDataset(
            paths["patch_dir"], paths["context_dir"], valid_slices, cfg,
            is_train=False, return_meta=True, expected_fold=held_out,
        )
        train_idx, val_idx, purged, split = choose_validation_blocks(
            val_base,
            float(cfg.train.spatial_val_ratio), int(cfg.train.spatial_block_size),
            int(cfg.train.spatial_purge_margin), int(cfg.train.split_seed) + dataset_index * 997,
        )
        if not train_idx:
            raise RuntimeError(f"Empty training split for {name}")
        if val_idx:
            val_parts.append(Subset(val_base, val_idx))
        train_parts.append(Subset(train_base, train_idx))
        manifest["datasets"][name] = split
        print(f"[{name}] train={len(train_idx)} val={len(val_idx)} purge={len(purged)} slices={valid_slices}")
    if not train_parts:
        raise RuntimeError("Training dataset is empty")
    if not val_parts:
        raise RuntimeError("In-domain validation dataset is empty")
    return ConcatDataset(train_parts), ConcatDataset(val_parts), train_parts, manifest


def build_domain_val_dataset(cfg, held_out: str, domain_val_slices: Sequence[str]):
    """Whole-slice validation set built from the held-out *domain* slices.

    No spatial blocking is needed: these slices are excluded from the training
    loss, so every patch is a valid (leakage-free) cross-domain validation
    sample. Returns None if no such slice resolves for this fold.
    """
    wanted = set(str(s) for s in domain_val_slices)
    if not wanted:
        return None
    configs = dataset_configs(cfg, held_out)
    parts = []
    covered: List[str] = []
    for name, paths in configs.items():
        slices = [sid for sid in paths["slices"] if sid in wanted and sid != held_out]
        if not slices:
            continue
        base = PathSTDataset(
            paths["patch_dir"], paths["context_dir"], slices, cfg,
            is_train=False, return_meta=True, expected_fold=held_out,
        )
        parts.append(base)
        covered.extend(slices)
        print(f"[domain-val:{name}] patches={len(base)} slices={slices}")
    missing = wanted - set(covered)
    if missing:
        raise RuntimeError(f"Could not build domain-val dataset for slices {sorted(missing)} (fold {held_out})")
    if not parts:
        return None
    return ConcatDataset(parts)


def build_weighted_sampler(train_parts: Sequence[Subset], cfg):
    edges = np.asarray(cfg.train.cancer_bin_edges, dtype=np.float64)
    bin_weights = np.asarray(cfg.train.cancer_bin_weights, dtype=np.float64)
    power = float(cfg.train.slice_balance_power)
    metas = []
    for subset in train_parts:
        for index in subset.indices:
            metas.append(subset.dataset.get_meta(int(index)))
    slice_counts = Counter(str(meta["slice_id"]) for meta in metas)
    weights = []
    for meta in metas:
        bin_id = int(np.digitize(float(meta["cancer_ratio"]), edges, right=False))
        weights.append(float(bin_weights[bin_id] * slice_counts[str(meta["slice_id"])] ** (-power)))
    array = np.asarray(weights, dtype=np.float64)
    array /= max(array.mean(), 1e-12)
    generator = torch.Generator().manual_seed(int(cfg.train.seed))
    return WeightedRandomSampler(torch.as_tensor(array, dtype=torch.double), len(array), replacement=True, generator=generator)


def move_batch(batch: Dict[str, Any], device: torch.device, channels_last: bool):
    hr = batch["hr"].to(device, non_blocking=True)
    he_context = batch["he_context"].to(device, non_blocking=True)
    context_tissue = batch["context_tissue"].to(device, non_blocking=True)
    spot_features = batch["spot_features"].to(device, non_blocking=True)
    spot_positions = batch["spot_positions"].to(device, non_blocking=True)
    spot_mask = batch["spot_mask"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    tissue = batch["tissue"].to(device, non_blocking=True)
    if channels_last:
        hr = hr.contiguous(memory_format=torch.channels_last)
        he_context = he_context.contiguous(memory_format=torch.channels_last)
        context_tissue = context_tissue.contiguous(memory_format=torch.channels_last)
    return (
        hr,
        he_context,
        spot_features,
        spot_positions,
        spot_mask,
        context_tissue,
        mask,
        tissue,
    )


def dice_from_counts(tp, fp, fn):
    return (2.0 * tp) / max(1e-8, 2.0 * tp + fp + fn)


def summarize_counts(counts):
    per_slice = []
    total = defaultdict(float)
    for sid, values in counts.items():
        per_slice.append(dice_from_counts(values["tp"], values["fp"], values["fn"]))
        for key, value in values.items():
            total[key] += value
    precision = total["tp"] / max(1e-8, total["tp"] + total["fp"])
    recall = total["tp"] / max(1e-8, total["tp"] + total["fn"])
    pred_frac = (total["tp"] + total["fp"]) / max(1.0, total["valid"])
    gt_frac = (total["tp"] + total["fn"]) / max(1.0, total["valid"])
    return {
        "macro_fg_dice": float(np.mean(per_slice)) if per_slice else 0.0,
        "precision": float(precision), "recall": float(recall),
        "pred_frac": float(pred_frac), "gt_frac": float(gt_frac),
    }


def _histogram_auprc(histograms):
    values = []
    for values_by_class in histograms.values():
        pos = values_by_class["pos"]
        neg = values_by_class["neg"]
        total_pos = float(pos.sum())
        if total_pos <= 0:
            continue
        tp = np.cumsum(pos[::-1])
        fp = np.cumsum(neg[::-1])
        precision = tp / np.maximum(tp + fp, 1e-12)
        recall = tp / total_pos
        delta_recall = np.diff(np.concatenate(([0.0], recall)))
        values.append(float(np.sum(precision * delta_recall)))
    return float(np.mean(values)) if values else 0.0


def _update_pr_histograms(histograms, probability, target, valid, slice_ids, bins: int):
    for index, sid in enumerate(slice_ids):
        mask = valid[index].reshape(-1)
        if not bool(mask.any()):
            continue
        p = probability[index].reshape(-1)[mask].detach().float().clamp(0.0, 1.0)
        y = target[index].reshape(-1)[mask].detach().bool()
        bin_index = torch.clamp((p * (bins - 1)).long(), 0, bins - 1)
        pos = torch.bincount(bin_index[y], minlength=bins).cpu().numpy().astype(np.float64)
        neg = torch.bincount(bin_index[~y], minlength=bins).cpu().numpy().astype(np.float64)
        histograms[str(sid)]["pos"] += pos
        histograms[str(sid)]["neg"] += neg


def threshold_grid(cfg):
    search = cfg.train.threshold_search
    if not bool(search.enabled):
        return np.asarray([0.5], dtype=np.float32)
    return np.arange(float(search.min), float(search.max) + 1e-8, float(search.step), dtype=np.float32)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch: int, channels_last: bool, cfg,
             desc: str = "Validation"):
    model.eval()
    thresholds = threshold_grid(cfg)
    threshold_counts = {float(t): defaultdict(lambda: defaultdict(float)) for t in thresholds}
    bins = int(getattr(cfg.train, "pr_bins", 512))
    pr_histograms = defaultdict(
        lambda: {"pos": np.zeros(bins, dtype=np.float64), "neg": np.zeros(bins, dtype=np.float64)}
    )
    loss_sum, batches = 0.0, 0
    component_sums = defaultdict(float)
    gate_sum, gate_n = 0.0, 0
    coverage_sum, coverage_n = 0.0, 0
    scale_sum, scale_n = 0.0, 0
    context_gate_sum, context_gate_n = 0.0, 0
    context_scale_sum, context_scale_n = 0.0, 0
    joint_scale_sum, joint_scale_n = 0.0, 0
    position_scale_sum, position_scale_n = 0.0, 0
    brier_sum, brier_n = 0.0, 0

    for batch in tqdm(loader, desc=desc, ncols=110, leave=False):
        (
            hr, he_context, spot_features, spot_positions, spot_mask,
            context_tissue, mask, tissue,
        ) = move_batch(batch, device, channels_last)
        outputs = model(
            hr,
            he_context=he_context,
            spot_features=spot_features,
            spot_positions=spot_positions,
            spot_mask=spot_mask,
            context_tissue=context_tissue,
        )
        loss, logs = criterion(outputs, mask, tissue, epoch=epoch)
        probability = torch.sigmoid(main_logits(outputs))
        valid = tissue > 0.5
        target = (mask > 0.5) & valid
        slice_ids = batch.get("slice_id", ["all"] * hr.shape[0])

        _update_pr_histograms(pr_histograms, probability, target, valid, slice_ids, bins)
        valid_float = valid.float()
        brier_sum += float((((probability - target.float()) ** 2) * valid_float).sum())
        brier_n += float(valid_float.sum())

        for threshold in thresholds:
            pred = (probability >= float(threshold)) & valid
            counts = threshold_counts[float(threshold)]
            for i, sid in enumerate(slice_ids):
                p, y, v = pred[i], target[i], valid[i]
                c = counts[str(sid)]
                c["tp"] += (p & y).sum().item()
                c["fp"] += (p & (~y) & v).sum().item()
                c["fn"] += ((~p) & y).sum().item()
                c["valid"] += v.sum().item()

        loss_sum += float(loss)
        for key in ("focal", "dice", "bound", "spec", "brier", "st_aux"):
            component_sums[key] += float(logs.get(key, 0.0))
        gate = outputs.get("fusion_gate")
        if gate is not None:
            gate_sum += float(gate.mean())
            gate_n += 1
        coverage = outputs.get("st_coverage")
        if coverage is not None:
            coverage_sum += float(coverage.mean())
            coverage_n += 1
        scale = outputs.get("fusion_scale")
        if scale is not None:
            scale_sum += float(scale)
            scale_n += 1
        context_gate = outputs.get("context_detail_gate")
        if context_gate is not None:
            context_gate_sum += float(context_gate.mean())
            context_gate_n += 1
        context_scale = outputs.get("context_detail_scale")
        if context_scale is not None:
            context_scale_sum += float(context_scale)
            context_scale_n += 1
        joint_scale = outputs.get("context_joint_scale")
        if joint_scale is not None:
            joint_scale_sum += float(joint_scale)
            joint_scale_n += 1
        position_scale = outputs.get("spot_position_scale")
        if position_scale is not None:
            position_scale_sum += float(position_scale)
            position_scale_n += 1
        batches += 1

    threshold_metrics = {threshold: summarize_counts(counts) for threshold, counts in threshold_counts.items()}
    best_threshold = max(threshold_metrics, key=lambda t: threshold_metrics[t]["macro_fg_dice"])
    best = dict(threshold_metrics[best_threshold])
    fixed = threshold_metrics[min(threshold_metrics, key=lambda t: abs(t - 0.5))]
    best.update({
        "threshold": float(best_threshold),
        "fixed_05_dice": float(fixed["macro_fg_dice"]),
        "macro_auprc": _histogram_auprc(pr_histograms),
        "brier": brier_sum / max(1.0, brier_n),
        "loss": loss_sum / max(1, batches),
        "gate_mean": gate_sum / max(1, gate_n),
        "coverage_mean": coverage_sum / max(1, coverage_n),
        "fusion_scale": scale_sum / max(1, scale_n),
        "context_gate_mean": context_gate_sum / max(1, context_gate_n),
        "context_fusion_scale": context_scale_sum / max(1, context_scale_n),
        "context_joint_scale": joint_scale_sum / max(1, joint_scale_n),
        "position_scale": position_scale_sum / max(1, position_scale_n),
        "components": {key: value / max(1, batches) for key, value in component_sums.items()},
    })
    lower, upper = float(thresholds.min()), float(thresholds.max())
    best["threshold_at_boundary"] = bool(
        abs(best["threshold"] - lower) < 1e-8 or abs(best["threshold"] - upper) < 1e-8
    )
    target_threshold = float(getattr(cfg.train, "selection_threshold_target", 0.50))
    best["threshold_distance"] = float(abs(best["threshold"] - target_threshold))
    brier_penalty = float(getattr(cfg.train, "selection_brier_penalty", 0.50))
    boundary_penalty = float(getattr(cfg.train, "selection_threshold_boundary_penalty", 0.08))
    distance_penalty = float(getattr(cfg.train, "selection_threshold_distance_penalty", 0.20))
    auprc_weight = float(getattr(cfg.train, "selection_auprc_weight", 0.10))
    dice_weight = float(getattr(cfg.train, "selection_dice_weight", 0.45))
    fixed05_weight = float(getattr(cfg.train, "selection_fixed05_weight", 0.45))
    best["selection_score"] = float(
        dice_weight * best["macro_fg_dice"]
        + fixed05_weight * best["fixed_05_dice"]
        + auprc_weight * best["macro_auprc"]
        - brier_penalty * best["brier"]
        - distance_penalty * best["threshold_distance"]
        - (boundary_penalty if best["threshold_at_boundary"] else 0.0)
    )
    return best



def append_csv(path: str, row: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _metric_or_none(metrics: Optional[Dict[str, Any]], key: str):
    if not metrics:
        return None
    value = metrics.get(key)
    return float(value) if value is not None else None


def save_checkpoint(model, ema, cfg, held_out: str, epoch: int, metrics: Dict[str, Any],
                    manifest_path: str,
                    monitor_metrics: Optional[Dict[str, Any]] = None,
                    domain_val_slices: Optional[Sequence[str]] = None,
                    selection_metric_name: str = "fixed_05_dice",
                    selection_source: str = "in_domain"):
    """Save a checkpoint. ``metrics`` are the *selection* metrics (cross-domain
    when out-of-domain selection is active, otherwise the in-domain metrics).
    ``monitor_metrics`` are the in-domain spatial-validation metrics kept for
    monitoring, and may be None when the in-domain monitor is disabled.
    """
    raw = copy.deepcopy(model.state_dict())
    ema.copy_to(model)
    ema_state = strip_uni_weights(model.state_dict())
    model.load_state_dict(raw)
    path = os.path.join(str(cfg.paths.checkpoint_dir), f"pathst_{held_out}_best.pth")

    raw_threshold = float(metrics["threshold"])
    threshold_distance = float(metrics.get("threshold_distance", abs(raw_threshold - 0.5)))
    unstable_threshold = bool(metrics.get("threshold_at_boundary", False)) or (
        threshold_distance > float(getattr(cfg.train, "threshold_max_distance_for_calibration", 0.20))
    )
    if bool(getattr(cfg.train, "threshold_fallback_when_unstable", True)) and unstable_threshold:
        saved_threshold = float(getattr(cfg.train, "threshold_fallback_value", 0.50))
        threshold_source = (
            "fallback threshold because selection threshold was unstable "
            f"(raw={raw_threshold:.3f}, distance={threshold_distance:.3f})"
        )
    else:
        saved_threshold = raw_threshold
        threshold_source = f"{selection_source} selection"

    torch.save({
        "model": strip_uni_weights(raw), "ema": ema_state, "epoch": epoch,
        # --- selection metrics (cross-domain when selection_source == 'domain_val') ---
        "val_macro_fg_dice": metrics["macro_fg_dice"],
        "val_fixed_05_dice": metrics["fixed_05_dice"],
        "val_macro_auprc": metrics["macro_auprc"],
        "val_brier": metrics["brier"],
        "selection_score": metrics.get("selection_score"),
        "selection_source": selection_source,
        "selection_metric": selection_metric_name,
        "domain_val_slices": list(domain_val_slices or []),
        # --- in-domain spatial-validation snapshot (monitoring only) ---
        "monitor_macro_fg_dice": _metric_or_none(monitor_metrics, "macro_fg_dice"),
        "monitor_fixed_05_dice": _metric_or_none(monitor_metrics, "fixed_05_dice"),
        "monitor_macro_auprc": _metric_or_none(monitor_metrics, "macro_auprc"),
        "checkpoint_metric": str(getattr(cfg.train, "checkpoint_metric", "selection_score")),
        "raw_validation_threshold": raw_threshold,
        "threshold_distance": threshold_distance,
        "calibrated_threshold": saved_threshold,
        "threshold_source": threshold_source,
        "test_slice": held_out,
        "train_slices": [
            sid for sid in cfg.experiment.active_slices
            if sid != held_out and sid not in set(domain_val_slices or [])
        ],
        "all_slices": list(cfg.experiment.active_slices),
        "split_manifest": manifest_path,
        "random_seed": int(cfg.train.seed),
        "split_seed": int(cfg.train.split_seed),
        "model_version": "PathST 512px detail + 4x context + OOD best-checkpoint selection",
        "config": cfg.to_dict(),
    }, path)
    print(f"Saved: {path}  (selected by {selection_source}:{selection_metric_name}="
          f"{float(metrics[selection_metric_name]):.4f})")



def train_one_fold(cfg, held_out: str, device: torch.device, channels_last: bool, use_compile: bool):
    active = list(cfg.experiment.active_slices)
    domain_val_slices = resolve_domain_val_slices(cfg, held_out)
    train_slices = [sid for sid in active if sid != held_out and sid not in set(domain_val_slices)]

    print("\n" + "=" * 100)
    print(f"Held out (TEST): {held_out}")
    print(f"Domain-val (model selection, NOT in loss): {domain_val_slices if domain_val_slices else '<none, in-domain selection>'}")
    print(f"Train slices ({len(train_slices)}): {train_slices}")
    print("=" * 100)

    train_ds, val_ds, train_parts, manifest = build_spatial_datasets(
        cfg, held_out, exclude_slices=domain_val_slices
    )
    manifest["random_seed"] = int(cfg.train.seed)
    manifest["split_seed"] = int(cfg.train.split_seed)
    domain_val_ds = build_domain_val_dataset(cfg, held_out, domain_val_slices)

    dv_cfg = getattr(cfg.train, "domain_val", None)
    keep_monitor = bool(getattr(dv_cfg, "keep_indomain_monitor", True)) if dv_cfg is not None else True
    use_domain_selection = bool(getattr(dv_cfg, "enabled", False)) and domain_val_ds is not None
    dv_selection_metric = str(getattr(dv_cfg, "selection_metric", "fixed_05_dice")) if dv_cfg is not None else "fixed_05_dice"
    fallback_metric = str(getattr(cfg.train, "checkpoint_metric", "fixed_05_dice"))

    if bool(getattr(dv_cfg, "enabled", False)) and domain_val_ds is None:
        print("[warn] domain_val.enabled=True but no domain-val slice resolved for this fold; "
              "falling back to in-domain selection.")

    os.makedirs(str(cfg.paths.checkpoint_dir), exist_ok=True)
    manifest_path = os.path.join(str(cfg.paths.checkpoint_dir), f"split_{held_out}.json")
    manifest["model_selection"] = {
        "mode": "out_of_domain" if use_domain_selection else "in_domain",
        "domain_val_slices": domain_val_slices,
        "selection_metric": dv_selection_metric if use_domain_selection else fallback_metric,
        "keep_indomain_monitor": keep_monitor,
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    sampler = build_weighted_sampler(train_parts, cfg) if bool(cfg.train.stratified_sampling) else None
    workers = int(cfg.train.num_workers)
    loader_args = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    if workers > 0:
        loader_args["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.train.batch_size), sampler=sampler,
        shuffle=sampler is None, drop_last=True, **loader_args,
    )
    # In-domain monitor loader: built when it drives selection (fallback) OR when
    # keep_indomain_monitor is on (so the divergence stays visible in the logs).
    build_monitor = keep_monitor or (not use_domain_selection)
    monitor_loader = DataLoader(
        val_ds, batch_size=int(cfg.train.batch_size), shuffle=False, drop_last=False, **loader_args,
    ) if build_monitor else None
    domain_val_loader = DataLoader(
        domain_val_ds, batch_size=int(cfg.train.batch_size), shuffle=False, drop_last=False, **loader_args,
    ) if domain_val_ds is not None else None

    model = build_model(cfg, build_uni=True).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if use_compile:
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile disabled after failure: {exc}")
    criterion = CombinedLoss(cfg).to(device)
    optimizer = torch.optim.AdamW(
        build_param_groups(model, float(cfg.train.base_lr), float(cfg.train.st_lr_mult)),
        lr=float(cfg.train.base_lr), weight_decay=float(cfg.train.weight_decay),
    )
    scheduler = make_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.amp and device.type == "cuda"))
    ema = EMA(model, float(cfg.train.ema_decay))

    log_path = os.path.join(str(cfg.paths.checkpoint_dir), f"train_{held_out}.csv")
    if os.path.exists(log_path):
        os.remove(log_path)

    selection_desc = (
        f"domain_val[{','.join(domain_val_slices)}]:{dv_selection_metric}"
        if use_domain_selection else f"in_domain:{fallback_metric}"
    )
    print(f"Model selection driven by -> {selection_desc}")

    best, bad_epochs, global_step = -1.0, 0, 0
    for epoch in range(int(cfg.train.epochs)):
        model.train()
        start, running, batches = time.time(), 0.0, 0
        components = defaultdict(float)
        progress = tqdm(train_loader, desc=f"[{held_out}] {epoch + 1}/{cfg.train.epochs}", ncols=110, leave=False)
        for batch in progress:
            (
                hr, he_context, spot_features, spot_positions, spot_mask,
                context_tissue, mask, tissue,
            ) = move_batch(batch, device, channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(cfg.train.amp and device.type == "cuda")):
                outputs = model(
                    hr,
                    he_context=he_context,
                    spot_features=spot_features,
                    spot_positions=spot_positions,
                    spot_mask=spot_mask,
                    context_tissue=context_tissue,
                )
                loss, logs = criterion(outputs, mask, tissue, epoch=epoch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(cfg.train.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            ema.update(model, global_step)
            running += float(loss.detach())
            batches += 1
            for key in ("focal", "dice", "bound", "spec", "brier", "st_aux"):
                components[key] += float(logs.get(key, 0.0))
            progress.set_postfix(loss=f"{float(loss):.3f}")
        scheduler.step()

        # Evaluate the EMA weights on both validation signals, then restore the
        # raw trainable weights before the next optimizer step.
        raw_state = copy.deepcopy(model.state_dict())
        ema.copy_to(model)
        monitor = validate(model, monitor_loader, criterion, device, epoch, channels_last, cfg,
                           desc="InDomainVal") if monitor_loader is not None else None
        domain = validate(model, domain_val_loader, criterion, device, epoch, channels_last, cfg,
                          desc="DomainVal") if domain_val_loader is not None else None

        if use_domain_selection:
            selection_metrics = domain
            sel_name = dv_selection_metric
            selection_source = "domain_val"
        else:
            selection_metrics = monitor
            sel_name = fallback_metric
            selection_source = "in_domain"
        if selection_metrics is None:
            raise RuntimeError("Selection metrics are unavailable; check validation loader construction.")
        if sel_name not in selection_metrics:
            raise KeyError(f"Selection metric '{sel_name}' not found; available={sorted(selection_metrics)}")


        model.load_state_dict(raw_state)

        train_loss = running / max(1, batches)
        lr_main = float(optimizer.param_groups[0]["lr"])
        lr_st = float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else lr_main

        mon = monitor or {}
        dom = domain or {}
        indomain_dice = mon.get("fixed_05_dice", float("nan"))
        domain_dice = dom.get("fixed_05_dice", float("nan"))
        domain_auprc = dom.get("macro_auprc", float("nan"))
        star = "  <== selected" if float(selection_metrics[sel_name]) > best + 1e-5 else ""
        print(
            f"[{held_out}] epoch={epoch + 1:02d} train={train_loss:.4f} "
            f"| in-domain Dice@0.5={indomain_dice:.4f} "
            f"| DOMAIN-VAL Dice@0.5={domain_dice:.4f} AUPRC={domain_auprc:.4f} "
            f"| sel({selection_source}:{sel_name})={float(selection_metrics[sel_name]):.4f} "
            f"| stgate={dom.get('gate_mean', mon.get('gate_mean', 0.0)):.3f} "
            f"ctxgate={dom.get('context_gate_mean', mon.get('context_gate_mean', 0.0)):.3f} "
            f"stscale={dom.get('fusion_scale', mon.get('fusion_scale', 0.0)):.3f} "
            f"ctxscale={dom.get('context_fusion_scale', mon.get('context_fusion_scale', 0.0)):.3f} "
            f"lr={lr_main:.2e}/{lr_st:.2e}{star}"
        )
        append_csv(log_path, {
            "epoch": epoch + 1, "train_loss": train_loss,
            "selection_source": selection_source, "selection_metric": sel_name,
            "selection_value": float(selection_metrics[sel_name]),
            # in-domain (monitor) snapshot
            "indomain_loss": mon.get("loss"),
            "indomain_dice_0.5": mon.get("fixed_05_dice"),
            "indomain_dice_best": mon.get("macro_fg_dice"),
            "indomain_auprc": mon.get("macro_auprc"),
            "indomain_precision": mon.get("precision"),
            "indomain_recall": mon.get("recall"),
            "indomain_brier": mon.get("brier"),
            # domain-val (cross-domain) snapshot
            "domainval_slices": ",".join(domain_val_slices),
            "domainval_loss": dom.get("loss"),
            "domainval_dice_0.5": dom.get("fixed_05_dice"),
            "domainval_dice_best": dom.get("macro_fg_dice"),
            "domainval_auprc": dom.get("macro_auprc"),
            "domainval_precision": dom.get("precision"),
            "domainval_recall": dom.get("recall"),
            "domainval_brier": dom.get("brier"),
            "gate_mean": dom.get("gate_mean", mon.get("gate_mean")),
            "coverage_mean": dom.get("coverage_mean", mon.get("coverage_mean")),
            "fusion_scale": dom.get("fusion_scale", mon.get("fusion_scale")),
            "lr_main": lr_main, "lr_st": lr_st,
            "seconds": time.time() - start,
        })

        score = float(selection_metrics[sel_name])
        if score > best + 1e-5:
            best, bad_epochs = score, 0
            save_checkpoint(
                model, ema, cfg, held_out, epoch, selection_metrics, manifest_path,
                monitor_metrics=monitor, domain_val_slices=domain_val_slices,
                selection_metric_name=sel_name, selection_source=selection_source,
            )
        elif epoch + 1 >= int(cfg.train.min_epochs):
            bad_epochs += 1
            if bad_epochs >= int(cfg.train.early_stop_patience):
                print(f"Early stopping on {selection_source}:{sel_name}: best={best:.4f}")
                break
    return best


def _resolve_run_seeds(cfg, cli_seeds):
    configured = list(getattr(cfg.experiment, "seeds", [42, 123, 567]))
    seeds = cli_seeds if cli_seeds is not None else configured
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("At least one random seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate random seeds are not allowed: {seeds}")
    return seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cross_dataset.yaml")
    parser.add_argument("--folds", nargs="+", default=['DuCIS'],
                        help="External test folds. Omit to run ViHBC, DuCIS and XeHBC.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Random seeds. Omit to use experiment.seeds from YAML.")
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--overwrite", default=True,action="store_true",
                        help="Retrain and overwrite an existing seed/fold checkpoint.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).resolve().parent / args.config
    base_cfg = ensure_runtime_defaults(load_config(config_path))
    seeds = _resolve_run_seeds(base_cfg, args.seeds)
    folds = [str(fold) for fold in (args.folds or list(base_cfg.experiment.external_folds))]
    unknown = sorted(set(folds) - set(str(x) for x in base_cfg.experiment.external_folds))
    if unknown:
        raise ValueError(f"Unsupported folds: {unknown}")

    print(
        f"Data config: min_tissue_ratio={float(base_cfg.data.min_tissue_ratio):.3f}, "
        f"strict_context_cache={bool(base_cfg.data.strict_context_cache)}"
    )
    print(
        f"Checkpoint metric (fallback): {base_cfg.train.checkpoint_metric}; "
        f"ST lr multiplier: {float(base_cfg.train.st_lr_mult):.2f}; "
        f"EMA decay: {float(base_cfg.train.ema_decay):.4f}"
    )
    dv = getattr(base_cfg.train, "domain_val", None)
    print(
        f"Out-of-domain selection: enabled={bool(getattr(dv, 'enabled', False))}; "
        f"selection_metric={getattr(dv, 'selection_metric', 'n/a')}; "
        f"keep_indomain_monitor={bool(getattr(dv, 'keep_indomain_monitor', True))}"
    )
    aug = getattr(base_cfg.train, "strong_color_aug", None)
    print(
        f"Strong color augmentation: enabled={bool(getattr(aug, 'enabled', False))}; "
        "UNI extraction=online frozen backbone"
    )
    print(f"Random seeds: {seeds} | fixed split seed: {int(base_cfg.train.split_seed)}")
    print(f"External folds: {folds}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_root = str(base_cfg.paths.checkpoint_dir)
    for seed in seeds:
        print("\n" + "#" * 108)
        print(f"Cross-dataset multi-seed run: seed={seed}")
        print("#" * 108)
        for fold in folds:
            cfg = base_cfg.clone()
            cfg.train.seed = int(seed)
            seed_dir = os.path.join(checkpoint_root, f"seed_{seed}")
            fold_dir = os.path.join(seed_dir, f"fold_{fold}")
            cfg.paths.checkpoint_dir = fold_dir
            checkpoint = os.path.join(fold_dir, f"pathst_{fold}_best.pth")
            if os.path.exists(checkpoint) and not args.overwrite:
                print(f"Skip existing checkpoint: {checkpoint}")
                continue
            seed_everything(int(seed), deterministic=bool(cfg.train.deterministic))
            print(f"\n[seed={seed}] fold={fold} -> {fold_dir}")
            train_one_fold(cfg, str(fold), device, args.channels_last, args.compile)


if __name__ == "__main__":
    main()
