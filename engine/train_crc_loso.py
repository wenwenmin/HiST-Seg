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
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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


CRC_LOSO_SLICES = (
    "SN048_1", "SN048_2", "SN048_3", "SN048_4",
    "SN84_1", "SN84_2", "SN123_1", "SN123_2",
    "SN123_4", "SN124_1", "SN124_2", "SN124_4",
)
CRC_SEED_OFFSETS = {slice_id: (i + 1) * 997 for i, slice_id in enumerate(CRC_LOSO_SLICES)}


def configure_crc_loso(cfg):
    """Force the experiment to CRC-only leave-one-slice-out cross-validation."""
    if not hasattr(cfg, "experiment"):
        cfg.experiment = ConfigNode()
    if not hasattr(cfg, "datasets") or not hasattr(cfg.datasets, "CRC"):
        raise KeyError("The config must contain datasets.CRC for CRC LOSO.")
    cfg.experiment.active_slices = list(CRC_LOSO_SLICES)
    cfg.experiment.external_folds = list(CRC_LOSO_SLICES)
    cfg.datasets.CRC.slices = list(CRC_LOSO_SLICES)
    return cfg


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
    return cfg


def dataset_configs(cfg, held_out: str):
    """Return only the CRC dataset paths for the current LOSO fold.

    Each fold uses one CRC slice as the held-out test slice and the
    remaining eleven CRC slices for training plus internal spatial validation.
    """
    active = set(str(s) for s in cfg.experiment.active_slices) & set(CRC_LOSO_SLICES)
    if held_out not in active:
        raise ValueError(f"Held-out slice {held_out!r} is not one of the CRC LOSO slices: {list(CRC_LOSO_SLICES)}")
    result = {}
    for dataset_name, ds_cfg in cfg.datasets.items():
        if str(dataset_name) != "CRC":
            continue
        slices = [str(s) for s in ds_cfg.slices if str(s) in active]
        if not slices:
            continue
        result["CRC"] = {
            "slices": slices,
            "patch_dir": str(ds_cfg.patch_dir),
            "context_dir": resolve_context_cache_dir(cfg, ds_cfg, held_out),
        }
    if "CRC" not in result:
        raise RuntimeError("No valid CRC dataset configuration was found.")
    return result

class EMA:
    """Exponential moving average over trainable parameters only.

    The previous implementation averaged every floating-point entry in
    state_dict(), including BatchNorm running_mean/running_var buffers. For
    small-batch cross-slice training this can corrupt validation calibration.
    This version leaves all buffers untouched and only smooths trainable
    parameters.
    """

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


def build_spatial_datasets(cfg, held_out: str):
    configs = dataset_configs(cfg, held_out)
    active = list(cfg.experiment.active_slices)
    train_parts, val_parts = [], []
    manifest = {
        "protocol": "CRC leave-one-slice-out cross-validation",
        "held_out": held_out,
        "train_slices": [sid for sid in active if sid != held_out],
        "datasets": {},
    }
    for dataset_index, (name, paths) in enumerate(configs.items()):
        valid_slices = [sid for sid in paths["slices"] if sid != held_out]
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
    if not train_parts or not val_parts:
        raise RuntimeError("Training or validation dataset is empty")
    return ConcatDataset(train_parts), ConcatDataset(val_parts), train_parts, manifest


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
    """Macro AUPRC from fixed probability histograms, without storing all pixels."""
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
def validate(model, loader, criterion, device, epoch: int, channels_last: bool, cfg):
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

    for batch in tqdm(loader, desc="Validation", ncols=110, leave=False):
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
    # AUPRC can keep improving while the sigmoid scale drifts and the Dice-optimal
    # threshold collapses to the search boundary. Select checkpoints with an
    # explicit preference for stable 0.5-calibrated masks.
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


def save_checkpoint(model, ema, cfg, held_out: str, epoch: int, metrics: Dict[str, Any],
                    manifest_path: str):
    raw = copy.deepcopy(model.state_dict())
    ema.copy_to(model)
    ema_state = strip_uni_weights(model.state_dict())
    model.load_state_dict(raw)
    path = os.path.join(str(cfg.paths.checkpoint_dir), f"pathst_crc_loso_{held_out}_best.pth")

    raw_threshold = float(metrics["threshold"])
    threshold_distance = float(metrics.get("threshold_distance", abs(raw_threshold - 0.5)))
    unstable_threshold = bool(metrics.get("threshold_at_boundary", False)) or (
        threshold_distance > float(getattr(cfg.train, "threshold_max_distance_for_calibration", 0.20))
    )
    if bool(getattr(cfg.train, "threshold_fallback_when_unstable", True)) and unstable_threshold:
        saved_threshold = float(getattr(cfg.train, "threshold_fallback_value", 0.50))
        threshold_source = (
            "fallback threshold because internal validation threshold was unstable "
            f"(raw={raw_threshold:.3f}, distance={threshold_distance:.3f})"
        )
    else:
        saved_threshold = raw_threshold
        threshold_source = "internal spatial validation only"

    torch.save({
        "model": strip_uni_weights(raw), "ema": ema_state, "epoch": epoch,
        "val_macro_fg_dice": metrics["macro_fg_dice"],
        "val_fixed_05_dice": metrics["fixed_05_dice"],
        "val_macro_auprc": metrics["macro_auprc"],
        "val_brier": metrics["brier"],
        "selection_score": metrics.get("selection_score"),
        "checkpoint_metric": str(getattr(cfg.train, "checkpoint_metric", "selection_score")),
        "raw_validation_threshold": raw_threshold,
        "threshold_distance": threshold_distance,
        "calibrated_threshold": saved_threshold,
        "threshold_source": threshold_source,
        "test_slice": held_out,
        "train_slices": [sid for sid in cfg.experiment.active_slices if sid != held_out],
        "all_slices": list(cfg.experiment.active_slices),
        "split_manifest": manifest_path,
        "random_seed": int(cfg.train.seed),
        "split_seed": int(cfg.train.split_seed),
        "model_version": "CRC LOSO aligned H&E/ST multimodal context + gated detail fusion v1",
        "config": cfg.to_dict(),
    }, path)
    print(f"Saved: {path}")


def train_one_fold(cfg, held_out: str, device: torch.device, channels_last: bool, use_compile: bool):
    train_slices = [sid for sid in cfg.experiment.active_slices if sid != held_out]
    print("\n" + "=" * 100)
    print(f"Held out: {held_out} | train/internal validation on {len(train_slices)} slices")
    print(train_slices)
    print("=" * 100)
    train_ds, val_ds, train_parts, manifest = build_spatial_datasets(cfg, held_out)
    manifest["random_seed"] = int(cfg.train.seed)
    manifest["split_seed"] = int(cfg.train.split_seed)
    os.makedirs(str(cfg.paths.checkpoint_dir), exist_ok=True)
    manifest_path = os.path.join(str(cfg.paths.checkpoint_dir), f"split_{held_out}.json")
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
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg.train.batch_size), shuffle=False, drop_last=False, **loader_args,
    )

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

        raw_state = copy.deepcopy(model.state_dict())
        ema.copy_to(model)
        metrics = validate(model, val_loader, criterion, device, epoch, channels_last, cfg)
        model.load_state_dict(raw_state)
        train_loss = running / max(1, batches)
        boundary_flag = " [THRESHOLD BOUNDARY]" if metrics["threshold_at_boundary"] else ""
        lr_main = float(optimizer.param_groups[0]["lr"])
        lr_st = float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else lr_main
        print(
            f"[{held_out}] epoch={epoch + 1:02d} train={train_loss:.4f} val={metrics['loss']:.4f} "
            f"AUPRC={metrics['macro_auprc']:.4f} Dice@best={metrics['macro_fg_dice']:.4f} "
            f"thr={metrics['threshold']:.2f} dthr={metrics.get('threshold_distance', 0.0):.2f}{boundary_flag} "
            f"Dice@0.5={metrics['fixed_05_dice']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"pred={metrics['pred_frac']:.4f} gt={metrics['gt_frac']:.4f} "
            f"brier={metrics['brier']:.4f} sel={metrics['selection_score']:.4f} "
            f"focal={metrics['components']['focal']:.4f} reg={metrics['components']['dice']:.4f} "
            f"bd={metrics['components']['bound']:.4f} spec={metrics['components']['spec']:.4f} "
            f"lossB={metrics['components']['brier']:.4f} STaux={metrics['components']['st_aux']:.4f} "
            f"stgate={metrics['gate_mean']:.4f} cov={metrics['coverage_mean']:.4f} "
            f"stscale={metrics['fusion_scale']:.4f} "
            f"ctxgate={metrics['context_gate_mean']:.4f} "
            f"ctxscale={metrics['context_fusion_scale']:.4f} "
            f"joint={metrics['context_joint_scale']:.4f} "
            f"pos={metrics['position_scale']:.4f} lr={lr_main:.2e}/{lr_st:.2e}"
        )
        append_csv(log_path, {
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": metrics["loss"],
            "macro_auprc": metrics["macro_auprc"], "macro_fg_dice": metrics["macro_fg_dice"],
            "fixed_05_dice": metrics["fixed_05_dice"], "selection_score": metrics["selection_score"],
            "threshold": metrics["threshold"], "threshold_at_boundary": metrics["threshold_at_boundary"],
            "threshold_distance": metrics.get("threshold_distance", abs(float(metrics["threshold"]) - 0.5)),
            "precision": metrics["precision"], "recall": metrics["recall"],
            "pred_frac": metrics["pred_frac"], "gt_frac": metrics["gt_frac"],
            "brier": metrics["brier"],
            "gate_mean": metrics["gate_mean"], "coverage_mean": metrics["coverage_mean"],
            "fusion_scale": metrics["fusion_scale"],
            "context_gate_mean": metrics["context_gate_mean"],
            "context_fusion_scale": metrics["context_fusion_scale"],
            "context_joint_scale": metrics["context_joint_scale"],
            "position_scale": metrics["position_scale"],
            "val_focal": metrics["components"]["focal"],
            "val_region": metrics["components"]["dice"],
            "val_boundary": metrics["components"]["bound"],
            "val_specificity": metrics["components"]["spec"],
            "val_loss_brier": metrics["components"]["brier"],
            "st_aux": metrics["components"]["st_aux"],
            "lr_main": lr_main, "lr_st": lr_st,
            "seconds": time.time() - start,
        })

        metric_name = str(getattr(cfg.train, "checkpoint_metric", "macro_auprc"))
        if metric_name not in metrics:
            raise KeyError(f"Unknown train.checkpoint_metric={metric_name}; available={sorted(metrics)}")
        score = float(metrics[metric_name])
        if score > best + 1e-5:
            best, bad_epochs = score, 0
            save_checkpoint(model, ema, cfg, held_out, epoch, metrics, manifest_path)
        elif epoch + 1 >= int(cfg.train.min_epochs):
            bad_epochs += 1
            if bad_epochs >= int(cfg.train.early_stop_patience):
                print(f"Early stopping on {metric_name}: best={best:.4f}")
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
    parser.add_argument("--config", default="crc_loso.yaml")
    parser.add_argument("--folds", nargs="+", default=None,
                        help="CRC LOSO folds to run. Omit to run all 12 folds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Random seeds. Omit to use experiment.seeds from YAML.")
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--overwrite", default=False,action="store_true",
                        help="Retrain and overwrite an existing seed/fold checkpoint.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).resolve().parent / args.config
    base_cfg = configure_crc_loso(ensure_runtime_defaults(load_config(config_path)))
    seeds = _resolve_run_seeds(base_cfg, args.seeds)
    folds = [str(fold) for fold in (args.folds or list(CRC_LOSO_SLICES))]
    unknown = sorted(set(folds) - set(CRC_LOSO_SLICES))
    if unknown:
        raise ValueError(f"Unsupported STHBC folds: {unknown}; supported={list(CRC_LOSO_SLICES)}")

    print(
        f"Data config: min_tissue_ratio={float(base_cfg.data.min_tissue_ratio):.3f}, "
        f"strict_context_cache={bool(base_cfg.data.strict_context_cache)}"
    )
    print(
        f"Checkpoint metric: {base_cfg.train.checkpoint_metric}; "
        f"ST lr multiplier: {float(base_cfg.train.st_lr_mult):.2f}; "
        f"EMA decay: {float(base_cfg.train.ema_decay):.4f}"
    )
    print(f"Random seeds: {seeds} | fixed split seed: {int(base_cfg.train.split_seed)}")
    print(f"CRC LOSO folds: {folds}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_root = str(base_cfg.paths.checkpoint_dir)
    for seed in seeds:
        print("\n" + "#" * 108)
        print(f"CRC multi-seed run: seed={seed}")
        print("#" * 108)
        for fold in folds:
            cfg = base_cfg.clone()
            cfg.train.seed = int(seed)
            seed_dir = os.path.join(checkpoint_root, f"seed_{seed}")
            fold_dir = os.path.join(seed_dir, f"fold_{fold}")
            cfg.paths.checkpoint_dir = fold_dir
            checkpoint = os.path.join(fold_dir, f"pathst_crc_loso_{fold}_best.pth")
            if os.path.exists(checkpoint) and not args.overwrite:
                print(f"Skip existing checkpoint: {checkpoint}")
                continue
            seed_everything(int(seed), deterministic=bool(cfg.train.deterministic))
            print(f"\n[seed={seed}] fold={fold} -> {fold_dir}")
            train_one_fold(cfg, str(fold), device, args.channels_last, args.compile)


if __name__ == "__main__":
    main()
