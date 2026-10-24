# Path configuration:
# Replace the repository-relative defaults (data/, outputs/, and workspaces/)
# for your environment. See readme_baseline.txt at the baseline package root.
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from cgs_sthbc_lib import (
    DualPatchDataset,
    SinglePatchDataset,
    append_csv,
    build_optimizer_dual,
    build_optimizer_single,
    check_dataset,
    ensure_dir,
    eval_patch_single,
    evaluate_whole_slide,
    eval_patch_dual,
    get_device,
    load_all_slides,
    load_manifest,
    load_yaml,
    make_dual_model_from_singles,
    make_single_model,
    sample_patch_manifest,
    save_manifest,
    set_requires_grad,
    set_seed,
    summarize_metrics,
    train_one_epoch_dual,
    train_one_epoch_single,
    write_rows_csv,
)


def get_fold_lists(cfg: dict, test_slice: str):
    slices = list(cfg["slices"])
    val_slice = cfg["fold_val_map"][test_slice]
    train_slices = [s for s in slices if s not in {test_slice, val_slice}]
    return train_slices, val_slice, test_slice


def maybe_build_manifests(cfg: dict, slides: dict, fold_dir: Path, train_slices: List[str], val_slice: str, fold_seed: int):
    train_manifest = fold_dir / "manifests" / "train_manifest.csv"
    val_manifest = fold_dir / "manifests" / "val_manifest.csv"
    if train_manifest.exists() and val_manifest.exists():
        return load_manifest(train_manifest), load_manifest(val_manifest)
    rows_train = sample_patch_manifest(
        slides, train_slices, cfg, cfg["sampling"]["train_per_slide"], seed=fold_seed
    )
    rows_val = sample_patch_manifest(
        slides, [val_slice], cfg, cfg["sampling"]["val_per_slide"], seed=fold_seed + 10000
    )
    save_manifest(rows_train, train_manifest)
    save_manifest(rows_val, val_manifest)
    return rows_train, rows_val


def make_loader(ds, cfg: dict, train: bool):
    return DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=train,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def train_single_stage(
    cfg: dict,
    slides: dict,
    rows_train: list,
    rows_val: list,
    fold_dir: Path,
    fold_name: str,
    input_level: str,
    device: torch.device,
) -> Path:
    assert input_level in {"detail", "context"}
    stage_dir = ensure_dir(fold_dir / f"single_{input_level}")
    ckpt_path = stage_dir / f"single_{input_level}_best.pth"
    log_path = stage_dir / f"single_{input_level}_log.csv"
    if ckpt_path.exists():
        print(f"[{fold_name}] Reusing existing {input_level} checkpoint: {ckpt_path}")
        return ckpt_path

    ds_train = SinglePatchDataset(rows_train, slides, cfg, input_level=input_level, train=True)
    ds_val = SinglePatchDataset(rows_val, slides, cfg, input_level=input_level, train=False)
    train_loader = make_loader(ds_train, cfg, train=True)
    val_loader = make_loader(ds_val, cfg, train=False)

    model = make_single_model(cfg, freeze_encoder=True).to(device)
    optimizer = build_optimizer_single(model, cfg, backbone_trainable=False)
    best_score = -1.0
    best_epoch = -1
    no_improve = 0
    epochs = int(cfg["train"][f"{input_level}_epochs"])
    unfreeze_epoch = int(cfg["train"].get("single_unfreeze_epoch", 10))
    amp = bool(cfg["train"].get("amp", True))
    patience = int(cfg["train"].get("patience", 12))

    for epoch in range(1, epochs + 1):
        if epoch == unfreeze_epoch + 1:
            print(f"[{fold_name}][single-{input_level}] unfreezing encoder at epoch {epoch}")
            set_requires_grad(model.encoder, True)
            optimizer = build_optimizer_single(model, cfg, backbone_trainable=True)
        train_loss = train_one_epoch_single(model, train_loader, optimizer, device, amp=amp)
        val_metrics = eval_patch_single(model, val_loader, device)
        score = 0.6 * val_metrics.get("dice", 0.0) + 0.4 * (0.0 if math.isnan(val_metrics.get("ap", float("nan"))) else val_metrics.get("ap", 0.0))
        row = {"fold": fold_name, "stage": f"single_{input_level}", "epoch": epoch, "train_loss": train_loss, "score": score, **val_metrics}
        append_csv(log_path, row)
        print(f"[{fold_name}][single-{input_level}] epoch={epoch:03d} train={train_loss:.4f} val_dice={val_metrics['dice']:.4f} val_ap={val_metrics['ap']:.4f} score={score:.4f}")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            with open(stage_dir / "best_info.json", "w", encoding="utf-8") as f:
                json.dump({"best_epoch": best_epoch, "best_score": best_score, "checkpoint": str(ckpt_path)}, f, indent=2)
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"[{fold_name}][single-{input_level}] early stopping at epoch {epoch}")
            break
    return ckpt_path


def train_cgs_stage(
    cfg: dict,
    slides: dict,
    rows_train: list,
    rows_val: list,
    fold_dir: Path,
    fold_name: str,
    val_slice: str,
    test_slice: str,
    detail_ckpt: Path,
    context_ckpt: Path,
    device: torch.device,
    run_test: bool,
):
    stage_dir = ensure_dir(fold_dir / "cgs")
    ckpt_path = stage_dir / "cgs_best.pth"
    log_path = stage_dir / "cgs_log.csv"

    ds_train = DualPatchDataset(rows_train, slides, cfg, train=True)
    ds_val = DualPatchDataset(rows_val, slides, cfg, train=False)
    train_loader = make_loader(ds_train, cfg, train=True)
    val_patch_loader = make_loader(ds_val, cfg, train=False)

    model = make_dual_model_from_singles(cfg, detail_ckpt, context_ckpt, freeze_base=True).to(device)
    optimizer = build_optimizer_dual(model, cfg, base_trainable=False)
    best_score = -1.0
    best_epoch = -1
    no_improve = 0
    amp = bool(cfg["train"].get("amp", True))
    patience = int(cfg["train"].get("patience", 12))
    epochs = int(cfg["train"].get("cgs_epochs", 60))
    unfreeze_epoch = int(cfg["train"].get("cgs_unfreeze_epoch", 15))
    val_every = int(cfg["train"].get("validate_whole_slide_every", 1))

    for epoch in range(1, epochs + 1):
        if epoch == unfreeze_epoch + 1:
            print(f"[{fold_name}][CGS] unfreezing encoders and lv2 decoder at epoch {epoch}")
            set_requires_grad(model.encoder.encoder_detail, True)
            set_requires_grad(model.encoder.encoder_context, True)
            set_requires_grad(model.decoder.lv2decoder, True)
            set_requires_grad(model.decoder.mca, True)
            optimizer = build_optimizer_dual(model, cfg, base_trainable=True)
        train_loss = train_one_epoch_dual(model, train_loader, optimizer, device, amp=amp)
        patch_metrics = {}
        if val_every > 0 and epoch % val_every == 0:
            whole_metrics = evaluate_whole_slide(
                model, slides[val_slice], cfg, device, mode="dual", save_dir=stage_dir / "val_whole_last"
            )
        else:
            whole_metrics = {"selection_score": float("nan"), "dice": float("nan"), "ap": float("nan")}
        # Lightweight patch metrics provide extra debugging but do not select the final model.
        patch_metrics = eval_patch_dual(model, val_patch_loader, device)
        score = whole_metrics.get("selection_score", float("nan"))
        if math.isnan(score):
            score = 0.6 * patch_metrics.get("dice", 0.0) + 0.4 * (0.0 if math.isnan(patch_metrics.get("ap", float("nan"))) else patch_metrics.get("ap", 0.0))
        row = {
            "fold": fold_name,
            "stage": "cgs",
            "epoch": epoch,
            "train_loss": train_loss,
            "patch_val_dice": patch_metrics.get("dice"),
            "patch_val_ap": patch_metrics.get("ap"),
            "whole_val_dice": whole_metrics.get("dice"),
            "whole_val_iou": whole_metrics.get("iou"),
            "whole_val_ap": whole_metrics.get("ap"),
            "whole_val_auc": whole_metrics.get("auc"),
            "selection_score": score,
        }
        append_csv(log_path, row)
        print(f"[{fold_name}][CGS] epoch={epoch:03d} train={train_loss:.4f} patch_dice={patch_metrics['dice']:.4f} whole_dice={whole_metrics.get('dice', float('nan')):.4f} whole_ap={whole_metrics.get('ap', float('nan')):.4f} sel={score:.4f}")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            with open(stage_dir / "best_info.json", "w", encoding="utf-8") as f:
                json.dump({"best_epoch": best_epoch, "best_score": best_score, "checkpoint": str(ckpt_path), "val_slice": val_slice}, f, indent=2)
            # Keep a snapshot of the selected validation visualization.
            evaluate_whole_slide(model, slides[val_slice], cfg, device, mode="dual", save_dir=stage_dir / "val_best")
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"[{fold_name}][CGS] early stopping at epoch {epoch}")
            break

    if not ckpt_path.exists():
        raise RuntimeError(f"CGS checkpoint was not saved for {fold_name}")
    if run_test:
        model = make_dual_model_from_singles(cfg, detail_ckpt, context_ckpt, freeze_base=False).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
        test_metrics = evaluate_whole_slide(
            model, slides[test_slice], cfg, device, mode="dual", save_dir=stage_dir / "test"
        )
        test_row = {"fold": fold_name, "test_slice": test_slice, "val_slice": val_slice, **test_metrics}
        with open(stage_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_row, f, indent=2)
        append_csv(Path(cfg["output_root"]) / "sthbc_loso_test_summary.csv", test_row)
        print(f"[{fold_name}][TEST] Dice={test_metrics['dice']:.4f} IoU={test_metrics['iou']:.4f} AP={test_metrics['ap']:.4f} AUC={test_metrics['auc']:.4f} HD95={test_metrics.get('hd95', float('nan')):.2f}")
    return ckpt_path


def train_fold(cfg: dict, slides: dict, test_slice: str, run_test: bool = True):
    train_slices, val_slice, test_slice = get_fold_lists(cfg, test_slice)
    fold_name = f"fold_{test_slice}"
    fold_dir = ensure_dir(Path(cfg["output_root"]) / fold_name)
    with open(fold_dir / "fold_split.json", "w", encoding="utf-8") as f:
        json.dump({"fold": fold_name, "train": train_slices, "val": val_slice, "test": test_slice}, f, indent=2)
    print("=" * 100)
    print(f"{fold_name}: train={train_slices}, val={val_slice}, test={test_slice}")
    print("=" * 100)
    fold_seed = int(cfg["train"].get("seed", 42)) + list(cfg["slices"]).index(test_slice) * 1000
    rows_train, rows_val = maybe_build_manifests(cfg, slides, fold_dir, train_slices, val_slice, fold_seed)
    device = get_device(cfg)
    print(f"Using device: {device}")
    detail_ckpt = train_single_stage(cfg, slides, rows_train, rows_val, fold_dir, fold_name, "detail", device)
    context_ckpt = train_single_stage(cfg, slides, rows_train, rows_val, fold_dir, fold_name, "context", device)
    cgs_ckpt = train_cgs_stage(cfg, slides, rows_train, rows_val, fold_dir, fold_name, val_slice, test_slice, detail_ckpt, context_ckpt, device, run_test=run_test)
    return cgs_ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sthbc_loso.yaml")
    parser.add_argument("--fold", default="G2", help="all or one of C1,D1,H1,E1,F1,G2,A1,B1")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-test", action="store_true", help="train only; do not evaluate test slice after each fold")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["train"].get("seed", 42)))
    ensure_dir(cfg["output_root"])
    print("Checking STHBC data...")
    check_dataset(cfg, out_dir=Path(cfg["output_root"]) / "dataset_check")
    if args.check_only:
        print("Check-only finished.")
        return
    slides = load_all_slides(cfg)
    folds = list(cfg["slices"]) if args.fold == "all" else [args.fold]
    all_rows = []
    for test_slice in folds:
        train_fold(cfg, slides, test_slice, run_test=not args.no_test)
    summary_csv = Path(cfg["output_root"]) / "sthbc_loso_test_summary.csv"
    if summary_csv.exists():
        import csv
        with open(summary_csv, "r", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))
        metrics = ["dice", "iou", "precision", "recall", "auc", "ap", "hd95"]
        summary = summarize_metrics(all_rows, metrics)
        write_rows_csv(Path(cfg["output_root"]) / "sthbc_loso_mean_std.csv", summary)
        print("Saved final summary:", Path(cfg["output_root"]) / "sthbc_loso_mean_std.csv")


if __name__ == "__main__":
    main()
