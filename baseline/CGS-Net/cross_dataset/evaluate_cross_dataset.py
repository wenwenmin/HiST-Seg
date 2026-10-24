# Path configuration:
# Replace the repository-relative defaults (data/, outputs/, and workspaces/)
# for your environment. See readme_baseline.txt at the baseline package root.
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

try:
    from .cgs_cross_dataset_lib import (
        append_csv,
        ensure_dir,
        evaluate_whole_slide,
        get_device,
        load_all_slides,
        load_yaml,
        make_dual_model_from_singles,
        summarize_metrics,
        write_rows_csv,
    )
except ImportError:
    from cgs_cross_dataset_lib import (
        append_csv,
        ensure_dir,
        evaluate_whole_slide,
        get_device,
        load_all_slides,
        load_yaml,
        make_dual_model_from_singles,
        summarize_metrics,
        write_rows_csv,
    )


def fold_lists(cfg: dict, test_slice: str):
    val_slice = cfg["fold_val_map"][test_slice]
    train_slices = [s for s in cfg["slices"] if s not in {test_slice, val_slice}]
    return train_slices, val_slice, test_slice


def infer_one_fold(cfg: dict, slides: dict, test_slice: str):
    fold_name = f"fold_{test_slice}"
    fold_dir = Path(cfg["output_root"]) / fold_name
    detail_ckpt = fold_dir / "single_detail" / "single_detail_best.pth"
    context_ckpt = fold_dir / "single_context" / "single_context_best.pth"
    cgs_ckpt = fold_dir / "cgs" / "cgs_best.pth"
    for p in [detail_ckpt, context_ckpt, cgs_ckpt]:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
    _, val_slice, test_slice = fold_lists(cfg, test_slice)
    device = get_device(cfg)
    model = make_dual_model_from_singles(cfg, detail_ckpt, context_ckpt, freeze_base=False).to(device)
    model.load_state_dict(torch.load(cgs_ckpt, map_location=device), strict=True)
    save_dir = fold_dir / "cgs" / "test_reinfer"
    metrics = evaluate_whole_slide(model, slides[test_slice], cfg, device, mode="dual", save_dir=save_dir)
    row = {"fold": fold_name, "test_slice": test_slice, "val_slice": val_slice, **metrics}
    with open(save_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    experiment_name = cfg.get("experiment_name", "sthbc_loso")
    append_csv(Path(cfg["output_root"]) / f"{experiment_name}_test_summary_reinfer.csv", row)
    print(f"[{fold_name}] Dice={metrics['dice']:.4f} IoU={metrics['iou']:.4f} AP={metrics['ap']:.4f} AUC={metrics['auc']:.4f} HD95={metrics.get('hd95', float('nan')):.2f}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cross_dataset.yaml")
    parser.add_argument("--fold", default="all", help="all configured evaluation slides, or one slide ID")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    ensure_dir(cfg["output_root"])
    slides = load_all_slides(cfg)
    # overwrite previous reinfer summary for a clean run
    experiment_name = cfg.get("experiment_name", "sthbc_loso")
    summary_path = Path(cfg["output_root"]) / f"{experiment_name}_test_summary_reinfer.csv"
    if summary_path.exists():
        summary_path.unlink()
    evaluation_slices = list(cfg.get("evaluation_slices", cfg["slices"]))
    folds = evaluation_slices if args.fold == "all" else [args.fold]
    rows = [infer_one_fold(cfg, slides, f) for f in folds]
    summary = summarize_metrics(rows, ["dice", "iou", "precision", "recall", "auc", "ap", "hd95"])
    mean_std_path = Path(cfg["output_root"]) / f"{experiment_name}_mean_std_reinfer.csv"
    write_rows_csv(mean_std_path, summary)
    print("Saved:", mean_std_path)


if __name__ == "__main__":
    main()
