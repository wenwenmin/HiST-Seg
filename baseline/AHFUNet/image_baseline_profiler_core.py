from __future__ import annotations

"""Unified computational-cost profiler for ResNet50-UNet and AHFUNet.

Protocol
--------
- supervised training data: STHBC patches;
- training fold: A1, matching the checkpoints used by the supplied external
  ViHBC inference scripts;
- random seed: 42;
- target inference section: ViHBC;
- inference: one complete-slide warm-up and three measured complete-slide runs;
- parameters: all instantiated model parameters;
- GPU memory: peak absolute NVIDIA memory of this process and its descendants;
- excluded from inference time: model/checkpoint loading, metrics,
  visualization, and output-file writing.

The training timer starts after the prepared STHBC dataset objects and loaders
have been constructed. Batch loading and augmentation during every epoch remain
inside the timed training phase.
"""

import argparse
import csv
import gc
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from unified_gpu_monitor import ProcessTreeGPUMonitor


SEED = 42
TRAIN_FOLD = "A1"
STHBC_SLICES = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]
WARMUP_RUNS = 1
INFERENCE_REPEATS = 3


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def cv_imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if image is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    return image


def cv_imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )
    return total, trainable


def load_local_modules(architecture: str, baseline_dir: Path):
    baseline_dir = baseline_dir.resolve()
    if str(baseline_dir) not in sys.path:
        sys.path.insert(0, str(baseline_dir))

    # Avoid importing a model.py left over from another baseline.
    for module_name in (
        "model",
        "dataset_ST1",
        "ahf_unet_baseline",
    ):
        sys.modules.pop(module_name, None)

    dataset_module = importlib.import_module("dataset_ST1")
    if architecture == "resnet50_unet":
        model_module = importlib.import_module("model")
        model_class = getattr(model_module, "ResNet50_UNet")
    elif architecture == "ahfunet":
        model_module = importlib.import_module("ahf_unet_baseline")
        model_class = getattr(model_module, "AHFUnet_Baseline")
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return dataset_module.HEDataset, dataset_module.get_transforms, model_class


def build_model(model_class, architecture: str, pretrained: bool) -> nn.Module:
    if architecture == "resnet50_unet":
        return model_class(pretrained=bool(pretrained))
    return model_class()


def unwrap_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        if torch.is_tensor(output[0]):
            return output[0]
    if isinstance(output, dict):
        for key in ("logits", "out", "prediction", "pred"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise TypeError(f"Unable to extract logits from output type {type(output)!r}")


class MaskedBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tissue_mask: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-7
        probabilities = torch.sigmoid(logits)
        bce = self.bce(logits, labels)
        masked_bce = (bce * tissue_mask).sum() / (
            tissue_mask.sum() + eps
        )
        intersection = (
            probabilities * labels * tissue_mask
        ).sum()
        union = (
            (probabilities * tissue_mask).sum()
            + (labels * tissue_mask).sum()
        )
        dice_loss = 1.0 - (2.0 * intersection + eps) / (union + eps)
        return self.bce_weight * masked_bce + self.dice_weight * dice_loss


def ahf_training_loss(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    tissue_masks: torch.Tensor,
) -> torch.Tensor:
    output = model(
        x=images,
        targets=masks,
        tissue_masks=tissue_masks,
    )
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        loss = output[1]
        if torch.is_tensor(loss):
            return loss
    if isinstance(output, dict) and torch.is_tensor(output.get("loss")):
        return output["loss"]
    raise TypeError(
        "AHFUNet training forward did not return a tensor loss in the "
        "expected tuple/dictionary format."
    )


def make_loaders(
    HEDataset,
    get_transforms,
    patch_dir: Path,
    h5ad_base_dir: Path,
    batch_size: int,
    num_workers: int,
):
    train_slices = [
        slide for slide in STHBC_SLICES if slide != TRAIN_FOLD
    ]
    full_train = HEDataset(
        str(patch_dir),
        str(h5ad_base_dir),
        allowed_slices=train_slices,
        transform=get_transforms(is_train=True),
    )
    full_validation = HEDataset(
        str(patch_dir),
        str(h5ad_base_dir),
        allowed_slices=train_slices,
        transform=get_transforms(is_train=False),
    )

    count = len(full_train)
    if count < 2:
        raise RuntimeError(f"Too few STHBC patches: {count}")

    indices = np.arange(count)
    np.random.shuffle(indices)
    validation_size = max(1, int(count * 0.2))
    training_size = count - validation_size

    train_loader = DataLoader(
        Subset(full_train, indices[:training_size]),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    validation_loader = DataLoader(
        Subset(full_validation, indices[training_size:]),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader, train_slices


def train_worker(args) -> None:
    set_seed(SEED)
    baseline_dir = Path(args.baseline_dir).resolve()
    output_json = Path(args.worker_output).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    HEDataset, get_transforms, model_class = load_local_modules(
        args.architecture, baseline_dir
    )

    # Resolve/download pretrained backbone weights outside the measured phase.
    if args.architecture == "resnet50_unet":
        preflight = build_model(
            model_class, args.architecture, pretrained=True
        )
        del preflight
        gc.collect()

    train_loader, validation_loader, train_slices = make_loaders(
        HEDataset=HEDataset,
        get_transforms=get_transforms,
        patch_dir=Path(args.patch_dir),
        h5ad_base_dir=Path(args.h5ad_base_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device(
        f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for unified GPU profiling.")

    clear_cuda()
    monitor = ProcessTreeGPUMonitor(
        root_pid=os.getpid(),
        gpu_index=args.gpu_index,
        interval_s=0.05,
    ).start()
    start = time.perf_counter()

    model = build_model(
        model_class,
        args.architecture,
        pretrained=(args.architecture == "resnet50_unet"),
    ).to(device)
    total_params, trainable_params = count_parameters(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    scaler = GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )
    resnet_loss = MaskedBCEDiceLoss()

    best_validation_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        progress = tqdm(
            train_loader,
            desc=(
                f"{args.display_name} epoch "
                f"{epoch}/{args.epochs} [train]"
            ),
            ncols=110,
        )
        for images, masks, tissue_masks, *_ in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(
                device, non_blocking=True
            ).float()
            tissue_masks = tissue_masks.to(
                device, non_blocking=True
            ).float()

            optimizer.zero_grad(set_to_none=True)
            with autocast(
                "cuda",
                enabled=(device.type == "cuda"),
            ):
                if args.architecture == "resnet50_unet":
                    logits = unwrap_logits(model(images))
                    loss = resnet_loss(
                        logits, masks, tissue_masks
                    )
                else:
                    loss = ahf_training_loss(
                        model, images, masks, tissue_masks
                    )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.detach().item())
            train_batches += 1
            progress.set_postfix(
                loss=f"{float(loss.detach().item()):.4f}"
            )

        model.eval()
        validation_loss_sum = 0.0
        validation_batches = 0
        with torch.inference_mode():
            for images, masks, tissue_masks, *_ in validation_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(
                    device, non_blocking=True
                ).float()
                tissue_masks = tissue_masks.to(
                    device, non_blocking=True
                ).float()

                with autocast(
                    "cuda",
                    enabled=(device.type == "cuda"),
                ):
                    if args.architecture == "resnet50_unet":
                        logits = unwrap_logits(model(images))
                        loss = resnet_loss(
                            logits, masks, tissue_masks
                        )
                    else:
                        loss = ahf_training_loss(
                            model, images, masks, tissue_masks
                        )

                validation_loss_sum += float(loss.detach().item())
                validation_batches += 1

        average_validation_loss = validation_loss_sum / max(
            1, validation_batches
        )
        scheduler.step(average_validation_loss)

        print(
            f"Epoch {epoch:03d}: "
            f"train={train_loss_sum / max(1, train_batches):.6f}, "
            f"validation={average_validation_loss:.6f}"
        )

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    torch.cuda.synchronize(device)
    training_time_s = float(time.perf_counter() - start)
    memory = monitor.stop()

    if not checkpoint_path.is_file():
        raise RuntimeError("No checkpoint was saved during training.")

    result = {
        "Method": args.display_name,
        "Architecture": args.architecture,
        "Seed": SEED,
        "Training_Fold": TRAIN_FOLD,
        "Training_Slices": train_slices,
        "Epochs": int(args.epochs),
        "Batch_Size": int(args.batch_size),
        "Params": total_params,
        "Params_M": total_params / 1e6,
        "Trainable_Params": trainable_params,
        "Trainable_Params_M": trainable_params / 1e6,
        "Training_Time_s": training_time_s,
        "Training_Memory_MB": float(
            memory.peak_family_memory_mb
        ),
        "Best_Epoch": int(best_epoch),
        "Best_Validation_Loss": float(best_validation_loss),
        "Checkpoint": str(checkpoint_path),
    }
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def gaussian_window(size: int, sigma_factor: float = 0.125) -> torch.Tensor:
    sigma = max(1.0, float(size) * float(sigma_factor))
    coordinates = (
        np.arange(size, dtype=np.float32) - size // 2
    )
    one_dimensional = np.exp(
        -(coordinates ** 2) / (2.0 * sigma ** 2)
    )
    return torch.from_numpy(
        np.outer(one_dimensional, one_dimensional).astype(np.float32)
    )


def sliding_positions(length: int, patch: int, stride: int):
    positions = list(range(0, length - patch + 1, stride))
    if not positions:
        raise ValueError(
            f"Slide dimension {length} is smaller than patch size {patch}."
        )
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions


def whole_slide_predict(
    model: nn.Module,
    image_bgr: np.ndarray,
    tissue_mask: np.ndarray,
    device: torch.device,
    patch_size: int,
    overlap: float,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    height, width = image_bgr.shape[:2]
    stride = max(
        1, int(round(patch_size * (1.0 - overlap)))
    )
    y_positions = sliding_positions(height, patch_size, stride)
    x_positions = sliding_positions(width, patch_size, stride)

    transform = A.Compose(
        [
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )
    image_rgb = cv2.cvtColor(
        image_bgr, cv2.COLOR_BGR2RGB
    )
    blend = gaussian_window(patch_size).to(device)

    probability_sum = torch.zeros(
        (height, width),
        dtype=torch.float32,
        device=device,
    )
    weight_sum = torch.zeros_like(probability_sum)

    model.eval()
    with torch.inference_mode():
        for y in y_positions:
            for x in x_positions:
                patch = image_rgb[
                    y : y + patch_size,
                    x : x + patch_size,
                ]
                tensor = transform(image=patch)["image"]
                tensor = tensor.unsqueeze(0).to(
                    device, non_blocking=True
                )

                with autocast(
                    "cuda",
                    enabled=(device.type == "cuda"),
                ):
                    logits = unwrap_logits(model(tensor))
                    probability = torch.sigmoid(
                        logits
                    ).squeeze()

                probability_sum[
                    y : y + patch_size,
                    x : x + patch_size,
                ] += probability.float() * blend
                weight_sum[
                    y : y + patch_size,
                    x : x + patch_size,
                ] += blend

    probability = (
        probability_sum
        / torch.clamp(weight_sum, min=1e-8)
    ).float().cpu().numpy()
    probability *= tissue_mask.astype(np.float32)
    prediction = (
        probability > float(threshold)
    ).astype(np.uint8)
    return probability, prediction


def inference_worker(args) -> None:
    set_seed(SEED)
    baseline_dir = Path(args.baseline_dir).resolve()
    output_json = Path(args.worker_output).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    _, _, model_class = load_local_modules(
        args.architecture, baseline_dir
    )

    vihbc_dir = Path(args.vihbc_dir).resolve()
    image_path = vihbc_dir / "he_resized.png"
    tissue_path = vihbc_dir / "mask_resized.png"
    image_bgr = cv_imread(image_path)
    tissue_gray = cv_imread(
        tissue_path, cv2.IMREAD_GRAYSCALE
    )
    tissue_mask = (
        tissue_gray > 127
    ).astype(np.uint8)
    if tissue_mask.shape != image_bgr.shape[:2]:
        tissue_mask = cv2.resize(
            tissue_mask,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device(
        f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for unified GPU profiling.")

    model = build_model(
        model_class,
        args.architecture,
        pretrained=False,
    ).to(device)
    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    clear_cuda()

    monitor = ProcessTreeGPUMonitor(
        root_pid=os.getpid(),
        gpu_index=args.gpu_index,
        interval_s=0.05,
    ).start()

    for warmup_index in range(WARMUP_RUNS):
        print(
            f"Complete-slide warm-up "
            f"{warmup_index + 1}/{WARMUP_RUNS}"
        )
        warmup_probability, warmup_prediction = whole_slide_predict(
            model=model,
            image_bgr=image_bgr,
            tissue_mask=tissue_mask,
            device=device,
            patch_size=args.patch_size,
            overlap=args.overlap,
            threshold=args.threshold,
        )
        del warmup_probability, warmup_prediction
        torch.cuda.synchronize(device)

    inference_times = []
    final_probability = None
    final_prediction = None

    for repeat in range(INFERENCE_REPEATS):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        final_probability, final_prediction = whole_slide_predict(
            model=model,
            image_bgr=image_bgr,
            tissue_mask=tissue_mask,
            device=device,
            patch_size=args.patch_size,
            overlap=args.overlap,
            threshold=args.threshold,
        )
        torch.cuda.synchronize(device)
        elapsed = float(time.perf_counter() - start)
        inference_times.append(elapsed)
        print(
            f"Inference repeat {repeat + 1}/"
            f"{INFERENCE_REPEATS}: {elapsed:.4f} s"
        )

    memory = monitor.stop()

    probability_path = (
        output_json.parent
        / f"{args.architecture}_ViHBC_seed42_probability.npz"
    )
    mask_path = (
        output_json.parent
        / f"{args.architecture}_ViHBC_seed42_pred_mask.png"
    )
    np.savez_compressed(
        probability_path,
        probability=final_probability.astype(np.float32),
        seed=np.asarray(SEED, dtype=np.int32),
        fold=np.asarray("ViHBC"),
    )
    cv_imwrite(
        mask_path,
        final_prediction.astype(np.uint8) * 255,
    )

    result = {
        "Method": args.display_name,
        "Architecture": args.architecture,
        "Seed": SEED,
        "Inference_Slice": "ViHBC",
        "Params": total_params,
        "Params_M": total_params / 1e6,
        "Trainable_Params": trainable_params,
        "Trainable_Params_M": trainable_params / 1e6,
        "Inference_Time_s": float(
            np.mean(inference_times)
        ),
        "Inference_Time_Std_s": float(
            np.std(inference_times, ddof=0)
        ),
        "Inference_Times_s": inference_times,
        "Inference_Memory_MB": float(
            memory.peak_family_memory_mb
        ),
        "Inference_Warmup_Runs": WARMUP_RUNS,
        "Inference_Repeats": INFERENCE_REPEATS,
        "Patch_Size": int(args.patch_size),
        "Overlap": float(args.overlap),
        "Threshold": float(args.threshold),
        "Checkpoint": str(checkpoint_path),
        "Probability_Map": str(probability_path),
        "Prediction_Mask": str(mask_path),
    }
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_worker(script: Path, worker: str, args, worker_output: Path):
    command = [
        sys.executable,
        str(script),
        "--worker",
        worker,
        "--architecture",
        args.architecture,
        "--display_name",
        args.display_name,
        "--baseline_dir",
        str(Path(args.baseline_dir).resolve()),
        "--patch_dir",
        str(Path(args.patch_dir).resolve()),
        "--h5ad_base_dir",
        str(Path(args.h5ad_base_dir).resolve()),
        "--vihbc_dir",
        str(Path(args.vihbc_dir).resolve()),
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--worker_output",
        str(worker_output.resolve()),
        "--output_dir",
        str(Path(args.output_dir).resolve()),
        "--gpu_index",
        str(args.gpu_index),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--learning_rate",
        str(args.learning_rate),
        "--patch_size",
        str(args.patch_size),
        "--overlap",
        str(args.overlap),
        "--threshold",
        str(args.threshold),
    ]
    subprocess.run(command, check=True)


def controller(args) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()

    training_json = output_dir / "training_worker.json"
    inference_json = output_dir / "inference_worker.json"

    if not args.skip_training:
        if checkpoint_path.exists() and args.overwrite:
            checkpoint_path.unlink()
        run_worker(
            Path(__file__).resolve(),
            "train",
            args,
            training_json,
        )
    elif not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"--skip_training was used but checkpoint is missing: "
            f"{checkpoint_path}"
        )

    run_worker(
        Path(__file__).resolve(),
        "infer",
        args,
        inference_json,
    )

    training: Dict[str, Any] = {}
    if training_json.is_file():
        training = json.loads(
            training_json.read_text(encoding="utf-8")
        )
    inference = json.loads(
        inference_json.read_text(encoding="utf-8")
    )

    summary = dict(inference)
    summary.update(
        {
            "Training_Time_s": training.get(
                "Training_Time_s", float("nan")
            ),
            "Training_Memory_MB": training.get(
                "Training_Memory_MB", float("nan")
            ),
            "Best_Epoch": training.get("Best_Epoch"),
            "Best_Validation_Loss": training.get(
                "Best_Validation_Loss"
            ),
            "Training_Fold": TRAIN_FOLD,
            "Training_Time_Definition": (
                "Model initialization, optimizer construction, "
                "10-epoch fitting, validation, scheduler updates, "
                "and best-checkpoint saving from prepared STHBC "
                "patch datasets; held-out A1 test evaluation excluded"
            ),
            "Inference_Time_Definition": (
                "Mean of three complete ViHBC sliding-window "
                "predictions after one complete-slide warm-up; "
                "model loading, metrics, visualization, and file "
                "writing excluded"
            ),
            "GPU_Memory_Definition": (
                "Peak absolute NVIDIA memory used by the profiled "
                "process and descendants, sampled with nvidia-smi "
                "every 0.05 s"
            ),
            "Parameter_Definition": (
                "All instantiated neural-network parameters"
            ),
        }
    )

    summary_json = output_dir / (
        f"{args.architecture}_ViHBC_efficiency.json"
    )
    summary_csv = output_dir / (
        f"{args.architecture}_ViHBC_efficiency.csv"
    )
    summary_txt = output_dir / (
        f"{args.architecture}_ViHBC_efficiency.txt"
    )

    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with open(
        summary_csv, "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary.keys())
        )
        writer.writeheader()
        writer.writerow(summary)

    latex_row = (
        f"{args.display_name} & "
        f"{summary['Params_M']:.2f}M & "
        f"{summary['Training_Time_s']:.2f} & "
        f"{summary['Inference_Time_s']:.2f} & "
        f"{summary['Training_Memory_MB']:.2f} & "
        f"{summary['Inference_Memory_MB']:.2f} \\\\"
    )
    report = "\n".join(
        [
            "=" * 88,
            f"{args.display_name} unified ViHBC computational cost",
            "=" * 88,
            f"Params                  : {summary['Params_M']:.4f} M",
            f"Trainable Params        : {summary['Trainable_Params_M']:.4f} M",
            f"Training Time           : {summary['Training_Time_s']:.2f} s",
            (
                f"Inference Time          : "
                f"{summary['Inference_Time_s']:.2f} ± "
                f"{summary['Inference_Time_Std_s']:.2f} s"
            ),
            (
                f"Training GPU Memory     : "
                f"{summary['Training_Memory_MB']:.2f} MB"
            ),
            (
                f"Inference GPU Memory    : "
                f"{summary['Inference_Memory_MB']:.2f} MB"
            ),
            "",
            "LaTeX table row:",
            latex_row,
            "=" * 88,
        ]
    )
    summary_txt.write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    print(f"\nJSON: {summary_json}")
    print(f"CSV : {summary_csv}")
    print(f"TXT : {summary_txt}")


def build_parser(default_architecture: Optional[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=["resnet50_unet", "ahfunet"],
        default=default_architecture,
        required=(default_architecture is None),
    )
    parser.add_argument("--display_name", default=None)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument(
        "--patch_dir",
        default=r"D:\Project_2\Pixel_Seg\data\STHBC\patches(xy)",
    )
    parser.add_argument(
        "--h5ad_base_dir",
        default=r"D:\Project_2\Pixel_Seg\data\STHBC\ST",
    )
    parser.add_argument(
        "--vihbc_dir",
        default=r"D:\Project_2\Pixel_Seg\data\ViHBC\resized_data",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--worker",
        choices=["train", "infer"],
        default=None,
    )
    parser.add_argument("--worker_output", default=None)
    return parser


def main(default_architecture: Optional[str] = None) -> None:
    parser = build_parser(default_architecture)
    args = parser.parse_args()

    if args.display_name is None:
        args.display_name = (
            "ResNet50-UNet"
            if args.architecture == "resnet50_unet"
            else "AHFUNet"
        )

    baseline_dir = Path(args.baseline_dir).resolve()
    if args.output_dir is None:
        args.output_dir = str(
            baseline_dir / "efficiency_results_unified"
        )
    if args.checkpoint is None:
        checkpoint_name = (
            "best_model_A1_resnet50_seed42_unified.pth"
            if args.architecture == "resnet50_unet"
            else "best_model_A1_ahfunet_seed42_unified.pth"
        )
        args.checkpoint = str(
            Path(args.output_dir) / "checkpoints" / checkpoint_name
        )

    if args.worker:
        if not args.worker_output:
            raise ValueError("--worker_output is required in worker mode")
        if args.worker == "train":
            train_worker(args)
        else:
            inference_worker(args)
    else:
        controller(args)


if __name__ == "__main__":
    main()
