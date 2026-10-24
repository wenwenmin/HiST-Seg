# Path configuration:
# Replace the repository-relative defaults (data/, outputs/, and workspaces/)
# for your environment. See readme_baseline.txt at the baseline package root.
from __future__ import annotations

"""
Profile CGS-Net computational cost on the STHBC A1 LOSO fold.

Protocol
--------
Training:
- seed: 42;
- held-out test slice: A1;
- validation slice: cfg["fold_val_map"]["A1"];
- training slices: all remaining STHBC slices;
- includes the original three CGS-Net stages:
    1) detail single-stream training;
    2) context single-stream training;
    3) dual-stream CGS training;
- includes validation, early stopping and best-checkpoint writing;
- excludes the final held-out A1 test inference.

Inference:
- uses the final fold_A1 checkpoints;
- performs whole-slide inference on the STHBC A1 section;
- one complete-slide warm-up;
- three complete-slide measured runs;
- reported time starts before evaluate_whole_slide and ends immediately after
  the final top-level model forward, excluding the metric/file-writing tail.

Parameters:
- trainable parameters of the final dual-stream CGS-Net.

GPU memory:
- peak PyTorch CUDA reserved memory during each phase;
- peak allocated memory is also saved for auditing;
- this avoids unstable high-frequency nvidia-smi polling on Windows.

Required project files
----------------------
Place this evaluator in the CGS-Net project folder containing:
- cgs_sthbc_lib.py
- train_sthbc_loso.py
- sthbc_loso.yaml
"""

import argparse
import csv
import gc
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch


METHOD = "CGS-Net"
SEED = 42
DEFAULT_FOLD = "A1"
WARMUP_RUNS = 1
INFERENCE_REPEATS = 3
SCRIPT_VERSION = "2026-07-25-STHBC-A1-v1"


# =============================================================================
# General utilities
# =============================================================================
def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_existing_file(
    explicit: Optional[str],
    *,
    script_dir: Path,
    candidates: Sequence[str],
    description: str,
) -> Path:
    checked = []

    if explicit:
        raw = Path(explicit).expanduser()
        paths = [raw] if raw.is_absolute() else [
            Path.cwd() / raw,
            script_dir / raw,
        ]
        for path in paths:
            path = path.resolve()
            checked.append(path)
            if path.is_file():
                return path

    for name in candidates:
        path = (script_dir / name).resolve()
        checked.append(path)
        if path.is_file():
            return path

    for name in candidates:
        matches = sorted(
            (p.resolve() for p in script_dir.rglob(name) if p.is_file()),
            key=lambda p: (len(p.parts), str(p).lower()),
        )
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Cannot find {description}. Checked:\n  "
        + "\n  ".join(str(path) for path in checked)
    )


def clear_cuda(device: Optional[torch.device] = None) -> None:
    gc.collect()
    if torch.cuda.is_available():
        if device is not None:
            torch.cuda.synchronize(device)
        else:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )
    return total, trainable


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def require_cuda_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError(
            "CUDA is required for CGS-Net GPU memory profiling. "
            f"The configuration selected device={device}."
        )


def reset_torch_peak_memory(device: torch.device) -> None:
    require_cuda_device(device)
    clear_cuda(device)
    torch.cuda.reset_peak_memory_stats(device)


def read_torch_peak_memory(device: torch.device) -> Dict[str, float]:
    require_cuda_device(device)
    torch.cuda.synchronize(device)
    allocated = float(
        torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    )
    reserved = float(
        torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
    )
    if reserved <= 0.0:
        raise RuntimeError(
            "PyTorch reported 0 MB peak reserved GPU memory despite CUDA "
            "execution. Check the CUDA device and PyTorch installation."
        )
    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
    }


# =============================================================================
# Inference timing boundary
# =============================================================================
class ForwardBoundaryTimer:
    """Measure call start through the final top-level model forward."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.call_start: Optional[float] = None
        self.last_post: Optional[float] = None
        self._pre_handle = None
        self._post_handle = None

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _pre_hook(self, module, inputs) -> None:
        self._sync()

    def _post_hook(self, module, inputs, output) -> None:
        self._sync()
        self.last_post = time.perf_counter()

    def start_call(self) -> None:
        self.last_post = None
        self._sync()
        self.call_start = time.perf_counter()
        self._pre_handle = self.model.register_forward_pre_hook(
            self._pre_hook
        )
        self._post_handle = self.model.register_forward_hook(
            self._post_hook
        )

    def finish_call(self) -> float:
        self._sync()

        if self._pre_handle is not None:
            self._pre_handle.remove()
        if self._post_handle is not None:
            self._post_handle.remove()
        self._pre_handle = None
        self._post_handle = None

        if self.call_start is None or self.last_post is None:
            raise RuntimeError(
                "evaluate_whole_slide did not invoke the top-level CGS-Net "
                "model forward."
            )
        return float(self.last_post - self.call_start)


# =============================================================================
# Checkpoint paths
# =============================================================================
def checkpoint_paths(training_output_root: Path, fold: str) -> Dict[str, Path]:
    fold_dir = training_output_root / f"fold_{fold}"
    return {
        "fold_dir": fold_dir,
        "detail": (
            fold_dir / "single_detail" / "single_detail_best.pth"
        ),
        "context": (
            fold_dir / "single_context" / "single_context_best.pth"
        ),
        "cgs": fold_dir / "cgs" / "cgs_best.pth",
    }


def require_checkpoints(paths: Dict[str, Path]) -> None:
    for key in ("detail", "context", "cgs"):
        path = paths[key]
        if not path.is_file():
            raise FileNotFoundError(
                f"Required {key} checkpoint is missing: {path}"
            )


# =============================================================================
# Training worker
# =============================================================================
def train_worker(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    train_script = Path(args.train_script).resolve()
    train_module = load_module(
        train_script,
        "cgs_train_sthbc_a1_profiled",
    )

    config_path = Path(args.train_config).resolve()
    output_root = Path(args.training_output_root).resolve()
    output_json = Path(args.worker_output).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    cfg = train_module.load_yaml(str(config_path))
    cfg["output_root"] = str(output_root)
    cfg.setdefault("train", {})
    cfg["train"]["seed"] = SEED

    if args.fold not in list(cfg["slices"]):
        raise ValueError(
            f"Fold {args.fold!r} is not present in cfg['slices']: "
            f"{list(cfg['slices'])}"
        )

    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"Training output already exists: {output_root}\n"
                "Use --overwrite for a clean full three-stage profile, or "
                "--skip_training to reuse existing checkpoints."
            )
    output_root.mkdir(parents=True, exist_ok=True)

    train_module.set_seed(SEED)

    # Dataset loading is excluded from the fitting timer.
    slides = train_module.load_all_slides(cfg)
    train_slices, validation_slice, test_slice = (
        train_module.get_fold_lists(cfg, args.fold)
    )

    if test_slice != args.fold:
        raise RuntimeError(
            f"Unexpected fold resolution: requested={args.fold}, "
            f"resolved test={test_slice}"
        )

    device = train_module.get_device(cfg)
    require_cuda_device(device)

    reset_torch_peak_memory(device)
    start = time.perf_counter()

    # run_test=False prevents A1 inference from entering Training Time.
    returned_checkpoint = train_module.train_fold(
        cfg,
        slides,
        args.fold,
        run_test=False,
    )

    torch.cuda.synchronize(device)
    training_time_s = float(time.perf_counter() - start)
    peak = read_torch_peak_memory(device)

    paths = checkpoint_paths(output_root, args.fold)
    require_checkpoints(paths)

    returned_checkpoint = Path(returned_checkpoint).resolve()
    if returned_checkpoint != paths["cgs"].resolve():
        print(
            "[Warning] train_fold returned a checkpoint path different from "
            f"the conventional path:\n  returned={returned_checkpoint}\n"
            f"  expected={paths['cgs']}"
        )

    # Build the final model only for parameter auditing, after the measured
    # training phase has finished.
    model = train_module.make_dual_model_from_singles(
        cfg,
        paths["detail"],
        paths["context"],
        freeze_base=False,
    ).to(device)
    model.load_state_dict(
        safe_torch_load(paths["cgs"], device),
        strict=True,
    )
    total_params, trainable_params = count_parameters(model)

    result = {
        "Method": METHOD,
        "Seed": SEED,
        "Dataset": "STHBC",
        "Training_Fold": args.fold,
        "Training_Slices": list(train_slices),
        "Validation_Slice": validation_slice,
        "Held_Out_Test_Slice": test_slice,
        "Params": trainable_params,
        "Params_M": trainable_params / 1e6,
        "Trainable_Params": trainable_params,
        "Trainable_Params_M": trainable_params / 1e6,
        "Total_Params": total_params,
        "Total_Params_M": total_params / 1e6,
        "Training_Time_s": training_time_s,
        "Training_Memory_MB": peak["reserved_mb"],
        "Training_Peak_Allocated_MB": peak["allocated_mb"],
        "Training_Peak_Reserved_MB": peak["reserved_mb"],
        "Detail_Checkpoint": str(paths["detail"]),
        "Context_Checkpoint": str(paths["context"]),
        "CGS_Checkpoint": str(paths["cgs"]),
        "Training_Output_Root": str(output_root),
    }
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# A1 inference worker
# =============================================================================
def inference_worker(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    import cgs_sthbc_lib as lib

    config_path = Path(args.train_config).resolve()
    output_root = Path(args.training_output_root).resolve()
    output_json = Path(args.worker_output).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    cfg = lib.load_yaml(str(config_path))
    cfg["output_root"] = str(output_root)
    cfg.setdefault("train", {})
    cfg["train"]["seed"] = SEED

    slides = lib.load_all_slides(cfg)
    if args.fold not in slides:
        raise KeyError(
            f"STHBC slide {args.fold!r} is unavailable. "
            f"Available slide keys: {list(slides.keys())}"
        )
    test_slide = slides[args.fold]

    train_slices, validation_slice, test_slice = (
        _get_fold_lists_from_cfg(cfg, args.fold)
    )

    paths = checkpoint_paths(output_root, args.fold)
    require_checkpoints(paths)

    device = lib.get_device(cfg)
    require_cuda_device(device)

    model = lib.make_dual_model_from_singles(
        cfg,
        paths["detail"],
        paths["context"],
        freeze_base=False,
    ).to(device)
    model.load_state_dict(
        safe_torch_load(paths["cgs"], device),
        strict=True,
    )
    model.eval()

    total_params, trainable_params = count_parameters(model)

    # Reset after model loading. PyTorch initializes the new peak to the
    # current allocator state, so model-resident memory remains represented.
    reset_torch_peak_memory(device)

    temporary_root = Path(
        tempfile.mkdtemp(prefix=f"cgs_sthbc_{args.fold}_profile_")
    )

    inference_times = []
    try:
        for warmup_index in range(WARMUP_RUNS):
            timer = ForwardBoundaryTimer(model, device)
            timer.start_call()
            lib.evaluate_whole_slide(
                model,
                test_slide,
                cfg,
                device,
                mode="dual",
                save_dir=(
                    temporary_root / f"warmup_{warmup_index + 1}"
                ),
            )
            _ = timer.finish_call()
            print(
                f"Complete-slide warm-up "
                f"{warmup_index + 1}/{WARMUP_RUNS} finished"
            )

        # Warm-up allocations are intentionally retained because the reported
        # peak describes the complete inference worker after model loading.
        for repeat in range(INFERENCE_REPEATS):
            timer = ForwardBoundaryTimer(model, device)
            timer.start_call()
            lib.evaluate_whole_slide(
                model,
                test_slide,
                cfg,
                device,
                mode="dual",
                save_dir=(
                    temporary_root / f"repeat_{repeat + 1}"
                ),
            )
            elapsed = timer.finish_call()
            inference_times.append(elapsed)
            print(
                f"Inference repeat {repeat + 1}/{INFERENCE_REPEATS}: "
                f"{elapsed:.4f} s"
            )

        peak = read_torch_peak_memory(device)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    time_array = np.asarray(inference_times, dtype=np.float64)

    result = {
        "Method": METHOD,
        "Seed": SEED,
        "Dataset": "STHBC",
        "Inference_Slice": test_slice,
        "Training_Slices": list(train_slices),
        "Validation_Slice": validation_slice,
        "Params": trainable_params,
        "Params_M": trainable_params / 1e6,
        "Trainable_Params": trainable_params,
        "Trainable_Params_M": trainable_params / 1e6,
        "Total_Params": total_params,
        "Total_Params_M": total_params / 1e6,
        "Inference_Time_s": float(time_array.mean()),
        "Inference_Time_Std_s": float(time_array.std(ddof=0)),
        "Inference_Times_s": [
            float(value) for value in time_array
        ],
        "Inference_Memory_MB": peak["reserved_mb"],
        "Inference_Peak_Allocated_MB": peak["allocated_mb"],
        "Inference_Peak_Reserved_MB": peak["reserved_mb"],
        "Inference_Warmup_Runs": WARMUP_RUNS,
        "Inference_Repeats": INFERENCE_REPEATS,
        "Inference_Timing_Boundary": (
            "Start of the STHBC A1 evaluate_whole_slide call through "
            "the final top-level CGS-Net forward. Metric computation and "
            "result writing after the final forward are excluded."
        ),
        "Detail_Checkpoint": str(paths["detail"]),
        "Context_Checkpoint": str(paths["context"]),
        "CGS_Checkpoint": str(paths["cgs"]),
    }
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_fold_lists_from_cfg(
    cfg: dict,
    test_slice: str,
) -> Tuple[list, str, str]:
    slices = list(cfg["slices"])
    validation_slice = cfg["fold_val_map"][test_slice]
    training_slices = [
        name
        for name in slices
        if name not in {test_slice, validation_slice}
    ]
    return training_slices, validation_slice, test_slice


# =============================================================================
# Controller
# =============================================================================
def run_worker(
    script: Path,
    worker_name: str,
    args: argparse.Namespace,
    output_json: Path,
) -> None:
    command = [
        sys.executable,
        str(script),
        "--worker",
        worker_name,
        "--project_dir",
        str(Path(args.project_dir).resolve()),
        "--train_script",
        str(Path(args.train_script).resolve()),
        "--train_config",
        str(Path(args.train_config).resolve()),
        "--training_output_root",
        str(Path(args.training_output_root).resolve()),
        "--output_dir",
        str(Path(args.output_dir).resolve()),
        "--fold",
        str(args.fold),
        "--gpu_index",
        str(int(args.gpu_index)),
        "--worker_output",
        str(output_json.resolve()),
    ]
    if args.overwrite:
        command.append("--overwrite")

    print("\n" + "=" * 100)
    print(f"Launching {worker_name} worker")
    print(" ".join(command))
    print("=" * 100)
    subprocess.run(command, check=True)


def controller(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_json = output_dir / "training_worker_A1.json"
    inference_json = output_dir / "inference_worker_A1.json"

    if not args.skip_training:
        run_worker(
            Path(__file__).resolve(),
            "train",
            args,
            training_json,
        )
    else:
        print(
            "[Skip] Training is skipped. Existing fold_A1 checkpoints "
            "will be used; training time and memory will be NaN."
        )

    paths = checkpoint_paths(
        Path(args.training_output_root).resolve(),
        args.fold,
    )
    require_checkpoints(paths)

    run_worker(
        Path(__file__).resolve(),
        "infer",
        args,
        inference_json,
    )

    if not inference_json.is_file():
        raise FileNotFoundError(
            f"Inference worker output not found: {inference_json}"
        )
    inference = json.loads(
        inference_json.read_text(encoding="utf-8")
    )

    if not args.skip_training:
        if not training_json.is_file():
            raise FileNotFoundError(
                f"Training worker output not found: {training_json}"
            )
        training = json.loads(
            training_json.read_text(encoding="utf-8")
        )
        training_time = float(training["Training_Time_s"])
        training_memory = float(training["Training_Memory_MB"])
        training_allocated = float(
            training["Training_Peak_Allocated_MB"]
        )
    else:
        training = {}
        training_time = float("nan")
        training_memory = float("nan")
        training_allocated = float("nan")

    summary: Dict[str, Any] = dict(inference)
    summary.update(
        {
            "Training_Time_s": training_time,
            "Training_Memory_MB": training_memory,
            "Training_Peak_Allocated_MB": training_allocated,
            "Training_Fold": args.fold,
            "Parameter_Table_Field": "Trainable Params",
            "Training_Time_Definition": (
                "Original CGS-Net detail single-stream, context "
                "single-stream and dual-stream CGS stages for the STHBC "
                "A1 LOSO fold. Validation, early stopping and best-checkpoint "
                "writing are included; A1 test inference is excluded."
            ),
            "Inference_Time_Definition": (
                "Mean of three whole-slide STHBC A1 inference runs after "
                "one complete-slide warm-up. Timing ends after the final "
                "top-level model forward, excluding the metric and "
                "file-writing tail."
            ),
            "GPU_Memory_Definition": (
                "Peak PyTorch CUDA reserved memory during the corresponding "
                "phase. Peak allocated memory is also saved for auditing."
            ),
            "Parameter_Definition": (
                "Parameters with requires_grad=True in the final dual-stream "
                "CGS-Net. Temporary single-stream pretraining models are not "
                "added to the parameter field."
            ),
            "Script_Version": SCRIPT_VERSION,
        }
    )

    summary_json = output_dir / "CGSNet_STHBC_A1_costs.json"
    summary_csv = output_dir / "CGSNet_STHBC_A1_costs.csv"
    summary_txt = output_dir / "CGSNet_STHBC_A1_costs.txt"

    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary.keys()),
        )
        writer.writeheader()
        writer.writerow(summary)

    latex_row = (
        f"CGS-Net & {summary['Trainable_Params_M']:.2f}M & "
        f"{summary['Training_Time_s']:.2f} & "
        f"{summary['Inference_Time_s']:.2f} & "
        f"{summary['Training_Memory_MB']:.2f} & "
        f"{summary['Inference_Memory_MB']:.2f} \\\\"
    )

    report = "\n".join(
        [
            "=" * 96,
            "CGS-Net computational cost on STHBC A1",
            "=" * 96,
            f"Seed / Fold             : {SEED} / {args.fold}",
            f"Training slices         : {summary['Training_Slices']}",
            f"Validation slice        : {summary['Validation_Slice']}",
            f"Inference slice         : {summary['Inference_Slice']}",
            "",
            f"Trainable Params        : {summary['Trainable_Params_M']:.4f} M",
            f"Total Params            : {summary['Total_Params_M']:.4f} M",
            f"Training Time           : {summary['Training_Time_s']:.2f} s",
            (
                f"Inference Time          : "
                f"{summary['Inference_Time_s']:.2f} ± "
                f"{summary['Inference_Time_Std_s']:.2f} s"
            ),
            (
                f"Training GPU Memory     : "
                f"{summary['Training_Memory_MB']:.2f} MB reserved "
                f"(allocated peak: "
                f"{summary['Training_Peak_Allocated_MB']:.2f} MB)"
            ),
            (
                f"Inference GPU Memory    : "
                f"{summary['Inference_Memory_MB']:.2f} MB reserved "
                f"(allocated peak: "
                f"{summary['Inference_Peak_Allocated_MB']:.2f} MB)"
            ),
            "",
            "LaTeX table row:",
            latex_row,
            "=" * 96,
        ]
    )
    summary_txt.write_text(report + "\n", encoding="utf-8")

    print("\n" + report)
    print(f"\nJSON: {summary_json}")
    print(f"CSV : {summary_csv}")
    print(f"TXT : {summary_txt}")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Profile CGS-Net training and whole-slide inference cost on "
            "the STHBC A1 LOSO fold."
        )
    )
    parser.add_argument(
        "--project_dir",
        default=str(script_dir),
        help="CGS-Net project folder.",
    )
    parser.add_argument(
        "--train_script",
        default=None,
        help="Path to train_sthbc_loso.py.",
    )
    parser.add_argument(
        "--train_config",
        default=None,
        help="Path to sthbc_loso.yaml.",
    )
    parser.add_argument(
        "--training_output_root",
        default=str(
            script_dir / "efficiency_training_STHBC_A1_seed42"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(
            script_dir / "efficiency_results_STHBC_A1"
        ),
    )
    parser.add_argument(
        "--fold",
        default=DEFAULT_FOLD,
        help="Held-out STHBC test slice. Default and intended value: A1.",
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help=(
            "Kept for output metadata. Actual CUDA device follows the "
            "STHBC YAML configuration."
        ),
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Reuse existing fold_A1 checkpoints and profile inference only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the profiling training output and retrain all stages.",
    )
    parser.add_argument(
        "--worker",
        choices=["train", "infer"],
        default=None,
    )
    parser.add_argument("--worker_output", default=None)

    args = parser.parse_args()

    args.project_dir = str(
        Path(args.project_dir).expanduser().resolve()
    )
    project_dir = Path(args.project_dir)

    args.train_script = str(
        resolve_existing_file(
            args.train_script,
            script_dir=project_dir,
            candidates=[
                "train_sthbc_loso.py",
                "train_sthbc_loso(9).py",
                "train_sthbc_loso(8).py",
                "train_sthbc_loso(7).py",
            ],
            description="CGS-Net STHBC LOSO training script",
        )
    )
    args.train_config = str(
        resolve_existing_file(
            args.train_config,
            script_dir=project_dir,
            candidates=[
                "sthbc_loso.yaml",
                "sthbc_loso/sthbc_loso.yaml",
            ],
            description="CGS-Net STHBC LOSO YAML configuration",
        )
    )

    if args.fold != "A1":
        print(
            f"[Warning] This evaluator was requested for A1, but --fold "
            f"was set to {args.fold!r}."
        )

    return args


def main() -> None:
    args = parse_args()

    print(f"Evaluator version: {SCRIPT_VERSION}")
    print(f"Project dir       : {args.project_dir}")
    print(f"Train script      : {args.train_script}")
    print(f"Train config      : {args.train_config}")
    print(f"Fold / seed       : {args.fold} / {SEED}")

    if args.worker:
        if not args.worker_output:
            raise ValueError(
                "--worker_output is required in worker mode."
            )
        if args.worker == "train":
            train_worker(args)
        else:
            inference_worker(args)
    else:
        controller(args)


if __name__ == "__main__":
    main()
