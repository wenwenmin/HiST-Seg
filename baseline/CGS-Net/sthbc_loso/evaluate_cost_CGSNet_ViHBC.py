from __future__ import annotations

"""Unified CGS-Net computational-cost profiler on ViHBC.

The supplied CGS-Net implementation trains three stages:
1. a detail single-stream model;
2. a context single-stream model;
3. the final dual-stream CGS model initialized from both single streams.

This profiler includes all three stages in Training Time and counts parameters
of the final dual-stream model.

For whole-slide inference, the original ``evaluate_whole_slide`` function also
computes metrics and writes result files. To retain the common efficiency
protocol without modifying ``cgs_sthbc_lib.py``, inference time is measured
from the beginning of the whole-slide call through the final top-level model
forward. The metric and file-writing tail after the final model forward is
therefore excluded.

Requirements
------------
Place this script and ``unified_gpu_monitor.py`` in the CGS-Net project, which
must also contain:
- cgs_sthbc_lib.py
- the supplied train_sthbc_loso.py
- a training YAML
- an external YAML loadable by cgs_sthbc_lib.load_all_slides that contains
  a slide with key ``ViHBC``.
"""

import argparse
import csv
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Set

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def _parse_memory_value(text: str) -> float:
    value = text.strip().replace("MiB", "").replace("MB", "").strip()
    if not value or value.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return 0.0
    return float(value)


def query_gpu_process_memory(gpu_index: int = 0) -> Dict[int, float]:
    """Return {pid: used_gpu_memory_MiB} for one NVIDIA GPU."""
    field_variants = ("used_gpu_memory", "used_memory")
    for field in field_variants:
        command = [
            "nvidia-smi",
            "-i",
            str(int(gpu_index)),
            f"--query-compute-apps=pid,{field}",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(
                command,
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            continue

        result: Dict[int, float] = {}
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                memory = _parse_memory_value(parts[1])
            except (TypeError, ValueError):
                continue
            result[pid] = result.get(pid, 0.0) + memory
        return result

    raise RuntimeError(
        "Unable to query NVIDIA process memory. Confirm that nvidia-smi is "
        "available and that the selected GPU supports compute-process queries."
    )


def process_family(root_pid: int) -> Set[int]:
    pids: Set[int] = {int(root_pid)}
    if psutil is None:
        return pids
    try:
        process = psutil.Process(int(root_pid))
        for child in process.children(recursive=True):
            pids.add(int(child.pid))
    except (psutil.Error, OSError):
        pass
    return pids


@dataclass
class GPUMemorySample:
    peak_family_memory_mb: float
    peak_root_memory_mb: float
    samples: int


class ProcessTreeGPUMonitor:
    """Poll absolute NVIDIA memory used by a process and its descendants.

    This avoids mixing torch.cuda.max_memory_allocated(), reserved-memory
    values, and whole-GPU deltas. Each profiled phase should run in a clean
    process for the most reproducible result.
    """

    def __init__(
        self,
        root_pid: Optional[int] = None,
        gpu_index: int = 0,
        interval_s: float = 0.05,
    ) -> None:
        self.root_pid = int(root_pid if root_pid is not None else os.getpid())
        self.gpu_index = int(gpu_index)
        self.interval_s = float(interval_s)
        self.peak_family_memory_mb = 0.0
        self.peak_root_memory_mb = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None

    def _sample_once(self) -> None:
        memory_by_pid = query_gpu_process_memory(self.gpu_index)
        family = process_family(self.root_pid)
        family_memory = sum(memory_by_pid.get(pid, 0.0) for pid in family)
        root_memory = memory_by_pid.get(self.root_pid, 0.0)
        self.peak_family_memory_mb = max(self.peak_family_memory_mb, family_memory)
        self.peak_root_memory_mb = max(self.peak_root_memory_mb, root_memory)
        self.samples += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except BaseException as error:  # store and report on stop
                self._error = error
                return
            self._stop.wait(self.interval_s)

    def start(self) -> "ProcessTreeGPUMonitor":
        if self._thread is not None:
            raise RuntimeError("GPU monitor has already been started.")
        # nvidia-smi may not show a newly launched process immediately.
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> GPUMemorySample:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        try:
            self._sample_once()
        except BaseException as error:
            if self._error is None:
                self._error = error
        if self._error is not None and self.samples == 0:
            raise RuntimeError(f"GPU memory monitoring failed: {self._error}")
        return GPUMemorySample(
            peak_family_memory_mb=float(self.peak_family_memory_mb),
            peak_root_memory_mb=float(self.peak_root_memory_mb),
            samples=int(self.samples),
        )

    def __enter__(self) -> "ProcessTreeGPUMonitor":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()



SEED = 42
DEFAULT_TRAIN_FOLD = "A1"
WARMUP_RUNS = 1
INFERENCE_REPEATS = 3


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
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


class ForwardBoundaryTimer:
    """Measure from the whole-slide call start through the last model forward."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.first_pre: Optional[float] = None
        self.last_post: Optional[float] = None
        self.call_start: Optional[float] = None
        self._pre_handle = None
        self._post_handle = None

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _pre_hook(self, module, inputs):
        self._sync()
        now = time.perf_counter()
        if self.first_pre is None:
            self.first_pre = now

    def _post_hook(self, module, inputs, output):
        self._sync()
        self.last_post = time.perf_counter()

    def start_call(self) -> None:
        self.first_pre = None
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
                "CGS-Net whole-slide evaluation did not invoke the "
                "top-level model forward."
            )
        return float(self.last_post - self.call_start)


def train_worker(args) -> None:
    project_dir = Path(args.project_dir).resolve()
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    train_script = Path(args.train_script).resolve()
    train_module = load_module(
        train_script, "cgs_train_profiled"
    )

    cfg = train_module.load_yaml(str(Path(args.train_config).resolve()))
    cfg["output_root"] = str(Path(args.training_output_root).resolve())
    cfg.setdefault("train", {})
    cfg["train"]["seed"] = SEED

    output_root = Path(cfg["output_root"])
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_module.set_seed(SEED)

    # Slide loading and dataset checking are input preparation, not fitting.
    slides = train_module.load_all_slides(cfg)
    train_slices, validation_slice, test_slice = (
        train_module.get_fold_lists(cfg, args.fold)
    )

    device = train_module.get_device(cfg)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for unified GPU profiling.")

    clear_cuda()
    monitor = ProcessTreeGPUMonitor(
        root_pid=os.getpid(),
        gpu_index=args.gpu_index,
        interval_s=0.05,
    ).start()
    start = time.perf_counter()

    cgs_checkpoint = train_module.train_fold(
        cfg,
        slides,
        args.fold,
        run_test=False,
    )

    torch.cuda.synchronize(device)
    training_time_s = float(time.perf_counter() - start)
    memory = monitor.stop()

    fold_dir = output_root / f"fold_{args.fold}"
    detail_checkpoint = (
        fold_dir / "single_detail" / "single_detail_best.pth"
    )
    context_checkpoint = (
        fold_dir / "single_context" / "single_context_best.pth"
    )
    cgs_checkpoint = Path(cgs_checkpoint)

    for checkpoint in (
        detail_checkpoint,
        context_checkpoint,
        cgs_checkpoint,
    ):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    model = train_module.make_dual_model_from_singles(
        cfg,
        detail_checkpoint,
        context_checkpoint,
        freeze_base=False,
    ).to(device)
    model.load_state_dict(
        torch.load(
            cgs_checkpoint,
            map_location=device,
            weights_only=True,
        ),
        strict=True,
    )
    total_params, trainable_params = count_parameters(model)

    result = {
        "Method": "CGS-Net",
        "Seed": SEED,
        "Training_Fold": args.fold,
        "Training_Slices": train_slices,
        "Validation_Slice": validation_slice,
        "Held_Out_STHBC_Slice": test_slice,
        "Params": total_params,
        "Params_M": total_params / 1e6,
        "Trainable_Params": trainable_params,
        "Trainable_Params_M": trainable_params / 1e6,
        "Training_Time_s": training_time_s,
        "Training_Memory_MB": float(
            memory.peak_family_memory_mb
        ),
        "Detail_Checkpoint": str(detail_checkpoint),
        "Context_Checkpoint": str(context_checkpoint),
        "CGS_Checkpoint": str(cgs_checkpoint),
        "Training_Output_Root": str(output_root),
    }
    Path(args.worker_output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def inference_worker(args) -> None:
    project_dir = Path(args.project_dir).resolve()
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    import cgs_sthbc_lib as lib

    train_cfg = lib.load_yaml(str(Path(args.train_config).resolve()))
    train_cfg["output_root"] = str(
        Path(args.training_output_root).resolve()
    )
    train_cfg.setdefault("train", {})
    train_cfg["train"]["seed"] = SEED

    external_cfg = lib.load_yaml(
        str(Path(args.external_config).resolve())
    )
    external_slides = lib.load_all_slides(external_cfg)
    if args.vihbc_key not in external_slides:
        raise KeyError(
            f"External CGS config does not expose slide "
            f"{args.vihbc_key!r}. Available keys: "
            f"{list(external_slides.keys())}"
        )
    vihbc_slide = external_slides[args.vihbc_key]

    fold_dir = (
        Path(args.training_output_root).resolve()
        / f"fold_{args.fold}"
    )
    detail_checkpoint = (
        fold_dir / "single_detail" / "single_detail_best.pth"
    )
    context_checkpoint = (
        fold_dir / "single_context" / "single_context_best.pth"
    )
    cgs_checkpoint = fold_dir / "cgs" / "cgs_best.pth"

    for checkpoint in (
        detail_checkpoint,
        context_checkpoint,
        cgs_checkpoint,
    ):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    device = lib.get_device(train_cfg)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for unified GPU profiling.")

    model = lib.make_dual_model_from_singles(
        train_cfg,
        detail_checkpoint,
        context_checkpoint,
        freeze_base=False,
    ).to(device)
    model.load_state_dict(
        torch.load(
            cgs_checkpoint,
            map_location=device,
            weights_only=True,
        ),
        strict=True,
    )
    model.eval()
    total_params, trainable_params = count_parameters(model)

    clear_cuda()
    monitor = ProcessTreeGPUMonitor(
        root_pid=os.getpid(),
        gpu_index=args.gpu_index,
        interval_s=0.05,
    ).start()

    temporary_root = Path(
        tempfile.mkdtemp(prefix="cgs_vihbc_profile_")
    )

    try:
        for warmup_index in range(WARMUP_RUNS):
            timer = ForwardBoundaryTimer(model, device)
            timer.start_call()
            lib.evaluate_whole_slide(
                model,
                vihbc_slide,
                external_cfg,
                device,
                mode="dual",
                save_dir=temporary_root / f"warmup_{warmup_index}",
            )
            _ = timer.finish_call()
            print(
                f"Complete-slide warm-up "
                f"{warmup_index + 1}/{WARMUP_RUNS} finished"
            )

        inference_times = []
        for repeat in range(INFERENCE_REPEATS):
            timer = ForwardBoundaryTimer(model, device)
            timer.start_call()
            lib.evaluate_whole_slide(
                model,
                vihbc_slide,
                external_cfg,
                device,
                mode="dual",
                save_dir=temporary_root / f"repeat_{repeat}",
            )
            elapsed = timer.finish_call()
            inference_times.append(elapsed)
            print(
                f"Inference repeat {repeat + 1}/"
                f"{INFERENCE_REPEATS}: {elapsed:.4f} s"
            )
    finally:
        memory = monitor.stop()
        shutil.rmtree(temporary_root, ignore_errors=True)

    result = {
        "Method": "CGS-Net",
        "Seed": SEED,
        "Inference_Slice": args.vihbc_key,
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
        "Inference_Timing_Boundary": (
            "Start of evaluate_whole_slide call through the final "
            "top-level model forward; metric and file-writing tail "
            "after the final forward excluded"
        ),
        "Detail_Checkpoint": str(detail_checkpoint),
        "Context_Checkpoint": str(context_checkpoint),
        "CGS_Checkpoint": str(cgs_checkpoint),
    }
    Path(args.worker_output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_worker(script: Path, worker: str, args, output_json: Path):
    command = [
        sys.executable,
        str(script),
        "--worker",
        worker,
        "--project_dir",
        str(Path(args.project_dir).resolve()),
        "--train_script",
        str(Path(args.train_script).resolve()),
        "--train_config",
        str(Path(args.train_config).resolve()),
        "--external_config",
        str(Path(args.external_config).resolve()),
        "--training_output_root",
        str(Path(args.training_output_root).resolve()),
        "--output_dir",
        str(Path(args.output_dir).resolve()),
        "--fold",
        args.fold,
        "--vihbc_key",
        args.vihbc_key,
        "--gpu_index",
        str(args.gpu_index),
        "--worker_output",
        str(output_json.resolve()),
    ]
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)


def controller(args) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_json = output_dir / "training_worker.json"
    inference_json = output_dir / "inference_worker.json"

    if not args.skip_training:
        run_worker(
            Path(__file__).resolve(),
            "train",
            args,
            training_json,
        )

    fold_dir = (
        Path(args.training_output_root).resolve()
        / f"fold_{args.fold}"
    )
    required = [
        fold_dir / "single_detail" / "single_detail_best.pth",
        fold_dir / "single_context" / "single_context_best.pth",
        fold_dir / "cgs" / "cgs_best.pth",
    ]
    for checkpoint in required:
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Required CGS checkpoint is missing: {checkpoint}"
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
            "Training_Fold": args.fold,
            "Training_Time_Definition": (
                "All three original CGS-Net stages: detail single "
                "model, context single model, and final dual CGS "
                "model; validation, early stopping, and best "
                "checkpoint saving included; held-out test inference "
                "excluded"
            ),
            "Inference_Time_Definition": (
                "Mean of three ViHBC whole-slide calls after one "
                "warm-up, measured through the final model forward; "
                "metric computation and result writing after the "
                "final forward excluded"
            ),
            "GPU_Memory_Definition": (
                "Peak absolute NVIDIA memory used by the profiled "
                "process and descendants, sampled with nvidia-smi "
                "every 0.05 s"
            ),
            "Parameter_Definition": (
                "All parameters of the final dual-stream CGS-Net; "
                "temporary single-stream pretraining models are not "
                "added again"
            ),
        }
    )

    summary_json = output_dir / "cgsnet_ViHBC_efficiency.json"
    summary_csv = output_dir / "cgsnet_ViHBC_efficiency.csv"
    summary_txt = output_dir / "cgsnet_ViHBC_efficiency.txt"

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
        f"CGS-Net & {summary['Params_M']:.2f}M & "
        f"{summary['Training_Time_s']:.2f} & "
        f"{summary['Inference_Time_s']:.2f} & "
        f"{summary['Training_Memory_MB']:.2f} & "
        f"{summary['Inference_Memory_MB']:.2f} \\\\"
    )
    report = "\n".join(
        [
            "=" * 88,
            "CGS-Net unified ViHBC computational cost",
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir",
        default=str(Path(__file__).resolve().parent),
        help="CGS-Net folder. Default: folder containing this script.",
    )
    parser.add_argument(
        "--train_script",
        default=str(Path(__file__).resolve().parent / "train_sthbc_loso.py"),
        help="CGS-Net training script.",
    )
    parser.add_argument(
        "--train_config",
        default=str(Path(__file__).resolve().parent / "sthbc_loso.yaml"),
        help="STHBC training YAML.",
    )
    parser.add_argument(
        "--external_config",
        default=str(Path(__file__).resolve().parent / "vihbc.yaml"),
        help=(
            "ViHBC YAML readable by cgs_sthbc_lib.load_all_slides. "
            "The loaded slide dictionary must contain key ViHBC."
        ),
    )
    parser.add_argument(
        "--training_output_root",
        default=str(Path(__file__).resolve().parent / "efficiency_training_seed42"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parent / "efficiency_results"),
    )
    parser.add_argument("--fold", default=DEFAULT_TRAIN_FOLD)
    parser.add_argument("--vihbc_key", default="ViHBC")
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--worker",
        choices=["train", "infer"],
        default=None,
    )
    parser.add_argument("--worker_output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
