# Path configuration:
# Replace the repository-relative defaults (data/, outputs/, and workspaces/)
# for your environment. See readme_baseline.txt at the baseline package root.
from __future__ import annotations

'Uniform computational cost assessment of TESLA on ViHBC.\n\nPlace this file directly into the TESLA project root directory and run it. The project root directory should contain:\n TESLA_package/\n\nJust check DATA_ROOT below. Script fixed:\n slide = ViHBC\n seed = 42\n one warm-up + three measured inference runs\n\nOutput:\n Params_M\n Training_Time_s\n Inference_Time_s\n Training_Memory_MB\n Inference_Memory_MB'

import csv
import gc
import json
import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import cv2
import numpy as np
import scanpy as sc
import torch

try:
    import psutil
except ImportError:
    psutil = None


# =============================================================================

# =============================================================================

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(r"data")  # Configure this path for your environment; see readme_baseline.txt.
TESLA_PACKAGE_DIR = HERE / "TESLA_package"

SLIDE = "ViHBC"
SEED = 42
GPU_INDEX = 0

RESIZE_FACTOR = 0.2
TESLA_RES = 50
MARKER_GENES = ["ERBB2", "GATA3", "PIP", "SCGB2A2", "KRT5", "KRT7", "KRT14", "KRT18"]



USE_CUDA = True

WARMUP_RUNS = 1
INFERENCE_REPEATS = 3
OUTPUT_DIR = HERE / "efficiency_profile_output"


# =============================================================================

# =============================================================================

def _parse_memory(text: str) -> float:
    value = text.strip().replace("MiB", "").replace("MB", "").strip()
    if not value or value.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return 0.0
    return float(value)


def query_gpu_process_memory(gpu_index: int) -> Dict[int, float]:
    for field in ("used_gpu_memory", "used_memory"):
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu_index),
                    f"--query-compute-apps=pid,{field}",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            continue

        result: Dict[int, float] = {}
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                memory = _parse_memory(parts[1])
            except Exception:
                continue
            result[pid] = result.get(pid, 0.0) + memory
        return result
    raise RuntimeError('Unable to query GPU process memory via nvidia-smi.')


def process_family(root_pid: int) -> Set[int]:
    pids = {int(root_pid)}
    if psutil is None:
        return pids
    try:
        process = psutil.Process(int(root_pid))
        pids.update(int(child.pid) for child in process.children(recursive=True))
    except Exception:
        pass
    return pids


@dataclass
class MemoryResult:
    peak_mb: float
    samples: int


class GPUMonitor:
    def __init__(self, gpu_index: int, interval: float = 0.05):
        self.gpu_index = int(gpu_index)
        self.interval = float(interval)
        self.root_pid = os.getpid()
        self.peak_mb = 0.0
        self.samples = 0
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def _sample(self):
        values = query_gpu_process_memory(self.gpu_index)
        family = process_family(self.root_pid)
        current = sum(values.get(pid, 0.0) for pid in family)
        self.peak_mb = max(self.peak_mb, current)
        self.samples += 1

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self._sample()
            except Exception:
                pass
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> MemoryResult:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        try:
            self._sample()
        except Exception:
            pass
        return MemoryResult(float(self.peak_mb), int(self.samples))


# =============================================================================

# =============================================================================

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_tesla():
    import sys

    if str(TESLA_PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(TESLA_PACKAGE_DIR))

    from TESLA.imputation import imputation
    from TESLA.TESLA import TESLA
    from TESLA.annotation import refine_clusters

    return imputation, TESLA, refine_clusters


def prepare_vihbc():
    resized = DATA_ROOT / SLIDE / "resized_data"
    image_path = resized / "he_resized.png"
    adata_path = resized / "adata_resized.h5ad"
    tissue_path = resized / "mask_resized.png"

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    tissue = cv2.imread(str(tissue_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    if tissue is None:
        raise FileNotFoundError(tissue_path)

    tissue = (tissue > 127).astype(np.uint8)
    if tissue.shape != image.shape[:2]:
        tissue = cv2.resize(
            tissue,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    adata = sc.read_h5ad(str(adata_path))
    adata.var_names_make_unique()

    valid_genes = [gene for gene in MARKER_GENES if gene in adata.var_names]
    if not valid_genes:
        raise RuntimeError('No TESLA marker genes were found in ViHBC.')

    adata = adata[:, valid_genes].copy()
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()

    spatial_key = "spatial_resized" if "spatial_resized" in adata.obsm else "spatial"
    coordinates = np.asarray(adata.obsm[spatial_key], dtype=np.float64)

    
    image_h, image_w = image.shape[:2]
    c0, c1 = coordinates[:, 0], coordinates[:, 1]

    def in_range(values, upper):
        return np.nanmin(values) >= -5 and np.nanmax(values) <= upper + 5

    row_col = in_range(c0, image_h) and in_range(c1, image_w)
    col_row = in_range(c0, image_w) and in_range(c1, image_h)
    if col_row and not row_col:
        adata.obs["pixel_x"] = c1
        adata.obs["pixel_y"] = c0
    else:
        adata.obs["pixel_x"] = c0
        adata.obs["pixel_y"] = c1

    contours, _ = cv2.findContours(
        tissue,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise RuntimeError('There are no contours in the ViHBC tissue mask.')
    contour = max(contours, key=cv2.contourArea)
    return image, tissue, adata, valid_genes, contour


def build_annotation_input(image, tissue, sudo_adata, genes):
    resize_width = int(image.shape[1] * RESIZE_FACTOR)
    resize_height = int(image.shape[0] * RESIZE_FACTOR)

    masked_image = (image * np.dstack([(tissue != 0)] * 3)).astype(np.uint8)
    resized_image = cv2.resize(
        masked_image,
        (resize_width, resize_height),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = gray.reshape(gray.shape + (1,))
    image_band = (gray - gray.min()) / max(float(gray.max() - gray.min()), 1e-8)

    genes = list(set(gene for gene in genes if gene in sudo_adata.var.index))
    matrix = np.asarray(sudo_adata[:, genes].X)
    matrix = (matrix - matrix.min(axis=0)) / np.maximum(
        matrix.max(axis=0) - matrix.min(axis=0),
        1e-8,
    )
    sudo_adata = sudo_adata.copy()
    sudo_adata.obs["meta"] = np.min(matrix, axis=1)

    gene_image = np.zeros(image.shape[:2] + (1,), dtype=np.float32)
    for _, row in sudo_adata.obs.iterrows():
        x = int(row["x"])
        y = int(row["y"])
        expression = float(row["meta"])
        gene_image[
            int(x - TESLA_RES / 2) : int(x + TESLA_RES / 2),
            int(y - TESLA_RES / 2) : int(y + TESLA_RES / 2),
            0,
        ] = expression

    gene_image[tissue == 0] = 0
    gene_band = cv2.resize(
        gene_image[:, :, 0],
        (resize_width, resize_height),
        interpolation=cv2.INTER_AREA,
    )
    gene_band = (gene_band - gene_band.min()) / max(
        float(gene_band.max() - gene_band.min()),
        1e-8,
    )
    gene_band = gene_band.reshape(gene_band.shape + (1,))

    data = np.concatenate((image_band, gene_band), axis=2).astype(np.float32)
    return data, gene_band, resize_height, resize_width


def model_predict(tesla, data):
    tensor = torch.from_numpy(
        np.array([data.transpose((2, 0, 1)).astype("float32")])
    )
    device = next(tesla.model.parameters()).device
    tensor = tensor.to(device)

    tesla.model.eval()
    with torch.inference_mode():
        output = tesla.model(tensor)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, tesla.nChannel)
        probability = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1)

    return probability.detach().cpu().numpy(), prediction.detach().cpu().numpy()


def complete_inference(
    tesla,
    data,
    gene_band,
    tissue,
    refine_clusters,
    resize_height,
    resize_width,
):
    _, prediction = model_predict(tesla, data)

    refined = refine_clusters(
        pred=prediction,
        resize_height=resize_height,
        resize_width=resize_width,
        threshold=1000,
        radius=3,
    )

    marker = gene_band.flatten()
    clusters = refined.flatten()
    cluster_marker = {
        int(cluster): float(np.mean(marker[clusters == cluster]))
        for cluster in np.unique(clusters)
    }
    ranked = sorted(cluster_marker.items(), key=lambda item: item[1], reverse=True)
    target_clusters = [
        cluster
        for cluster, score in ranked
        if score > ranked[0][1] * 0.5
    ]

    refined_2d = refined.reshape(resize_height, resize_width).astype(np.int32)
    refined_full = cv2.resize(
        refined_2d.astype(np.float32),
        (tissue.shape[1], tissue.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)

    mask = np.isin(refined_full, target_clusters).astype(np.uint8)
    mask[tissue == 0] = 0
    return mask


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    return total, trainable


def main():
    if USE_CUDA and not torch.cuda.is_available():
        raise RuntimeError('USE_CUDA=True, but CUDA is not available in the current environment.')
    if not TESLA_PACKAGE_DIR.is_dir():
        raise FileNotFoundError(TESLA_PACKAGE_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    run_imputation, TESLA, refine_clusters = load_tesla()
    image, tissue, adata, genes, contour = prepare_vihbc()

    
    monitor = GPUMonitor(GPU_INDEX).start()
    synchronize()
    training_start = time.perf_counter()

    sudo_adata = run_imputation(
        img=image,
        raw=adata,
        cnt=contour,
        genes=genes,
        res=TESLA_RES,
    )
    data, gene_band, resize_height, resize_width = build_annotation_input(
        image,
        tissue,
        sudo_adata,
        genes,
    )

    set_seed(SEED)
    tesla = TESLA()
    tesla.train(
        input=data,
        use_cuda=USE_CUDA,
        train_refine=True,
        radius=3,
        nChannel=100,
        lr=0.1,
        minLabels=30,
        maxIter=30,
        stepsize_sim=1,
        stepsize_con=5,
        threshold=1000,
        plot_intermedium=False,
        plot_dir=None,
    )
    synchronize()
    training_time = float(time.perf_counter() - training_start)
    training_memory = monitor.stop().peak_mb

    total_params, trainable_params = count_parameters(tesla.model)

    
    for index in range(WARMUP_RUNS):
        _ = complete_inference(
            tesla,
            data,
            gene_band,
            tissue,
            refine_clusters,
            resize_height,
            resize_width,
        )
        synchronize()
        print(f"Warm-up {index + 1}/{WARMUP_RUNS} completed.")

    monitor = GPUMonitor(GPU_INDEX).start()
    inference_times = []
    final_mask = None

    for index in range(INFERENCE_REPEATS):
        synchronize()
        start = time.perf_counter()
        final_mask = complete_inference(
            tesla,
            data,
            gene_band,
            tissue,
            refine_clusters,
            resize_height,
            resize_width,
        )
        synchronize()
        elapsed = float(time.perf_counter() - start)
        inference_times.append(elapsed)
        print(f"Inference {index + 1}/{INFERENCE_REPEATS}: {elapsed:.4f} s")

    inference_memory = monitor.stop().peak_mb

    mask_path = OUTPUT_DIR / "ViHBC_seed42_TESLA_pred_mask.png"
    cv2.imwrite(str(mask_path), final_mask.astype(np.uint8) * 255)

    result = {
        "Model": "TESLA",
        "Slide": SLIDE,
        "Seed": SEED,
        "Params": total_params,
        "Params_M": total_params / 1e6,
        "Trainable_Params": trainable_params,
        "Trainable_Params_M": trainable_params / 1e6,
        "Training_Time_s": training_time,
        "Inference_Time_s": float(np.mean(inference_times)),
        "Inference_Time_Std_s": float(np.std(inference_times, ddof=0)),
        "Training_Memory_MB": float(training_memory),
        "Inference_Memory_MB": float(inference_memory),
        "Warmup_Runs": WARMUP_RUNS,
        "Inference_Repeats": INFERENCE_REPEATS,
        "TESLA_Use_CUDA": USE_CUDA,
        "Prediction_Mask": str(mask_path),
        "Training_Definition": (
            "TESLA imputation + annotation-input construction + Spatial_CNN fitting"
        ),
        "Inference_Definition": (
            "trained Spatial_CNN forward + cluster refinement + target-cluster "
            "selection + resize + final cancer mask"
        ),
        "Memory_Definition": (
            "peak absolute NVIDIA memory of the current process tree sampled "
            "with nvidia-smi every 0.05 seconds"
        ),
    }

    json_path = OUTPUT_DIR / "TESLA_ViHBC_cost.json"
    csv_path = OUTPUT_DIR / "TESLA_ViHBC_cost.csv"
    txt_path = OUTPUT_DIR / "TESLA_ViHBC_cost.txt"

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    row = (
        f"TESLA & {result['Params_M']:.2f}M & "
        f"{result['Training_Time_s']:.2f} & "
        f"{result['Inference_Time_s']:.2f} & "
        f"{result['Training_Memory_MB']:.2f} & "
        f"{result['Inference_Memory_MB']:.2f} \\\\"
    )
    report = "\n".join(
        [
            "=" * 78,
            "TESLA computational cost on ViHBC",
            "=" * 78,
            f"Params            : {result['Params_M']:.4f} M",
            f"Training Time     : {result['Training_Time_s']:.2f} s",
            (
                f"Inference Time    : {result['Inference_Time_s']:.2f} "
                f"± {result['Inference_Time_Std_s']:.2f} s"
            ),
            f"Training Memory   : {result['Training_Memory_MB']:.2f} MB",
            f"Inference Memory  : {result['Inference_Memory_MB']:.2f} MB",
            "",
            "LaTeX row:",
            row,
            "=" * 78,
        ]
    )
    txt_path.write_text(report + "\n", encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
