"""
Profile iStar computational cost on the ViHBC section only.

Outputs
-------
1. Parameter count:
   - trainable iStar imputation network
   - pretrained HIPT/HViT feature extractor, when it can be instantiated
   - total pipeline parameter count
2. Training time (s)
3. Inference time (s)
4. Peak training GPU memory (MB)
5. Peak inference GPU memory (MB)

Benchmark definition
--------------------
- Data preparation and pretrained histology-feature extraction are performed
  once and reported separately as preprocessing time.
- Training time covers optimization of the iStar imputation network.
- Inference time covers:
    (1) super-resolution prediction with the trained imputation network,
    (2) clustering of super-resolution embeddings,
    (3) conversion of clusters to the final cancer mask/probability map.
- Only ViHBC and seed 42 are processed.

Run this script from the iStar conda environment:
    python 1profile_istar_ViHBC_cost.py

Required local files are the same as in the original ViHBC iStar pipeline.
"""

from __future__ import annotations

import csv
import gc
import importlib.util
import inspect
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import scanpy as sc
import torch


# ============================================================================
# User configuration
# ============================================================================

SLIDE = "ViHBC"
SEED = 42
EPOCHS = 400
DEVICE = "cuda"
GPU_INDEX = 0

# Keep one trained iStar state for one independent seed.
# Set this to the same number used by your local experimental configuration
# if your modified impute.py ensembles multiple internal states.
N_STATES = 1

BASE_DATA_DIR = Path(r"D:\Project_2\Pixel_Seg\data")
ISTAR_REPO_DIR = Path(r"D:\Project_2\iStar")
WORKSPACE_ROOT = ISTAR_REPO_DIR / "istar_workspace_efficiency"
PREFIX_DIR = WORKSPACE_ROOT / f"{SLIDE}_seed_{SEED}"

# Original iStar settings used in the supplied pipeline.
TOP_GENES = 1000
N_CLUSTERS = 10
FILTER_SIZE = 8
MIN_CLUSTER_SIZE = 20

BRCA_MARKERS = ["ERBB2", "EPCAM", "MKI67", "CDH1"]
CANCER_CLUSTER_RATIO = 0.5

# Reuse expensive image preprocessing and HIPT feature extraction when present.
FORCE_PREPROCESS = False

# Delete old trainable states and generated inference outputs before profiling.
RESET_TRAINING_STATE = True

# Count the pretrained HIPT/HViT feature-extractor parameters as part of the
# total pipeline parameter count. This does not affect timing measurements.
COUNT_HIPT_PARAMS = True

POLL_INTERVAL_SECONDS = 0.05


# ============================================================================
# General utilities
# ============================================================================

def seed_everything(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_trailing_separator(path: Path) -> str:
    text = str(path.resolve())
    return text if text.endswith(os.sep) else text + os.sep


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        synchronize_cuda()


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        clear_cuda()
        torch.cuda.reset_peak_memory_stats()


def peak_cuda_memory_mb() -> Tuple[float, float]:
    """Return peak allocated and reserved memory in MiB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    reserved = torch.cuda.max_memory_reserved() / (1024.0 ** 2)
    return float(allocated), float(reserved)


def command_supports_option(script_path: Path, option: str) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=str(ISTAR_REPO_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return option in (result.stdout or "")
    except Exception:
        return False


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Optional[Path] = None,
) -> float:
    """Run one command and return elapsed wall-clock time."""
    print("\n[Command]")
    print(" ".join(map(str, command)))

    log_file = None
    stdout_target = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
        stdout_target = log_file

    start = time.perf_counter()
    try:
        subprocess.run(
            list(map(str, command)),
            cwd=str(cwd),
            stdout=stdout_target,
            stderr=subprocess.STDOUT if log_file is not None else None,
            check=True,
        )
    finally:
        if log_file is not None:
            log_file.close()

    elapsed = time.perf_counter() - start
    print(f"[Elapsed] {elapsed:.2f} s")
    return float(elapsed)


def query_total_gpu_memory_mb(gpu_index: int) -> Optional[float]:
    """Read total memory currently used on a physical GPU with nvidia-smi."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        value = output.strip().splitlines()[0].strip()
        return float(value)
    except Exception:
        return None


def run_subprocess_with_gpu_profile(
    command: Sequence[str],
    *,
    cwd: Path,
    gpu_index: int,
    log_path: Path,
) -> Tuple[float, float]:
    """Profile elapsed time and GPU-memory increase for a child process.

    The reported GPU memory is the peak increase over the idle baseline.
    Run on a GPU without unrelated jobs for the cleanest measurement.
    """
    baseline = query_total_gpu_memory_mb(gpu_index)
    baseline = float(baseline) if baseline is not None else 0.0
    peak_delta = 0.0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n[Profiled command]")
    print(" ".join(map(str, command)))

    start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(map(str, command)),
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            current = query_total_gpu_memory_mb(gpu_index)
            if current is not None:
                peak_delta = max(peak_delta, float(current) - baseline)
            time.sleep(POLL_INTERVAL_SECONDS)

        return_code = process.wait()

    elapsed = time.perf_counter() - start
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    current = query_total_gpu_memory_mb(gpu_index)
    if current is not None:
        peak_delta = max(peak_delta, float(current) - baseline)

    peak_delta = max(0.0, peak_delta)
    print(f"[Elapsed] {elapsed:.2f} s | [Peak GPU delta] {peak_delta:.2f} MB")
    return float(elapsed), float(peak_delta)


def find_existing_file(
    directory: Path,
    candidates: Sequence[str],
    description: str,
) -> Path:
    for name in candidates:
        path = directory / name
        if path.exists():
            return path
    checked = "\n".join(str(directory / name) for name in candidates)
    raise FileNotFoundError(f"Cannot find {description}. Checked:\n{checked}")


def safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


# ============================================================================
# ViHBC preprocessing; not included in train/inference table values
# ============================================================================

def prepare_vihbc_inputs(prefix_dir: Path) -> float:
    """Prepare iStar inputs and HIPT features for ViHBC only."""
    prefix_dir.mkdir(parents=True, exist_ok=True)
    resized_dir = BASE_DATA_DIR / SLIDE / "resized_data"

    he_source = find_existing_file(
        resized_dir,
        ["he_resized.png", "he.png"],
        "ViHBC H&E image",
    )
    mask_source = find_existing_file(
        resized_dir,
        ["mask_resized.png", "tissue_mask_resized.png", "mask.png"],
        "ViHBC tissue mask",
    )
    adata_path = find_existing_file(
        resized_dir,
        ["adata_resized.h5ad", f"{SLIDE}_resized.h5ad", f"{SLIDE}.h5ad"],
        "ViHBC AnnData",
    )

    he_raw = prefix_dir / "he-raw.jpg"
    mask_scaled = prefix_dir / "mask-scaled.png"
    cnts_path = prefix_dir / "cnts.tsv"
    locs_path = prefix_dir / "locs-raw.tsv"
    radius_raw = prefix_dir / "radius-raw.txt"
    pixel_size = prefix_dir / "pixel-size.txt"
    pixel_size_raw = prefix_dir / "pixel-size-raw.txt"

    if FORCE_PREPROCESS:
        for path in [
            prefix_dir / "he.jpg",
            prefix_dir / "mask-small.png",
            prefix_dir / "embeddings-hist.pickle",
            prefix_dir / "_embeddings-hist.pickle",
            prefix_dir / "gene-names.txt",
        ]:
            safe_remove(path)

    if not he_raw.exists():
        image = cv2.imread(str(he_source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read H&E image: {he_source}")
        if not cv2.imwrite(str(he_raw), image):
            raise IOError(f"Cannot save: {he_raw}")

    if not mask_scaled.exists():
        shutil.copy2(mask_source, mask_scaled)

    if not cnts_path.exists() or not locs_path.exists():
        print("[Preprocess] Reading AnnData and exporting count/coordinate files...")
        adata = sc.read_h5ad(str(adata_path))
        adata.var_names_make_unique()

        expression = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        expression_df = pd.DataFrame(
            expression,
            index=adata.obs_names,
            columns=adata.var_names,
        )
        expression_df.to_csv(cnts_path, sep="\t")

        spatial_key = "spatial_resized" if "spatial_resized" in adata.obsm else "spatial"
        coordinates = np.asarray(adata.obsm[spatial_key])
        locations = pd.DataFrame(
            coordinates,
            index=adata.obs_names,
            columns=["x", "y"],
        )
        locations.to_csv(locs_path, sep="\t")

    radius_raw.write_text("10\n", encoding="utf-8")
    pixel_size.write_text("0.5\n", encoding="utf-8")
    pixel_size_raw.write_text("0.5\n", encoding="utf-8")

    prefix = ensure_trailing_separator(prefix_dir)
    log_dir = prefix_dir / "benchmark_logs"
    prep_start = time.perf_counter()

    required_preprocessed = [
        prefix_dir / "he.jpg",
        prefix_dir / "mask-small.png",
    ]
    if FORCE_PREPROCESS or not all(path.exists() for path in required_preprocessed):
        run_command(
            [sys.executable, "rescale.py", prefix, "--image"],
            cwd=ISTAR_REPO_DIR,
            log_path=log_dir / "01_rescale_image.log",
        )
        run_command(
            [sys.executable, "preprocess.py", prefix, "--image", "--mask"],
            cwd=ISTAR_REPO_DIR,
            log_path=log_dir / "02_preprocess.log",
        )

    embedding_path = prefix_dir / "embeddings-hist.pickle"
    if FORCE_PREPROCESS or not embedding_path.exists():
        run_command(
            [
                sys.executable,
                "extract_features.py",
                "--img_path",
                str(prefix_dir / "he.jpg"),
                "--out_prefix",
                prefix,
                f"--device={DEVICE}",
            ],
            cwd=ISTAR_REPO_DIR,
            log_path=log_dir / "03_extract_features.log",
        )

        wrong_name = prefix_dir / "_embeddings-hist.pickle"
        if wrong_name.exists() and not embedding_path.exists():
            wrong_name.rename(embedding_path)

    gene_names = prefix_dir / "gene-names.txt"
    if FORCE_PREPROCESS or not gene_names.exists():
        run_command(
            [
                sys.executable,
                "select_genes.py",
                f"--n-top={TOP_GENES}",
                str(cnts_path),
                str(gene_names),
            ],
            cwd=ISTAR_REPO_DIR,
            log_path=log_dir / "04_select_genes.log",
        )

    radius_path = prefix_dir / "radius.txt"
    locs_scaled_path = prefix_dir / "locs.tsv"
    if FORCE_PREPROCESS or not radius_path.exists() or not locs_scaled_path.exists():
        run_command(
            [sys.executable, "rescale.py", prefix, "--locs", "--radius"],
            cwd=ISTAR_REPO_DIR,
            log_path=log_dir / "05_rescale_locs_radius.log",
        )

    prep_time = time.perf_counter() - prep_start
    print(f"[Preprocessing/HIPT feature extraction] {prep_time:.2f} s")
    return float(prep_time)


# ============================================================================
# Directly split the original iStar imputation routine into train/inference
# ============================================================================

def load_local_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def count_unique_parameters(objects: Iterable[Any]) -> int:
    """Count unique nn.Parameter objects across one or more modules."""
    seen: set[int] = set()
    total = 0

    def visit(obj: Any) -> None:
        nonlocal total
        if isinstance(obj, torch.nn.Module):
            for parameter in obj.parameters():
                identity = id(parameter)
                if identity not in seen:
                    seen.add(identity)
                    total += int(parameter.numel())
            return

        if isinstance(obj, dict):
            for value in obj.values():
                visit(value)
            return

        if isinstance(obj, (list, tuple, set)):
            for value in obj:
                visit(value)
            return

        if hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                if isinstance(value, (torch.nn.Module, dict, list, tuple, set)):
                    visit(value)

    for item in objects:
        visit(item)
    return int(total)


def accepted_kwargs(function, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(function)
    has_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if has_var_kwargs:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def load_radius(prefix_dir: Path) -> float:
    radius_path = prefix_dir / "radius.txt"
    text = radius_path.read_text(encoding="utf-8").strip()
    # Original iStar converts the rescaled radius to the embedding-grid radius.
    return int(float(text)) / 16.0


def train_and_predict_imputation(
    prefix_dir: Path,
) -> Dict[str, Any]:
    """Train iStar and separately profile the trained-model prediction."""
    if str(ISTAR_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ISTAR_REPO_DIR))
    os.chdir(ISTAR_REPO_DIR)

    impute_module = load_local_module(
        "istar_impute_profiled",
        ISTAR_REPO_DIR / "impute.py",
    )

    prefix = ensure_trailing_separator(prefix_dir)
    print("[Load iStar data]")
    embeddings, counts_df, locations = impute_module.get_data(prefix)

    names = np.asarray(counts_df.columns)
    counts = counts_df.to_numpy().astype(np.float32)

    _, normalized_counts, _, (counts_min, counts_max) = impute_module.normalize(
        embeddings,
        counts,
    )

    radius = load_radius(prefix_dir)
    n_train = normalized_counts.shape[0]
    batch_size = min(128, max(1, n_train // 16))

    states_dir = prefix_dir / "states"
    if RESET_TRAINING_STATE:
        safe_remove(states_dir)
        for generated in [
            prefix_dir / "cnts-super",
            prefix_dir / "embeddings-gene.pickle",
            prefix_dir / "clusters-gene",
        ]:
            safe_remove(generated)

    states_dir.mkdir(parents=True, exist_ok=True)

    models: List[torch.nn.Module] = []
    datasets: List[Any] = []

    # ------------------------------ Training ------------------------------
    clear_cuda()
    reset_cuda_peak()
    synchronize_cuda()
    training_start = time.perf_counter()

    for state_index in range(N_STATES):
        state_seed = SEED + state_index
        seed_everything(state_seed)

        state_prefix = ensure_trailing_separator(states_dir / f"{state_index:02d}")
        Path(state_prefix).mkdir(parents=True, exist_ok=True)

        kwargs = {
            "x": embeddings,
            "y": normalized_counts,
            "locs": locations,
            "radius": radius,
            "prefix": state_prefix,
            "batch_size": batch_size,
            "epochs": EPOCHS,
            "lr": 1e-4,
            "load_saved": False,
            "device": DEVICE,
            "seed": state_seed,
        }
        kwargs = accepted_kwargs(impute_module.get_model, kwargs)
        model, dataset = impute_module.get_model(**kwargs)
        models.append(model)
        datasets.append(dataset)

    synchronize_cuda()
    training_time = time.perf_counter() - training_start
    train_allocated_mb, train_reserved_mb = peak_cuda_memory_mb()

    one_model_params = count_unique_parameters([models[0]])
    ensemble_params = count_unique_parameters(models)

    print(
        f"[Training] time={training_time:.2f} s | "
        f"peak allocated={train_allocated_mb:.2f} MB | "
        f"one model params={one_model_params / 1e6:.4f} M"
    )

    # Move models off GPU so inference memory starts from a clean baseline.
    for model in models:
        model.eval()
        model.to("cpu")
    clear_cuda()

    # Prepare prediction metadata before the timed forward pass.
    mask_size = float(datasets[0].mask.sum())
    counts_range = np.stack([counts_min, counts_max], axis=-1)
    counts_range = counts_range / max(mask_size, 1.0)

    row_batch_size = 50
    num_row_batches = embeddings.shape[0] // row_batch_size + 1
    embedding_batches = np.array_split(embeddings, num_row_batches)

    (prefix_dir / "cnts-super").mkdir(parents=True, exist_ok=True)

    # ------------------------------ Inference -----------------------------
    reset_cuda_peak()
    synchronize_cuda()
    inference_start = time.perf_counter()

    with torch.inference_mode():
        impute_module.predict(
            model_states=models,
            x_batches=embedding_batches,
            name_list=names,
            y_range=counts_range,
            prefix=prefix,
            device=DEVICE,
        )

    synchronize_cuda()
    imputation_inference_time = time.perf_counter() - inference_start
    infer_allocated_mb, infer_reserved_mb = peak_cuda_memory_mb()

    print(
        f"[Imputation inference] time={imputation_inference_time:.2f} s | "
        f"peak allocated={infer_allocated_mb:.2f} MB"
    )

    return {
        "training_time_s": float(training_time),
        "training_gpu_allocated_mb": float(train_allocated_mb),
        "training_gpu_reserved_mb": float(train_reserved_mb),
        "imputation_inference_time_s": float(imputation_inference_time),
        "imputation_inference_gpu_allocated_mb": float(infer_allocated_mb),
        "imputation_inference_gpu_reserved_mb": float(infer_reserved_mb),
        "imputation_one_model_params": int(one_model_params),
        "imputation_ensemble_params": int(ensemble_params),
        "number_of_internal_states": int(len(models)),
    }


# ============================================================================
# Clustering and zero-shot conversion to the final cancer-region prediction
# ============================================================================

def run_clustering(prefix_dir: Path) -> Tuple[float, float]:
    cluster_dir = prefix_dir / "clusters-gene"
    safe_remove(cluster_dir)
    cluster_dir.mkdir(parents=True, exist_ok=True)

    prefix = ensure_trailing_separator(prefix_dir)
    cluster_script = ISTAR_REPO_DIR / "cluster.py"

    command = [
        sys.executable,
        str(cluster_script),
        f"--filter-size={FILTER_SIZE}",
        f"--min-cluster-size={MIN_CLUSTER_SIZE}",
        f"--n-clusters={N_CLUSTERS}",
        f"--mask={prefix_dir / 'mask-small.png'}",
        str(prefix_dir / "embeddings-gene.pickle"),
        ensure_trailing_separator(cluster_dir),
    ]
    if command_supports_option(cluster_script, "--seed"):
        command.append(f"--seed={SEED}")

    return run_subprocess_with_gpu_profile(
        command,
        cwd=ISTAR_REPO_DIR,
        gpu_index=GPU_INDEX,
        log_path=prefix_dir / "benchmark_logs" / "06_cluster.log",
    )


def resize_nearest(array: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(
        array.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return np.rint(resized).astype(np.int32)


def zero_shot_cluster_scores(
    adata,
    cluster_map: np.ndarray,
    marker_genes: Sequence[str],
    spatial_key: str,
) -> Tuple[List[int], Dict[int, float], List[str]]:
    adata.var_names_make_unique()
    valid_markers = [gene for gene in marker_genes if gene in adata.var_names]
    if not valid_markers:
        raise RuntimeError(
            f"None of the requested markers are present: {list(marker_genes)}"
        )

    coordinates = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    height, width = cluster_map.shape
    x = np.rint(coordinates[:, 0]).astype(np.int64)
    y = np.rint(coordinates[:, 1]).astype(np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    spot_clusters = np.full(adata.n_obs, -1, dtype=np.int32)
    spot_clusters[valid] = cluster_map[y[valid], x[valid]]

    cluster_scores: Dict[int, float] = {}
    for cluster_id in sorted(int(v) for v in np.unique(spot_clusters) if int(v) >= 0):
        spot_mask = spot_clusters == cluster_id
        expression = adata[spot_mask, valid_markers].X
        expression = expression.toarray() if hasattr(expression, "toarray") else np.asarray(expression)
        score = float(np.nanmean(expression))
        if np.isfinite(score):
            cluster_scores[cluster_id] = score

    if not cluster_scores:
        raise RuntimeError("No valid iStar cluster marker scores were obtained.")

    ranked = sorted(cluster_scores.items(), key=lambda item: item[1], reverse=True)
    n_select = max(1, int(np.ceil(len(ranked) * CANCER_CLUSTER_RATIO)))
    selected = [int(cluster_id) for cluster_id, _ in ranked[:n_select]]
    return selected, cluster_scores, valid_markers


def build_probability_map(
    cluster_map: np.ndarray,
    cluster_scores: Dict[int, float],
    tissue_mask: np.ndarray,
) -> np.ndarray:
    probability = np.zeros(cluster_map.shape, dtype=np.float32)
    scores = np.asarray(list(cluster_scores.values()), dtype=np.float64)
    score_min = float(scores.min())
    score_max = float(scores.max())

    if score_max > score_min:
        normalized = {
            cluster_id: float((score - score_min) / (score_max - score_min))
            for cluster_id, score in cluster_scores.items()
        }
    else:
        normalized = {cluster_id: 0.5 for cluster_id in cluster_scores}

    for cluster_id, score in normalized.items():
        probability[cluster_map == cluster_id] = score

    probability[tissue_mask == 0] = 0.0
    return probability


def create_final_prediction(prefix_dir: Path) -> Tuple[float, Path, Path]:
    """Create the ViHBC cancer mask/probability map from iStar clusters."""
    start = time.perf_counter()
    resized_dir = BASE_DATA_DIR / SLIDE / "resized_data"

    tissue_path = find_existing_file(
        resized_dir,
        ["mask_resized.png", "tissue_mask_resized.png", "mask.png"],
        "ViHBC tissue mask",
    )
    adata_path = find_existing_file(
        resized_dir,
        ["adata_resized.h5ad", f"{SLIDE}_resized.h5ad", f"{SLIDE}.h5ad"],
        "ViHBC AnnData",
    )

    tissue_image = cv2.imread(str(tissue_path), cv2.IMREAD_GRAYSCALE)
    if tissue_image is None:
        raise ValueError(f"Cannot read tissue mask: {tissue_path}")
    tissue_mask = (tissue_image > 127).astype(np.uint8)

    labels_path = prefix_dir / "clusters-gene" / "labels.pickle"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing iStar cluster labels: {labels_path}")
    with open(labels_path, "rb") as file:
        cluster_map = np.asarray(pickle.load(file))

    if cluster_map.ndim != 2:
        raise ValueError(f"Expected a 2-D cluster map, got {cluster_map.shape}")

    cluster_map = resize_nearest(
        cluster_map,
        width=tissue_mask.shape[1],
        height=tissue_mask.shape[0],
    )

    adata = sc.read_h5ad(str(adata_path))
    spatial_key = "spatial_resized" if "spatial_resized" in adata.obsm else "spatial"

    selected_clusters, cluster_scores, valid_markers = zero_shot_cluster_scores(
        adata,
        cluster_map,
        BRCA_MARKERS,
        spatial_key,
    )

    prediction = np.isin(cluster_map, selected_clusters).astype(np.uint8)
    prediction[tissue_mask == 0] = 0
    probability = build_probability_map(cluster_map, cluster_scores, tissue_mask)

    output_dir = prefix_dir / "final_prediction"
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{SLIDE}_seed{SEED}_istar_pred_mask.png"
    probability_path = output_dir / f"{SLIDE}_seed{SEED}_istar_probability.npz"

    if not cv2.imwrite(str(mask_path), prediction * 255):
        raise IOError(f"Cannot save mask: {mask_path}")
    np.savez_compressed(
        probability_path,
        probability=probability.astype(np.float32),
        slide=np.asarray(SLIDE),
        seed=np.asarray(SEED, dtype=np.int32),
        selected_clusters=np.asarray(selected_clusters, dtype=np.int32),
        valid_markers=np.asarray(valid_markers),
    )

    elapsed = time.perf_counter() - start
    print(
        f"[Zero-shot postprocessing] time={elapsed:.2f} s | "
        f"selected clusters={selected_clusters}"
    )
    return float(elapsed), mask_path, probability_path


# ============================================================================
# HIPT/HViT and result reporting
# ============================================================================

def count_hipt_parameters() -> Optional[int]:
    """Instantiate the pretrained HIPT extractor on CPU and count parameters."""
    if not COUNT_HIPT_PARAMS:
        return None

    model256_path = ISTAR_REPO_DIR / "checkpoints" / "vit256_small_dino.pth"
    model4k_path = ISTAR_REPO_DIR / "checkpoints" / "vit4k_xs_dino.pth"
    if not model256_path.exists() or not model4k_path.exists():
        print(
            "[Warning] HIPT checkpoint(s) not found; total pipeline parameter "
            "count will fall back to the trainable imputation network."
        )
        return None

    try:
        if str(ISTAR_REPO_DIR) not in sys.path:
            sys.path.insert(0, str(ISTAR_REPO_DIR))
        from hipt_4k import HIPT_4K

        model = HIPT_4K(
            model256_path=str(model256_path),
            model4k_path=str(model4k_path),
            device256="cpu",
            device4k="cpu",
        )
        parameters = count_unique_parameters([model])
        del model
        gc.collect()
        return int(parameters)
    except Exception as error:
        print(f"[Warning] Could not count HIPT parameters: {error}")
        return None


def save_results(
    preprocessing_time_s: float,
    imputation_profile: Dict[str, Any],
    cluster_time_s: float,
    cluster_gpu_memory_mb: float,
    postprocess_time_s: float,
    mask_path: Path,
    probability_path: Path,
) -> Dict[str, Any]:
    hipt_params = count_hipt_parameters()
    imputation_params = int(imputation_profile["imputation_ensemble_params"])

    total_pipeline_params = (
        imputation_params + int(hipt_params)
        if hipt_params is not None
        else imputation_params
    )

    inference_time = (
        float(imputation_profile["imputation_inference_time_s"])
        + float(cluster_time_s)
        + float(postprocess_time_s)
    )
    inference_memory = max(
        float(imputation_profile["imputation_inference_gpu_allocated_mb"]),
        float(cluster_gpu_memory_mb),
    )

    result: Dict[str, Any] = {
        "Model": "iStar",
        "Slide": SLIDE,
        "Seed": SEED,
        "Epochs": EPOCHS,
        "Internal_States": N_STATES,
        # Use Total_Pipeline_Params_M for the paper table when all models are
        # counted consistently, including frozen pretrained encoders.
        "Params_M": total_pipeline_params / 1e6,
        "Total_Pipeline_Params_M": total_pipeline_params / 1e6,
        "Trainable_Imputation_Params_M": (
            int(imputation_profile["imputation_one_model_params"]) / 1e6
        ),
        "Imputation_Ensemble_Params_M": imputation_params / 1e6,
        "HIPT_Params_M": (
            int(hipt_params) / 1e6 if hipt_params is not None else None
        ),
        "Training_Time_s": float(imputation_profile["training_time_s"]),
        "Inference_Time_s": float(inference_time),
        "Training_Memory_MB": float(
            imputation_profile["training_gpu_allocated_mb"]
        ),
        "Inference_Memory_MB": float(inference_memory),
        "Training_Reserved_Memory_MB": float(
            imputation_profile["training_gpu_reserved_mb"]
        ),
        "Inference_Reserved_Memory_MB": float(
            imputation_profile["imputation_inference_gpu_reserved_mb"]
        ),
        "Preprocessing_and_HIPT_Time_s": float(preprocessing_time_s),
        "Imputation_Inference_Time_s": float(
            imputation_profile["imputation_inference_time_s"]
        ),
        "Clustering_Time_s": float(cluster_time_s),
        "Postprocessing_Time_s": float(postprocess_time_s),
        "Cluster_GPU_Memory_MB": float(cluster_gpu_memory_mb),
        "Prediction_Mask": str(mask_path),
        "Probability_Map": str(probability_path),
        "Memory_Unit": "MiB",
    }

    output_dir = prefix_dir = PREFIX_DIR / "efficiency_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "istar_ViHBC_efficiency.json"
    csv_path = output_dir / "istar_ViHBC_efficiency.csv"
    txt_path = output_dir / "istar_ViHBC_efficiency.txt"

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    table_lines = [
        "=" * 76,
        "iStar computational cost on ViHBC",
        "=" * 76,
        f"Params                 : {result['Params_M']:.2f} M",
        f"Training Time          : {result['Training_Time_s']:.2f} s",
        f"Inference Time         : {result['Inference_Time_s']:.2f} s",
        f"Training Memory        : {result['Training_Memory_MB']:.2f} MB",
        f"Inference Memory       : {result['Inference_Memory_MB']:.2f} MB",
        "-" * 76,
        f"Trainable imputation   : {result['Trainable_Imputation_Params_M']:.4f} M",
        f"HIPT/HViT params       : {result['HIPT_Params_M']}",
        f"Preprocessing time     : {result['Preprocessing_and_HIPT_Time_s']:.2f} s",
        f"Imputation inference   : {result['Imputation_Inference_Time_s']:.2f} s",
        f"Clustering             : {result['Clustering_Time_s']:.2f} s",
        f"Zero-shot postprocess  : {result['Postprocessing_Time_s']:.2f} s",
        "=" * 76,
    ]
    txt_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(table_lines))
    print(f"\nJSON: {json_path}")
    print(f"CSV : {csv_path}")
    print(f"TXT : {txt_path}")
    return result


def validate_environment() -> None:
    required_scripts = [
        "rescale.py",
        "preprocess.py",
        "extract_features.py",
        "select_genes.py",
        "impute.py",
        "cluster.py",
    ]
    missing = [
        str(ISTAR_REPO_DIR / script)
        for script in required_scripts
        if not (ISTAR_REPO_DIR / script).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing iStar scripts:\n" + "\n".join(missing)
        )

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DEVICE='cuda', but CUDA is not available.")

    if SLIDE != "ViHBC":
        raise ValueError("This profiler is intentionally restricted to ViHBC.")


def main() -> None:
    validate_environment()
    seed_everything(SEED)
    PREFIX_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("iStar ViHBC computational-cost profiling")
    print(f"Repository : {ISTAR_REPO_DIR}")
    print(f"Data       : {BASE_DATA_DIR / SLIDE / 'resized_data'}")
    print(f"Workspace  : {PREFIX_DIR}")
    print(f"Seed       : {SEED}")
    print(f"Epochs     : {EPOCHS}")
    print("=" * 90)

    preprocessing_time = prepare_vihbc_inputs(PREFIX_DIR)

    profile = train_and_predict_imputation(PREFIX_DIR)

    cluster_time, cluster_gpu_memory = run_clustering(PREFIX_DIR)

    postprocess_time, mask_path, probability_path = create_final_prediction(
        PREFIX_DIR
    )

    save_results(
        preprocessing_time_s=preprocessing_time,
        imputation_profile=profile,
        cluster_time_s=cluster_time,
        cluster_gpu_memory_mb=cluster_gpu_memory,
        postprocess_time_s=postprocess_time,
        mask_path=mask_path,
        probability_path=probability_path,
    )


if __name__ == "__main__":
    main()
