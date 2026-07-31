"""Extract 512-D scGPT-spatial embeddings from all available slice genes.

This script removes the fixed-gene dictionary restriction and removes PCA.

Final workflow
--------------
For each slice:

1. Read the complete spot-by-gene matrix.
2. Normalize the COMPLETE matrix.
3. Retain every gene whose symbol is present in the scGPT vocabulary.
4. Optionally remove genes detected in too few spots.
5. Pass the resulting per-slice all-gene matrix to scGPT-spatial.
6. Save the raw 512-dimensional spot embedding directly.
7. Write the same 512-dimensional embedding into ``{slice}_ctx.npz``.

There is:
- no fixed 512/600-gene dictionary;
- no zero-filling to a cross-slice gene panel;
- no fold-specific gene selection;
- no PCA;
- no reduction from 512 to 64 dimensions.

Important context-length note
-----------------------------
scGPT-spatial was pretrained with a maximum context of 600 genes per spot.
Therefore, this script keeps ALL vocabulary-compatible genes as the candidate
input pool, but by default ``--max-length 601`` limits an individual spot to
at most 600 expressed gene tokens plus CLS, matching model pretraining.

Use ``--max-length 0`` to automatically pass every expressed gene of every spot.
This can require much more GPU memory and extends beyond the model's original
training context, so it is not recommended unless explicitly tested.

Preprocessing modes
-------------------
``paper_mean`` (default)
    Approximate the two-level normalization described in scGPT-spatial:
    (1) divide each slide by its non-zero mean over its vocabulary-compatible
        QC-passing genes;
    (2) divide each gene by a non-zero mean estimated only from the configured
        normalization-training slices (normally the eight STHBC slices).
    No log1p is applied.

``scanpy_full``
    Normalize every spot to ``target_sum`` using the COMPLETE source matrix,
    apply log1p, and only then retain all scGPT-vocabulary genes.

``none``
    Use stored expression values directly, then retain vocabulary genes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import certifi
import cv2
import numpy as np
import scipy.sparse as sp
import yaml

# Windows/scGPT compatibility.
if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid: []
if not hasattr(np, "float"):
    np.float = float

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_STHBC_SLICES = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]

GENE_COLUMN_CANDIDATES = (
    "feature_name",
    "gene_name",
    "gene_symbols",
    "symbol",
    "gene_symbol",
    "features",
    "SYMBOL",
)


# =============================================================================
# SSL / basic I/O
# =============================================================================

def _patched_ssl_context(
    purpose=ssl.Purpose.SERVER_AUTH,
    *,
    cafile=None,
    capath=None,
    cadata=None,
):
    protocol = (
        ssl.PROTOCOL_TLS_CLIENT
        if purpose == ssl.Purpose.SERVER_AUTH
        else ssl.PROTOCOL_TLS_SERVER
    )
    context = ssl.SSLContext(protocol)
    context.load_verify_locations(cafile=certifi.where())
    return context


ssl.create_default_context = _patched_ssl_context


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return cfg


def resolve_config_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()

    local = Path(__file__).resolve().parent / value
    if local.exists():
        return local.resolve()

    raise FileNotFoundError(value)


def cv_imread(path: str, flags: int = cv2.IMREAD_COLOR):
    if not os.path.exists(path):
        return None
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def clean_gene_name(value: Any) -> str:
    name = str(value).strip()
    return "" if name.lower() in {"", "nan", "none"} else name


def canonical_gene_name(value: Any) -> str:
    return clean_gene_name(value).upper()


# =============================================================================
# Dataset registry
# =============================================================================

def slice_registry(cfg: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    datasets = cfg.get("datasets", {})

    if not isinstance(datasets, Mapping):
        raise KeyError("cfg.datasets is required")

    for dataset_name, dataset_cfg_obj in datasets.items():
        dataset_cfg = dict(dataset_cfg_obj)
        slices = [str(value) for value in dataset_cfg.get("slices", [])]
        resized_dir = str(dataset_cfg.get("resized_dir", ""))

        if not resized_dir:
            raise KeyError(
                f"datasets.{dataset_name}.resized_dir is required"
            )

        per_slice_subdir = bool(
            dataset_cfg.get("per_slice_subdir", False)
        )
        configured_h5ad = str(
            dataset_cfg.get("h5ad_name", "adata_resized.h5ad")
        )

        for slice_id in slices:
            base = (
                os.path.join(resized_dir, slice_id)
                if per_slice_subdir
                else resized_dir
            )

            if str(dataset_name).upper() == "STHBC":
                names = [
                    "adata_HR_resized.h5ad",
                    configured_h5ad,
                    "adata_resized.h5ad",
                ]
            else:
                names = [
                    configured_h5ad,
                    "adata_HR_resized.h5ad",
                    "adata_resized.h5ad",
                ]

            names = list(dict.fromkeys(names))

            registry[slice_id] = {
                "dataset": str(dataset_name),
                "base": base,
                "h5ad_candidates": [
                    os.path.join(base, name) for name in names
                ],
                "context_root": str(
                    dataset_cfg.get("context_cache_root", "")
                ),
                "raw_root": str(
                    dataset_cfg.get("raw_scgpt_root", "")
                ),
            }

    return registry


def resolve_h5ad_path(info: Mapping[str, Any]) -> str:
    candidates = [str(path) for path in info["h5ad_candidates"]]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No h5ad file found. Tried:\n  " + "\n  ".join(candidates)
    )


# =============================================================================
# Gene symbols and expression matrices
# =============================================================================

def resolve_gene_names(adata: Any) -> np.ndarray:
    for column in GENE_COLUMN_CANDIDATES:
        if column in adata.var.columns:
            values = adata.var[column].astype(str).to_numpy()
            cleaned = np.asarray(
                [clean_gene_name(value) for value in values],
                dtype=object,
            )
            if np.count_nonzero(cleaned != "") > 0:
                return cleaned

    return np.asarray(
        [clean_gene_name(value) for value in adata.var_names],
        dtype=object,
    )


def get_expression_matrix(
    adata: Any,
    counts_layer: Optional[str],
) -> sp.csr_matrix:
    if counts_layer:
        if counts_layer not in adata.layers:
            raise KeyError(
                f"Requested counts layer {counts_layer!r} is absent"
            )
        matrix = adata.layers[counts_layer]
    else:
        matrix = adata.X

    if sp.issparse(matrix):
        result = matrix.tocsr().astype(np.float32, copy=True)
    else:
        result = sp.csr_matrix(
            np.asarray(matrix, dtype=np.float32)
        )

    if result.data.size and not np.isfinite(result.data).all():
        raise ValueError("Expression matrix contains NaN/Inf")
    if result.data.size and float(result.data.min()) < 0:
        raise ValueError("Expression matrix contains negative values")

    result.eliminate_zeros()
    return result


def load_vocab(model_dir: str) -> Tuple[set[str], Dict[str, str]]:
    path = Path(model_dir) / "vocab.json"
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        vocab_obj = json.load(file)

    canonical_to_vocab_name: Dict[str, str] = {}
    for raw_name in vocab_obj.keys():
        display = clean_gene_name(raw_name)
        canonical = canonical_gene_name(raw_name)
        if canonical and canonical not in canonical_to_vocab_name:
            canonical_to_vocab_name[canonical] = display

    return set(canonical_to_vocab_name), canonical_to_vocab_name


def aggregate_duplicate_symbols(
    matrix: sp.csr_matrix,
    source_gene_names: Sequence[Any],
) -> Tuple[sp.csr_matrix, List[str], Dict[str, int]]:
    """Merge duplicate gene symbols while preserving first-seen order."""
    canonical_to_column: Dict[str, int] = {}
    output_names: List[str] = []
    source_rows: List[int] = []
    target_cols: List[int] = []
    hit_count: Dict[str, int] = {}

    for source_index, raw_name in enumerate(source_gene_names):
        display = clean_gene_name(raw_name)
        canonical = canonical_gene_name(raw_name)
        if not canonical:
            continue

        if canonical not in canonical_to_column:
            canonical_to_column[canonical] = len(output_names)
            output_names.append(display)

        target_index = canonical_to_column[canonical]
        source_rows.append(source_index)
        target_cols.append(target_index)
        hit_count[canonical] = hit_count.get(canonical, 0) + 1

    if not output_names:
        raise RuntimeError("No valid gene symbols were found")

    aggregator = sp.csr_matrix(
        (
            np.ones(len(source_rows), dtype=np.float32),
            (
                np.asarray(source_rows, dtype=np.int64),
                np.asarray(target_cols, dtype=np.int64),
            ),
        ),
        shape=(matrix.shape[1], len(output_names)),
    )

    merged = (matrix @ aggregator).tocsr().astype(np.float32)
    merged.eliminate_zeros()

    duplicate_symbols = {
        output_names[canonical_to_column[canonical]]: count
        for canonical, count in hit_count.items()
        if count > 1
    }

    return merged, output_names, duplicate_symbols


def retain_vocab_genes(
    matrix: sp.csr_matrix,
    gene_names: Sequence[str],
    vocab: set[str],
    canonical_to_vocab_name: Mapping[str, str],
    min_gene_fraction: float,
) -> Tuple[sp.csr_matrix, List[str], Dict[str, Any]]:
    """Keep all scGPT-vocabulary genes passing the detection threshold."""
    if not 0.0 <= float(min_gene_fraction) <= 1.0:
        raise ValueError("min_gene_fraction must be in [0, 1]")

    vocab_indices: List[int] = []
    output_names: List[str] = []

    for index, gene in enumerate(gene_names):
        canonical = canonical_gene_name(gene)
        if canonical in vocab:
            vocab_indices.append(index)
            # Use the exact spelling stored in vocab.json.
            output_names.append(canonical_to_vocab_name[canonical])

    if not vocab_indices:
        raise RuntimeError("No slice genes overlap the scGPT vocabulary")

    selected = matrix[:, np.asarray(vocab_indices, dtype=np.int64)].tocsr()

    min_spots = max(
        1,
        int(np.ceil(float(min_gene_fraction) * matrix.shape[0])),
    )
    detection = np.asarray(selected.getnnz(axis=0)).ravel()
    keep = detection >= min_spots

    if not np.any(keep):
        raise RuntimeError(
            f"No scGPT-vocabulary genes pass min detection count {min_spots}"
        )

    kept_indices = np.flatnonzero(keep)
    retained = selected[:, kept_indices].tocsr().astype(np.float32)
    retained.eliminate_zeros()
    retained_names = [output_names[index] for index in kept_indices]

    return retained, retained_names, {
        "unique_source_genes": int(matrix.shape[1]),
        "vocab_overlap_genes_before_qc": int(len(vocab_indices)),
        "min_gene_fraction": float(min_gene_fraction),
        "min_gene_spots": int(min_spots),
        "retained_vocab_genes": int(len(retained_names)),
    }


def library_normalize_log1p_full(
    matrix: sp.csr_matrix,
    target_sum: float,
) -> Tuple[sp.csr_matrix, Dict[str, Any]]:
    """Normalize with COMPLETE source libraries and then apply log1p."""
    if target_sum <= 0:
        raise ValueError("target_sum must be positive")

    row_sums = np.asarray(
        matrix.sum(axis=1)
    ).ravel().astype(np.float64)
    nonempty = row_sums > 0

    scale = np.zeros_like(row_sums, dtype=np.float64)
    scale[nonempty] = float(target_sum) / row_sums[nonempty]

    normalized = sp.diags(
        scale.astype(np.float32)
    ) @ matrix
    normalized = normalized.tocsr().astype(np.float32)
    normalized.data = np.log1p(normalized.data).astype(np.float32)
    normalized.eliminate_zeros()

    return normalized, {
        "operation": "complete_matrix_normalize_total_then_log1p",
        "target_sum": float(target_sum),
        "empty_spots_before_normalization": int(
            np.count_nonzero(~nonempty)
        ),
        "median_full_library_size": (
            float(np.median(row_sums[nonempty]))
            if np.any(nonempty)
            else 0.0
        ),
    }


def slide_nonzero_mean(
    matrix: sp.csr_matrix,
) -> float:
    if matrix.data.size == 0:
        raise RuntimeError("Expression matrix has no non-zero values")

    value = float(np.mean(matrix.data, dtype=np.float64))
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError(f"Invalid slide non-zero mean: {value}")

    return value


def divide_columns_by_gene_means(
    matrix: sp.csr_matrix,
    gene_names: Sequence[str],
    gene_means: Mapping[str, float],
) -> Tuple[sp.csr_matrix, List[str]]:
    scale = np.ones(len(gene_names), dtype=np.float32)
    unseen_genes: List[str] = []

    for index, gene in enumerate(gene_names):
        canonical = canonical_gene_name(gene)
        value = float(gene_means.get(canonical, 1.0))

        if canonical not in gene_means:
            unseen_genes.append(gene)

        if not np.isfinite(value) or value <= 0:
            value = 1.0

        scale[index] = np.float32(1.0 / value)

    result = matrix @ sp.diags(scale)
    result = result.tocsr().astype(np.float32)
    result.eliminate_zeros()
    return result, unseen_genes


def extract_coordinates(
    adata: Any,
    slice_id: str,
) -> Tuple[np.ndarray, str]:
    for key in ("spatial_resized", "spatial"):
        if key in adata.obsm:
            coordinates = np.asarray(
                adata.obsm[key],
                dtype=np.float32,
            )
            if coordinates.shape != (adata.n_obs, 2):
                raise ValueError(
                    f"[{slice_id}] invalid {key}: {coordinates.shape}"
                )
            if not np.isfinite(coordinates).all():
                raise ValueError(f"[{slice_id}] {key} contains NaN/Inf")
            return coordinates, key

    raise KeyError(
        f"[{slice_id}] missing spatial_resized/spatial coordinates"
    )


# =============================================================================
# Paper-like training-only gene means
# =============================================================================

def compute_paper_gene_means(
    registry: Mapping[str, Mapping[str, Any]],
    training_slices: Sequence[str],
    counts_layer: Optional[str],
    vocab: set[str],
    canonical_to_vocab_name: Mapping[str, str],
    min_gene_fraction: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Estimate non-zero means for all training-observed vocab genes."""
    import scanpy as sc

    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    per_slice: Dict[str, Any] = {}

    for slice_id in training_slices:
        path = resolve_h5ad_path(registry[slice_id])
        adata = sc.read_h5ad(path)

        matrix = get_expression_matrix(adata, counts_layer)
        raw_gene_names = resolve_gene_names(adata)
        matrix, merged_names, duplicate_map = aggregate_duplicate_symbols(
            matrix,
            raw_gene_names,
        )

        vocab_matrix, vocab_names, qc_meta = retain_vocab_genes(
            matrix=matrix,
            gene_names=merged_names,
            vocab=vocab,
            canonical_to_vocab_name=canonical_to_vocab_name,
            min_gene_fraction=min_gene_fraction,
        )

        slide_mean = slide_nonzero_mean(vocab_matrix)
        normalized = vocab_matrix.copy().astype(np.float32)
        normalized.data /= np.float32(slide_mean)

        csc = normalized.tocsc()

        for index, gene in enumerate(vocab_names):
            start, end = csc.indptr[index], csc.indptr[index + 1]
            values = csc.data[start:end]
            if values.size == 0:
                continue

            canonical = canonical_gene_name(gene)
            sums[canonical] = sums.get(canonical, 0.0) + float(
                np.sum(values, dtype=np.float64)
            )
            counts[canonical] = counts.get(canonical, 0) + int(values.size)

        per_slice[slice_id] = {
            "source_h5ad": path,
            **qc_meta,
            "slide_nonzero_mean": float(slide_mean),
            "duplicate_source_symbols": duplicate_map,
        }

        print(
            f"  paper_mean {slice_id}: genes={len(vocab_names)}, "
            f"slide_nonzero_mean={slide_mean:.6g}"
        )

    gene_means = {
        gene: float(sums[gene] / counts[gene])
        for gene in sums
        if counts.get(gene, 0) > 0
    }

    if not gene_means:
        raise RuntimeError("No training gene means could be estimated")

    return gene_means, {
        "training_slices": list(training_slices),
        "genes_with_training_mean": int(len(gene_means)),
        "per_slice": per_slice,
    }


# =============================================================================
# scGPT
# =============================================================================

def load_scgpt_embedder(repo_dir: str):
    if not os.path.isdir(repo_dir):
        raise NotADirectoryError(repo_dir)

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    import scgpt_spatial
    import scgpt_spatial.data_collator
    import scgpt_spatial.preprocess

    original_binning = scgpt_spatial.preprocess.binning

    def safe_binning(row, *args, **kwargs):
        return row if len(row) == 0 else original_binning(
            row,
            *args,
            **kwargs,
        )

    scgpt_spatial.preprocess.binning = safe_binning
    scgpt_spatial.data_collator.binning = safe_binning

    return scgpt_spatial.tasks.embed_data


def make_scgpt_input(
    source_adata: Any,
    matrix: sp.csr_matrix,
    gene_names: Sequence[str],
    slice_id: str,
):
    from anndata import AnnData
    import pandas as pd

    var = pd.DataFrame(
        index=pd.Index(list(gene_names), name="gene")
    )
    var["feature_name"] = list(gene_names)

    obs = source_adata.obs.copy()
    obs["batch_id"] = slice_id

    return AnnData(
        X=matrix,
        obs=obs,
        var=var,
    )


def resolve_effective_max_length(
    matrix: sp.csr_matrix,
    requested_max_length: int,
) -> Tuple[int, Dict[str, Any]]:
    nonzero_per_spot = np.asarray(
        matrix.getnnz(axis=1)
    ).ravel().astype(np.int64)

    maximum = int(nonzero_per_spot.max()) if len(nonzero_per_spot) else 0
    median = float(np.median(nonzero_per_spot)) if len(nonzero_per_spot) else 0.0

    if int(requested_max_length) == 0:
        effective = max(2, maximum + 1)
        mode = "auto_full_nonzero_plus_cls"
    else:
        if requested_max_length < 2:
            raise ValueError("--max-length must be 0 or at least 2")
        effective = int(requested_max_length)
        mode = "fixed"

    gene_capacity = effective - 1
    truncated_spots = int(
        np.count_nonzero(nonzero_per_spot > gene_capacity)
    )

    return effective, {
        "requested_max_length": int(requested_max_length),
        "effective_max_length": int(effective),
        "gene_token_capacity": int(gene_capacity),
        "median_nonzero_genes_per_spot": median,
        "max_nonzero_genes_per_spot": maximum,
        "spots_exceeding_context_capacity": truncated_spots,
        "context_length_mode": mode,
    }


def embed_slice(
    embed_data,
    scgpt_input: Any,
    model_dir: str,
    batch_size: int,
    max_length: int,
    slice_id: str,
) -> np.ndarray:
    embedded = embed_data(
        scgpt_input,
        model_dir,
        gene_col="feature_name",
        obs_to_save="batch_id",
        max_length=int(max_length),
        batch_size=int(batch_size),
        return_new_adata=True,
        use_fast_transformer=False,
    )

    features = (
        embedded.X.toarray()
        if sp.issparse(embedded.X)
        else np.asarray(embedded.X)
    )
    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2:
        raise RuntimeError(
            f"[{slice_id}] invalid scGPT feature shape: {features.shape}"
        )
    if features.shape[0] != scgpt_input.n_obs:
        raise RuntimeError(
            f"[{slice_id}] feature rows {features.shape[0]} != "
            f"spots {scgpt_input.n_obs}"
        )
    if features.shape[1] != 512:
        raise RuntimeError(
            f"[{slice_id}] expected 512-D scGPT embeddings, "
            f"got {features.shape[1]}"
        )
    if not np.isfinite(features).all():
        raise RuntimeError(
            f"[{slice_id}] scGPT features contain NaN/Inf"
        )

    return features


# =============================================================================
# Outputs
# =============================================================================

def save_raw_features(
    output_root: str,
    cache_tag: str,
    slice_id: str,
    coordinates: np.ndarray,
    features: np.ndarray,
    input_genes: Sequence[str],
    input_gene_sha256: str,
    preprocessing_mode: str,
    source_h5ad: str,
    context_meta: Mapping[str, Any],
) -> str:
    if not output_root:
        raise KeyError(
            f"[{slice_id}] raw_scgpt_root is empty in the YAML"
        )

    output_dir = os.path.join(output_root, cache_tag)
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, f"{slice_id}_stfeat.npz")

    np.savez_compressed(
        path,
        coords=coordinates.astype(np.float32),
        features=features.astype(np.float32),
        feature_type=np.asarray("scgpt_all_available_genes"),
        feature_dim=np.int64(512),
        input_genes=np.asarray(input_genes, dtype="U"),
        input_gene_count=np.int64(len(input_genes)),
        input_gene_sha256=np.asarray(input_gene_sha256),
        preprocessing_mode=np.asarray(preprocessing_mode),
        source_h5ad=np.asarray(source_h5ad),
        max_length=np.int64(context_meta["effective_max_length"]),
        spots_exceeding_context_capacity=np.int64(
            context_meta["spots_exceeding_context_capacity"]
        ),
    )

    print(f"  saved raw scGPT: {path} | {features.shape}")
    return path


def write_context_cache(
    info: Mapping[str, Any],
    slice_id: str,
    coordinates: np.ndarray,
    features: np.ndarray,
    downsample: int,
    cache_tag: str,
    input_genes: Sequence[str],
    input_gene_sha256: str,
    preprocessing_mode: str,
    source_h5ad: str,
    context_meta: Mapping[str, Any],
) -> str:
    he_path = os.path.join(str(info["base"]), "he_resized.png")
    mask_path = os.path.join(str(info["base"]), "mask_resized.png")

    he = cv_imread(he_path, cv2.IMREAD_COLOR)
    if he is None:
        raise FileNotFoundError(he_path)

    tissue = cv_imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if tissue is None:
        raise FileNotFoundError(mask_path)

    height, width = he.shape[:2]
    thumb_h = max(1, height // int(downsample))
    thumb_w = max(1, width // int(downsample))

    thumb = cv2.cvtColor(
        cv2.resize(
            he,
            (thumb_w, thumb_h),
            interpolation=cv2.INTER_AREA,
        ),
        cv2.COLOR_BGR2RGB,
    )
    tissue_thumb = cv2.resize(
        tissue,
        (thumb_w, thumb_h),
        interpolation=cv2.INTER_NEAREST,
    )

    coords_thumb = coordinates.astype(np.float32).copy()
    coords_thumb[:, 0] *= float(thumb_w) / float(width)
    coords_thumb[:, 1] *= float(thumb_h) / float(height)

    context_root = str(info["context_root"])
    if not context_root:
        raise KeyError(
            f"[{slice_id}] context_cache_root is empty in the YAML"
        )

    output_dir = os.path.join(context_root, cache_tag)
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, f"{slice_id}_ctx.npz")

    np.savez_compressed(
        path,
        thumb=thumb,
        tissue_thumb=tissue_thumb,
        coords_thumb=coords_thumb,
        features=features.astype(np.float32),
        full_h=np.int64(height),
        full_w=np.int64(width),
        ds=np.int64(downsample),
        cache_tag=np.asarray(cache_tag),
        feature_type=np.asarray("scgpt_all_available_genes"),
        feature_dim=np.int64(512),
        input_gene_count=np.int64(len(input_genes)),
        input_gene_sha256=np.asarray(input_gene_sha256),
        preprocessing_mode=np.asarray(preprocessing_mode),
        source_h5ad=np.asarray(source_h5ad),
        max_length=np.int64(context_meta["effective_max_length"]),
        spots_exceeding_context_capacity=np.int64(
            context_meta["spots_exceeding_context_capacity"]
        ),
    )

    print(f"  saved context cache: {path} | {features.shape}")
    return path


# =============================================================================
# CLI and main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract direct 512-D scGPT-spatial embeddings from every "
            "vocabulary-compatible gene in each original slice. No dictionary "
            "and no PCA."
        )
    )

    parser.add_argument(
        "--config",
        default="../engine/cross_dataset.yaml",
    )
    parser.add_argument(
        "--slices",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--normalization-train-slices",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=("paper_mean", "scanpy_full", "none"),
        default="paper_mean",
    )
    parser.add_argument(
        "--input-state",
        choices=("raw", "normalized", "log1p"),
        default="raw",
    )
    parser.add_argument(
        "--counts-layer",
        default=None,
    )
    parser.add_argument(
        "--target-sum",
        type=float,
        default=1e4,
    )
    parser.add_argument(
        "--min-gene-fraction",
        type=float,
        default=0.0003,
        help=(
            "Retain vocabulary genes expressed in at least this fraction "
            "of spots within each slice. Use 0 to retain every detected "
            "vocabulary gene."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=601,
        help=(
            "Total sequence length including CLS. Default 601 matches the "
            "model's 600-gene pretraining context. Use 0 to automatically "
            "include every expressed gene per spot."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--cache-tag",
        default="all_genes_scgpt512_ds4",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= float(args.min_gene_fraction) <= 1.0:
        raise ValueError("--min-gene-fraction must be in [0, 1]")

    config_path = resolve_config_path(args.config)
    cfg = load_yaml(config_path)
    registry = slice_registry(cfg)

    active_slices = [
        str(value)
        for value in cfg.get(
            "experiment",
            {},
        ).get("active_slices", [])
    ]
    slices = [
        str(value)
        for value in (args.slices or active_slices)
    ]

    if not slices:
        raise ValueError("No slices selected")

    missing_registry = sorted(set(slices) - set(registry))
    if missing_registry:
        raise KeyError(
            f"Slices missing from dataset registry: {missing_registry}"
        )

    normalization_train_slices = [
        str(value)
        for value in (
            args.normalization_train_slices
            or [
                slice_id
                for slice_id in DEFAULT_STHBC_SLICES
                if slice_id in slices
            ]
        )
    ]

    if (
        args.preprocess_mode == "paper_mean"
        and not normalization_train_slices
    ):
        raise ValueError(
            "paper_mean requires normalization-training slices"
        )

    unknown_train = sorted(
        set(normalization_train_slices) - set(slices)
    )
    if unknown_train:
        raise ValueError(
            "Normalization-training slices are not selected: "
            f"{unknown_train}"
        )

    preprocess_cfg = dict(cfg.get("preprocess", {}))
    model_dir = str(preprocess_cfg.get("scgpt_model_dir", ""))
    repo_dir = str(preprocess_cfg.get("scgpt_repo_dir", ""))

    if not model_dir or not repo_dir:
        raise KeyError(
            "cfg.preprocess.scgpt_model_dir and scgpt_repo_dir are required"
        )

    batch_size = int(
        args.batch_size or preprocess_cfg.get("batch_size", 64)
    )

    configured_st_dim = int(
        cfg.get("dims", {}).get("st_dim", 512)
    )
    if configured_st_dim != 512:
        print(
            f"[warn] cfg.dims.st_dim={configured_st_dim}, but this script "
            "writes direct 512-D scGPT embeddings. Set dims.st_dim: 512."
        )

    if (
        args.preprocess_mode == "paper_mean"
        and args.input_state != "raw"
    ):
        raise ValueError(
            "paper_mean should be applied to raw/count-scale values"
        )

    vocab, canonical_to_vocab_name = load_vocab(model_dir)
    embed_data = load_scgpt_embedder(repo_dir)

    metadata_root = Path(
        str(
            cfg.get("paths", {}).get(
                "preprocess_metadata_dir",
                config_path.parent,
            )
        )
    ) / args.cache_tag
    metadata_root.mkdir(parents=True, exist_ok=True)

    done_path = metadata_root / "preprocess_metadata.json"
    if done_path.exists() and not args.overwrite:
        print(f"Skip existing metadata: {done_path}")
        print("Use --overwrite to regenerate features and caches.")
        return

    print("=" * 100)
    print("All-available-gene scGPT-spatial extraction")
    print("=" * 100)
    print(f"Config                        : {config_path}")
    print(f"Slices                        : {slices}")
    print(f"Normalization training slices : {normalization_train_slices}")
    print(f"Preprocessing mode            : {args.preprocess_mode}")
    print(f"Input state                   : {args.input_state}")
    print(f"Minimum gene fraction         : {args.min_gene_fraction}")
    print(f"Requested max length          : {args.max_length}")
    print(f"Cache tag                     : {args.cache_tag}")
    print("Fixed dictionary               : disabled")
    print("PCA                            : disabled")
    print("Output feature dimension       : 512")

    paper_gene_means: Optional[Dict[str, float]] = None
    paper_statistics: Optional[Dict[str, Any]] = None

    if args.preprocess_mode == "paper_mean":
        print(
            "\nEstimating training-only gene means for all "
            "vocabulary-compatible genes..."
        )
        paper_gene_means, paper_statistics = compute_paper_gene_means(
            registry=registry,
            training_slices=normalization_train_slices,
            counts_layer=args.counts_layer,
            vocab=vocab,
            canonical_to_vocab_name=canonical_to_vocab_name,
            min_gene_fraction=float(args.min_gene_fraction),
        )

        mean_genes = sorted(paper_gene_means)
        np.savez_compressed(
            metadata_root / "paper_mean_normalization_stats.npz",
            genes=np.asarray(
                [canonical_to_vocab_name.get(gene, gene) for gene in mean_genes],
                dtype="U",
            ),
            canonical_genes=np.asarray(mean_genes, dtype="U"),
            gene_nonzero_means=np.asarray(
                [paper_gene_means[gene] for gene in mean_genes],
                dtype=np.float32,
            ),
            training_slices=np.asarray(
                normalization_train_slices,
                dtype="U",
            ),
        )

    import scanpy as sc

    slice_audit: Dict[str, Any] = {}

    for slice_id in slices:
        print("\n" + "-" * 100)
        print(f"Processing {slice_id}")

        info = registry[slice_id]
        source_h5ad = resolve_h5ad_path(info)
        print(f"  source h5ad: {source_h5ad}")

        adata = sc.read_h5ad(source_h5ad)
        coordinates, coordinate_key = extract_coordinates(
            adata,
            slice_id,
        )

        full_matrix = get_expression_matrix(
            adata,
            args.counts_layer,
        )
        source_gene_names = resolve_gene_names(adata)

        if len(source_gene_names) != full_matrix.shape[1]:
            raise ValueError(
                f"[{slice_id}] gene-name count {len(source_gene_names)} "
                f"!= matrix columns {full_matrix.shape[1]}"
            )

        # Merge duplicate symbols before vocabulary matching.
        merged_matrix, merged_names, duplicate_map = (
            aggregate_duplicate_symbols(
                full_matrix,
                source_gene_names,
            )
        )

        normalization_meta: Dict[str, Any] = {}
        unseen_training_mean_genes: List[str] = []

        if args.preprocess_mode == "scanpy_full":
            if args.input_state == "raw":
                transformed_full, normalization_meta = (
                    library_normalize_log1p_full(
                        merged_matrix,
                        args.target_sum,
                    )
                )
            elif args.input_state == "normalized":
                transformed_full = merged_matrix.copy().astype(
                    np.float32
                )
                transformed_full.data = np.log1p(
                    transformed_full.data
                ).astype(np.float32)
                transformed_full.eliminate_zeros()
                normalization_meta = {
                    "operation": "complete_matrix_log1p_only"
                }
            else:
                transformed_full = merged_matrix
                normalization_meta = {
                    "operation": "none_input_already_log1p"
                }

            scgpt_matrix, scgpt_genes, qc_meta = retain_vocab_genes(
                matrix=transformed_full,
                gene_names=merged_names,
                vocab=vocab,
                canonical_to_vocab_name=canonical_to_vocab_name,
                min_gene_fraction=float(args.min_gene_fraction),
            )

        elif args.preprocess_mode == "paper_mean":
            scgpt_matrix, scgpt_genes, qc_meta = retain_vocab_genes(
                matrix=merged_matrix,
                gene_names=merged_names,
                vocab=vocab,
                canonical_to_vocab_name=canonical_to_vocab_name,
                min_gene_fraction=float(args.min_gene_fraction),
            )

            slide_mean_value = slide_nonzero_mean(scgpt_matrix)
            scgpt_matrix = scgpt_matrix.copy().astype(np.float32)
            scgpt_matrix.data /= np.float32(slide_mean_value)

            assert paper_gene_means is not None
            scgpt_matrix, unseen_training_mean_genes = (
                divide_columns_by_gene_means(
                    scgpt_matrix,
                    scgpt_genes,
                    paper_gene_means,
                )
            )

            normalization_meta = {
                "operation": "two_level_nonzero_mean_normalization",
                "slide_nonzero_mean": float(slide_mean_value),
                "genes_without_training_mean": int(
                    len(unseen_training_mean_genes)
                ),
            }

        else:
            scgpt_matrix, scgpt_genes, qc_meta = retain_vocab_genes(
                matrix=merged_matrix,
                gene_names=merged_names,
                vocab=vocab,
                canonical_to_vocab_name=canonical_to_vocab_name,
                min_gene_fraction=float(args.min_gene_fraction),
            )
            normalization_meta = {"operation": "none"}

        zero_spots = int(
            np.count_nonzero(
                np.asarray(scgpt_matrix.getnnz(axis=1)).ravel() == 0
            )
        )
        if zero_spots:
            print(
                f"  [warn] {zero_spots}/{adata.n_obs} spots have no "
                "non-zero vocabulary-compatible genes"
            )

        effective_max_length, context_meta = (
            resolve_effective_max_length(
                scgpt_matrix,
                args.max_length,
            )
        )

        if context_meta["spots_exceeding_context_capacity"] > 0:
            print(
                f"  [info] {context_meta['spots_exceeding_context_capacity']}/"
                f"{adata.n_obs} spots exceed the "
                f"{context_meta['gene_token_capacity']}-gene context capacity; "
                "scGPT will select/truncate genes for those spots."
            )

        if args.max_length == 0 and effective_max_length > 601:
            print(
                f"  [warn] auto full-gene max_length={effective_max_length} "
                "exceeds the model's 601-token pretraining setting and may "
                "consume substantial GPU memory."
            )

        scgpt_input = make_scgpt_input(
            source_adata=adata,
            matrix=scgpt_matrix,
            gene_names=scgpt_genes,
            slice_id=slice_id,
        )

        features = embed_slice(
            embed_data=embed_data,
            scgpt_input=scgpt_input,
            model_dir=model_dir,
            batch_size=batch_size,
            max_length=effective_max_length,
            slice_id=slice_id,
        )

        input_gene_sha256 = hashlib.sha256(
            "\n".join(scgpt_genes).encode("utf-8")
        ).hexdigest()

        raw_path = save_raw_features(
            output_root=str(info["raw_root"]),
            cache_tag=args.cache_tag,
            slice_id=slice_id,
            coordinates=coordinates,
            features=features,
            input_genes=scgpt_genes,
            input_gene_sha256=input_gene_sha256,
            preprocessing_mode=args.preprocess_mode,
            source_h5ad=source_h5ad,
            context_meta=context_meta,
        )

        context_path = write_context_cache(
            info=info,
            slice_id=slice_id,
            coordinates=coordinates,
            features=features,
            downsample=int(
                cfg.get("geometry", {}).get("downsample", 4)
            ),
            cache_tag=args.cache_tag,
            input_genes=scgpt_genes,
            input_gene_sha256=input_gene_sha256,
            preprocessing_mode=args.preprocess_mode,
            source_h5ad=source_h5ad,
            context_meta=context_meta,
        )

        feature_std = np.std(features, axis=0)

        slice_audit[slice_id] = {
            "dataset": info["dataset"],
            "source_h5ad": source_h5ad,
            "spots": int(adata.n_obs),
            "source_gene_entries": int(adata.n_vars),
            "unique_merged_genes": int(len(merged_names)),
            "coordinate_key": coordinate_key,
            "input_gene_count": int(len(scgpt_genes)),
            "input_gene_sha256": input_gene_sha256,
            "input_genes": list(scgpt_genes),
            "duplicate_source_symbols": duplicate_map,
            "zero_input_spots": zero_spots,
            "normalization": {
                **qc_meta,
                **normalization_meta,
            },
            "genes_without_training_mean": list(
                unseen_training_mean_genes
            ),
            "context_length": context_meta,
            "feature_shape": list(features.shape),
            "feature_global_mean": float(np.mean(features)),
            "feature_global_std": float(np.std(features)),
            "zero_variance_feature_dimensions": int(
                np.count_nonzero(feature_std < 1e-8)
            ),
            "raw_feature_path": raw_path,
            "context_cache_path": context_path,
        }

        print(
            f"  completed {slice_id}: input_genes={len(scgpt_genes)}, "
            f"features={features.shape}, "
            f"feature_std={np.std(features):.6g}"
        )

    metadata = {
        "protocol": (
            "all per-slice scGPT-vocabulary genes; no fixed dictionary; "
            "no fold-specific selection; no PCA"
        ),
        "cache_tag": args.cache_tag,
        "feature_type": "scgpt_all_available_genes",
        "feature_dim": 512,
        "selected_slices": slices,
        "normalization_training_slices": normalization_train_slices,
        "normalization_uses_external_slices": bool(
            set(normalization_train_slices)
            - set(DEFAULT_STHBC_SLICES)
        ),
        "preprocessing_mode": args.preprocess_mode,
        "input_state": args.input_state,
        "normalization_order": (
            "complete source matrix -> normalization -> merge symbols -> "
            "retain all scGPT-vocabulary genes"
        ),
        "min_gene_fraction": float(args.min_gene_fraction),
        "requested_max_length": int(args.max_length),
        "model_native_context_note": (
            "The model was pretrained with a maximum 600-gene context. "
            "All vocabulary-compatible slice genes are retained as candidates, "
            "while individual spots are capped by max_length unless max_length=0."
        ),
        "fixed_dictionary_used": False,
        "pca_used": False,
        "paper_mean_statistics": paper_statistics,
        "slice_audit": slice_audit,
    }

    with done_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("Completed all-gene scGPT preprocessing")
    print("=" * 100)
    print(f"Metadata: {done_path}")
    print(f"Cache tag: {args.cache_tag}")
    print(
        "The downstream ST encoder must use input dimension 512 and read "
        "the `features` array directly from each context cache."
    )


if __name__ == "__main__":
    main()
