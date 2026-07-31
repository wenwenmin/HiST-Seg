"""Generate CRC all-gene 512-D scGPT-spatial context caches.

This preprocessor is designed for the multimodal-context PathST model.  It
replaces the legacy CRC pipeline that selected per-slice HVGs and later reduced
scGPT features with PCA.

For every CRC slice it performs:

1. read ``{slice}_cropped.h5ad`` and ``obsm['spatial_cropped']``;
2. read the complete expression matrix;
3. merge duplicate gene symbols;
4. normalize each spot on the complete matrix to ``target_sum`` and apply
   ``log1p`` (slice-local preprocessing, so no held-out slice contributes to
   another fold's statistics);
5. retain every QC-passing gene present in the scGPT vocabulary;
6. run scGPT-spatial and keep the native 512-D spot embedding without PCA;
7. save a raw feature file and the exact ``*_ctx.npz`` cache consumed by
   the current PathST project.

The output cache layout is:

``context_cache_root/all_genes_scgpt512_ds4/{slice}_ctx.npz``
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
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import certifi
import cv2
import numpy as np
import scipy.sparse as sp
import yaml


CRC_SLICES = (
    "SN048_1", "SN048_2", "SN048_3", "SN048_4",
    "SN84_1", "SN84_2", "SN123_1", "SN123_2",
    "SN123_4", "SN124_1", "SN124_2", "SN124_4",
)

GENE_COLUMN_CANDIDATES = (
    "feature_name", "gene_name", "gene_symbols", "symbol",
    "gene_symbol", "features", "SYMBOL",
)

if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid: []
if not hasattr(np, "float"):
    np.float = float
warnings.filterwarnings("ignore", category=RuntimeWarning)


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


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML root: {path}")
    return cfg


def cv_imread(path: str, flags: int) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def clean_gene_name(value: Any) -> str:
    name = str(value).strip()
    return "" if name.lower() in {"", "nan", "none"} else name


def canonical_gene_name(value: Any) -> str:
    return clean_gene_name(value).upper()


def resolve_gene_names(adata: Any) -> np.ndarray:
    for column in GENE_COLUMN_CANDIDATES:
        if column in adata.var.columns:
            values = adata.var[column].astype(str).to_numpy()
            cleaned = np.asarray([clean_gene_name(value) for value in values], dtype=object)
            if np.count_nonzero(cleaned != ""):
                return cleaned
    return np.asarray([clean_gene_name(value) for value in adata.var_names], dtype=object)


def get_expression_matrix(adata: Any, counts_layer: str | None) -> sp.csr_matrix:
    if counts_layer:
        if counts_layer not in adata.layers:
            raise KeyError(f"Requested layer {counts_layer!r} is absent")
        matrix = adata.layers[counts_layer]
    else:
        matrix = adata.X
    if sp.issparse(matrix):
        result = matrix.tocsr().astype(np.float32, copy=True)
    else:
        result = sp.csr_matrix(np.asarray(matrix, dtype=np.float32))
    if result.data.size and not np.isfinite(result.data).all():
        raise ValueError("Expression matrix contains NaN/Inf")
    if result.data.size and float(result.data.min()) < 0:
        raise ValueError("Expression matrix contains negative values; use raw counts")
    result.eliminate_zeros()
    return result


def aggregate_duplicate_symbols(
    matrix: sp.csr_matrix,
    source_gene_names: Sequence[Any],
) -> Tuple[sp.csr_matrix, List[str], Dict[str, int]]:
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
        raise RuntimeError("No valid gene symbols found")

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
    duplicates = {
        output_names[canonical_to_column[canonical]]: count
        for canonical, count in hit_count.items()
        if count > 1
    }
    return merged, output_names, duplicates


def normalize_total_log1p(
    matrix: sp.csr_matrix,
    target_sum: float,
) -> Tuple[sp.csr_matrix, Dict[str, Any]]:
    if target_sum <= 0:
        raise ValueError("target_sum must be positive")
    row_sums = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    nonempty = row_sums > 0
    scale = np.zeros_like(row_sums, dtype=np.float64)
    scale[nonempty] = float(target_sum) / row_sums[nonempty]
    normalized = (sp.diags(scale.astype(np.float32)) @ matrix).tocsr().astype(np.float32)
    normalized.data = np.log1p(normalized.data).astype(np.float32)
    normalized.eliminate_zeros()
    return normalized, {
        "operation": "complete_matrix_normalize_total_then_log1p",
        "target_sum": float(target_sum),
        "empty_spots": int(np.count_nonzero(~nonempty)),
        "median_library_size": float(np.median(row_sums[nonempty])) if np.any(nonempty) else 0.0,
    }


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


def retain_vocab_genes(
    matrix: sp.csr_matrix,
    gene_names: Sequence[str],
    vocab: set[str],
    canonical_to_vocab_name: Mapping[str, str],
    min_gene_fraction: float,
) -> Tuple[sp.csr_matrix, List[str], Dict[str, Any]]:
    indices: List[int] = []
    names: List[str] = []
    for index, gene in enumerate(gene_names):
        canonical = canonical_gene_name(gene)
        if canonical in vocab:
            indices.append(index)
            names.append(canonical_to_vocab_name[canonical])
    if not indices:
        raise RuntimeError("No genes overlap the scGPT vocabulary")

    selected = matrix[:, np.asarray(indices, dtype=np.int64)].tocsr()
    min_spots = max(1, int(np.ceil(float(min_gene_fraction) * matrix.shape[0])))
    detection = np.asarray(selected.getnnz(axis=0)).ravel()
    keep = detection >= min_spots
    if not np.any(keep):
        raise RuntimeError(f"No genes pass min detection count {min_spots}")
    kept = np.flatnonzero(keep)
    retained = selected[:, kept].tocsr().astype(np.float32)
    retained.eliminate_zeros()
    retained_names = [names[index] for index in kept]
    return retained, retained_names, {
        "vocab_overlap_before_qc": int(len(indices)),
        "min_gene_fraction": float(min_gene_fraction),
        "min_gene_spots": int(min_spots),
        "retained_vocab_genes": int(len(retained_names)),
    }


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
        return row if len(row) == 0 else original_binning(row, *args, **kwargs)

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

    obs = source_adata.obs.copy()
    obs["batch_id"] = slice_id
    var = pd.DataFrame(index=pd.Index(list(gene_names), name="gene"))
    var["feature_name"] = list(gene_names)
    return AnnData(X=matrix, obs=obs, var=var)


def resolve_max_length(matrix: sp.csr_matrix, requested: int) -> Tuple[int, Dict[str, Any]]:
    nonzero = np.asarray(matrix.getnnz(axis=1)).ravel().astype(np.int64)
    maximum = int(nonzero.max()) if len(nonzero) else 0
    median = float(np.median(nonzero)) if len(nonzero) else 0.0
    if requested == 0:
        effective = max(2, maximum + 1)
        mode = "all_nonzero_plus_cls"
    else:
        if requested < 2:
            raise ValueError("max_length must be 0 or >=2")
        effective = int(requested)
        mode = "fixed"
    capacity = effective - 1
    return effective, {
        "requested_max_length": int(requested),
        "effective_max_length": int(effective),
        "gene_token_capacity": int(capacity),
        "median_nonzero_genes_per_spot": median,
        "max_nonzero_genes_per_spot": maximum,
        "spots_exceeding_context_capacity": int(np.count_nonzero(nonzero > capacity)),
        "context_length_mode": mode,
    }


def embed_slice(
    embed_data,
    adata: Any,
    model_dir: str,
    batch_size: int,
    max_length: int,
    slice_id: str,
) -> np.ndarray:
    embedded = embed_data(
        adata,
        model_dir,
        gene_col="feature_name",
        obs_to_save="batch_id",
        max_length=int(max_length),
        batch_size=int(batch_size),
        return_new_adata=True,
        use_fast_transformer=False,
    )
    features = embedded.X.toarray() if sp.issparse(embedded.X) else np.asarray(embedded.X)
    features = np.asarray(features, dtype=np.float32)
    if features.shape != (adata.n_obs, 512):
        raise RuntimeError(f"[{slice_id}] expected {(adata.n_obs, 512)}, got {features.shape}")
    if not np.isfinite(features).all():
        raise RuntimeError(f"[{slice_id}] embedding contains NaN/Inf")
    return features


def save_outputs(
    slice_id: str,
    coordinates: np.ndarray,
    features: np.ndarray,
    genes: Sequence[str],
    image: np.ndarray,
    tissue: np.ndarray,
    source_h5ad: str,
    raw_root: str,
    context_root: str,
    cache_tag: str,
    downsample: int,
    context_meta: Mapping[str, Any],
) -> Tuple[str, str]:
    raw_dir = os.path.join(raw_root, cache_tag)
    context_dir = os.path.join(context_root, cache_tag)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(context_dir, exist_ok=True)

    gene_hash = hashlib.sha256("\n".join(genes).encode("utf-8")).hexdigest()
    raw_path = os.path.join(raw_dir, f"{slice_id}_stfeat.npz")
    np.savez_compressed(
        raw_path,
        coords=coordinates.astype(np.float32),
        features=features.astype(np.float32),
        feature_type=np.asarray("scgpt_all_available_genes"),
        feature_dim=np.int64(512),
        input_genes=np.asarray(genes, dtype="U"),
        input_gene_count=np.int64(len(genes)),
        input_gene_sha256=np.asarray(gene_hash),
        preprocessing_mode=np.asarray("scanpy_full"),
        source_h5ad=np.asarray(source_h5ad),
        max_length=np.int64(context_meta["effective_max_length"]),
        spots_exceeding_context_capacity=np.int64(context_meta["spots_exceeding_context_capacity"]),
    )

    height, width = image.shape[:2]
    thumb_h = max(1, height // int(downsample))
    thumb_w = max(1, width // int(downsample))
    thumb = cv2.cvtColor(
        cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA),
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

    context_path = os.path.join(context_dir, f"{slice_id}_ctx.npz")
    np.savez_compressed(
        context_path,
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
        input_gene_count=np.int64(len(genes)),
        input_gene_sha256=np.asarray(gene_hash),
        preprocessing_mode=np.asarray("scanpy_full"),
        source_h5ad=np.asarray(source_h5ad),
        max_length=np.int64(context_meta["effective_max_length"]),
        spots_exceeding_context_capacity=np.int64(context_meta["spots_exceeding_context_capacity"]),
    )
    return raw_path, context_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../engine/crc_loso.yaml")
    parser.add_argument("--slices", nargs="+", default=None)
    parser.add_argument("--counts_layer", default=None)
    parser.add_argument("--target_sum", type=float, default=1e4)
    parser.add_argument("--min_gene_fraction", type=float, default=0.0003)
    parser.add_argument("--max_length", type=int, default=601)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--cache_tag", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).resolve().parent / args.config
    cfg = load_yaml(config_path)
    dataset_cfg = dict(dict(cfg["datasets"])["CRC"])
    preprocess_cfg = dict(cfg.get("preprocess", {}))

    slices = [str(value) for value in (args.slices or dataset_cfg.get("slices", CRC_SLICES))]
    unknown = sorted(set(slices) - set(CRC_SLICES))
    if unknown:
        raise ValueError(f"Unknown CRC slices: {unknown}")

    repo_dir = str(preprocess_cfg["scgpt_repo_dir"])
    model_dir = str(preprocess_cfg["scgpt_model_dir"])
    batch_size = int(args.batch_size or preprocess_cfg.get("batch_size", 64))
    cache_tag = str(args.cache_tag or preprocess_cfg.get("cache_tag", "all_genes_scgpt512_ds4"))
    downsample = int(dict(cfg["geometry"]).get("downsample", 4))

    h5ad_dir = str(dataset_cfg["h5ad_dir"])
    image_dir = str(dataset_cfg["image_dir"])
    tissue_dir = str(dataset_cfg["tissue_dir"])
    raw_root = str(dataset_cfg["raw_scgpt_root"])
    context_root = str(dataset_cfg["context_cache_root"])

    vocab, canonical_to_vocab_name = load_vocab(model_dir)
    embed_data = load_scgpt_embedder(repo_dir)

    metadata_dir = Path(str(dict(cfg.get("paths", {})).get(
        "preprocess_metadata_dir",
        os.path.join(context_root, cache_tag),
    )))
    metadata_dir.mkdir(parents=True, exist_ok=True)
    audit: Dict[str, Any] = {}

    import scanpy as sc

    print("=" * 100)
    print("CRC all-gene scGPT-spatial 512-D preprocessing")
    print("=" * 100)
    print(f"Slices: {slices}")
    print("Gene intersection: disabled")
    print("Per-slice HVG selection: disabled")
    print("PCA: disabled")
    print("Output dimension: 512")
    print("Normalization: complete matrix normalize_total + log1p (per slice)")

    for slice_id in slices:
        source_h5ad = os.path.join(h5ad_dir, f"{slice_id}_cropped.h5ad")
        image_path = os.path.join(image_dir, f"{slice_id}.png")
        tissue_path = os.path.join(tissue_dir, f"{slice_id}.png")
        raw_path_expected = os.path.join(raw_root, cache_tag, f"{slice_id}_stfeat.npz")
        context_path_expected = os.path.join(context_root, cache_tag, f"{slice_id}_ctx.npz")
        if (
            not args.overwrite
            and os.path.exists(raw_path_expected)
            and os.path.exists(context_path_expected)
        ):
            print(f"[{slice_id}] skip existing outputs")
            continue

        if not os.path.exists(source_h5ad):
            raise FileNotFoundError(source_h5ad)
        image = cv_imread(image_path, cv2.IMREAD_COLOR)
        tissue = cv_imread(tissue_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        if tissue is None:
            tissue = np.full(image.shape[:2], 255, dtype=np.uint8)
        if tissue.shape != image.shape[:2]:
            raise ValueError(f"[{slice_id}] tissue/image shape mismatch")

        print("\n" + "-" * 100)
        print(f"Processing {slice_id}: {source_h5ad}")
        adata = sc.read_h5ad(source_h5ad)
        if "spatial_cropped" not in adata.obsm:
            raise KeyError(f"[{slice_id}] missing adata.obsm['spatial_cropped']")
        coordinates = np.asarray(adata.obsm["spatial_cropped"], dtype=np.float32)
        if coordinates.shape != (adata.n_obs, 2):
            raise ValueError(f"[{slice_id}] invalid coordinates: {coordinates.shape}")
        if not np.isfinite(coordinates).all():
            raise ValueError(f"[{slice_id}] coordinates contain NaN/Inf")

        matrix = get_expression_matrix(adata, args.counts_layer)
        gene_names = resolve_gene_names(adata)
        if len(gene_names) != matrix.shape[1]:
            raise ValueError(f"[{slice_id}] gene-name count mismatch")
        merged, merged_names, duplicates = aggregate_duplicate_symbols(matrix, gene_names)
        normalized, normalization_meta = normalize_total_log1p(merged, args.target_sum)
        scgpt_matrix, scgpt_genes, qc_meta = retain_vocab_genes(
            matrix=normalized,
            gene_names=merged_names,
            vocab=vocab,
            canonical_to_vocab_name=canonical_to_vocab_name,
            min_gene_fraction=float(args.min_gene_fraction),
        )
        zero_spots = int(np.count_nonzero(np.asarray(scgpt_matrix.getnnz(axis=1)).ravel() == 0))
        effective_max_length, context_meta = resolve_max_length(scgpt_matrix, int(args.max_length))
        if context_meta["spots_exceeding_context_capacity"]:
            print(
                f"[{slice_id}] {context_meta['spots_exceeding_context_capacity']}/{adata.n_obs} "
                f"spots exceed {context_meta['gene_token_capacity']} gene tokens"
            )

        scgpt_input = make_scgpt_input(
            source_adata=adata,
            matrix=scgpt_matrix,
            gene_names=scgpt_genes,
            slice_id=slice_id,
        )
        features = embed_slice(
            embed_data=embed_data,
            adata=scgpt_input,
            model_dir=model_dir,
            batch_size=batch_size,
            max_length=effective_max_length,
            slice_id=slice_id,
        )
        raw_path, context_path = save_outputs(
            slice_id=slice_id,
            coordinates=coordinates,
            features=features,
            genes=scgpt_genes,
            image=image,
            tissue=tissue,
            source_h5ad=source_h5ad,
            raw_root=raw_root,
            context_root=context_root,
            cache_tag=cache_tag,
            downsample=downsample,
            context_meta=context_meta,
        )

        in_bounds = (
            (coordinates[:, 0] >= 0)
            & (coordinates[:, 0] < image.shape[1])
            & (coordinates[:, 1] >= 0)
            & (coordinates[:, 1] < image.shape[0])
        )
        audit[slice_id] = {
            "source_h5ad": source_h5ad,
            "spots": int(adata.n_obs),
            "source_gene_entries": int(adata.n_vars),
            "unique_merged_genes": int(len(merged_names)),
            "duplicate_source_symbols": duplicates,
            "input_gene_count": int(len(scgpt_genes)),
            "zero_input_spots": zero_spots,
            "coordinates_out_of_bounds": int(np.count_nonzero(~in_bounds)),
            "normalization": {**normalization_meta, **qc_meta},
            "context_length": context_meta,
            "feature_shape": list(features.shape),
            "feature_mean": float(np.mean(features)),
            "feature_std": float(np.std(features)),
            "raw_feature_path": raw_path,
            "context_cache_path": context_path,
        }
        print(
            f"[{slice_id}] genes={len(scgpt_genes)}, embedding={features.shape}, "
            f"out_of_bounds={np.count_nonzero(~in_bounds)}"
        )
        print(f"  raw: {raw_path}")
        print(f"  ctx: {context_path}")

    metadata = {
        "protocol": "CRC all available vocabulary-compatible genes; no common-gene intersection; no per-slice HVG; no PCA",
        "feature_dim": 512,
        "cache_tag": cache_tag,
        "selected_slices": slices,
        "preprocessing_mode": "scanpy_full",
        "target_sum": float(args.target_sum),
        "min_gene_fraction": float(args.min_gene_fraction),
        "requested_max_length": int(args.max_length),
        "slice_audit": audit,
    }
    metadata_path = metadata_dir / "crc_all_genes_scgpt512_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("CRC scGPT preprocessing completed")
    print("=" * 100)
    print(f"Metadata: {metadata_path}")
    print(f"Context cache: {os.path.join(context_root, cache_tag)}")


if __name__ == "__main__":
    main()
