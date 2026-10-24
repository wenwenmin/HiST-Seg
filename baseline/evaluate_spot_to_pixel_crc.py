# Path configuration:
# Replace the repository-relative defaults (data/, outputs/, and workspaces/)
# for your environment. See readme_baseline.txt at the baseline package root.
# Evaluation protocol:
# Predictions must first be generated from the native, original-scale spot data.
# This script uses cropped_data and adata.obsm['spatial_cropped'] only to map
# those predictions into the common aligned CRC pixel-level evaluation space.
'Public baseline experiment module. Configure file-system paths as described in readme_baseline.txt.'

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import scanpy as sc
from scipy import ndimage
from scipy.spatial import cKDTree


# =============================================================================
# Constants
# =============================================================================

CRC_SLICES = [
    "SN048_1", "SN048_2", "SN048_3", "SN048_4",
    "SN84_1", "SN84_2",
    "SN123_1", "SN123_2", "SN123_4",
    "SN124_1", "SN124_2", "SN124_4",
]

SUPPORTED_MODELS = [
    "BLEEP",
    "MCGAE",
    "MEATRD",
    "SpaCRD",
]

DEFAULT_SEEDS = [
    42,
    123,
    567,
]

METRIC_KEYS = [
    "Target_IoU",
    "Dice",
    "Recall",
    "Precision",
    "HD95",
]


# =============================================================================
# Basic I/O
# =============================================================================

def cv_imread(
    path: str,
    flags: int = cv2.IMREAD_COLOR,
) -> np.ndarray:
    'Implementation of cv_imread for this baseline experiment.'
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found:{path}"
        )

    array = np.fromfile(
        path,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        array,
        flags,
    )

    if image is None:
        raise RuntimeError(
            f"OpenCV image decoding failed:{path}"
        )

    return image


def cv_imwrite(
    path: str,
    image: np.ndarray,
) -> None:
    'Implementation of cv_imwrite for this baseline experiment.'
    path = str(path)

    output_dir = os.path.dirname(
        path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    extension = os.path.splitext(
        path
    )[1]

    if not extension:
        raise ValueError(
            f"The output file is missing an extension:{path}"
        )

    ok, buffer = cv2.imencode(
        extension,
        image,
    )

    if not ok:
        raise RuntimeError(
            f"OpenCV image encoding failed:{path}"
        )

    buffer.tofile(
        path
    )


def write_csv(
    path: str,
    rows: List[Dict[str, object]],
) -> None:
    'Implementation of write_csv for this baseline experiment.'
    if not rows:
        return

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(
                    key
                )
                fieldnames.append(
                    key
                )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )


# =============================================================================
# CRC paths
# =============================================================================

def resolve_crc_paths(
    slice_name: str,
    adata_base_dir: str,
    he_base_dir: str,
    label_base_dir: str,
    tissue_base_dir: str,
) -> Dict[str, str]:
    """
    Resolve one CRC slice using the same path layout as the original script.
    """
    return {
        "he": os.path.join(
            he_base_dir,
            f"{slice_name}.png",
        ),
        "label": os.path.join(
            label_base_dir,
            f"{slice_name}.png",
        ),
        "tissue": os.path.join(
            tissue_base_dir,
            f"{slice_name}.png",
        ),
        "adata": os.path.join(
            adata_base_dir,
            f"{slice_name}_cropped.h5ad",
        ),
    }


def validate_crc_paths(
    slice_name: str,
    paths: Dict[str, str],
) -> None:
    required = {
        "H&E": paths["he"],
        "Label": paths["label"],
        "Tissue mask": paths["tissue"],
        "AnnData": paths["adata"],
    }

    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not os.path.exists(path)
    ]

    if missing:
        details = "\n".join(
            f"  - {item}"
            for item in missing
        )

        raise FileNotFoundError(
            f"[{slice_name}] Missing necessary files:\n"
            f"{details}"
        )


# =============================================================================
# Metrics
# =============================================================================

def compute_hd95(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
) -> float:
    'Implementation of compute_hd95 for this baseline experiment.'
    p = (
        pred.astype(bool)
        & valid.astype(bool)
    )

    g = (
        gt.astype(bool)
        & valid.astype(bool)
    )

    if (
        not p.any()
        or not g.any()
    ):
        return float(
            "nan"
        )

    structure = (
        ndimage.generate_binary_structure(
            2,
            1,
        )
    )

    p_surface = (
        p
        ^ ndimage.binary_erosion(
            p,
            structure=structure,
            border_value=0,
        )
    )

    g_surface = (
        g
        ^ ndimage.binary_erosion(
            g,
            structure=structure,
            border_value=0,
        )
    )

    if (
        not p_surface.any()
        or not g_surface.any()
    ):
        return float(
            "nan"
        )

    dist_to_g = (
        ndimage.distance_transform_edt(
            ~g_surface
        )
    )

    dist_to_p = (
        ndimage.distance_transform_edt(
            ~p_surface
        )
    )

    distances = np.concatenate(
        [
            dist_to_g[
                p_surface
            ],
            dist_to_p[
                g_surface
            ],
        ]
    )

    return float(
        np.percentile(
            distances,
            95.0,
        )
    )


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
) -> Dict[str, float]:
    """
    Pixel-level main-table metrics:
        Target_IoU, Dice, Recall, Precision, HD95.
    """
    if pred.shape != gt.shape:
        raise ValueError(
            f"Predictions are inconsistent with label dimensions:"
            f"pred={pred.shape}, gt={gt.shape}"
        )

    if valid.shape != gt.shape:
        raise ValueError(
            f"Organization mask is inconsistent with label dimensions:"
            f"valid={valid.shape}, gt={gt.shape}"
        )

    eps = 1e-7

    valid = valid.astype(
        bool
    )

    p = (
        (pred == 1)
        & valid
    )

    g = (
        (gt == 1)
        & valid
    )

    tp = int(
        np.sum(
            p & g
        )
    )

    fp = int(
        np.sum(
            p
            & ~g
            & valid
        )
    )

    fn = int(
        np.sum(
            ~p
            & g
            & valid
        )
    )

    tn = int(
        np.sum(
            ~p
            & ~g
            & valid
        )
    )

    target_iou = (
        tp
        / (
            tp
            + fp
            + fn
            + eps
        )
    )

    dice = (
        2.0
        * tp
        / (
            2.0
            * tp
            + fp
            + fn
            + eps
        )
    )

    recall = (
        tp
        / (
            tp
            + fn
            + eps
        )
    )

    precision = (
        tp
        / (
            tp
            + fp
            + eps
        )
    )

    hd95 = compute_hd95(
        p,
        g,
        valid,
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "Target_IoU": float(
            target_iou
        ),
        "Dice": float(
            dice
        ),
        "Recall": float(
            recall
        ),
        "Precision": float(
            precision
        ),
        "HD95": float(
            hd95
        ),
    }


# =============================================================================
# Prediction loading
# =============================================================================

def build_prediction_path(
    pred_root: str,
    model_name: str,
    slice_name: str,
    seed: int,
) -> str:
    """
    Keep exactly the same CRC prediction paths as the old script.
    """
    pred_base_dir = os.path.join(
        pred_root,
        model_name,
        "CRC",
    )

    if model_name == "BLEEP":
        return os.path.join(
            pred_base_dir,
            f"probs_{slice_name}_seed={seed}.txt",
        )

    if model_name == "SpaCRD":
        return os.path.join(
            pred_base_dir,
            f"{slice_name}_preds(Final)_seed={seed}.txt",
        )

    return os.path.join(
        pred_base_dir,
        f"{slice_name}_pred_seed={seed}.txt",
    )


def load_spot_outputs(
    pred_path: str,
    threshold: float = 0.5,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    bool,
]:
    """
    Load spot-level prediction values.

    Continuous probability outputs are thresholded at 0.5.
    Native binary 0/1 outputs are kept unchanged.
    """
    raw = np.asarray(
        np.loadtxt(
            pred_path,
            dtype=float,
        )
    )

    if raw.ndim == 0:
        raw = raw.reshape(
            1
        )

    elif raw.ndim == 2:
        
        raw = raw[
            :,
            -1,
        ]

    raw = raw.reshape(
        -1
    )

    if raw.size == 0:
        raise ValueError(
            f"Prediction file is empty:{pred_path}"
        )

    if not np.all(
        np.isfinite(
            raw
        )
    ):
        raise ValueError(
            f"Prediction file contains NaN or Inf:{pred_path}"
        )

    spot_scores = np.clip(
        raw,
        0.0,
        1.0,
    )

    is_binary_input = bool(
        np.all(
            np.isclose(
                spot_scores,
                0.0,
            )
            | np.isclose(
                spot_scores,
                1.0,
            )
        )
    )

    if is_binary_input:
        spot_predictions = (
            spot_scores > 0.0
        ).astype(
            np.uint8
        )

    else:
        spot_predictions = (
            spot_scores
            >= float(
                threshold
            )
        ).astype(
            np.uint8
        )

    return (
        spot_scores,
        spot_predictions,
        not is_binary_input,
    )


# =============================================================================
# Parameter-free nearest-spot Voronoi rasterization
# =============================================================================

def build_nearest_spot_index_map(
    coordinates: np.ndarray,
    tissue_mask: np.ndarray,
    row_chunk_size: int = 256,
) -> Tuple[
    np.ndarray,
    Dict[str, float],
]:
    """
    Assign every tissue pixel to its nearest ST spot.

    No:
        - spot radius;
        - circle rendering;
        - morphology;
        - Gaussian sigma;
        - GT-dependent parameter.

    The returned map stores the ORIGINAL spot index for every tissue pixel.
    Pixels outside tissue are -1.
    """
    coords = np.asarray(
        coordinates,
        dtype=np.float32,
    )

    if (
        coords.ndim != 2
        or coords.shape[1] < 2
    ):
        raise ValueError(
            f"coordinates should be [N,2+],"
            f"current shape={coords.shape}"
        )

    coords = coords[
        :,
        :2,
    ]

    height, width = (
        tissue_mask.shape
    )

    in_bounds = (
        np.isfinite(
            coords[
                :,
                0,
            ]
        )
        & np.isfinite(
            coords[
                :,
                1,
            ]
        )
        & (
            coords[
                :,
                0,
            ]
            >= 0
        )
        & (
            coords[
                :,
                0,
            ]
            < width
        )
        & (
            coords[
                :,
                1,
            ]
            >= 0
        )
        & (
            coords[
                :,
                1,
            ]
            < height
        )
    )

    valid_spot_ids = np.flatnonzero(
        in_bounds
    )

    if valid_spot_ids.size == 0:
        raise ValueError(
            'There are no valid ST spot coordinates within the image extent.'
        )

    tree = cKDTree(
        coords[
            valid_spot_ids
        ]
    )

    nearest_index = np.full(
        (
            height,
            width,
        ),
        -1,
        dtype=np.int32,
    )

    tissue_pixels = int(
        tissue_mask.sum()
    )

    assigned_pixels = 0

    for y0 in range(
        0,
        height,
        int(
            row_chunk_size
        ),
    ):
        y1 = min(
            height,
            y0
            + int(
                row_chunk_size
            ),
        )

        tissue_chunk = (
            tissue_mask[
                y0:y1
            ]
        )

        yy, xx = np.nonzero(
            tissue_chunk
        )

        if xx.size == 0:
            continue

        query_xy = np.column_stack(
            [
                xx.astype(
                    np.float32
                ),
                (
                    yy
                    + y0
                ).astype(
                    np.float32
                ),
            ]
        )

        try:
            _, local_nn = tree.query(
                query_xy,
                k=1,
                workers=-1,
            )

        except TypeError:
            # Compatibility with older SciPy.
            _, local_nn = tree.query(
                query_xy,
                k=1,
            )

        global_spot_ids = (
            valid_spot_ids[
                np.asarray(
                    local_nn,
                    dtype=np.int64,
                )
            ]
        )

        nearest_index[
            yy + y0,
            xx,
        ] = (
            global_spot_ids.astype(
                np.int32
            )
        )

        assigned_pixels += int(
            xx.size
        )

    diagnostics = {
        "Num_Spots_Total": int(
            len(
                coords
            )
        ),
        "Num_Spots_InBounds": int(
            valid_spot_ids.size
        ),
        "Tissue_Pixels": int(
            tissue_pixels
        ),
        "Assigned_Tissue_Pixels": int(
            assigned_pixels
        ),
        "Assigned_Fraction": float(
            assigned_pixels
            / max(
                1,
                tissue_pixels,
            )
        ),
    }

    return (
        nearest_index,
        diagnostics,
    )


def rasterize_binary_prediction(
    nearest_spot_index: np.ndarray,
    spot_predictions: np.ndarray,
) -> np.ndarray:
    """
    Convert spot-level binary predictions to a dense piecewise-constant mask.
    """
    pred = np.zeros(
        nearest_spot_index.shape,
        dtype=np.uint8,
    )

    valid = (
        nearest_spot_index
        >= 0
    )

    pred[
        valid
    ] = spot_predictions[
        nearest_spot_index[
            valid
        ]
    ]

    return pred


# =============================================================================
# Visualization
# =============================================================================

def apply_cancer_mask_overlay(
    image_bgr: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.45,
    color: Tuple[int, int, int] = (
        0,
        0,
        255,
    ),
) -> np.ndarray:
    """Overlay predicted cancer region on the H&E image."""
    if (
        image_bgr.shape[
            :2
        ]
        != pred_mask.shape
    ):
        raise ValueError(
            f"Image and mask dimensions are inconsistent:"
            f"image={image_bgr.shape[:2]}, "
            f"mask={pred_mask.shape}"
        )

    overlay = (
        image_bgr.copy()
        .astype(
            np.float32
        )
    )

    mask = (
        pred_mask.astype(
            bool
        )
    )

    color_array = np.asarray(
        color,
        dtype=np.float32,
    )

    overlay[
        mask
    ] = (
        overlay[
            mask
        ]
        * (
            1.0
            - alpha
        )
        + color_array
        * alpha
    )

    return np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )


# =============================================================================
# Statistics
# =============================================================================

def sample_std(
    values: Sequence[float],
) -> float:
    """
    Sample standard deviation across random seeds.
    """
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = array[
        np.isfinite(
            array
        )
    ]

    if array.size <= 1:
        return 0.0

    return float(
        np.std(
            array,
            ddof=1,
        )
    )


def metric_mean_std(
    seed_metrics: Sequence[
        Dict[
            str,
            float,
        ]
    ],
    key: str,
) -> Tuple[
    float,
    float,
]:
    values = np.asarray(
        [
            float(
                metrics[
                    key
                ]
            )
            for metrics
            in seed_metrics
            if np.isfinite(
                float(
                    metrics[
                        key
                    ]
                )
            )
        ],
        dtype=np.float64,
    )

    if values.size == 0:
        return (
            float(
                "nan"
            ),
            float(
                "nan"
            ),
        )

    return (
        float(
            np.mean(
                values
            )
        ),
        sample_std(
            values
        ),
    )


def format_mean_std(
    mean: float,
    std: float,
    hd95: bool = False,
) -> str:
    if not np.isfinite(
        mean
    ):
        return "NaN"

    if hd95:
        return (
            f"{mean:.2f}"
            f"±"
            f"{std:.2f}"
        )

    return (
        f"{mean:.4f}"
        f"±"
        f"{std:.4f}"
    )


# =============================================================================
# Single-slice evaluation
# =============================================================================

def evaluate_slice(
    slice_name: str,
    model_names: Sequence[str],
    pred_root: str,
    adata_base_dir: str,
    he_base_dir: str,
    label_base_dir: str,
    tissue_base_dir: str,
    output_root: str,
    seeds: Sequence[int],
    threshold: float,
    voronoi_row_chunk: int,
) -> Tuple[
    Dict[
        str,
        Dict[
            int,
            Dict[
                str,
                float,
            ],
        ],
    ],
    Dict[
        str,
        object,
    ],
]:
    """
    Evaluate one CRC slice.

    The expensive nearest-spot assignment is constructed once per slice
    and reused for all models and seeds.
    """
    print(
        "\n"
        + "="
        * 100
    )

    print(
        f"Slice: {slice_name}"
    )

    print(
        "="
        * 100
    )

    paths = resolve_crc_paths(
        slice_name=slice_name,
        adata_base_dir=adata_base_dir,
        he_base_dir=he_base_dir,
        label_base_dir=label_base_dir,
        tissue_base_dir=tissue_base_dir,
    )

    validate_crc_paths(
        slice_name,
        paths,
    )

    he_image = cv_imread(
        paths[
            "he"
        ],
        cv2.IMREAD_COLOR,
    )

    true_mask = (
        cv_imread(
            paths[
                "label"
            ],
            cv2.IMREAD_GRAYSCALE,
        )
        > 127
    ).astype(
        np.uint8
    )

    tissue_mask = (
        cv_imread(
            paths[
                "tissue"
            ],
            cv2.IMREAD_GRAYSCALE,
        )
        > 127
    )

    if (
        he_image.shape[
            :2
        ]
        != true_mask.shape
    ):
        raise ValueError(
            f"[{slice_name}] H&E does not match label size:"
            f"H&E={he_image.shape[:2]}, "
            f"label={true_mask.shape}"
        )

    if (
        tissue_mask.shape
        != true_mask.shape
    ):
        raise ValueError(
            f"[{slice_name}] tissue mask is inconsistent with label size:"
            f"tissue={tissue_mask.shape}, "
            f"label={true_mask.shape}"
        )

    if not np.any(
        tissue_mask
    ):
        raise ValueError(
            f"[{slice_name}] There is no valid tissue area in the tissue mask."
        )

    adata = sc.read_h5ad(
        paths[
            "adata"
        ]
    )

    spatial_key = (
        "spatial_cropped"
    )

    if (
        spatial_key
        not in adata.obsm
    ):
        raise KeyError(
            f"[{slice_name}] adata.obsm does not contain {spatial_key!r}; "
            f"available keys: {list(adata.obsm.keys())}"
        )

    coordinates = np.asarray(
        adata.obsm[
            spatial_key
        ],
        dtype=np.float32,
    )

    if (
        coordinates.ndim
        != 2
        or coordinates.shape[
            1
        ]
        < 2
    ):
        raise ValueError(
            f"[{slice_name}] spatial_cropped coordinate shape is abnormal:"
            f"{coordinates.shape}"
        )

    coordinates = (
        coordinates[
            :,
            :2,
        ]
    )

    nearest_index, geometry_info = (
        build_nearest_spot_index_map(
            coordinates=coordinates,
            tissue_mask=tissue_mask,
            row_chunk_size=voronoi_row_chunk,
        )
    )

    geometry_row: Dict[
        str,
        object,
    ] = {
        "Slice": slice_name,
        "HE_Path": paths[
            "he"
        ],
        "Label_Path": paths[
            "label"
        ],
        "Tissue_Mask_Path": paths[
            "tissue"
        ],
        "AnnData_Path": paths[
            "adata"
        ],
        "Spatial_Key": spatial_key,
        **geometry_info,
    }

    print(
        f"  image={true_mask.shape[1]}"
        f"x"
        f"{true_mask.shape[0]} | "
        f"spots="
        f"{geometry_info['Num_Spots_Total']} | "
        f"in-bounds="
        f"{geometry_info['Num_Spots_InBounds']} | "
        f"tissue pixels="
        f"{geometry_info['Tissue_Pixels']} | "
        f"assigned="
        f"{geometry_info['Assigned_Fraction']:.4f}"
    )

    slice_results: Dict[
        str,
        Dict[
            int,
            Dict[
                str,
                float,
            ],
        ],
    ] = {}

    for model_name in model_names:
        print(
            f"\n  [{model_name}]"
        )

        slice_results[
            model_name
        ] = {}

        for seed in seeds:
            pred_path = (
                build_prediction_path(
                    pred_root=pred_root,
                    model_name=model_name,
                    slice_name=slice_name,
                    seed=int(
                        seed
                    ),
                )
            )

            if not os.path.exists(
                pred_path
            ):
                print(
                    f"    seed={seed}: "
                    f"missing -> {pred_path}"
                )
                continue

            try:
                (
                    spot_scores,
                    spot_predictions,
                    has_continuous_scores,
                ) = load_spot_outputs(
                    pred_path=pred_path,
                    threshold=threshold,
                )

            except Exception as exc:
                print(
                    f"    seed={seed}: "
                    f"Read failed:{exc}"
                )
                continue

            
            
            if (
                len(
                    spot_predictions
                )
                != len(
                    coordinates
                )
            ):
                print(
                    f"    seed={seed}: "
                    f"Number of coordinates ({len(coordinates)})"
                    f"with the predicted number ({len(spot_predictions)})"
                    f"Inconsistent, skip."
                )
                continue

            pred_mask = (
                rasterize_binary_prediction(
                    nearest_index,
                    spot_predictions,
                )
            )

            result = compute_metrics(
                pred=pred_mask,
                gt=true_mask,
                valid=tissue_mask,
            )

            slice_results[
                model_name
            ][
                int(
                    seed
                )
            ] = result

            seed_output_dir = os.path.join(
                output_root,
                model_name,
                slice_name,
                f"seed_{seed}",
            )

            os.makedirs(
                seed_output_dir,
                exist_ok=True,
            )

            mask_255 = (
                pred_mask
                * 255
            ).astype(
                np.uint8
            )

            cv_imwrite(
                os.path.join(
                    seed_output_dir,
                    f"{slice_name}_{model_name}_voronoi_mask.png",
                ),
                mask_255,
            )

            overlay = apply_cancer_mask_overlay(
                he_image,
                pred_mask,
                alpha=0.45,
            )

            cv_imwrite(
                os.path.join(
                    seed_output_dir,
                    f"{slice_name}_{model_name}_voronoi_overlay.jpg",
                ),
                overlay,
            )

            hd_text = (
                f"{result['HD95']:.2f}"
                if np.isfinite(
                    result[
                        "HD95"
                    ]
                )
                else "NaN"
            )

            prediction_kind = (
                "continuous"
                if has_continuous_scores
                else "binary"
            )

            print(
                f"    seed={seed} | "
                f"input={prediction_kind} | "
                f"Target_IoU="
                f"{result['Target_IoU']:.4f} | "
                f"Dice="
                f"{result['Dice']:.4f} | "
                f"Recall="
                f"{result['Recall']:.4f} | "
                f"Precision="
                f"{result['Precision']:.4f} | "
                f"HD95="
                f"{hd_text}"
            )

    return (
        slice_results,
        geometry_row,
    )


# =============================================================================
# Summary
# =============================================================================

def build_summary_tables(
    all_metrics: Dict[
        str,
        Dict[
            str,
            Dict[
                int,
                Dict[
                    str,
                    float,
                ],
            ],
        ],
    ],
    slice_order: Sequence[str],
    model_order: Sequence[str],
) -> Tuple[
    List[
        Dict[
            str,
            object,
        ]
    ],
    List[
        Dict[
            str,
            object,
        ]
    ],
]:
    """
    Build:
        1. per-slice mean±sample-SD table;
        2. model-level macro summary across slices.
    """
    per_slice_rows: List[
        Dict[
            str,
            object,
        ]
    ] = []

    model_macro_rows: List[
        Dict[
            str,
            object,
        ]
    ] = []

    for model_name in model_order:
        slice_mean_values = {
            key: []
            for key in METRIC_KEYS
        }

        for slice_name in slice_order:
            seed_dict = (
                all_metrics
                .get(
                    slice_name,
                    {},
                )
                .get(
                    model_name,
                    {},
                )
            )

            if not seed_dict:
                continue

            seed_metrics = list(
                seed_dict.values()
            )

            row: Dict[
                str,
                object,
            ] = {
                "Method": model_name,
                "Slice": slice_name,
                "Num_Seeds": len(
                    seed_metrics
                ),
            }

            for key in METRIC_KEYS:
                mean, std = metric_mean_std(
                    seed_metrics,
                    key,
                )

                row[
                    f"{key}_Mean"
                ] = mean

                row[
                    f"{key}_Std"
                ] = std

                if np.isfinite(
                    mean
                ):
                    slice_mean_values[
                        key
                    ].append(
                        mean
                    )

            per_slice_rows.append(
                row
            )

        macro_row: Dict[
            str,
            object,
        ] = {
            "Method": model_name,
        }

        for key in METRIC_KEYS:
            values = np.asarray(
                slice_mean_values[
                    key
                ],
                dtype=np.float64,
            )

            if values.size == 0:
                macro_mean = float(
                    "nan"
                )

                macro_std = float(
                    "nan"
                )

            else:
                macro_mean = float(
                    np.mean(
                        values
                    )
                )

                macro_std = sample_std(
                    values
                )

            macro_row[
                f"Macro_{key}_Mean"
            ] = macro_mean

            macro_row[
                f"Macro_{key}_Std_Across_Slices"
            ] = macro_std

        model_macro_rows.append(
            macro_row
        )

    return (
        per_slice_rows,
        model_macro_rows,
    )


def lookup_slice_summary(
    per_slice_rows: Sequence[
        Dict[
            str,
            object,
        ]
    ],
    model_name: str,
    slice_name: str,
) -> Optional[
    Dict[
        str,
        object,
    ]
]:
    for row in per_slice_rows:
        if (
            str(
                row.get(
                    "Method"
                )
            )
            == str(
                model_name
            )
            and str(
                row.get(
                    "Slice"
                )
            )
            == str(
                slice_name
            )
        ):
            return row

    return None


def build_wide_report_rows(
    per_slice_rows: Sequence[
        Dict[
            str,
            object,
        ]
    ],
    slice_order: Sequence[str],
    model_order: Sequence[str],
) -> List[
    List[
        str
    ]
]:
    """
    Paper-friendly wide table:
        Slice | Metric | BLEEP | MCGAE | MEATRD | SpaCRD
    """
    report_rows: List[
        List[
            str
        ]
    ] = []

    for slice_name in slice_order:
        for metric_index, key in enumerate(
            METRIC_KEYS
        ):
            row = [
                (
                    slice_name
                    if metric_index == 0
                    else ""
                ),
                key,
            ]

            for model_name in model_order:
                summary = (
                    lookup_slice_summary(
                        per_slice_rows,
                        model_name,
                        slice_name,
                    )
                )

                if summary is None:
                    value = 'Not evaluated'

                else:
                    mean = float(
                        summary.get(
                            f"{key}_Mean",
                            np.nan,
                        )
                    )

                    std = float(
                        summary.get(
                            f"{key}_Std",
                            np.nan,
                        )
                    )

                    value = format_mean_std(
                        mean,
                        std,
                        hd95=(
                            key
                            == "HD95"
                        ),
                    )

                row.append(
                    value
                )

            report_rows.append(
                row
            )

    return report_rows


def print_wide_report(
    rows: Sequence[
        Sequence[
            str
        ]
    ],
    model_order: Sequence[str],
) -> None:
    headers = [
        "Slice",
        "Metric",
        *model_order,
    ]

    table = [
        headers,
        *[
            list(
                row
            )
            for row in rows
        ],
    ]

    widths = [
        max(
            len(
                str(
                    row[
                        column
                    ]
                )
            )
            for row in table
        )
        for column in range(
            len(
                headers
            )
        )
    ]

    def format_row(
        row: Sequence[str],
    ) -> str:
        return (
            "| "
            + " | ".join(
                str(
                    value
                ).ljust(
                    widths[
                        index
                    ]
                )
                for index, value
                in enumerate(
                    row
                )
            )
            + " |"
        )

    separator = (
        "|-"
        + "-|-".join(
            "-" * width
            for width in widths
        )
        + "-|"
    )

    print(
        "\n\n"
        + "="
        * 150
    )

    print(
        "CRC spot-level baselines -> "
        "nearest-spot Voronoi pixel-level evaluation"
    )

    print(
        "Metrics: "
        "Target_IoU, Dice, Recall, Precision, HD95"
    )

    print(
        "="
        * 150
    )

    print(
        format_row(
            headers
        )
    )

    print(
        separator
    )

    for row in rows:
        print(
            format_row(
                row
            )
        )

    print(
        "="
        * 150
    )


def save_wide_report_csv(
    path: str,
    rows: Sequence[
        Sequence[
            str
        ]
    ],
    model_order: Sequence[str],
) -> None:
    os.makedirs(
        os.path.dirname(
            path
        ),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "Slice",
                "Metric",
                *model_order,
            ]
        )

        writer.writerows(
            rows
        )


# =============================================================================
# Main evaluation
# =============================================================================

def evaluate_crc_baselines(
    test_slices: Sequence[str],
    model_names: Sequence[str] = SUPPORTED_MODELS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    pred_root: str = r"outputs",
    adata_base_dir: str = (
        r"data/CRC"
        r"\cropped_data\h5ad_cropped"
    ),
    he_base_dir: str = (
        r"data/CRC"
        r"\cropped_data\he_cropped"
    ),
    label_base_dir: str = (
        r"data/CRC"
        r"\cropped_data\label_cropped"
    ),
    tissue_base_dir: str = (
        r"data/CRC"
        r"\cropped_data\tissue_mask_cropped"
    ),
    output_root: str = (
        r"data/CRC"
        r"\predictions_Baseline_Voronoi"
    ),
    threshold: float = 0.5,
    voronoi_row_chunk: int = 256,
) -> Dict[
    str,
    Dict[
        str,
        Dict[
            int,
            Dict[
                str,
                float,
            ],
        ],
    ],
]:
    """
    Unified CRC spot-baseline evaluation entry point.
    """
    invalid_slices = [
        slice_name
        for slice_name in test_slices
        if slice_name not in CRC_SLICES
    ]

    invalid_models = [
        model_name
        for model_name in model_names
        if model_name not in SUPPORTED_MODELS
    ]

    if invalid_slices:
        raise ValueError(
            f"Unsupported CRC sections: {invalid_slices}; "
            f"supported sections: {CRC_SLICES}"
        )

    if invalid_models:
        raise ValueError(
            f"Unsupported models: {invalid_models}; "
            f"supported models: {SUPPORTED_MODELS}"
        )

    if not test_slices:
        raise ValueError(
            'test_slices cannot be empty.'
        )

    if not model_names:
        raise ValueError(
            'model_names cannot be empty.'
        )

    if not seeds:
        raise ValueError(
            'seeds cannot be empty.'
        )

    if not (
        0.0
        <= float(
            threshold
        )
        <= 1.0
    ):
        raise ValueError(
            'threshold must be in [0, 1].'
        )

    if int(
        voronoi_row_chunk
    ) <= 0:
        raise ValueError(
            'voronoi_row_chunk must be greater than 0.'
        )

    os.makedirs(
        output_root,
        exist_ok=True,
    )

    all_metrics: Dict[
        str,
        Dict[
            str,
            Dict[
                int,
                Dict[
                    str,
                    float,
                ],
            ],
        ],
    ] = {}

    per_seed_rows: List[
        Dict[
            str,
            object,
        ]
    ] = []

    geometry_rows: List[
        Dict[
            str,
            object,
        ]
    ] = []

    print(
        "\n"
        + "="
        * 110
    )

    print(
        "CRC spot-level baseline pixel evaluation"
    )

    print(
        "Protocol: "
        "tissue-clipped nearest-spot Voronoi rasterization"
    )

    print(
        "Post-processing: NONE"
    )

    print(
        f"Threshold for continuous spot probabilities: "
        f"{float(threshold):.2f}"
    )

    print(
        f"Models: "
        f"{list(model_names)}"
    )

    print(
        f"Slices: "
        f"{list(test_slices)}"
    )

    print(
        f"Seeds: "
        f"{list(seeds)}"
    )

    print(
        "="
        * 110
    )

    for slice_name in test_slices:
        (
            slice_results,
            geometry_row,
        ) = evaluate_slice(
            slice_name=slice_name,
            model_names=model_names,
            pred_root=pred_root,
            adata_base_dir=adata_base_dir,
            he_base_dir=he_base_dir,
            label_base_dir=label_base_dir,
            tissue_base_dir=tissue_base_dir,
            output_root=output_root,
            seeds=seeds,
            threshold=threshold,
            voronoi_row_chunk=voronoi_row_chunk,
        )

        all_metrics[
            slice_name
        ] = slice_results

        geometry_rows.append(
            geometry_row
        )

        for (
            model_name,
            seed_dict,
        ) in slice_results.items():
            for (
                seed,
                result,
            ) in seed_dict.items():
                per_seed_rows.append(
                    {
                        "Method": model_name,
                        "Slice": slice_name,
                        "Seed": int(
                            seed
                        ),
                        "Prediction_Path": (
                            build_prediction_path(
                                pred_root=pred_root,
                                model_name=model_name,
                                slice_name=slice_name,
                                seed=int(
                                    seed
                                ),
                            )
                        ),
                        "Rasterization": (
                            "Nearest-spot Voronoi"
                        ),
                        "Postprocess": "None",
                        **result,
                    }
                )

    (
        per_slice_rows,
        model_macro_rows,
    ) = build_summary_tables(
        all_metrics=all_metrics,
        slice_order=test_slices,
        model_order=model_names,
    )

    wide_rows = build_wide_report_rows(
        per_slice_rows=per_slice_rows,
        slice_order=test_slices,
        model_order=model_names,
    )

    print_wide_report(
        rows=wide_rows,
        model_order=model_names,
    )

    write_csv(
        os.path.join(
            output_root,
            "pixel_metrics_per_seed.csv",
        ),
        per_seed_rows,
    )

    write_csv(
        os.path.join(
            output_root,
            "pixel_metrics_per_slice_mean_std.csv",
        ),
        per_slice_rows,
    )

    write_csv(
        os.path.join(
            output_root,
            "pixel_metrics_model_macro_summary.csv",
        ),
        model_macro_rows,
    )

    write_csv(
        os.path.join(
            output_root,
            "voronoi_geometry_diagnostics.csv",
        ),
        geometry_rows,
    )

    wide_csv_path = os.path.join(
        output_root,
        'CRC_Voronoi_Inference_results.csv',
    )

    save_wide_report_csv(
        path=wide_csv_path,
        rows=wide_rows,
        model_order=model_names,
    )

    protocol = {
        "task": (
            "Pixel-level CTR segmentation comparison"
        ),
        "dataset": "CRC",
        "slices": list(
            test_slices
        ),
        "models": list(
            model_names
        ),
        "seeds": [
            int(
                seed
            )
            for seed in seeds
        ],
        "spot_prediction_space": (
            "Method-native original-scale H&E/ST data; one value per original spot"
        ),
        "pixel_evaluation_space": (
            "CRC cropped_data with adata.obsm['spatial_cropped']"
        ),
        "spot_to_pixel_conversion": (
            "Tissue-clipped nearest-spot Voronoi rasterization"
        ),
        "postprocessing": "None",
        "continuous_probability_threshold": float(
            threshold
        ),
        "spatial_key": (
            "adata.obsm['spatial_cropped']"
        ),
        "data_layout": {
            "he": (
                r"data/CRC"
                r"\cropped_data\he_cropped\{slice}.png"
            ),
            "label": (
                r"data/CRC"
                r"\cropped_data\label_cropped\{slice}.png"
            ),
            "tissue": (
                r"data/CRC"
                r"\cropped_data\tissue_mask_cropped\{slice}.png"
            ),
            "adata": (
                r"data/CRC"
                r"\cropped_data\h5ad_cropped\{slice}_cropped.h5ad"
            ),
        },
        "metrics": METRIC_KEYS,
        "seed_statistic": (
            "mean ± sample standard deviation (ddof=1)"
        ),
        "important_note": (
            "Use the same tissue mask and the same HD95 implementation "
            "for HiST-Seg and all baseline methods."
        ),
    }

    with open(
        os.path.join(
            output_root,
            "evaluation_protocol.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            protocol,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nResults saved to:\n"
        f"{output_root}"
    )

    print(
        f"Paper-style wide table:\n"
        f"{wide_csv_path}"
    )

    return all_metrics


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Use nearest-spot Voronoi rasterization'
            'Uniformly evaluate CRC spot-level baselines on 12 slices.'
        )
    )

    parser.add_argument(
        "--slices",
        nargs="+",
        choices=CRC_SLICES,
        default=CRC_SLICES,
        help=(
            'CRC slices to evaluate; default is all 12.'
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=SUPPORTED_MODELS,
        help=(
            'Models to evaluate; defaults to all 4.'
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=(
            'Random seed; default 42 123 567.'
        ),
    )

    parser.add_argument(
        "--pred-root",
        default=r"outputs",
        help=(
            'baseline prediction result root directory.'
        ),
    )

    parser.add_argument(
        "--adata-base-dir",
        default=(
            r"data/CRC"
            r"\cropped_data\h5ad_cropped"
        ),
        help=(
            'CRC h5ad_cropped directory.'
        ),
    )

    parser.add_argument(
        "--he-base-dir",
        default=(
            r"data/CRC"
            r"\cropped_data\he_cropped"
        ),
        help=(
            'CRC he_cropped directory.'
        ),
    )

    parser.add_argument(
        "--label-base-dir",
        default=(
            r"data/CRC"
            r"\cropped_data\label_cropped"
        ),
        help=(
            'CRC label_cropped directory.'
        ),
    )

    parser.add_argument(
        "--tissue-base-dir",
        default=(
            r"data/CRC"
            r"\cropped_data\tissue_mask_cropped"
        ),
        help=(
            'CRC tissue_mask_cropped directory.'
        ),
    )

    parser.add_argument(
        "--output-root",
        default=(
            r"data/CRC"
            r"\predictions_Baseline_Voronoi"
        ),
        help=(
            'Output directory.'
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            'Continuous spot probability binarization threshold; default 0.5.'
        ),
    )

    parser.add_argument(
        "--voronoi-row-chunk",
        type=int,
        default=256,
        help=(
            'Row chunk size for Voronoi nearest neighbor queries; default 256.'
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluate_crc_baselines(
        test_slices=args.slices,
        model_names=args.models,
        seeds=args.seeds,
        pred_root=args.pred_root,
        adata_base_dir=args.adata_base_dir,
        he_base_dir=args.he_base_dir,
        label_base_dir=args.label_base_dir,
        tissue_base_dir=args.tissue_base_dir,
        output_root=args.output_root,
        threshold=args.threshold,
        voronoi_row_chunk=args.voronoi_row_chunk,
    )


if __name__ == "__main__":
    main()
