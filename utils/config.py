from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


class ConfigNode(dict):
    """Recursive mapping with both dict-style and attribute-style access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _convert(value)

    def clone(self) -> "ConfigNode":
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(self)


def _convert(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return value
    if isinstance(value, Mapping):
        return ConfigNode({str(key): _convert(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_convert(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_convert(item) for item in value)
    return value


def _to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _require(cfg: ConfigNode, dotted: str) -> Any:
    current: Any = cfg
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"Missing required config field: {dotted}")
        current = current[part]
    return current


def _derive_geometry(cfg: ConfigNode) -> None:
    """Derive all tensor sizes from patch/context geometry.

    Supplied configs use:
        detail patch        = 512 x 512
        context scale       = 4
        physical context    = 2048 x 2048
        cache downsample    = 4
        H&E context image   = 512 x 512
        context/ST grid     = 128 x 128
        UNI semantic grid   = 32 x 32
    """
    patch = int(_require(cfg, "geometry.patch_size"))
    cache_downsample = int(_require(cfg, "geometry.downsample"))
    context_scale = int(_require(cfg, "geometry.ctx_scale"))
    uni_patch = int(getattr(cfg.geometry, "uni_patch_size", 16))
    context_grid = int(_require(cfg, "geometry.context_grid_size"))

    if patch <= 0 or cache_downsample <= 0 or context_scale <= 0 or context_grid <= 0:
        raise ValueError("Geometry values must be positive")
    if patch % uni_patch != 0:
        raise ValueError("geometry.patch_size must be divisible by geometry.uni_patch_size")

    physical_context = patch * context_scale
    if physical_context % cache_downsample != 0:
        raise ValueError("patch_size * ctx_scale must be divisible by geometry.downsample")
    context_image = physical_context // cache_downsample
    if context_image % context_grid != 0:
        raise ValueError(
            "The context image must be an integer multiple of context_grid_size"
        )
    if context_grid % context_scale != 0:
        raise ValueError("context_grid_size must be divisible by ctx_scale")

    cfg.geometry.cache_downsample = cache_downsample
    cfg.geometry.physical_context_size = physical_context
    cfg.geometry.context_image_size = context_image
    cfg.geometry.context_encoder_stride = context_image // context_grid
    cfg.geometry.context_footprint_size = context_grid // context_scale
    cfg.geometry.sem_grid = patch // uni_patch


def _validate(cfg: ConfigNode) -> None:
    active = [str(value) for value in _require(cfg, "experiment.active_slices")]
    if len(active) != len(set(active)):
        raise ValueError("experiment.active_slices contains duplicates")
    if len(active) < 3:
        raise ValueError("At least three active slices are required")

    registered = []
    for dataset_cfg in _require(cfg, "datasets").values():
        registered.extend(str(value) for value in dataset_cfg.slices)
    unknown = sorted(set(active) - set(registered))
    if unknown:
        raise ValueError(f"Unknown active slices: {unknown}")

    if int(cfg.dims.st_dim) != 512:
        raise ValueError("The current scGPT spot encoder requires dims.st_dim=512")
    if int(cfg.dims.sem_dim) <= 0:
        raise ValueError("dims.sem_dim must be positive")
    st_drop = float(cfg.model.st_drop_rate)
    if not 0.0 <= st_drop < 1.0:
        raise ValueError("model.st_drop_rate must be in [0, 1)")


def load_config(path: str | Path) -> ConfigNode:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")

    cfg = _convert(raw)
    cfg.config_path = str(path.resolve())
    _derive_geometry(cfg)
    _validate(cfg)
    return cfg


def save_config(cfg: ConfigNode, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, allow_unicode=True, sort_keys=False)
