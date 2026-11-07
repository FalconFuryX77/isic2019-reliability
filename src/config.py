"""
Central configuration loader for the ISIC 2019 reliability pipeline.
Loads YAML config files, resolves paths relative to the project root,
and exposes a simple attribute-style config object.
"""
import copy
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"

ISIC_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_FULL_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AK": "Actinic keratosis",
    "BKL": "Benign keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion",
    "SCC": "Squamous cell carcinoma",
}


class Config:
    """Attribute-style wrapper around a nested dict of settings."""

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name):
        data = object.__getattribute__(self, "_data")
        if name in data:
            value = data[name]
            return Config(value) if isinstance(value, dict) else value
        raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name, value):
        object.__getattribute__(self, "_data")[name] = value

    def get(self, name, default=None):
        data = object.__getattribute__(self, "_data")
        if name in data:
            value = data[name]
            return Config(value) if isinstance(value, dict) else value
        return default

    def to_dict(self):
        return copy.deepcopy(object.__getattribute__(self, "_data"))

    def __repr__(self):
        return f"Config({self._data!r})"


def load_config(path=None, overrides: dict = None) -> Config:
    """Load a YAML config file, optionally overlaying dict overrides."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if overrides:
        data = _deep_merge(data, overrides)

    # Resolve paths relative to project root
    data.setdefault("project_root", str(PROJECT_ROOT))
    paths = data.setdefault("paths", {})
    for key in ["splits_dir", "experiments_dir", "checkpoints_dir",
                "results_dir", "figures_dir", "tables_dir", "gradcam_dir",
                "logs_dir"]:
        if key in paths:
            p = Path(paths[key])
            paths[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)

    # Resolve dataset paths
    ds = data.setdefault("dataset", {})
    raw = Path(ds.get("raw_dir", "data/raw"))
    ds["raw_dir"] = str(raw if raw.is_absolute() else PROJECT_ROOT / raw)

    return Config(data)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict (returns new dict)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
