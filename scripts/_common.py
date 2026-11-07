"""
Entry-point helpers: load config, resolve model name, build device.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.utils import get_device  # noqa: E402

# Backbone name mapping (file-safe)
BACKBONE_KEYS = {
    "resnet50": "resnet50",
    "resnet": "resnet50",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnetb3": "efficientnet_b3",
    "effnet": "efficientnet_b3",
    "efficientnet": "efficientnet_b3",
}

MODEL_DISPLAY = {
    "resnet50": "ResNet-50",
    "efficientnet_b3": "EfficientNet-B3",
}


def resolve_backbone(name):
    key = BACKBONE_KEYS.get(str(name).lower())
    if key is None:
        raise ValueError(f"Unknown backbone: {name}. Use resnet50 or efficientnet_b3")
    return key