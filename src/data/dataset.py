"""
PyTorch Dataset + transforms for ISIC 2019.
============================================
- Augmentation is applied ONLY to training data.
- Validation/test receive preprocessing only (deterministic).
- Preprocessing is model-agnostic at the data layer; the model's own
  preprocessing (normalization) is handled here via configurable stats.
"""
import random

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

# ImageNet stats (used by ResNet / EfficientNet / most torchvision models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _pil_loader(path):
    with Image.open(path) as im:
        return im.convert("RGB")


class ISICDataset(Dataset):
    """Image dataset built from a split CSV produced by split.py."""

    def __init__(self, csv_path, image_dir, class_names,
                 image_size=224, augment=False, transform=None, cache_dir=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.augment = augment
        self.image_size = (image_size, image_size)

        self.df = self.df.reset_index(drop=True)
        if cache_dir:
            # Lossless pre-resized PNGs (scripts/build_image_cache.py). Resize on an
            # image that already has the target size is an identity copy in Pillow,
            # so network inputs are bit-identical to the uncached pipeline.
            self.paths = [str(Path(cache_dir) / f"{i}.png") for i in self.df["image_id"]]
        else:
            self.paths = self.df["image_path"].tolist()
        self.labels = [self.class_to_idx[l] for l in self.df["label"]]
        self.ids = self.df["image_id"].tolist()

        if transform is None:
            transform = make_transforms(image_size, augment=augment)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = _pil_loader(self.paths[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    def class_counts(self):
        counts = {}
        for l in self.labels:
            counts[self.class_names[l]] = counts.get(self.class_names[l], 0) + 1
        return counts


def make_transforms(image_size, augment=False, aug_params=None, mean=None, std=None):
    """Build the transform pipeline.

    Validation/test: resize -> center crop (optional) -> tensor -> normalize.
    Train: resize -> augmentations -> tensor -> normalize.
    """
    import torchvision.transforms as T

    mean = mean if mean is not None else IMAGENET_MEAN
    std = std if std is not None else IMAGENET_STD

    resize = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)
    normalize = T.Normalize(mean=mean, std=std)

    if not augment:
        return T.Compose([resize, T.ToTensor(), normalize])

    aug_params = aug_params or {}
    transforms = [resize]

    transforms.append(T.RandomHorizontalFlip(
        p=0.5 if aug_params.get("horizontal_flip", True) else 0.0))
    if aug_params.get("vertical_flip", False):
        transforms.append(T.RandomVerticalFlip(p=0.5))

    rotation = aug_params.get("rotation_deg", 15)
    if rotation and rotation > 0:
        transforms.append(T.RandomRotation(rotation))

    zoom = aug_params.get("zoom_range", 0.10)
    translation = aug_params.get("translation_pct", 0.05)
    if zoom or translation:
        transforms.append(T.RandomAffine(
            degrees=0,
            translate=(translation, translation) if translation else None,
            scale=(1 - zoom, 1 + zoom) if zoom else None,
        ))

    brightness = aug_params.get("brightness_jitter", 0.1)
    contrast = aug_params.get("contrast_jitter", 0.1)
    if brightness or contrast:
        transforms.append(T.ColorJitter(
            brightness=brightness if brightness else 0,
            contrast=contrast if contrast else 0,
        ))

    transforms.append(T.ToTensor())
    transforms.append(normalize)
    return T.Compose(transforms)


def compute_class_weights(dataset):
    """Balanced class weights from a training dataset ONLY."""
    counts = dataset.class_counts()
    total = sum(counts.values())
    n_classes = len(dataset.class_names)
    weights = np.zeros(n_classes, dtype=np.float64)
    for name, count in counts.items():
        idx = dataset.class_to_idx[name]
        weights[idx] = total / (n_classes * count)
    return torch.tensor(weights, dtype=torch.float32)


def load_splits(cfg, class_names):
    """Load train/val/test datasets from split CSVs.

    Returns (train_ds, val_ds, test_ds) with augmentation only on train.
    """
    splits_dir = str(cfg.paths.splits_dir)
    image_dir = str(Path(cfg.dataset.raw_dir) / cfg.dataset.train_input_dir)
    img_size = cfg.preprocessing.image_size

    aug_params = None
    if cfg.augmentation.enabled:
        aug_params = cfg.augmentation.to_dict() if hasattr(cfg.augmentation, "to_dict") else cfg.augmentation._data

    cache_dir = cfg.preprocessing.get("cache_dir", None)
    if cache_dir and not Path(cache_dir).is_absolute():
        cache_dir = str(Path(cfg.get("project_root", ".")) / cache_dir)

    train_ds = ISICDataset(
        f"{splits_dir}/train.csv", image_dir, class_names,
        image_size=img_size, augment=True, transform=make_transforms(
            img_size, augment=True, aug_params=aug_params), cache_dir=cache_dir)
    val_ds = ISICDataset(
        f"{splits_dir}/validation.csv", image_dir, class_names,
        image_size=img_size, augment=False, transform=make_transforms(img_size),
        cache_dir=cache_dir)
    test_ds = ISICDataset(
        f"{splits_dir}/test.csv", image_dir, class_names,
        image_size=img_size, augment=False, transform=make_transforms(img_size),
        cache_dir=cache_dir)
    return train_ds, val_ds, test_ds


def get_dataloaders(cfg, class_names):
    """Return (train_loader, val_loader, test_loader)."""
    train_ds, val_ds, test_ds = load_splits(cfg, class_names)

    def _loader(ds, shuffle):
        return torch.utils.data.DataLoader(
            ds, batch_size=cfg.training.batch_size,
            shuffle=shuffle, num_workers=cfg.training.num_workers,
            pin_memory=True, persistent_workers=cfg.training.num_workers > 0,
            drop_last=False)

    return _loader(train_ds, shuffle=True), _loader(val_ds, False), _loader(test_ds, False)