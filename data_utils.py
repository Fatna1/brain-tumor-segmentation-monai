# data_utils.py
import os
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    ToTensord, MapLabelValued, RandCropByPosNegLabeld,
)
from monai.apps import DecathlonDataset
from monai.data import DataLoader as MonaiDataLoader   # ← use this, not torch's

import config


def get_base_transforms():
    """Base transforms applied to every sample (train + val)."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
        MapLabelValued(
            keys=["label"],
            orig_labels=[0, 1, 2, 4],
            target_labels=[0, 1, 2, 3],
        ),
    ])


def get_train_transforms():
    """Base transforms + random positive/negative crop for training."""
    return Compose([
        get_base_transforms(),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=config.SPATIAL_SIZE,
            pos=1, neg=1,
            num_samples=config.NUM_SAMPLES,
            image_key="image",
            image_threshold=0,
        ),
    ])


def get_val_transforms():
    """Only the base transforms for validation (full volume)."""
    return get_base_transforms()


def get_datasets():
    """Build and return (train_ds, val_ds)."""
    os.makedirs(config.ROOT_DIR, exist_ok=True)

    train_ds = DecathlonDataset(
        root_dir=config.ROOT_DIR,
        task=config.TASK,
        section="training",
        download=True,
        cache_rate=config.CACHE_RATE,
        transform=get_train_transforms(),
    )

    val_ds = DecathlonDataset(
        root_dir=config.ROOT_DIR,
        task=config.TASK,
        section="validation",
        download=False,
        transform=get_val_transforms(),
    )
    return train_ds, val_ds


def get_dataloaders(train_ds=None, val_ds=None):
    """Return (train_loader, val_loader). Will build datasets if not passed."""
    if train_ds is None or val_ds is None:
        train_ds, val_ds = get_datasets()

    # ✅ MONAI's DataLoader handles list-of-dict crops from RandCropByPosNegLabeld
    train_loader = MonaiDataLoader(
        train_ds, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS,
    )
    val_loader = MonaiDataLoader(
        val_ds, batch_size=config.VAL_BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
    )
    return train_loader, val_loader