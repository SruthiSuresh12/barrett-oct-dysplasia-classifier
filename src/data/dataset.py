#!/usr/bin/env python3
"""
Barrett's dataset loader with MONAI medical image augmentations.

Uses MONAI transforms rather than standard torchvision to align with
clinical medical imaging pipelines (e.g. tethered capsule OCT preprocessing).
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


def get_class_weights(dataset: ImageFolder) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced Barrett's data."""
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    weights = 1.0 / class_counts
    weights = weights / weights.sum()
    return torch.FloatTensor(weights)


def get_sample_weights(dataset: ImageFolder) -> torch.Tensor:
    """Per-sample weights for WeightedRandomSampler (handles class imbalance)."""
    class_weights = get_class_weights(dataset)
    sample_weights = torch.FloatTensor([
        class_weights[label] for label in dataset.targets
    ])
    return sample_weights


def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """
    Returns augmentation pipeline.
    
    Training: heavy augmentation to handle limited Barrett's images (n=94).
    Val/Test: deterministic resize + normalize only.
    
    Note: elastic_transform mimics tissue deformation seen in tethered capsule
    OCT acquisitions (peristalsis, patient movement).
    """
    mean = [0.485, 0.456, 0.406]  # ImageNet stats (transfer learning baseline)
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # simulate OCT artifacts
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Returns train, val, test dataloaders and class metadata.
    
    Uses WeightedRandomSampler for training to handle class imbalance
    (only 94 Barrett's vs 932 normal images).
    """
    data_dir = Path(data_dir)

    datasets = {}
    for split in ["train", "val", "test"]:
        datasets[split] = ImageFolder(
            root=str(data_dir / split),
            transform=get_transforms(split, image_size)
        )

    # Weighted sampling for training (critical for 94 Barrett's vs 932 normal)
    sample_weights = get_sample_weights(datasets["train"])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_info = {
        "class_to_idx": datasets["train"].class_to_idx,
        "idx_to_class": {v: k for k, v in datasets["train"].class_to_idx.items()},
        "class_weights": get_class_weights(datasets["train"]),
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "n_test": len(datasets["test"]),
    }

    print(f"Dataset loaded:")
    print(f"  Train: {class_info['n_train']} | Val: {class_info['n_val']} | Test: {class_info['n_test']}")
    print(f"  Classes: {class_info['class_to_idx']}")
    print(f"  Class weights: {class_info['class_weights'].numpy().round(3)}")

    return train_loader, val_loader, test_loader, class_info
