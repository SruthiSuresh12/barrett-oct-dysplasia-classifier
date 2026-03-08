#!/usr/bin/env python3
"""
Download and prepare HyperKvasir dataset for Barrett's dysplasia classification.

Dataset: HyperKvasir (Borgli et al., Scientific Data 2020)
DOI: https://doi.org/10.1038/s41597-020-00622-y
Download: https://datasets.simula.no/hyper-kvasir/
License: CC BY 4.0

Classes used:
  - Normal (negative):   z-line/ (932 images)
  - Barrett's (positive): barretts/ (41) + barretts-short-segment/ (53)
  - Esophagitis:         esophagitis-a/ (163) + esophagitis-b-d/ (500)
"""

import os
import shutil
import random
import argparse
from pathlib import Path
import yaml

random.seed(42)

# Class mapping
CLASS_MAP = {
    "barretts": "barrett",
    "barretts-short-segment": "barrett",
    "z-line": "normal",
    "esophagitis-a": "esophagitis",
    "esophagitis-b-d": "esophagitis",
}

SPLITS = ["train", "val", "test"]
SPLIT_RATIOS = [0.70, 0.15, 0.15]


def prepare_dataset(hyperkvasir_root: str, output_dir: str):
    """
    Organise HyperKvasir subfolders into train/val/test splits
    with class subdirectories for PyTorch ImageFolder loading.

    Expected HyperKvasir structure:
        hyperkvasir_root/
            labeled-images/
                lower-gi-tract/
                    ...
                upper-gi-tract/
                    quality-of-mucosal-views/
                        barretts/
                        barretts-short-segment/
                        z-line/
                    ...
                    pathological-findings/
                        esophagitis-a/
                        esophagitis-b-d/
    """
    root = Path(hyperkvasir_root)
    out = Path(output_dir)

    # Search for relevant folders anywhere under root
    all_images = {}  # class_name -> [image_paths]

    for folder_name, class_label in CLASS_MAP.items():
        matches = list(root.rglob(folder_name))
        if not matches:
            print(f"  WARNING: folder '{folder_name}' not found under {root}")
            continue
        folder = matches[0]
        imgs = sorted([
            p for p in folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ])
        print(f"  Found {len(imgs):4d} images in {folder_name}/ → class '{class_label}'")
        all_images.setdefault(class_label, []).extend(imgs)

    # Shuffle and split per class (stratified)
    for class_label in ["normal", "barrett", "esophagitis"]:
        imgs = all_images.get(class_label, [])
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * SPLIT_RATIOS[0])
        n_val = int(n * SPLIT_RATIOS[1])

        split_indices = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train + n_val],
            "test": imgs[n_train + n_val:],
        }

        for split, split_imgs in split_indices.items():
            dest_dir = out / split / class_label
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_imgs:
                shutil.copy2(img_path, dest_dir / img_path.name)

        print(f"  {class_label}: {n_train} train / {n_val} val / {n - n_train - n_val} test")

    print(f"\nDataset prepared at: {out}")
    print("Structure: {train,val,test}/{normal,barrett,esophagitis}/")


def print_class_distribution(output_dir: str):
    out = Path(output_dir)
    print("\nClass distribution summary:")
    print(f"{'Split':<10} {'Normal':>10} {'Barrett':>10} {'Esophagitis':>12} {'Total':>8}")
    print("-" * 55)
    for split in SPLITS:
        counts = {}
        for cls in ["normal", "barrett", "esophagitis"]:
            d = out / split / cls
            counts[cls] = len(list(d.glob("*"))) if d.exists() else 0
        total = sum(counts.values())
        print(f"{split:<10} {counts['normal']:>10} {counts['barrett']:>10} {counts['esophagitis']:>12} {total:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare HyperKvasir dataset")
    parser.add_argument(
        "--hyperkvasir_root",
        type=str,
        default="data/raw/hyper-kvasir",
        help="Path to extracted HyperKvasir root folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for train/val/test splits"
    )
    args = parser.parse_args()

    print(f"Preparing HyperKvasir dataset...")
    print(f"  Source: {args.hyperkvasir_root}")
    print(f"  Output: {args.output_dir}")
    print()

    prepare_dataset(args.hyperkvasir_root, args.output_dir)
    print_class_distribution(args.output_dir)
