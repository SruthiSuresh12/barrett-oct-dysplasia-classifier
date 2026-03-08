#!/usr/bin/env python3
"""
Training script for Barrett's esophagus dysplasia classifier.

Usage:
    python src/models/train.py --data_dir data/processed --epochs 50

Outputs:
    results/checkpoints/best_model.pt
    results/tables/training_history.csv
    results/figures/training_curves.png
"""

import os
import sys
import argparse
import yaml
import time
import csv
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.dataset import get_dataloaders
from src.models.efficientnet_barrett import build_model


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, total_ce, total_drs = 0, 0, 0
    correct, n = 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        logits, drs = model(imgs)
        loss, ce, drs_l = loss_fn(logits, drs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_ce += ce.item() * len(labels)
        total_drs += drs_l.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)

    return total_loss / n, total_ce / n, total_drs / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    all_preds, all_labels, all_drs = [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, drs = model(imgs)
        loss, _, _ = loss_fn(logits, drs, labels)

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)

        all_preds.extend(logits.softmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_drs.extend(drs.cpu().numpy())

    return total_loss / n, correct / n, np.array(all_preds), np.array(all_labels), np.array(all_drs)


def main(args):
    # Load config
    with open("configs/train_params.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader, class_info = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=cfg["training"]["batch_size"],
        image_size=cfg["data"]["image_size"],
    )

    # Model
    model, loss_fn = build_model(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        pretrained=cfg["model"]["pretrained"],
        class_weights=class_info["class_weights"],
        device=device,
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0
    patience_counter = 0
    history = []

    Path("results/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        # Unfreeze all layers at epoch 10 for full fine-tuning
        if epoch == 10:
            model.unfreeze_all()
            print(f"  Epoch {epoch}: Unfreezing all layers for full fine-tuning")

        t0 = time.time()
        train_loss, train_ce, train_drs, train_acc = train_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_loss, val_acc, val_preds, val_labels, val_drs = eval_epoch(
            model, val_loader, loss_fn, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
            f"{elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_info": class_info,
            }, "results/checkpoints/best_model.pt")
            print(f"  ✓ New best model saved (val_acc={val_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Save training history
    with open("results/tables/training_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print("Checkpoint saved to: results/checkpoints/best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    main(args)
