#!/usr/bin/env python3
"""
Dysplasia Risk Score (DRS) computation and clinical report generation.

DRS combines:
  1. Barrett's class probability (from classifier)
  2. Texture entropy (GLCM-based mucosal irregularity)
  3. Gland irregularity proxy (edge density in high-activation regions)

Output: Per-image DRS + PDF-style clinical summary report.

Usage:
    python src/scoring/drs_report.py \
        --checkpoint results/checkpoints/best_model.pt \
        --patient_dir data/processed/test/barrett \
        --patient_id PATIENT_001
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from datetime import datetime
import yaml
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.feature import graycomatrix, graycoprops

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.efficientnet_barrett import BarrettClassifier


CLASS_NAMES = {0: "Normal", 1: "Barrett's", 2: "Esophagitis"}


def compute_texture_entropy(img_gray: np.ndarray) -> float:
    """
    Compute GLCM entropy as proxy for mucosal texture irregularity.
    
    Barrett's epithelium shows columnar mucosa with goblet cells — 
    distinct GLCM texture signature vs normal squamous epithelium.
    """
    img_uint8 = (img_gray * 255).astype(np.uint8)
    glcm = graycomatrix(
        img_uint8, distances=[1, 3], angles=[0, np.pi/4, np.pi/2],
        levels=256, symmetric=True, normed=True
    )
    # Entropy: -sum(p * log(p+eps))
    p = glcm + 1e-10
    entropy = -np.sum(p * np.log(p))
    # Normalize to [0, 1] range (typical range ~50-200)
    return float(np.clip(entropy / 200.0, 0, 1))


def compute_gland_irregularity(img_gray: np.ndarray) -> float:
    """
    Edge density as proxy for glandular architecture irregularity.
    
    Dysplastic Barrett's shows architectural disarray: irregular glands,
    back-to-back glands, loss of polarity → higher edge density.
    """
    img_uint8 = (img_gray * 255).astype(np.uint8)
    edges = cv2.Canny(img_uint8, threshold1=50, threshold2=150)
    return float(edges.mean() / 255.0)


def compute_drs(
    barrett_prob: float,
    texture_entropy: float,
    gland_irregularity: float,
    normal_prob: float,
    weights: dict,
) -> float:
    """
    Dysplasia Risk Score (DRS) computation.
    
    DRS = w1*P(barrett) + w2*texture_entropy + w3*gland_irregularity - w4*P(normal)
    Clipped to [0, 1].
    """
    drs = (
        weights["barrett_probability"] * barrett_prob
        + weights["texture_entropy"] * texture_entropy
        + weights["gland_irregularity"] * gland_irregularity
        - weights["normal_penalty"] * normal_prob
    )
    return float(np.clip(drs, 0, 1))


def get_risk_tier(drs: float, thresholds: dict) -> tuple:
    if drs >= thresholds["high_risk"]:
        return "HIGH", "#F44336", "Immediate biopsy referral recommended"
    elif drs >= thresholds["low_risk"]:
        return "MODERATE", "#FF9800", "6-month surveillance follow-up recommended"
    else:
        return "LOW", "#4CAF50", "Routine annual surveillance"


@torch.no_grad()
def score_image(model, img_path: str, device: str, weights: dict, thresholds: dict) -> dict:
    """Score a single endoscopy image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    img_gray = np.array(img_pil.convert("L").resize((224, 224))) / 255.0

    model.eval()
    logits, drs_model = model(img_tensor)
    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_class = int(probs.argmax())

    texture = compute_texture_entropy(img_gray)
    gland_irr = compute_gland_irregularity(img_gray)
    drs = compute_drs(
        barrett_prob=float(probs[1]),
        texture_entropy=texture,
        gland_irregularity=gland_irr,
        normal_prob=float(probs[0]),
        weights=weights,
    )

    risk_tier, risk_color, recommendation = get_risk_tier(drs, thresholds)

    return {
        "image": Path(img_path).name,
        "pred_class": CLASS_NAMES[pred_class],
        "p_normal": round(float(probs[0]), 4),
        "p_barrett": round(float(probs[1]), 4),
        "p_esophagitis": round(float(probs[2]), 4),
        "texture_entropy": round(texture, 4),
        "gland_irregularity": round(gland_irr, 4),
        "drs": round(drs, 4),
        "risk_tier": risk_tier,
        "recommendation": recommendation,
    }


def generate_report_figure(results: list, patient_id: str, output_path: str):
    """Generate clinical summary figure with DRS results per image."""
    df = pd.DataFrame(results)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Barrett's Esophagus Risk Assessment Report\nPatient ID: {patient_id} | "
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        fontsize=13, fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Panel A: DRS per image
    ax1 = fig.add_subplot(gs[0, :])
    colors = [
        "#F44336" if r == "HIGH" else "#FF9800" if r == "MODERATE" else "#4CAF50"
        for r in df["risk_tier"]
    ]
    bars = ax1.bar(range(len(df)), df["drs"], color=colors, alpha=0.85, edgecolor="white")
    ax1.axhline(y=0.30, color="orange", linestyle="--", alpha=0.8, label="Moderate threshold (0.30)")
    ax1.axhline(y=0.55, color="red", linestyle="--", alpha=0.8, label="High threshold (0.55)")
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels([r["image"][:12] for r in results], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Dysplasia Risk Score (DRS)", fontsize=10)
    ax1.set_title("A. Per-Image DRS", fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)

    # Panel B: Risk tier distribution
    ax2 = fig.add_subplot(gs[1, 0])
    tier_counts = df["risk_tier"].value_counts()
    tier_colors = {"HIGH": "#F44336", "MODERATE": "#FF9800", "LOW": "#4CAF50"}
    wedge_colors = [tier_colors.get(t, "#999") for t in tier_counts.index]
    ax2.pie(
        tier_counts.values,
        labels=tier_counts.index,
        colors=wedge_colors,
        autopct="%1.0f%%",
        startangle=90,
    )
    ax2.set_title("B. Risk Tier Distribution", fontweight="bold")

    # Panel C: DRS components scatter
    ax3 = fig.add_subplot(gs[1, 1])
    sc = ax3.scatter(
        df["texture_entropy"], df["gland_irregularity"],
        c=df["drs"], cmap="RdYlGn_r",
        s=100, alpha=0.8, edgecolors="white", linewidths=0.5,
        vmin=0, vmax=1
    )
    plt.colorbar(sc, ax=ax3, label="DRS")
    ax3.set_xlabel("Texture Entropy", fontsize=9)
    ax3.set_ylabel("Gland Irregularity", fontsize=9)
    ax3.set_title("C. DRS Components", fontweight="bold")

    # Summary stats annotation
    n_high = (df["risk_tier"] == "HIGH").sum()
    n_mod = (df["risk_tier"] == "MODERATE").sum()
    mean_drs = df["drs"].mean()
    summary = (
        f"Summary: {len(df)} images analyzed\n"
        f"High risk: {n_high} | Moderate: {n_mod}\n"
        f"Mean DRS: {mean_drs:.3f}"
    )
    fig.text(0.02, 0.02, summary, fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Report figure saved: {output_path}")


def main(args):
    # Load config
    with open("configs/drs_weights.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = BarrettClassifier(num_classes=3, dropout=0.0, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Score all images in patient directory
    img_dir = Path(args.patient_dir)
    image_paths = sorted([
        str(p) for p in img_dir.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    if not image_paths:
        print(f"No images found in {img_dir}")
        return

    print(f"Scoring {len(image_paths)} images for patient {args.patient_id}...")
    results = []
    for p in image_paths:
        r = score_image(model, p, device, cfg["weights"], cfg["thresholds"])
        results.append(r)
        print(f"  {r['image']}: DRS={r['drs']:.3f} ({r['risk_tier']})")

    # Save results table
    df = pd.DataFrame(results)
    out_csv = f"results/tables/drs_{args.patient_id}.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Results saved: {out_csv}")

    # Generate report figure
    generate_report_figure(
        results, args.patient_id,
        f"results/figures/drs_report_{args.patient_id}.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/checkpoints/best_model.pt")
    parser.add_argument("--patient_dir", default="data/processed/test/barrett")
    parser.add_argument("--patient_id", default="PATIENT_001")
    args = parser.parse_args()
    main(args)
