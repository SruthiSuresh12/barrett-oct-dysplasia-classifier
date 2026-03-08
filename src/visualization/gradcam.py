#!/usr/bin/env python3
"""
Grad-CAM saliency maps for Barrett's classifier.

Highlights which mucosal regions (glandular architecture, goblet cells,
vascular patterns) drove the model's classification decision.

This is the clinical interpretability layer — analogous to what a
physician using tethered capsule OCT would want to see highlighted.

Usage:
    python src/visualization/gradcam.py \
        --checkpoint results/checkpoints/best_model.pt \
        --image_dir data/processed/test/barrett \
        --n_samples 8
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.efficientnet_barrett import BarrettClassifier


CLASS_NAMES = {0: "Normal", 1: "Barrett's", 2: "Esophagitis"}
CLASS_COLORS = {0: "#2196F3", 1: "#F44336", 2: "#FF9800"}


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for EfficientNet-B0.
    Hooks into the last convolutional block (features[-1]).
    """

    def __init__(self, model: BarrettClassifier):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook into last feature block
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor: torch.Tensor, class_idx: int = None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            img_tensor: (1, 3, H, W) preprocessed image tensor
            class_idx: target class (None = predicted class)
        
        Returns:
            heatmap: (H, W) numpy array in [0, 1]
            pred_class: predicted class index
            drs_score: dysplasia risk score
            probs: class probabilities
        """
        self.model.eval()
        img_tensor.requires_grad = True

        logits, drs = self.model(img_tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().detach().numpy()

        pred_class = logits.argmax(1).item()
        target = class_idx if class_idx is not None else pred_class

        # Backward for target class
        self.model.zero_grad()
        logits[0, target].backward()

        # Pool gradients over spatial dimensions
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])  # (C,)

        # Weight activations by gradients
        activations = self.activations.squeeze(0)  # (C, h, w)
        for i, w in enumerate(pooled_grads):
            activations[i] *= w

        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        return heatmap, pred_class, drs.item(), probs


def overlay_heatmap(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image."""
    h, w = original_img.shape[:2]

    # Upsample heatmap to image size
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0

    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # RGB

    overlay = (1 - alpha) * original_img / 255.0 + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


def load_image(path: str, image_size: int = 224):
    """Load and preprocess image for inference."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert("RGB")
    img_np = np.array(img.resize((image_size, image_size)))
    img_tensor = transform(img).unsqueeze(0)
    return img_np, img_tensor


def visualize_batch(
    model: BarrettClassifier,
    image_paths: list,
    output_path: str,
    image_size: int = 224,
):
    """Generate Grad-CAM panel figure for a batch of images."""
    gradcam = GradCAM(model)
    n = len(image_paths)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[None, :]

    fig.suptitle(
        "Grad-CAM Saliency — Barrett's Esophagus Risk Regions",
        fontsize=14, fontweight="bold", y=0.98
    )

    col_titles = ["Original Image", "Grad-CAM Heatmap", "DRS Panel"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight="bold")

    for i, img_path in enumerate(image_paths):
        img_np, img_tensor = load_image(img_path, image_size)
        heatmap, pred_class, drs, probs = gradcam.generate(img_tensor)
        overlay = overlay_heatmap(img_np, heatmap)

        # Col 1: Original
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_ylabel(Path(img_path).stem[:20], fontsize=8)
        axes[i, 0].axis("off")

        # Col 2: Grad-CAM overlay
        axes[i, 1].imshow(overlay)
        pred_name = CLASS_NAMES[pred_class]
        color = CLASS_COLORS[pred_class]
        axes[i, 1].set_title(
            f"Pred: {pred_name} ({probs[pred_class]:.2f})",
            fontsize=9, color=color
        )
        axes[i, 1].axis("off")

        # Col 3: DRS bar
        ax3 = axes[i, 2]
        drs_color = "#F44336" if drs > 0.55 else "#FF9800" if drs > 0.30 else "#4CAF50"
        ax3.barh(["DRS"], [drs], color=drs_color, height=0.4)
        ax3.axvline(x=0.30, color="orange", linestyle="--", alpha=0.7, label="Moderate")
        ax3.axvline(x=0.55, color="red", linestyle="--", alpha=0.7, label="High")
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("Dysplasia Risk Score")

        risk_label = "HIGH RISK" if drs > 0.55 else "MODERATE" if drs > 0.30 else "Low Risk"
        ax3.set_title(f"DRS = {drs:.3f} ({risk_label})", fontsize=9, color=drs_color)
        if i == 0:
            ax3.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM panel saved: {output_path}")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = BarrettClassifier(num_classes=3, dropout=0.0, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Get image paths
    img_dir = Path(args.image_dir)
    image_paths = sorted([
        str(p) for p in img_dir.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])[:args.n_samples]

    if not image_paths:
        print(f"No images found in {img_dir}")
        return

    print(f"Generating Grad-CAM for {len(image_paths)} images...")
    visualize_batch(
        model=model,
        image_paths=image_paths,
        output_path="results/figures/gradcam_panel.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/checkpoints/best_model.pt")
    parser.add_argument("--image_dir", default="data/processed/test/barrett")
    parser.add_argument("--n_samples", type=int, default=6)
    args = parser.parse_args()
    main(args)
