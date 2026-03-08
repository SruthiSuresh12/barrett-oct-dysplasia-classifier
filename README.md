# barrett-oct-dysplasia-classifier

**AI-assisted dysplasia risk scoring for Barrett's esophagus endoscopy images.**

Fine-tuned EfficientNet-B0 with Grad-CAM saliency and a composite Dysplasia Risk Score (DRS), built to support clinical decision-making in tethered capsule OCT workflows.

---

## Motivation

Barrett's esophagus (BE) is the primary precursor to esophageal adenocarcinoma. Early detection of dysplasia — the transition from metaplastic Barrett's epithelium to neoplastic tissue — is clinically critical but requires expert endoscopist review. Tethered capsule OCT devices (Liang et al., *Gastroenterology* 2023; Chen et al., *Nature Medicine* 2019) can capture high-resolution mucosal images non-invasively, but lack a standardised computational risk scoring layer.

This project bridges that gap: a lightweight deep learning classifier deployable on the image output of tethered capsule devices, providing per-frame dysplasia risk scores with interpretable saliency maps.

---

## Dataset

**HyperKvasir** (Borgli et al., *Scientific Data* 2020; DOI: [10.1038/s41597-020-00622-y](https://doi.org/10.1038/s41597-020-00622-y))  
Open-access, CC BY 4.0 — download at [datasets.simula.no/hyper-kvasir](https://datasets.simula.no/hyper-kvasir/)

| Class | Folders used | N images |
|---|---|---|
| Barrett's | `barretts/` + `barretts-short-segment/` | 94 |
| Normal | `z-line/` | 932 |
| Esophagitis | `esophagitis-a/` + `esophagitis-b-d/` | 663 |

> **Note on class imbalance:** Only 94 Barrett's images vs 932 normal — a clinically realistic constraint (rare disease). This is handled via `WeightedRandomSampler` and heavy augmentation including elastic transforms that mimic peristaltic motion artefacts in capsule imaging.

---

## Pipeline

```
HyperKvasir (raw)
      │
      ▼
prepare_dataset.py      → stratified 70/15/15 train/val/test splits
      │
      ▼
EfficientNet-B0 (fine-tuned)
  ├── Classification head  → P(normal | barrett | esophagitis)
  └── DRS regression head  → auxiliary risk scalar [0, 1]
      │
      ▼
Grad-CAM                 → mucosal region saliency maps
      │
      ▼
DRS scoring              → texture entropy (GLCM) + gland irregularity (edge density)
      │
      ▼
Clinical report figure   → per-patient risk summary
```

---

## Model Architecture

- **Backbone:** EfficientNet-B0 (ImageNet pretrained, 5.3M params)
- **Training strategy:** Layers 1–4 frozen for first 10 epochs; full fine-tuning thereafter
- **Loss:** CrossEntropy + auxiliary DRS MSE (weighted 0.8 / 0.2)
- **Sampler:** WeightedRandomSampler (compensates for 10:1 normal:barrett imbalance)
- **Augmentation:** Random flip, rotation, elastic transform, colour jitter, Gaussian blur, random erasing

---

## Dysplasia Risk Score (DRS)

```
DRS = 0.50 × P(barrett) + 0.25 × texture_entropy + 0.15 × gland_irregularity − 0.10 × P(normal)
```

| DRS range | Risk tier | Clinical action |
|---|---|---|
| < 0.30 | Low | Routine annual surveillance |
| 0.30 – 0.55 | Moderate | 6-month follow-up |
| > 0.55 | High | Immediate biopsy referral |

**Texture entropy** is computed from Gray-Level Co-occurrence Matrix (GLCM) features — Barrett's columnar mucosa with goblet cells produces a distinct GLCM signature vs normal squamous epithelium.

**Gland irregularity** uses Canny edge density in high-activation regions as a proxy for architectural disarray (back-to-back glands, loss of polarity) associated with dysplastic progression.

---

## Results

*Results will be updated after full training run on HyperKvasir dataset.*

| Metric | Value |
|---|---|
| Val Accuracy | — |
| Barrett's F1 | — |
| Barrett's AUC-ROC | — |
| Mean DRS (barrett test set) | — |

---

## Grad-CAM Output

Saliency maps highlight the mucosal regions (glandular architecture, goblet cell distribution, vascular patterns) that most influenced each classification — analogous to regions an endoscopist would scrutinise during tethered capsule review.

*Figure: `results/figures/gradcam_panel.png` (generated after training)*

---

## Quickstart

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate barrett-classifier

# 2. Download HyperKvasir
# Visit: https://datasets.simula.no/hyper-kvasir/
# Extract to: data/raw/hyper-kvasir/

# 3. Prepare dataset
python src/data/prepare_dataset.py \
    --hyperkvasir_root data/raw/hyper-kvasir \
    --output_dir data/processed

# 4. Train
python src/models/train.py --data_dir data/processed --epochs 50

# 5. Grad-CAM
python src/visualization/gradcam.py \
    --checkpoint results/checkpoints/best_model.pt \
    --image_dir data/processed/test/barrett

# 6. DRS report
python src/scoring/drs_report.py \
    --checkpoint results/checkpoints/best_model.pt \
    --patient_dir data/processed/test/barrett \
    --patient_id PATIENT_001

# Or run the full pipeline with Snakemake:
snakemake --cores 4 all
```

---

## Repository Structure

```
barrett-oct-dysplasia-classifier/
├── configs/
│   ├── train_params.yaml        # Model + training hyperparameters
│   └── drs_weights.yaml         # DRS component weights and thresholds
├── src/
│   ├── data/
│   │   ├── prepare_dataset.py   # HyperKvasir → train/val/test splits
│   │   └── dataset.py           # DataLoader with WeightedRandomSampler
│   ├── models/
│   │   ├── efficientnet_barrett.py  # EfficientNet-B0 + DRS head + loss
│   │   └── train.py             # Training loop with early stopping
│   ├── visualization/
│   │   └── gradcam.py           # Grad-CAM saliency generation
│   └── scoring/
│       └── drs_report.py        # DRS computation + clinical report
├── results/
│   ├── checkpoints/             # Model weights
│   ├── figures/                 # Grad-CAM panels, DRS reports
│   └── tables/                  # Per-image DRS scores, training history
├── Snakefile                    # End-to-end pipeline
└── environment.yml
```

---

## References

- Borgli H et al. HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy. *Scientific Data* 2020. DOI: [10.1038/s41597-020-00622-y](https://doi.org/10.1038/s41597-020-00622-y)
- Tan M, Le QV. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML* 2019. arXiv: [1905.11946](https://arxiv.org/abs/1905.11946)
- Selvaraju RR et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV* 2017. DOI: [10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74)
