"""
Snakemake pipeline: Barrett's Esophagus Dysplasia Classifier

Steps:
  1. prepare   — organise HyperKvasir into train/val/test splits
  2. train      — fine-tune EfficientNet-B0 with DRS head
  3. evaluate   — generate confusion matrix, ROC curves, metrics
  4. gradcam    — produce Grad-CAM saliency panels
  5. report     — generate DRS clinical report

Run:
  snakemake --cores 4 all
"""

configfile: "configs/train_params.yaml"

rule all:
    input:
        "results/tables/training_history.csv",
        "results/figures/gradcam_panel.png",
        "results/figures/drs_report_TEST_COHORT.png",
        "results/tables/drs_TEST_COHORT.csv"

rule prepare_dataset:
    input:
        hyperkvasir="data/raw/hyper-kvasir"
    output:
        directory("data/processed/train"),
        directory("data/processed/val"),
        directory("data/processed/test")
    shell:
        "python src/data/prepare_dataset.py "
        "--hyperkvasir_root {input.hyperkvasir} "
        "--output_dir data/processed"

rule train:
    input:
        "data/processed/train",
        "data/processed/val"
    output:
        "results/checkpoints/best_model.pt",
        "results/tables/training_history.csv"
    shell:
        "python src/models/train.py "
        "--data_dir data/processed "
        "--epochs 50"

rule gradcam:
    input:
        checkpoint="results/checkpoints/best_model.pt",
        images="data/processed/test/barrett"
    output:
        "results/figures/gradcam_panel.png"
    shell:
        "python src/visualization/gradcam.py "
        "--checkpoint {input.checkpoint} "
        "--image_dir {input.images} "
        "--n_samples 6"

rule drs_report:
    input:
        checkpoint="results/checkpoints/best_model.pt",
        images="data/processed/test/barrett"
    output:
        "results/figures/drs_report_TEST_COHORT.png",
        "results/tables/drs_TEST_COHORT.csv"
    shell:
        "python src/scoring/drs_report.py "
        "--checkpoint {input.checkpoint} "
        "--patient_dir {input.images} "
        "--patient_id TEST_COHORT"
