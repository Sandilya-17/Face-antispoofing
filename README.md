# Face Anti-Spoofing (Presentation Attack Detection)

A face anti-spoofing / presentation attack detection (PAD) study on the
Real-and-Fake-Face-Detection corpus (CIPLAB, Yonsei University), combining
a classical, interpretable color-texture pipeline with a novel GAN-based
one-class reconstruction anomaly score.

Full methodology, results, discussion, and references: **[report/REPORT.md](report/REPORT.md)**.

## What's here

- **Classical pipeline** (`src/feature_extraction.py`, `src/build_dataset.py`,
  `src/train_evaluate.py`): multi-scale LBP + HOG + HSV/YCbCr color
  statistics -> PCA -> SVM-RBF / Logistic Regression / Random Forest.
  Best result: **69.1% accuracy, 0.736 AUC** (report Sec. 5).
- **Region-local features** (`src/region_features.py`,
  `src/train_hybrid_fusion.py`, `src/train_score_fusion.py`): eye/nose/mouth
  patch descriptors, targeting the "easy"-attack blind spot in the global
  pipeline (report Sec. 5.4).
- **GAN-based one-class reconstruction anomaly scoring** (`src/gan_reconstruction.py`,
  `src/train_gan_fusion.py`, `src/train_gan_score_fusion.py`): a small
  convolutional generator-discriminator pair trained *only* on genuine
  faces, whose reconstruction residual and discriminator score are used as
  a learned anomaly cue. Alone, this catches 94-98% of spoofed images at
  every attack difficulty (vs. 53.7% for the classical pipeline on "easy"
  attacks) but is poorly calibrated on held-out real faces; report Sec. 6
  covers the full methodology, fusion experiments, and honest discussion of
  this asymmetry.

## Data

This project uses the **Real and Fake Face Detection** dataset (CIPLAB,
Yonsei University), which is **not GAN-generated** but consists of expert
Photoshop-composited region splices. See [data/README_DATA.md](data/README_DATA.md)
for full provenance, licensing (CC-BY-NC-SA-4.0), and setup instructions.

## Quickstart

```bash
# 1. Get the dataset (see data/README_DATA.md for details/citation)
git clone https://github.com/Sandilya-17/dataset.git /tmp/dataset_repo
ln -s /tmp/dataset_repo/real_and_fake_face_detection/real_and_fake_face data/real_and_fake_face

# 2. Install dependencies
pip install -r requirements.txt

# 3. Classical pipeline
cd src
python3 build_dataset.py
python3 train_evaluate.py

# 4. GAN-based extension (Sec. 6 of the report)
python3 gan_reconstruction.py
python3 train_gan_fusion.py
python3 train_gan_score_fusion.py
```

See `report/REPORT.md` Section 9 for the full reproducibility guide.
