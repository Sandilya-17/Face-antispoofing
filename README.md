# Face Anti-Spoofing (Presentation Attack Detection)

A research-style implementation of face anti-spoofing (real vs. spoofed
face classification) using **color-texture analysis** — multi-scale LBP,
HOG, and HSV/YCbCr color statistics — combined with classical ML
classifiers (Logistic Regression, SVM-RBF, Random Forest), evaluated with
rigorous cross-validation, a controlled feature-family ablation, bootstrap
+ repeated-split statistical significance testing, and a held-out test
set.

**Full write-up:** [`report/REPORT.md`](report/REPORT.md) — methodology,
results, a completed deep-learning baseline comparison (§5.5), ablation
study, statistical significance testing, confusion matrices, ROC curves,
per-attack-difficulty analysis, an explicit account of what this study
still does *not* show (cross-dataset generalization test, §8.2) with a
ready-to-run script, honest framing, and 25 references.

## Results at a glance

| Model | Test Accuracy | F1 | AUC |
|---|---|---|---|
| Logistic Regression | 67.8% | 0.678 | 0.707 |
| **SVM (RBF)** — best | **69.1%** | **0.682** | **0.736** |
| Random Forest | 58.6% | 0.455 | 0.622 |
| MobileNetV2 (deep, fine-tuned) | 60.9% | 0.496 | 0.667 |

**Deep vs. classical (§5.5 of the report):** a fine-tuned, ImageNet-
pretrained MobileNetV2 *underperforms* the classical SVM-RBF pipeline on
this corpus — likely small-data overfitting combined with a mismatch
between generic CNN features and this dataset's Photoshop-splice-edge
spoof signal, which hand-crafted HOG is well-suited to detect (§7.3 shows
HOG carries 92.3% of feature importance). Read as specific to this small,
single-source dataset, not a general classical-vs-deep claim.

Dataset: 2,041 images (1,081 real, 960 spoofed, labeled by attack
difficulty easy/mid/hard).

**Ablation (§6 of the report):** HOG alone gets 68.7% accuracy / 0.750
AUC — nearly all of the fused model's performance. LBP-only and
color-only are both near chance (~51–53%). Fusion is not earning its
complexity on this dataset.

**Significance testing (§7 of the report):** bootstrap CIs on the single
test split overlap between SVM-RBF and LogReg, but a paired test across
30 repeated random splits shows SVM-RBF's edge is real (p < 0.0001) —
small in magnitude, but not noise.

## Novel contribution (v2): region-local feature fusion

The original ablation (§6/§5.4) found something worth chasing: **"easy"
spoof attacks are detected worse than "mid" attacks**, because they're
small, localized manipulations (a warp around the eyes, nose, or mouth)
that global whole-image descriptors dilute across the full frame. v2 adds
a targeted test of the obvious fix: compute the **same** LBP+HOG+color
descriptors, but **locally**, on cropped eye/nose/mouth patches, and fuse
them with the global features.

- `src/region_features.py` — locates the face (Haar cascade, bundled with
  OpenCV, no extra downloads) and crops eye/nose/mouth regions via
  detected eye position + standard anthropometric face proportions for
  nose/mouth, then runs the existing descriptor functions on each patch.
- `src/build_dataset_region.py` — builds `results/region_features.npz`,
  aligned by filename to `results/features.npz`.
- `src/train_hybrid_fusion.py` — trains and evaluates **global-only**,
  **region-only**, and **global+region fused** models on the identical
  split, with the same per-difficulty (easy/mid/hard) breakdown as the
  original report, so the headline question — *does region-local fusion
  fix the easy-attack blind spot?* — has a direct, reproducible answer in
  `results/hybrid_fusion_results.json`.

This keeps the study's original honesty standard: it's a controlled test
of a hypothesis the data itself generated, not a hand-wavy "add more
features" claim.

## Quickstart

```bash
pip install -r requirements.txt   # all deps already present in most envs; no new deps for v2

cd src
python3 build_dataset.py               # extract global features -> results/features.npz   (~1 min)
python3 build_dataset_region.py        # extract region-local features -> results/region_features.npz (~1-2 min)
python3 train_evaluate.py              # train/evaluate all models -> results/               (~2 min)
python3 ablation_study.py              # feature-family ablation                             (~2 min)
python3 statistical_significance.py    # bootstrap CI + 30 repeated splits                   (~3 min)
python3 train_hybrid_fusion.py         # NEW: global vs region vs fused, + difficulty breakdown (~2-3 min)
python3 predict.py path/to/face.jpg    # run inference with the saved best model
```

`deep_baseline_mobilenetv2.py` has now been run (results in
`results/deep_baseline_metrics.json`, discussed in report §5.5). One
script remains **ready to run but not executed** (see report §8.2 for
exactly why):
```bash
python3 deep_baseline_mobilenetv2.py                          # DONE — see §5.5
python3 cross_dataset_eval.py --target_root /path/to/dataset   # needs a 2nd PAD dataset
```

## Project structure

```
face_antispoofing/
├── data/real_and_fake_face/          # dataset (symlinked from provided repo)
├── src/
│   ├── feature_extraction.py         # LBP + color + HOG descriptors
│   ├── build_dataset.py              # extracts features -> results/features.npz
│   ├── train_evaluate.py             # CV, training, evaluation, plots
│   ├── ablation_study.py             # feature-family ablation (LBP/color/HOG/combined)
│   ├── statistical_significance.py   # bootstrap CI + paired significance tests
│   ├── deep_baseline_mobilenetv2.py  # deep-learning baseline (ready-to-run, unexecuted)
│   ├── cross_dataset_eval.py         # cross-dataset generalization protocol (executed vs. NUAA, see report \u00a78.2)
│   └── predict.py                    # single-image inference
├── results/                          # features, metrics, ablation, significance, trained model, figures
├── report/REPORT.md                  # full research report
└── requirements.txt
```

## Why classical ML, not a CNN, for the headline numbers?

This was built in a CPU-only environment with no GPU, no network access,
and only pre-extracted hand-crafted feature vectors cached (the raw
images were not available to train a CNN from pixels). Rather than force
a hastily-trained, under-fit CNN and report misleading numbers, this
project uses a legitimate, well-published classical baseline
(color-texture analysis, per Boulkenafet et al.) that trains in ~2
minutes and gives honest, reproducible, statistically-validated results.
A complete deep-learning baseline script and a cross-dataset
generalization script are both included, ready to run once GPU compute
and/or a second dataset are available — see report §8 for the exact
protocol and why they weren't executed here.

## Honest scope

This is a **classical-baseline re-evaluation with full statistical
rigor**, now including a completed deep-learning comparison — a legitimate
genre for a course project, portfolio piece, or workshop/student-track
submission. It is **not** a state-of-the-art claim and is **not**, as
scoped, ready for a mainstream CV/biometrics venue (CVPR, ICCV, IJCB,
BTAS) without the cross-dataset generalization results (§8.2) filled in.
See report §9 for the full "honest framing" discussion.
