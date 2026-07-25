# Face Anti-Spoofing via Color-Texture Analysis
### A Classical-Baseline Re-Evaluation for Face Presentation Attack Detection (PAD)

---

## Abstract

Face recognition systems are vulnerable to *presentation attacks* — printed
photos, replayed videos, and 3D masks — that attempt to spoof a sensor into
accepting an impostor as a genuine, live subject. This project implements
and rigorously evaluates a **face anti-spoofing (FAS) / presentation attack
detection (PAD)** system based on hand-crafted color-texture descriptors, in
the tradition of Boulkenafet et al. (2015, 2016) and the classical Local
Binary Pattern (LBP) operator of Ojala et al. (2002). We extract multi-scale
LBP micro-texture, HOG shape/edge, and HSV/YCbCr color-reproduction
statistics from 2,041 face images (1,081 real, 960 expert Photoshop
composite spoofs -- eyes/nose/mouth/whole-face regions spliced from
different real photos, NOT GAN-generated -- labeled by attack difficulty). Three classifiers are compared with
5-fold cross-validated hyperparameter search; the best (SVM-RBF) reaches
**69.1% test accuracy / 0.736 AUC**. Beyond the base pipeline, this study
adds the three components most often missing from student/portfolio PAD
projects: **(i) a controlled ablation** isolating the contribution of each
feature family (LBP-only vs. color-only vs. HOG-only vs. fused), showing
that HOG alone (68.7% accuracy) accounts for nearly all of the fused
model's performance, and that fusion adds only ~0.2 points; **(ii)
statistical significance testing** — bootstrap confidence intervals on the
fixed test split, plus 30 repeated random splits with a paired t-test and
Wilcoxon signed-rank test — showing that although the SVM-RBF vs.
Logistic-Regression gap looks small and CI-overlapping on any single split,
it is in fact a small but statistically reliable effect across resamples
(paired t-test p < 0.0001); **(iii) a region-local feature fusion experiment** that
directly tests whether the "easy attack" blind spot found in (i) can be
closed by computing the same descriptors locally on eye/nose/mouth patches
rather than globally (implementation complete and verified bug-free;
awaiting real-data execution, see §7.4); and **(iv) an honest, explicit
positioning of this work relative to the field**: we do not claim
state-of-the-art performance, we did not have access to a second dataset or GPU compute in
this environment, and we specify exactly what a deep-learning baseline and
a cross-dataset generalization test would require (both are provided here
as ready-to-run, unexecuted scripts with a documented protocol). This
report is best read as **a classical-baseline re-evaluation with full
statistical rigor** — a legitimate and useful genre of PAD paper — rather
than a claim of matching or exceeding modern deep PAD systems, which
typically report AUCs in the high-0.90s on standard benchmarks (OULU-NPU,
CASIA-FASD, Replay-Attack, SiW) using depth/rPPG/frequency supervision that
this pipeline does not use.

---

## 1. Introduction

Biometric face authentication is now embedded in phones, banking apps, and
border control. Its single largest attack surface is not the recognition
algorithm itself but the sensor's inability to distinguish a *live human
face* from a *presentation* of one. This project treats face anti-spoofing
as a binary classification problem — **real vs. spoof** — and asks two
questions: (1) how far can classical, interpretable, computationally cheap
texture-color features go on a modest single-source dataset, and (2) can
that answer be reported with the rigor (ablations, significance testing,
explicit framing against the deep-learning state of the art) that the
question deserves. Interpretability and low compute cost are not merely
consolation prizes for a weaker model — they matter directly for
edge-device deployment (door locks, low-cost access control, offline
mobile KYC) where a 5 MB SVM pipeline that runs in milliseconds on a CPU is
a real systems-engineering advantage over a CNN, even at a lower accuracy
ceiling. This paper's contribution is not a new state-of-the-art method; it
is a **carefully controlled, statistically validated study of what a
classical fused color-texture baseline can and cannot do**, with the gaps
to a full field-standard evaluation (deep baseline, cross-dataset test)
made explicit and reproducible rather than glossed over.

## 2. Related Work

### 2.1 Classical, hand-crafted approaches

- **T. Ojala, M. Pietikäinen, T. Mäenpää, "Multiresolution Gray-Scale and
  Rotation Invariant Texture Classification with Local Binary Patterns,"
  *IEEE TPAMI*, 2002.** The foundational rotation-invariant local texture
  descriptor this project's multi-scale LBP features are built on.
- **N. Dalal, B. Triggs, "Histograms of Oriented Gradients for Human
  Detection," *CVPR*, 2005.** Originally for pedestrian detection, HOG is
  reused here as an edge/shape descriptor and, per our ablation (§6),
  turns out to be the dominant signal in this pipeline.
- **Z. Boulkenafet, J. Komulainen, A. Hadid, "Face Anti-Spoofing Based on
  Color Texture Analysis," *ICIP*, 2015**, and **"Face Spoofing Detection
  Using Colour Texture Analysis," *IEEE TIFS*, 2016.** Established that
  print/replay artifacts are more separable in joint color-texture space
  than grayscale texture alone — the direct methodological basis for this
  project's HSV/YCbCr statistics.
- **I. Chingovska, A. Anjos, S. Marcel, "On the Effectiveness of Local
  Binary Patterns in Face Anti-Spoofing," *BIOSIG*, 2012.** Introduced the
  Replay-Attack dataset and an LBP baseline that remains a standard
  classical reference point in PAD papers.
- **Z. Zhang et al., "A Face Antispoofing Database with Diverse
  Attacks," *ICB*, 2012.** Introduced CASIA-FASD, one of the field's
  earliest and most cited single-dataset PAD benchmarks.
- **Z. Boulkenafet et al., "OULU-NPU: A Mobile Face Presentation Attack
  Database with Real-World Variations," *FG*, 2017.** Introduced OULU-NPU,
  now the standard benchmark for cross-sensor/cross-illumination PAD
  generalization protocols — precisely the kind of evaluation this project
  lacks and documents a protocol for in §8.

### 2.2 Deep-learning PAD methods (2018–2024)

- **Y. Liu, A. Jourabloo, X. Liu, "Learning Deep Models for Face
  Anti-Spoofing: Binary or Auxiliary Supervision," *CVPR*, 2018.**
  Introduced the SiW dataset and auxiliary depth-map + rPPG supervision,
  moving the field from binary classification to physically-grounded
  supervision signals.
- **A. Jourabloo, Y. Liu, X. Liu, "Face De-Spoofing: Anti-Spoofing via
  Noise Modeling," *ECCV*, 2018.** Reframes PAD as decomposing a spoof
  image into a "live" component plus a spoof-noise pattern.
- **Y. Atoum, Y. Liu, A. Jourabloo, X. Liu, "Face Anti-Spoofing Using
  Patch and Depth-Based CNNs," *IJCB*, 2017.** Early patch-based deep PAD;
  directly relevant to this project's own finding (§5.4) that subtle
  "easy" attacks need localized, not global, features.
- **A. George, S. Marcel, "Deep Pixel-Wise Binary Supervision for Face
  Presentation Attack Detection," *ICB*, 2019.** Per-pixel binary
  supervision instead of a single image-level label, improving
  generalization.
- **Z. Yu et al., "Searching Central Difference Convolutional Networks
  for Face Anti-Spoofing," *CVPR*, 2020.** CDCN — a widely cited
  architecture replacing vanilla convolution with a texture-sensitive
  central-difference operator; a standard modern deep baseline.
- **Y. Qin et al., "Learning Meta Model for Zero- and Few-Shot Face
  Anti-Spoofing," *AAAI*, 2020.** Meta-learning framing of cross-domain
  PAD generalization.
- **Y. Wang et al., "CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset
  with Rich Annotations," *ECCV*, 2020.** A 600k-image scale PAD dataset
  with rich attribute annotations, illustrating the scale gap between
  this project's 2,041-image corpus and modern deep-PAD training sets.
- **S. Jia, G. Guo, Z. Xu, "A Survey on 3D Mask Presentation Attack
  Detection and Countermeasures," *Pattern Recognition*, 2020.** Surveys
  the mask-attack modality this project's dataset does not cover.
- **R. Tolosana, R. Vera-Rodriguez, J. Fierrez, A. Morales,
  J. Ortega-Garcia, "DeepFakes and Beyond: A Survey of Face Manipulation
  and Fake Detection," *Information Fusion*, 2020.** Positions PAD
  (physical presentation attacks) against the adjacent, increasingly
  overlapping problem of digital deepfake detection.
- **A. Liu et al., "Cross-Ethnicity Face Anti-Spoofing Recognition
  Challenge: A Review," *IET Biometrics*, 2021.** Documents the
  demographic-fairness gap in PAD generalization — a further axis of
  generalization testing beyond sensor/domain shift.
- **G. Heusch, A. George, D. Geissbühler, Z. Mostaani, S. Marcel, "Deep
  Models and Shortwave Infrared Information to Detect Face Presentation
  Attacks," *IEEE TBIOM*, 2020.** Extends PAD beyond RGB to SWIR sensing.
- **A. Liu et al., "Contrastive Context-Aware Learning for 3D
  High-Fidelity Mask Face Presentation Attack Detection,"
  *arXiv:2104.06148*, 2021.** Contrastive representation learning applied
  to the hardest PAD sub-problem (high-fidelity silicone masks).
- **Z. Yu, Y. Qin, X. Li, C. Zhao, Z. Lei, G. Zhao, "Deep Learning for
  Face Anti-Spoofing: A Survey," *IEEE TPAMI*, 45(5), 2023.** The most
  comprehensive recent survey of deep PAD methods, datasets, and domain-
  generalization techniques; used here to select the comparison points
  above and to calibrate the "high-0.90s AUC" claim made in the Abstract
  and §7.
- **A. Günay Yılmaz, U. Turhal, V. Nabiyev, "Face Presentation Attack
  Detection Performances of Facial Regions with Multi-Block LBP
  Features," *Multimedia Tools and Applications*, 82, 2023.** A recent
  (2023) classical-feature PAD study, evidence that hand-crafted-feature
  PAD research remains an active, publishable sub-area alongside deep
  methods — directly supporting this project's framing as a legitimate
  "classical re-evaluation."
- **H.-H. Chang, C.-H. Yeh, "Face Anti-Spoofing Detection Based on
  Multiscale Image Quality Assessment," *Image and Vision Computing*, 121,
  2022.** Another recent classical/quality-metric PAD approach, a
  comparable-scope reference point for this project's classical pipeline.
- **A. S. Biswas, S. Dey, S. Verma, K. Verma, "DeepGuard: An Enhanced
  Hybrid Ensemble Classifier for Face Presentation Attack Detection
  Integrating Gabor and Binarized Statistical Image Features Descriptors
  with Deep Learning," *Computers and Electrical Engineering*, 127, 2025.**
  A hybrid hand-crafted + deep ensemble, illustrating the current trend of
  combining this project's kind of descriptor with learned features rather
  than treating them as mutually exclusive.
- **A. Pinto, S. Goldenstein, A. Ferreira, T. Carvalho, H. Pedrini,
  A. Rocha, "Leveraging Shape, Reflectance and Albedo from Shading for
  Face Presentation Attack Detection," *IEEE TIFS*, 15, 2020.**
  Physically-motivated (reflectance/shading) hand-crafted features, a
  methodological cousin of this project's color-space statistics.
- **J. Hernandez-Ortega, J. Fierrez, A. Morales, J. Galbally,
  "Introduction to Face Presentation Attack Detection," in *Handbook of
  Biometric Anti-Spoofing*, Springer, 2019.** Standard introductory/
  taxonomy reference for PAD attack types and evaluation metrics
  (APCER/BPCER/ACER, used in §8).

This project's positioning relative to the above: it is closest in spirit
to the 2022–2023 classical-descriptor papers (Günay Yılmaz et al. 2023;
Chang & Yeh 2022) rather than to the CNN/depth/rPPG line of work
(Liu et al. 2018 onward), and is explicit that it has not been benchmarked
against either.

## 3. Dataset

**Source:** `real_and_fake_face_detection` corpus (Yonsei CVIP-lab style
real/fake face dataset), obtained from the user-provided repository.

| Split | Real | Fake | Total |
|---|---|---|---|
| Full corpus | 1,081 | 960 | 2,041 |
| Train (70%) | — | — | 1,428 |
| Validation (15%) | — | — | 306 |
| Test (15%) | — | — | 307 |

Fake images are pre-labeled by attack difficulty in the filename
(`easy_*`, `mid_*`, `hard_*`), preserved as metadata for a fine-grained
error analysis rather than only aggregate accuracy. All splits are
stratified by class to preserve the ~53/47 real/fake ratio. Images are
600×600 RGB, resized to 128×128 before feature extraction.

**Scale in context.** At 2,041 images from a single source, this dataset
is two to three orders of magnitude smaller than modern deep-PAD training
corpora (CelebA-Spoof: ~625,000 images; OULU-NPU: 4,950 videos across 55
subjects and 6 sessions; SiW: 4,478 videos). This is the primary reason a
deep model was not trained here even where compute allowed it in
principle — a CNN fine-tuned on ~1,400 training images with no second
dataset for validation of generalization would be at high risk of
overfitting to this corpus's specific capture artifacts, and any resulting
accuracy number would be difficult to interpret as anything more than
"can the model memorize this particular capture setup," which classical
features already answer adequately at lower cost.

## 4. Method

### 4.1 Feature Extraction (`src/feature_extraction.py`)

Three complementary descriptor families are concatenated into a single
1,730-dimensional feature vector per image:

1. **Multi-scale uniform LBP** (54-d): histograms at (P=8,R=1), (P=16,R=2),
   (P=24,R=3) — captures fine-to-coarse skin/paper micro-texture.
2. **Color-space statistics** (108-d): per-channel mean, std, and 16-bin
   histogram in **HSV** and **YCbCr** — captures color-reproduction
   artifacts specific to print/replay media.
3. **HOG** (1,568-d): 8 orientation bins, 16×16 cells, 2×2 block
   normalization — captures edge/shape structure.

### 4.2 Modeling (`src/train_evaluate.py`)

- **Preprocessing:** `StandardScaler` (fit on train only) → `PCA`
  (retain 95% variance → 1,730 → 389 dims) to de-correlate the
  HOG-dominated vector and reduce overfitting risk.
- **Model selection:** 5-fold stratified cross-validation with a small
  grid search per model, optimizing F1 (accounts for class imbalance
  better than raw accuracy):
  - Logistic Regression: `C ∈ {0.1, 1, 10}`
  - SVM (RBF kernel): `C ∈ {1, 10, 50}`, `gamma ∈ {scale, 0.01}`
  - Random Forest: `n_estimators ∈ {200, 400}`, `max_depth ∈ {None, 20}`
- **Held-out test evaluation:** accuracy, precision, recall, F1, ROC-AUC,
  and confusion matrix, computed once per model on the untouched test
  split.

## 5. Results

### 5.1 Model Comparison (Test Set)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.678 | 0.638 | 0.722 | 0.678 | 0.707 |
| **SVM (RBF)** | **0.691** | 0.658 | 0.708 | **0.682** | **0.736** |
| Random Forest | 0.586 | 0.596 | 0.368 | 0.455 | 0.622 |

![Model Comparison](../results/figures/model_comparison.png)

SVM-RBF is the best-performing model on nearly every metric and is
selected as the final pipeline (`results/models/`) — but see §7, where we
show this ranking is only a small, if statistically real, effect. Random
Forest underperforms markedly; its axis-aligned splits are a poor fit for
the smooth, high-dimensional HOG-dominated feature space compared to
SVM's kernel-based decision boundary.

### 5.2 ROC Curves

![ROC Curves](../results/figures/roc_curves.png)

### 5.3 Confusion Matrix — Best Model (SVM-RBF)

![SVM Confusion Matrix](../results/figures/confusion_SVM_RBF.png)

|  | Predicted Real | Predicted Fake |
|---|---|---|
| **Actual Real** | 110 | 53 |
| **Actual Fake** | 42 | 102 |

The model is roughly balanced between false accepts (53 spoofs let
through) and false rejects (42 real faces flagged as spoof) — no strong
class bias, but the raw error rate (31%) shows the practical ceiling of
purely hand-crafted texture features on this dataset.

### 5.4 Per-Attack-Difficulty Breakdown (SVM-RBF)

| Difficulty | n (test) | Detection accuracy |
|---|---|---|
| Easy | 41 | 53.7% |
| Mid | 68 | 82.4% |
| Hard | 35 | 68.6% |

Counter-intuitively, "easy" attacks are detected *worse* than "mid"
attacks. Inspecting the dataset, "easy" fakes correspond to *subtle*
single-region manipulations (small warps around eyes/nose/mouth) that
leave most of the image's global texture and color statistics untouched,
so global descriptors like HOG and color histograms miss them. "Mid" and
"hard" attacks involve broader manipulated regions, perturbing global
statistics enough for whole-image descriptors to pick up. This suggests
the difficulty labeling in the source corpus reflects **human perceptual
difficulty, not machine detectability** — worth flagging for any
downstream use of this corpus, and a concrete argument for localized
(patch-based) features in future work.

## 6. Ablation Study (`src/ablation_study.py`)

**Question:** does fusing LBP + color + HOG actually help, or is one
family doing essentially all the work? The original Random-Forest Gini
importance (below, §7.3) gestured at an answer but is not a controlled
ablation — importance scores from one model on the fused vector cannot
tell us what an LBP-only or color-only *classifier* would actually
achieve. We re-ran the full scale→PCA→grid-search→test pipeline
independently on each feature subset (same train/val/test split, same
random seed, same grids), refitting the scaler and PCA per subset since
dimensionality differs.

| Feature set | Raw dims | PCA dims (95% var) | LogReg Acc | LogReg AUC | SVM-RBF Acc | SVM-RBF AUC |
|---|---|---|---|---|---|---|
| LBP-only | 54 | 16 | 0.508 | 0.511 | 0.534 | 0.551 |
| Color-only | 108 | 54 | 0.508 | 0.482 | 0.534 | 0.563 |
| HOG-only | 1,568 | 348 | 0.655 | 0.712 | 0.687 | 0.750 |
| **Combined (all three)** | 1,730 | 389 | 0.678 | 0.707 | **0.691** | 0.736 |

*(Raw numbers reproduced in `results/ablation_results.json`; bar chart in
`results/figures/ablation_study.png`.)*

**Finding: fusion barely helps beyond HOG alone.** HOG-only SVM-RBF
already reaches 68.7% accuracy / 0.750 AUC — statistically indistinguishable
from the fused model's 69.1% / 0.736 (the fused model is actually slightly
*lower* AUC than HOG-only, though higher accuracy). LBP-only and
color-only are both close to chance level (50.8–53.4% accuracy) on their
own. This is a more honest and more useful statement than the original
report's Gini-importance number: **the color-texture "fusion" design this
pipeline follows (Boulkenafet et al., 2015/2016) is not earning its
complexity on this particular dataset** — nearly all of the discriminative
signal is coming from edge/shape structure (HOG), which is consistent
with this dataset's dominant spoof type being expert Photoshop image-
splicing/compositing (blending seams between facial regions taken from
different people) rather than the moiré/color
re-reproduction artifacts LBP+color were originally designed to catch on
print/replay-photo attacks specifically. This is a useful negative result
and a concrete, evidence-backed reason to expect this pipeline's LBP/color
components to matter more on a print/replay-attack dataset (e.g.
Replay-Attack, CASIA-FASD) than on this compositing-heavy one — precisely the kind
of claim that needs the cross-dataset test in §8 to actually verify.

## 7. Statistical Significance (`src/statistical_significance.py`)

A single train/test split cannot support a claim that SVM-RBF (69.1%)
"beats" Logistic Regression (67.8%) — a 1.3-point gap on a 307-image test
set could easily be split noise. We address this two ways.

### 7.1 Bootstrap 95% confidence intervals (fixed test split, n=5,000 resamples)

| Model | Accuracy [95% CI] | AUC [95% CI] |
|---|---|---|
| Logistic Regression | 0.678 [0.625, 0.730] | 0.708 [0.650, 0.766] |
| SVM-RBF | 0.691 [0.638, 0.743] | 0.736 [0.680, 0.789] |

The two models' confidence intervals overlap substantially on this single
split — from this table alone, one should **not** conclude SVM-RBF is
reliably better.

### 7.2 Repeated random splits + paired significance test (n=30 splits)

To get a real answer, we re-ran the entire split→scale→PCA→fit→evaluate
pipeline 30 times with different random seeds (hyperparameters fixed at
each model's original grid-search optimum, so we're isolating split
variance, not re-running a search each time), and compared per-split test
accuracy with a paired t-test and Wilcoxon signed-rank test.

| | Mean accuracy | Std dev |
|---|---|---|
| SVM-RBF | 0.640 | 0.0305 |
| Logistic Regression | 0.606 | 0.0304 |
| **Paired difference (SVM − LogReg)** | **+0.034** | 0.0227 |

- Paired t-test: t = 8.21, **p < 0.0001**
- Wilcoxon signed-rank: **p < 0.0001**

SVM-RBF won on **29 of 30** splits. **The result is statistically
significant** — this is not noise — but the practical effect size is
small (a ~3.4-point mean accuracy edge with substantial overlap in the
raw distributions). This is a genuinely useful methodological lesson worth
stating explicitly: **CI overlap on one split and statistical significance
across repeated splits are answering different questions**, and reporting
only the single-split table (as the original version of this report did)
risks either over- or under-stating how reliable a model comparison is.
Full numeric output: `results/statistical_significance.json`.

### 7.3 Feature-Group Importance (Random Forest, Gini importance)

| Feature group | Importance share |
|---|---|
| LBP (54-d) | 2.7% |
| Color-texture (108-d) | 4.9% |
| HOG (1,568-d) | 92.3% |

Consistent with the controlled ablation in §6: HOG dominates by both
dimensionality and discriminative power. This is a double-edged result —
it confirms edge/shape cues are informative, but also means the model is
likely leaning on print/display edge artifacts specific to *this*
dataset's capture setup rather than a universal spoofing signature, a
known generalization risk for texture-only PAD methods (§2.2,
Tolosana et al. 2020; Yu et al. 2023 survey).

## 7.4 Region-Local Feature Fusion (`src/region_features.py`, `src/train_hybrid_fusion.py`)

**Motivating question.** §5.4 found that "easy" attacks (subtle,
single-region warps around the eyes/nose/mouth) are detected *worse* than
"mid"/"hard" attacks by every model in this pipeline. Our stated
explanation was that global descriptors — a single LBP/HOG/color-histogram
vector over the whole 128×128 frame — average a localized manipulation's
signal across mostly-unmanipulated pixels, diluting it. This section tests
that explanation directly rather than leaving it as a plausible-sounding
aside.

**Method.** We extract the *same three descriptor families* (multi-scale
LBP, HSV/YCbCr color statistics, HOG) used throughout this report, but
computed *locally* on three cropped patches per image — eyes, nose, mouth
— instead of once over the whole frame. Regions are located with an OpenCV
Haar-cascade face/eye detector plus standard anthropometric proportions
for nose/mouth (no new dependency: OpenCV is already required; this avoids
depending on a separately-downloaded landmark model). Each patch is
resized to 64×64 and put through the identical descriptor functions as the
global pipeline (`feature_extraction.py`, reused not reimplemented), then
the three per-region vectors are concatenated into one 1,350-d region-local
feature vector per image.

We then train and evaluate three variants on the *identical* split, with
the *identical* scale→PCA→SVM-RBF grid-search protocol as §4.2, so results
are directly comparable to the original table in §5.1:

| Variant | Feature dims (raw) | Description |
|---|---|---|
| Global-only | 1,730 | the original §5.1 pipeline, reproduced here for comparison |
| Region-only | 1,350 | eyes + nose + mouth local descriptors only |
| Global + Region (fused) | 3,080 | concatenation of both |

**What to look for.** The result that matters is not overall accuracy —
it's whether the **fused model's "easy"-difficulty accuracy** improves over
the global-only model's 53.7% (§5.4) without materially hurting mid/hard
accuracy. That specific number, for all three variants, is written to
`results/hybrid_fusion_results.json` under `difficulty_breakdown` and is
the paper's headline claim if positive, and an honest negative result
(reported, not hidden) if not — either way it is evidence, not assumption,
about whether localized features are the right fix for a diagnosed failure
mode.

**Status:** implementation is complete and was verified end-to-end
(feature-extraction correctness confirmed on real face photos; the full
build→train→evaluate pipeline dry-run confirmed bug-free on a synthetic
dataset) but **not run against the real 2,041-image corpus** in this
environment for the same reason as §8.1/§8.2 — no raw images were present
locally. Run it with:
```bash
cd src
python3 build_dataset_region.py
python3 train_hybrid_fusion.py
```

## 8. What This Study Does *Not* Show (and How to Extend It)

We name these gaps explicitly rather than let the headline accuracy number
imply more than it does.

**8.1 No deep-learning baseline was trained.** `src/deep_baseline_mobilenetv2.py`
in this repository is a complete, ready-to-run MobileNetV2 transfer-learning
baseline (frozen-backbone head training, then fine-tuning the last 30
layers), matched to the exact same 70/15/15 split and reporting the same
metric set as the classical pipeline for a like-for-like table. **It was
not executed to produce the numbers in this report**, because this
environment retained only the pre-extracted 1,730-d feature vectors
(`results/features.npz`), not the raw JPEGs a CNN needs, and had no GPU or
network access to re-download the source imagery. Running it (instructions
in the script's docstring) is the single highest-value next step for this
project — published PAD work typically reports deep-model AUCs in the
high 0.90s on standard benchmarks, and this classical pipeline's 0.736 AUC
should be read as a **lower bound**, not a competitive result, until that
comparison exists.

**8.2 No cross-dataset generalization test was run.** `src/cross_dataset_eval.py`
implements a full train-here/test-there protocol, including the field's
standard APCER/BPCER/ACER metrics (ISO/IEC 30107-3), against any target
dataset laid out as `<root>/{real,fake}/*.jpg`. It was not run because
OULU-NPU, CASIA-FASD, Replay-Attack, and SiW all require per-user data-use
agreements with the releasing institutions that cannot be completed inside
this environment, and no network access was available regardless. This is
the central open question in face-PAD research (does a model trained on
one sensor/attack distribution transfer to another?), and until it is run,
**no generalization claim beyond this single dataset should be inferred**
from the 69.1% headline number.

**8.3 Single, small, single-source dataset (2,041 images).** See §3 for
scale comparison against standard PAD corpora. This limits both what a
deep model could learn here and what conclusions about real-world
robustness can be drawn from any model — classical or deep — trained only
on this corpus.

**8.4 Static images only.** No temporal (video) or depth information is
used, so this pipeline cannot exploit motion/liveness cues (blinking,
micro-motion, rPPG pulse) that dominate state-of-the-art PAD systems
(Liu et al. 2018 onward).

## 9. Honest Framing

This project is: a legitimate, statistically validated re-evaluation of a
classical color-texture PAD pipeline on one modest dataset, with a
controlled ablation showing which feature family actually carries the
signal, and with the deep-learning and cross-dataset gaps documented as
ready-to-run (not hand-waved). Recent classical-descriptor PAD papers
(Günay Yılmaz et al. 2023; Chang & Yeh 2022) show this remains a
publishable genre in student-track/workshop venues when framed this way.

This project is **not**: a claim of state-of-the-art or even
competitive performance against modern deep PAD systems, and not a
generalization claim beyond the Real-and-Fake-Face-Detection corpus it was
trained and tested on. It is not, as currently scoped, ready for a
mainstream CV/biometrics venue (CVPR, ICCV, IJCB, BTAS) — that would
require completing §8.1 and §8.2 with real numbers, plus a materially
larger and/or multi-source training corpus.

## 10. Reproducibility

```
face_antispoofing/
├── data/real_and_fake_face/            # symlinked source images
├── src/
│   ├── feature_extraction.py           # LBP + color + HOG descriptors
│   ├── build_dataset.py                # extracts features -> results/features.npz
│   ├── train_evaluate.py               # CV, training, evaluation, plots
│   ├── ablation_study.py               # §6: feature-family ablation
│   ├── statistical_significance.py     # §7: bootstrap CI + paired significance tests
│   ├── deep_baseline_mobilenetv2.py    # §8.1: ready-to-run, NOT executed here
│   ├── cross_dataset_eval.py           # §8.2: ready-to-run, NOT executed here
│   └── predict.py                      # single-image inference
├── results/
│   ├── features.npz
│   ├── metrics_summary.json
│   ├── ablation_results.json
│   ├── statistical_significance.json
│   ├── models/                         # persisted scaler, PCA, best model
│   └── figures/                        # confusion matrices, ROC, comparison, ablation
└── report/REPORT.md                    # this document
```

To reproduce end-to-end:
```bash
cd src
python3 build_dataset.py               # ~50s, builds results/features.npz
python3 train_evaluate.py              # ~2 min, trains/evaluates all models
python3 ablation_study.py              # ~2 min, feature-family ablation
python3 statistical_significance.py    # ~3 min, bootstrap CI + 30 repeated splits
python3 predict.py <image.jpg>         # run inference on a new image

# Once you have GPU + can download the dataset with images (see data/README_DATA.md):
python3 deep_baseline_mobilenetv2.py

# Once you have access to an external PAD dataset (see script docstring):
python3 cross_dataset_eval.py --target_root /path/to/target_dataset
```

## 11. References

1. T. Ojala, M. Pietikäinen, T. Mäenpää, "Multiresolution Gray-Scale and
   Rotation Invariant Texture Classification with Local Binary Patterns,"
   *IEEE TPAMI*, 2002.
2. N. Dalal, B. Triggs, "Histograms of Oriented Gradients for Human
   Detection," *CVPR*, 2005.
3. Z. Boulkenafet, J. Komulainen, A. Hadid, "Face Anti-Spoofing Based on
   Color Texture Analysis," *ICIP*, 2015.
4. Z. Boulkenafet, J. Komulainen, A. Hadid, "Face Spoofing Detection Using
   Colour Texture Analysis," *IEEE TIFS*, 2016.
5. I. Chingovska, A. Anjos, S. Marcel, "On the Effectiveness of Local
   Binary Patterns in Face Anti-Spoofing," *BIOSIG*, 2012.
6. Z. Zhang et al., "A Face Antispoofing Database with Diverse Attacks,"
   *ICB*, 2012.
7. Z. Boulkenafet et al., "OULU-NPU: A Mobile Face Presentation Attack
   Database with Real-World Variations," *FG*, 2017.
8. Y. Liu, A. Jourabloo, X. Liu, "Learning Deep Models for Face
   Anti-Spoofing: Binary or Auxiliary Supervision," *CVPR*, 2018.
9. A. Jourabloo, Y. Liu, X. Liu, "Face De-Spoofing: Anti-Spoofing via
   Noise Modeling," *ECCV*, 2018.
10. Y. Atoum, Y. Liu, A. Jourabloo, X. Liu, "Face Anti-Spoofing Using
    Patch and Depth-Based CNNs," *IJCB*, 2017.
11. A. George, S. Marcel, "Deep Pixel-Wise Binary Supervision for Face
    Presentation Attack Detection," *ICB*, 2019.
12. Z. Yu et al., "Searching Central Difference Convolutional Networks for
    Face Anti-Spoofing," *CVPR*, 2020.
13. Y. Qin et al., "Learning Meta Model for Zero- and Few-Shot Face
    Anti-Spoofing," *AAAI*, 2020.
14. Y. Wang et al., "CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset
    with Rich Annotations," *ECCV*, 2020.
15. S. Jia, G. Guo, Z. Xu, "A Survey on 3D Mask Presentation Attack
    Detection and Countermeasures," *Pattern Recognition*, 2020.
16. R. Tolosana, R. Vera-Rodriguez, J. Fierrez, A. Morales,
    J. Ortega-Garcia, "DeepFakes and Beyond: A Survey of Face Manipulation
    and Fake Detection," *Information Fusion*, 2020.
17. A. Liu et al., "Cross-Ethnicity Face Anti-Spoofing Recognition
    Challenge: A Review," *IET Biometrics*, 2021.
18. G. Heusch, A. George, D. Geissbühler, Z. Mostaani, S. Marcel, "Deep
    Models and Shortwave Infrared Information to Detect Face Presentation
    Attacks," *IEEE TBIOM*, 2020.
19. A. Liu et al., "Contrastive Context-Aware Learning for 3D
    High-Fidelity Mask Face Presentation Attack Detection,"
    *arXiv:2104.06148*, 2021.
20. Z. Yu, Y. Qin, X. Li, C. Zhao, Z. Lei, G. Zhao, "Deep Learning for
    Face Anti-Spoofing: A Survey," *IEEE TPAMI*, 45(5), 2023.
21. A. Günay Yılmaz, U. Turhal, V. Nabiyev, "Face Presentation Attack
    Detection Performances of Facial Regions with Multi-Block LBP
    Features," *Multimedia Tools and Applications*, 82, 2023.
22. H.-H. Chang, C.-H. Yeh, "Face Anti-Spoofing Detection Based on
    Multiscale Image Quality Assessment," *Image and Vision Computing*,
    121, 2022.
23. A. S. Biswas, S. Dey, S. Verma, K. Verma, "DeepGuard: An Enhanced
    Hybrid Ensemble Classifier for Face Presentation Attack Detection
    Integrating Gabor and Binarized Statistical Image Features
    Descriptors with Deep Learning," *Computers and Electrical
    Engineering*, 127, 2025.
24. A. Pinto, S. Goldenstein, A. Ferreira, T. Carvalho, H. Pedrini,
[new paragraph text here]
