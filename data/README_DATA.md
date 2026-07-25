# Data

## Source (verified provenance)

This project uses the **"Real and Fake Face Detection"** dataset created by the
Computational Intelligence and Photography Lab (CIPLAB), Department of
Computer Science, Yonsei University.

- Canonical host (Kaggle): https://www.kaggle.com/ciplab/real-and-fake-face-detection
- Info/mirror page: https://github.com/minostauros/Real-and-Fake-Face-Detection
- Lab: https://sites.google.com/site/seonjookim/

**Suggested citation** (no formal paper is published for this dataset; cite the
dataset page directly):

> Computational Intelligence and Photography Lab, Department of Computer
> Science, Yonsei University. "Real and Fake Face Detection." Kaggle, 2019.
> https://www.kaggle.com/ciplab/real-and-fake-face-detection

## IMPORTANT: fake-image generation mechanism (corrects earlier assumption)

The fake images in this dataset are **NOT GAN-generated**. Per the dataset
creators' own documentation, they are **expert-generated Photoshop
composites** -- human retouching experts splicing/blending eyes, nose, mouth,
or whole-face regions from different real photographs together. The dataset
was built explicitly *as an alternative to* GAN-based fake datasets: the
creators' stated motivation is that a classifier trained on GAN artifacts may
learn GAN-specific statistical patterns that don't transfer to fakes made by
a "completely different process" (manual expert compositing), so this corpus
tests robustness to that different process.

Filenames encode which facial region was replaced (see
`filename_description.jpg` in the mirror repo). Fake images are also grouped
into `easy`/`mid`/`hard`, but the dataset creators explicitly note **these
groups were assigned subjectively and are not recommended for use as strict
categories** -- worth citing as a caveat anywhere the difficulty labels are
used as if they were an objective ground truth.

**Action item for the paper:** every place that currently says "GAN-manipulated,"
"GAN-generated," or "GAN blending seams" (Sections 8, 8.1, 13.6, Table 4, and
the abstract's "synthetic face image" framing) should be corrected to describe
**expert Photoshop image-splicing/compositing** instead. This does not weaken
the paper's HOG-dominance argument -- if anything it sharpens it: blending
seams between spliced facial regions from different people (mismatched skin
tone, lighting, and alignment at the splice boundary) are precisely the kind
of local edge/gradient discontinuity HOG is designed to detect, which is a
more mechanistically specific explanation than a generic "GAN artifact" claim.

## License (confirmed via Kaggle API)

**License: CC-BY-NC-SA-4.0** (Creative Commons Attribution-NonCommercial-
ShareAlike 4.0 International), confirmed directly from the dataset's
Kaggle metadata via `kaggle datasets metadata`.

This means: (1) attribution to the CIPLAB/Yonsei creators is required
(see citation below) whenever this dataset or derived results are
published; (2) **the dataset may not be used for commercial purposes**;
(3) any redistributed derivative of the dataset itself must carry the
same CC-BY-NC-SA-4.0 license. This is compatible with academic
publication (non-commercial research use), but should be stated
explicitly in the paper's data availability statement, and this project
should not be positioned for any commercial use case without separately
licensing the data from CIPLAB/Yonsei.

**Official citation** (from the dataset's own metadata):

> Seonghyeon Nam, Seoung Wug Oh, Jae Yeon Kang, Chang Ha Shin, Younghyun
> Jo, Young Hwi Kim, Kyungmin Kim, Minho Shim, Sungho Lee, Yunji Kim, Suho
> Han, Gunhee Nam, Dasol Lee, Subin Jeon, In Cho, Woongoh Cho, Sejong
> Yang, Dongyoung Kim, Hyolim Kang, Sukjun Hwang, and Seon Joo Kim.
> (2019, January). Real and Fake Face Detection, Version 1. Retrieved
> [Date Retrieved] from
> https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection.



This project expects the dataset at
`data/real_and_fake_face/{training_real,training_fake}/*.jpg`.

```bash
git clone https://github.com/Sandilya-17/dataset.git /tmp/dataset_repo
ln -s /tmp/dataset_repo/real_and_fake_face_detection/real_and_fake_face data/real_and_fake_face
```

(`Sandilya-17/dataset` is a personal re-upload of the CIPLAB/Yonsei corpus
above, used here only for convenient cloning; the citation and license terms
are governed by the original CIPLAB/Kaggle source, not the re-upload.)



## Second dataset: NUAA Photograph Imposter Database (cross-dataset test)

Used for the genuine cross-dataset generalization test in report §8.2.

- Source: NUAA Photograph Imposter Database (Tan et al., print-attack PAD
  corpus), via the pre-cropped/aligned Kaggle mirror
  `immada/cropped-and-align-nuaa`.
- License: **MIT** (confirmed via `kaggle datasets download` output at
  download time).
- Original citation: X. Tan, Y. Li, J. Liu, L. Jiang. "Face Liveness
  Detection from A Single Image with Sparse Low Rank Bilinear
  Discriminative Model." ECCV 2010.
- Usage here: train+test splits combined into a single held-out target
  set (`data/nuaa_final/{real,fake}/`) of 12,611 images, evaluated with
  `src/cross_dataset_eval.py`. The model was never trained on any NUAA
  images -- this is a pure out-of-distribution generalization test.
