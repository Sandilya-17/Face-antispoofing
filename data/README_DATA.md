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

## License (needs manual verification -- not yet confirmed)

The exact license/usage terms are listed on the Kaggle dataset page itself
(https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection), which
requires a logged-in browser session to render and could not be verified
via automated fetch. **Before submitting anywhere with a provenance/license
requirement, log in to Kaggle and copy the exact license field from the
dataset's "About this Dataset" / metadata panel into this file.** Do not
assume a specific license (e.g., CC0, CC-BY) without checking -- Kaggle
datasets vary widely in their stated terms, and misstating a license in a
publication is a real (and easily avoided) problem.



This project expects the dataset at
`data/real_and_fake_face/{training_real,training_fake}/*.jpg`.

```bash
git clone https://github.com/Sandilya-17/dataset.git /tmp/dataset_repo
ln -s /tmp/dataset_repo/real_and_fake_face_detection/real_and_fake_face data/real_and_fake_face
```

(`Sandilya-17/dataset` is a personal re-upload of the CIPLAB/Yonsei corpus
above, used here only for convenient cloning; the citation and license terms
are governed by the original CIPLAB/Kaggle source, not the re-upload.)

