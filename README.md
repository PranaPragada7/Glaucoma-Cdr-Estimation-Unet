# Glaucoma CDR Estimation using U-Net (Optic Disc/Cup Segmentation)

This repository implements **optic disc** and **optic cup** segmentation from retinal fundus images using a **U-Net** model (TensorFlow/Keras), and estimates the **Cup-to-Disc Ratio (CDR)** from the predicted masks.

> Educational/research project only — **not** a clinical diagnostic tool.

---

## Overview
**Goal:** Segment disc/cup → compute CDR, a common glaucoma-related measurement used in research pipelines.

**High-level pipeline**
1. Input fundus image → resize/normalize
2. U-Net predicts disc and cup masks
3. Postprocess masks (threshold + contour extraction)
4. Compute **CDR = cup vertical diameter / disc vertical diameter** (contour-based)

---

## Repository structure
```text
.
├── README.md
├── notebooks/
│   └── glaucoma_cdr_estimation_unet.ipynb
└── weights/
    └── model_weights.h5
