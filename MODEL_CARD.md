# Model card

## Summary

This repository contains a three-channel U-Net for optic disc and optic cup
segmentation from retinal fundus images. The segmentation output is used to
calculate a vertical cup-to-disc ratio (CDR).

The model is an educational research artifact. It has not been validated as a
medical device and is not intended for clinical decision-making.

## Architecture

- Input: `256 x 256 x 3` RGB image
- Encoder filters: 16, 32, 64, 128
- Bottleneck filters: 256
- Decoder: transposed convolutions with U-Net skip connections
- Output: three sigmoid channels
- Interpreted channels: optic disc (0), optic cup (1)

The third output channel is retained for compatibility with the original
training notebook but is not used by the packaged CDR calculation.

## Training provenance

The original notebook identifies
[ORIGA(-light)](https://pubmed.ncbi.nlm.nih.gov/21095735/) as the source dataset
and loads images, masks, and labels from private Google Drive paths. The
repository does not include the dataset, a machine-readable split manifest, or
a complete training environment lock file. The committed HDF5 artifact is
therefore treated as a legacy research weight file.

Weight file:

```text
weights/model_weights.h5
SHA-256: 907da603a355ce7d52e80c30337ef3f8a77f49add1e5d2cfc084087c7b2bd7cc
```

## Intended use

- Learning about medical-image segmentation and U-Net architectures
- Reproducing the software pipeline with an appropriately licensed dataset
- Comparing segmentation post-processing and CDR measurement approaches

## Out-of-scope use

- Diagnosing or screening for glaucoma
- Making treatment or referral decisions
- Processing patient data without the required privacy, security, and ethical
  controls
- Claiming clinical performance from the included notebook outputs

## Measurement behavior

The supported package thresholds disc and cup probabilities, constrains the
measured cup to the predicted disc, and calculates:

```text
vertical CDR = cup mask vertical extent / disc mask vertical extent
```

This differs from the legacy notebook, which calculates the greatest pairwise
Euclidean distance between contour points. The packaged definition matches the
vertical CDR terminology used in this repository and avoids an O(n^2) contour
distance calculation.

## Evaluation status

No independently reproducible benchmark is included. The notebook contains
historical training outputs, but the repository lacks the source dataset and a
fixed split manifest needed to verify those results. Accordingly, this project
makes no accuracy, sensitivity, specificity, Dice, IoU, or diagnostic claim.

## Risks and limitations

- Predictions may fail under changes in camera, illumination, resolution,
  ethnicity, disease distribution, or image quality.
- Segmentation artifacts directly affect the resulting CDR.
- A single CDR estimate does not capture the full clinical assessment of
  glaucoma.
- The original training and evaluation process may contain undocumented data
  leakage, labeling, or preprocessing assumptions.
- The included weights should be retrained and externally validated before any
  serious research comparison.
