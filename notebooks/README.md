# Notebook notes

`glaucoma_cdr_estimation_unet.ipynb` is the original Colab experiment retained
for provenance. It is not the supported inference entry point.

Before running it, note that it:

- mounts Google Drive;
- references `/content/drive/MyDrive/projects/origa/...` paths;
- expects precomputed `images.npy` and `t_mask.npy` arrays;
- expects a separately obtained ORIGA dataset and label workbook;
- trains for 50 epochs and can require a GPU;
- saves weights to the original author's Drive path; and
- uses maximum contour distance for its final CDR experiment, whereas the
  packaged library reports vertical mask extent.

Use the repository's `glaucoma_cdr` package for portable inference. Treat the
notebook as a record to refactor when the dataset and an explicit split
manifest are available.
