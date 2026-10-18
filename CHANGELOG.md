# Changelog

## Unreleased

### Added
- **Graphical interface** (`gui/`): runs the full workflow (Stages 1-3) on a folder of organoid videos without code, with run records for reproducibility and a command-line mode. Also analyses Cell Tracking Challenge datasets (annotations or your own results in the challenge format). Themes: Standard, Pink Panther, Forest Green. Tutorial: `gui/SAM-SPOT_GUI_tutorial.pdf`.

### Fixed
- `find_representative_images_per_cluster_PCA`: no longer crashes when a selected patch is larger than the mosaic cell (the cell grows to fit); clusters with fewer objects than mosaic cells leave cells empty (index -1, nan coordinates, size 0); colour patches and single-object clusters are handled.
- `compute_most_representative_image_patches`: handles fewer objects than mosaic cells, colour patches and a single SAM module.

## 2.0.0 (Oct 2026)

> **Compatibility.** Several defaults changed, so the default features differ from version 0.1.1. To compare with features computed by earlier SPOT versions, use the [`legacy`](https://github.com/fyz11/SPOT/tree/legacy) branch, the [`v0.1.1-natcomm`](https://github.com/fyz11/SPOT/releases/tag/v0.1.1-natcomm) release or `pip install SAM-SPOT==0.1.1`. Or restore the previous behaviour per function:
> - masked features: `use_mask=False`;
> - motion SIFT: `directional_motion_sift=False`;
> - EFD and shear-rate features: `elliptic_fourier_features=False` and `shear_features=False`;
> - module selection: `module_selection='legacy'`;
> - original implementation: `fast=False`.

### New defaults
- **Masked appearance and motion features** (`use_mask=True`): they describe only the object interior and no longer leak shape.
- **Directional / sign-aware motion SIFT** (`directional_motion_sift=True`): the flow vectors are SIFT's gradient field, curl and divergence are described sign aware, and shear has a principal-axis descriptor.
- **Elliptic Fourier descriptor** shape features (`EFD_*`) and **shear-rate** motion features (`*_shear_global_flow`). They have the same names and order as the PyPI 0.1.1 build.
- **Multi-scale SAM module selection** (`module_selection='multiscale'`, `parsimony=0.2`, `min_modules=3`).
- **Fast implementation** of the feature functions (`fast=True`, parallel with `n_jobs`), with the same outputs as the original.

### New features
- `*_torch` GPU variants of the shape, appearance and motion feature functions.
- PyTorch versions of the organoid detection and segmentation networks. The pretrained weights are archived on Zenodo ([doi:10.5281/zenodo.23438299](https://doi.org/10.5281/zenodo.23438299)) and downloaded automatically.
- `weight_sam_features`: equal weighting of the SAM sets and feature families, recommended for embeddings and clustering.
- `regress_out_confounders` / `ConfoundRegressor`: removes confounders, the batch by default, with continuous covariates per dataset.
- `SPOT.Features.augment.compute_rotation_averaged_features`: rotation-invariant features, with the equivalent SIFT bins collapsed.
- Demo notebook of the new features: `examples/SPOT_demo_new_features.ipynb`.

### Bug fixes
- The unmasked motion SIFT descriptors zeroed the flow inside, instead of outside, the object.
- The legacy SAM module selection picked a locally worst Davies-Bouldin score and crashed when the score curve had no peak.
- The shear features are now classified as motion; they were classified as shape.
- A constant speed now gives a uniform speed histogram.

### Packaging
- Python 3.9 - 3.13 and NumPy 2.
- PyTorch, UMAP, PaCMAP, statsmodels and fastcluster are installed by default.
- TensorFlow is optional (`pip install SAM-SPOT[keras-legacy]`), needed only for the original Keras networks.
