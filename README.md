# SPOT (Shape, appearance, motion Phenotype Observation Tool)

[![PyPI version](https://badge.fury.io/py/SAM-SPOT.svg)](https://badge.fury.io/py/SAM-SPOT)
[![Downloads](https://pepy.tech/badge/SAM-SPOT)](https://pepy.tech/project/SAM-SPOT)
[![Downloads](https://pepy.tech/badge/SAM-SPOT/month)](https://pepy.tech/project/SAM-SPOT)
[![Python version](https://img.shields.io/pypi/pyversions/SAM-SPOT)](https://pypistats.org/packages/SAM-SPOT)
[![GitHub stars](https://img.shields.io/github/stars/fyz11/SPOT?style=social)](https://github.com/fyz11/SPOT/)
[![GitHub forks](https://img.shields.io/github/forks/fyz11/SPOT?style=social)](https://github.com/fyz11/SPOT/)

SPOT turns movies of moving, changing objects (cells, organoids, ...) into a standardised **Shape, Appearance and Motion (SAM) phenome** per object and time point, and analyses it like single-cell sequencing data: embeddings, phenotype clusters, trajectories and **SAM modules** of co-varying features. It is push-and-go: no prior knowledge of the objects' behaviour is needed.

<p align="center">
  <img src="https://github.com/fyz11/SPOT/blob/main/docs/pictures/main_workflow.jpg" width=100%/>
</p>

<!-- TOC start -->
**Contents**
* [Quick start](#quick-start)
* [What's new](#whats-new)
* [The SAM phenome](#the-sam-phenome)
  + [Computing the phenome](#computing-the-phenome)
  + [Masked appearance and motion features](#masked-appearance-and-motion-features)
  + [Faster extraction](#faster-extraction)
  + [Rotation-invariant features](#rotation-invariant-features)
* [Analysing the SAM phenome](#analysing-the-sam-phenome)
  + [Equal weighting of the SAM sets and feature families](#equal-weighting-of-the-sam-sets-and-feature-families)
  + [Regressing out confounders](#regressing-out-confounders)
  + [SAM module selection](#sam-module-selection)
* [Pretrained networks for organoid detection and segmentation](#pretrained-networks-for-organoid-detection-and-segmentation)
* [Example scripts](#example-scripts)
* [To install](#to-install)
* [Associated papers](#associated-papers)
* [Associated datasets](#associated-datasets)
* [Documentation](#documentation)
* [Copyright information](#copyright-information)
<!-- TOC end -->

<!-- TOC --><a name="quick-start"></a>
## Quick start
```shell
conda create -n SAM-SPOT_env python=3.12
conda activate SAM-SPOT_env
conda install -c conda-forge python-annoy
pip install SAM-SPOT
```
Then open the demo notebook [`examples/SPOT_demo_new_features.ipynb`](examples/SPOT_demo_new_features.ipynb): a few minutes, no data download, it walks through feature extraction, normalisation, weighting, confound regression, rotation invariance, SAM modules and UMAP / PaCMAP embeddings on a synthetic movie. More installation options: [To install](#to-install).

<!-- TOC --><a name="whats-new"></a>
## What's new
**Oct 2026: version 2.0**
- **Faster:** an optimised implementation is now the default (`fast=True`, parallel with `n_jobs`), plus `*_torch` GPU variants, all with the same outputs. [More](#faster-extraction)
- **Masked appearance and motion** by default (`use_mask=True`): they describe only the object interior and no longer leak shape. [More](#masked-appearance-and-motion-features)
- **New features:** elliptic Fourier descriptors (shape) and the shear rate of the flow (motion), as in the PyPI 0.1.1 build. [More](#computing-the-phenome)
- **Analysis helpers:** equal weighting of SAM sets and feature families (recommended), confound regression (batch by default), rotation-invariant features. [More](#analysing-the-sam-phenome)
- **SAM modules:** a new multi-scale selection of the number of modules (the old rule picked the worst partition). [More](#sam-module-selection)
- **Pretrained networks** on Zenodo, downloaded automatically and run in PyTorch; TensorFlow is optional. [More](#pretrained-networks-for-organoid-detection-and-segmentation)
- **Python 3.9 - 3.13 and NumPy 2**; UMAP and PaCMAP are both installed.
- **Bug fix:** the unmasked motion SIFT descriptors zeroed the flow inside instead of outside the object.
- The code as published with the papers (version 0.1.1) is kept as release [v0.1.1-natcomm](https://github.com/fyz11/SPOT/releases/tag/v0.1.1-natcomm) and branch `legacy`.

<details>
<summary>Older news</summary>

- **Aug 2026:** both SPOT papers published in Nature Communications: [Methods paper](https://www.nature.com/articles/s41467-026-75505-8) and [Organoids paper](https://www.nature.com/articles/s41467-026-75506-7).
- **June 2026:** SPOT is available on PyPI (https://pypi.org/project/SAM-SPOT/0.1.1/): `pip install SAM-SPOT` (Python 3.9 or 3.10).
- **Jan 2026:** tutorial video for installing SPOT, [link](https://www.dropbox.com/scl/fi/yyebz7clz4m46791zigj6/SPOT-tutorial.mp4?rlkey=aio3zsr33l4szh8gk4290ku1a&e=1&st=sw9ist2l&dl=0). Optional `pacmap` dependency added (a core dependency since Oct 2026), which needs `python-annoy` from conda-forge.
</details>

<!-- TOC --><a name="the-sam-phenome"></a>
## The SAM phenome
Dynamic objects constantly change their behaviour. We hypothesise that three measurable properties, **Shape, Appearance and Motion**, together characterise the instantaneous phenotypic state of any object. SPOT computes one generalised SAM feature set for every object at every time point, like a transcriptome in single-cell sequencing, covering global, (local) regional and (local) distribution features:

<p align="center">
  <img src="https://github.com/fyz11/SPOT/blob/main/docs/pictures/SAM_motivation2.jpg" width=100%/>
</p>

| SAM set | feature families (feature name prefix) |
|---|---|
| **Shape** | curvature, contour geometry, binary morphology (area, eccentricity, Hu / Zernike moments, ...), Fourier (`Fourier_`), elliptic Fourier descriptors (`EFD_`), Euler characteristic transform (`ECT_`), shape context |
| **Appearance** | global and regional intensity, Haralick texture, SIFT (`sift_appear_`) |
| **Motion** | global motion (speed, curl, divergence, shear rate), regional speed and its distribution, SIFT of the speed, curl, divergence and shear-rate fields (`sift_motion_`, `curl_sift_`, `div_sift_`, `shear_sift_`) |

`SPOT.Analysis.preprocessing.sam_feature_groups(feature_names)` returns the set and family of every feature.

<!-- TOC --><a name="computing-the-phenome"></a>
### Computing the phenome
Inputs:
- `vid`: the video, (n_frames, n_rows, n_columns);
- `boundaries`: the tracked and segmented object contours, (n_objects, n_frames, n_contour_points, 2) in (y, x) image coordinates, `numpy.nan` where an object is absent;
- `pixel_res` (micrometers per pixel) and `time_res` (hours between frames).

Shape, appearance and motion are computed separately. Every function returns `(features, feature_names, is_normalised)`, with features of shape (n_objects, n_frames, n_features).

```python
import numpy as np
import skimage.exposure as skexposure
import SPOT.Features.features as SPOT_SAM_features
import SPOT.Tracking.optical_flow as SPOT_optical_flow

# shape
shape, shape_names, _ = SPOT_SAM_features.compute_boundary_morphology_features(boundaries, imshape, pixel_xy_res=pixel_res, n_jobs=-1)

# appearance (masks: optional foreground segmentation, used when the contours are coarser than the objects)
appearance, appearance_names, _ = SPOT_SAM_features.compute_boundary_texture_features(vid, boundaries, masks=masks, n_jobs=-1)

# motion: needs the optical flow first, so there are no motion features for the last frame
optical_flow_params = dict(pyr_scale=0.5, levels=5, winsize=15, iterations=5, poly_n=3, poly_sigma=1.2, flags=0)
vid_norm = np.array([skexposure.rescale_intensity(frame) for frame in vid])
flow = SPOT_optical_flow.extract_vid_optflow(vid_norm, flow_params=optical_flow_params, rescale_intensity=True)
motion, motion_names, _ = SPOT_SAM_features.compute_boundary_motion_features(flow, boundaries, masks=masks,
                                                                             pixel_res=pixel_res, time_res=time_res, n_jobs=-1)
```
The elliptic Fourier descriptors (`elliptic_fourier_features=True`, `elliptic_fourier_order=10`) and the shear-rate features (`shear_features=True`) are on by default, with the same names and order as the PyPI 0.1.1 build; set them to `False` for the features without them.

<!-- TOC --><a name="masked-appearance-and-motion-features"></a>
### Masked appearance and motion features
By default (`use_mask=True`):
- the Haralick and SIFT appearance features and the motion SIFT descriptors only use the pixels **inside** the object (intersected with `masks` if given), so appearance and motion do not double count shape;
- curl, divergence and shear rate are recomputed from the flow inside the object, so the flow discontinuity at the boundary does not leak in;
- the masked SIFT descriptors pool over shape-adapted ring x sector cells, on the object patch with its longest side capped at 64 pixels (`masked_sift_resize='cap'`; or `'square'`, `'native'`).

`use_mask=False` gives the original SAM features, computed on the bounding box crop (they also describe the outline).

<!-- TOC --><a name="faster-extraction"></a>
### Faster extraction
| implementation | how | notes |
|---|---|---|
| `compute_boundary_*_features(..., fast=True)` (default) | optimised CPU code: per-object crops, curl / divergence once per frame, vectorised ECT and shape context; parallel over objects and frames with `n_jobs` (-1 = all CPUs) | same outputs as the original (`fast=False`); in parallel, BLAS round-off can differ by ~1e-15 |
| `SPOT.Features.features_torch.compute_boundary_*_features_torch(..., device='cuda')` | rasterisation, Haralick, SIFT, ECT and curl / divergence batched on the GPU | needs a CUDA build of PyTorch ([To install](#to-install)); same outputs up to float64 round-off |

Timings: `benchmarks/benchmark_results.md`.

<!-- TOC --><a name="rotation-invariant-features"></a>
### Rotation-invariant features
SIFT orientation histograms, angular sector windows and the ECT directions depend on the orientation of the object in the image. `SPOT.Features.augment.compute_rotation_averaged_features` makes them rotation invariant:
- it rotates the video, contours and masks by `n_rotations` angles (multiples of 90 degrees are exact), recomputes the optical flow for motion, computes the features with any feature function and averages them;
- after averaging, the 4 SIFT bins of each quarter-turn orbit are equivalent, so they are collapsed into one feature per orbit (32 instead of 128 per descriptor; `collapse_sift=False` keeps all bins; only here, the feature functions always return all bins);
- `rotation_sensitivity` reports how much each feature changes with the angle.
```python
import SPOT.Features.augment as SPOT_augment

appearance, appearance_names, _ = SPOT_augment.compute_rotation_averaged_features(
    SPOT_SAM_features.compute_boundary_texture_features, boundaries, vid=vid, n_rotations=4, n_jobs=-1)
```

<!-- TOC --><a name="analysing-the-sam-phenome"></a>
## Analysing the SAM phenome
The Stage 3 example (`examples/SPOT_demo_single-cell-tracking-challenge_Step3_...py`) runs the full analysis: normalisation (StandardScaler + power transform, `SPOT.Analysis.preprocessing`), embedding (UMAP or PaCMAP), phenotype clusters and SAM modules. Recommended order: normalise, regress out confounders (optional), weight, then embed / cluster.

<!-- TOC --><a name="equal-weighting-of-the-sam-sets-and-feature-families"></a>
### Equal weighting of the SAM sets and feature families
The feature families have very different sizes (hundreds of shape context, ECT and SIFT bins against a handful of intensity or speed features), so without weighting the largest ones dominate every distance. `weight_sam_features` (after the per-feature normalisation; recommended for embeddings and clustering) gives
- shape, appearance and motion the same total variance (`set_weighting=True`), and
- within each, every feature family the same weight (`group_weighting=True`).
```python
import SPOT.Analysis.preprocessing as preprocess

embed_feats = preprocess.weight_sam_features(all_feats, feature_names, group_weighting=True, set_weighting=True)
```

<!-- TOC --><a name="regressing-out-confounders"></a>
### Regressing out confounders
`regress_out_confounders` removes undesired effects from the normalised features. **The default confounder is the batch** (plate, experiment, imaging session); continuous covariates such as size or intensity are chosen **per dataset**, only where they are nuisances and not part of the phenotype studied.
```python
# default: the batch only
all_feats_corrected, names_corrected = preprocess.regress_out_confounders(all_feats, feature_names=feature_names,
                                                                          categorical={'batch': batch_ids})

# per dataset, e.g. if illumination varies: add it (taken from the features)
all_feats_corrected, names_corrected = preprocess.regress_out_confounders(all_feats, feature_names=feature_names,
                                                                          categorical={'batch': batch_ids},
                                                                          covariate_features=['mean_intensity'])
```
- Batches: mean shift per batch (optionally also the scale, `batch_scale=True`). Continuous covariates: a cubic spline of their rank, so non-linear monotone effects are removed too.
- `reference=` (e.g. the control objects) estimates the effects on those rows only, so condition differences are not regressed away.
- Features still associated with a confounder afterwards (|Spearman rho| > 0.3 or batch eta^2 > 0.1) are dropped (`drop_associated=True`).
- `ConfoundRegressor` keeps the fitted model for new data; `confound_association` reports the association of every feature with every confounder.

<!-- TOC --><a name="sam-module-selection"></a>
### SAM module selection
`SPOT.Analysis.sam_analysis.hierarchical_cluster_features_into_SAM_modules` groups co-varying features into SAM modules by hierarchical clustering. The number of modules is chosen automatically (`module_selection='multiscale'`, default):
- **candidates:** the numbers of modules whose partition of the feature dendrogram lives longest compared with a null of permuted features (modules that are internally similar and far apart);
- **choice:** the stable candidate with the best consensus score, preferring fewer modules unless more are clearly better (`parsimony=0.2`), and at least `min_modules=3`;
- `scale='coarse'` / `'fine'` / an int overrides the choice; `return_scales=True` also returns the table of candidates.

<details>
<summary>Details of the selection</summary>

- Only modules of at least `min_module_size=3` features are counted, and a cut of the dendrogram is a scale only if its modules cover at least `min_coverage=0.75` of the features.
- The consensus score (0 - 1) averages the null-referenced partition lifetime; the silhouette, Davies-Bouldin, Calinski-Harabasz and within- minus between-module correlation, each relative to the null; and the bootstrap stability of the modules.
- Of the stable candidates (bootstrap ARI >= `min_stability=0.6`) whose score is within `parsimony` of the best, the one with the fewest modules is selected (as the one-standard-error rule). `parsimony=0` takes the best score.
- With fewer than `min_modules` modules, the next peak is used; if there is no other peak, the sole peak is returned with a warning.
- `module_selection='legacy'` reproduces the original rule (the first peak of the smoothed Davies-Bouldin score). Davies-Bouldin is lower-is-better, so that peak is a locally *worst* partition.
</details>

<!-- TOC --><a name="pretrained-networks-for-organoid-detection-and-segmentation"></a>
## Pretrained networks for organoid detection and segmentation
| model | use | download |
|---|---|---|
| YOLOv3 organoid bounding box detector | brightfield / phase contrast and fluorescence microscopy | [Zenodo](https://zenodo.org/records/23438299/files/keras_YOLOv3_organoid_detector2.h5?download=1) |
| attention U-Net organoid segmentation | segmentation of a bounding box crop | [Zenodo](https://zenodo.org/records/23438299/files/organoid-bbox_attn_seg-unet-master-v2_64x64-8feats_v2.h5?download=1), and in this repo: `models/segment_CNN_model/` |

Both are archived on Zenodo ([doi:10.5281/zenodo.23438299](https://doi.org/10.5281/zenodo.23438299)). They are downloaded on first use, checked against their sha256, cached in `~/.cache/SPOT/models` (or `$SPOT_MODELS_DIR`) and converted to PyTorch:
```python
from SPOT.Detection.pretrained import get_pretrained_weights

detector_pt = get_pretrained_weights('yolov3_organoid_detector')              # PyTorch .pt
detector_h5 = get_pretrained_weights('yolov3_organoid_detector', torch=False)  # original Keras .h5
segmenter_pt = get_pretrained_weights('attn_unet_organoid_segmentation')
```
The PyTorch loaders (`load_and_run_YOLOv3_weights_torch_detection_model_RGB`, `load_torch_segmentation_model`) use them when no weights file is given. For the Keras examples, copy the detector to `models/detect_CNN_model/`.

<!-- TOC --><a name="example-scripts"></a>
## Example scripts
Start with the demo [`examples/SPOT_demo_new_features.ipynb`](examples/SPOT_demo_new_features.ipynb) (or `examples/SPOT_demo_new_features.py`). Scripts for every step of the workflow, on the example data in `data/`, are in `examples/`; a detailed guide is the [Getting Started word document](https://github.com/fyz11/SPOT/blob/main/Getting%20Started%20with%20SPOT.docx).

| stage | scripts (`examples/`) |
|---|---|
| **1. Video acquisition and object segmentation** | `SPOT_Stage1_Step0_translation-register-RGB-confocal_video.py`, `SPOT_Stage1_Step0_unmix-RGB-confocal_video.py`, `SPOT_Stage1_Step1_detect_bbox-RGB-confocal_video.py` (or `..._torch.py`), `SPOT_Stage1_Step2_track_detect_bbox-RGB-confocal_video.py`, `SPOT_Stage1_Step3_segment_tracked_bbox-RGB-confocal_video.py` (or `..._torch.py`), `SPOT_Stage1_Step4_postprocess_segment_tracked_bbox-RGB-confocal_video.py` |
| **2. SAM phenome** (example fluorescent video) | `SPOT_Stage2_Step1_compute_SAM_phenomes.py`, `SPOT_Stage2_Step2_generate_metadata_table.py`, `SPOT_Stage2_Step3_compile_and_export_SAM_phenomes.py`, `SPOT_Stage2_Step4_compile_and_export_object_patch_imgs.py` |
| **2. SAM phenome** (cell tracking challenge) | `SPOT_demo_single-cell-tracking-challenge_Step0_Prepare_Single_Cell_Tracking_Challenge_Dataset_All.py`, `SPOT_demo_single-cell-tracking-challenge_Step1_Compute_SAM_phenome_Single_Cell_Tracking_Challenge_Dataset_All.py` |
| **3. Analysis of the SAM phenome** | `SPOT_demo_single-cell-tracking-challenge_Step2_Compile_All_SAM_phenomes_cell-tracking-challenge.py`, `SPOT_demo_single-cell-tracking-challenge_Step3_SPOT_analyze_all_SAM_phenomes_cell_tracking_challenge.py` |

Each example takes a few minutes; computing the SAM phenome of a full dataset can take tens of minutes.

<!-- TOC --><a name="to-install"></a>
## To install
Tested with Python 3.9 - 3.13 (NumPy 1.x and 2.x). We recommend a new conda environment, with the `annoy` package (needed by PaCMAP) from conda-forge, since pip has to compile it on some platforms (e.g. Windows):
```shell
conda create -n SAM-SPOT_env python=3.12
conda activate SAM-SPOT_env
conda install -c conda-forge python-annoy
```
Then install SPOT from PyPI:
```shell
pip install SAM-SPOT
```
or the latest version from GitHub (or from a clone of this repository, `pip install .`):
```shell
pip install "SAM-SPOT @ git+https://github.com/fyz11/SPOT.git"
```
- **GPU:** PyTorch is a core dependency, but pip installs its CPU build on Windows by default. For the GPU feature functions and networks, install a CUDA build of torch (>= 2.2) from [pytorch.org](https://pytorch.org/get-started/locally/) first; `pip install` keeps it.
- **Optional extras:** `keras-legacy` (TensorFlow, for the original Keras networks), `test` (pytest), `docs`, e.g. `pip install ".[keras-legacy]"`.
- **Version 0.1.1** (as published with the papers; Python 3.9 - 3.10): `pip install SAM-SPOT==0.1.1`, or the GitHub release [v0.1.1-natcomm](https://github.com/fyz11/SPOT/releases/tag/v0.1.1-natcomm). Its Keras functions need the pretrained weights downloaded manually.

<!-- TOC --><a name="associated-papers"></a>
## Associated papers
1. (Methods paper, 2D fixed and live-cell imaging application) **Development of a universal imaging “phenome” using shape, appearance and motion (SAM) features and the SAM Phenotype Observation Tool (SPOT)**, [Nat. Comms, 2026](https://www.nature.com/articles/s41467-026-75505-8), written by Felix Y. Zhou, Adam Norton-Steele, Lewis Marsh, Helen M. Byrne, Heather A. Harrington and Xin Lu.

2. (3D Organoid Timelapse Application) **Identifying phenotype-genotype-function coupling in 3D organoid imaging using 
Shape, Appearance and Motion Phenotype Observation Tool (SPOT)**, [Nat. Comms., 2026](https://www.nature.com/articles/s41467-026-75506-7), written by Felix Y. Zhou, Brittany-Amber Jacobs, Adam Norton-Steele, Xiaoyue Han, Thomas M. Carroll, Carlos Ruiz Puig, Joseph Chadwick, Xiao Qin, Richard Lisle, Lewis Marsh, Helen M. Byrne, Heather A. Harrington, Linna Zhou and Xin Lu.

<!-- TOC --><a name="associated-datasets"></a>
## Associated datasets
The [Synthetic Ground Truth Organoid (SGTO) datasets](https://zenodo.org/records/14265519) are used to validate SPOT sensitivity, robustness, and Z-plane effects in the associated 3D Organoid Timelapse Application paper (see above).

<!-- TOC --><a name="documentation"></a>
## Documentation
The documented API is available as html in the `docs/build/html` folder. To build up-to-date docs, go into `docs/` and run:
```shell
make html
```

<!-- TOC --><a name="copyright-information"></a>
## Copyright information

<!-- TOC --><a name="for-academic-and-non-profit-users"></a>
#### FOR ACADEMIC AND NON-PROFIT USERS
---
The software and scripts is made available as is under a Ludwig Software License for academic non-commercial research purposes. Please read the included software license agreement.

<!-- TOC --><a name="for-profit-users"></a>
#### FOR-PROFIT USERS
---
If you plan to use SPOT in any for-profit application, you are required to obtain a separate license. To do so, please contact Shayda Hemmati (shemmati@licr.org) or Par Olsson (polsson@licr.org) at the Ludwig Institute for  Cancer Research Ltd.
