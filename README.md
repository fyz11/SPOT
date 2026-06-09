# SPOT (Shape, appearance, motion Phenotype Observation Tool)

<p align="center">
  <img src="https://github.com/fyz11/SPOT/blob/main/docs/pictures/main_workflow.jpg" width=100%/>
</p>

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

   * [Introduction to SPOT](#introduction-to-spot)
   * [Introduction to the SAM phenome](#introduction-to-the-sam-phenome)
        + [Code snippet to compute phenome](#code-snippet-to-compute-phenome)
   * [Associated Paper](#associated-paper)
   * [Getting Started](#getting-started)
   * [SPOTapp](#spotapp)
   * [Pretrained Neural Network Models for organoid detection and segmentation](#pretrained-neural-network-models-for-organoid-detection-and-segmentation)
   * [Documentation](#documentation)
   * [To install](#to-install)
   * [COPYRIGHT INFORMATION:](#copyright-information)
        + [FOR ACADEMIC AND NON-PROFIT USERS](#for-academic-and-non-profit-users)
        + [FOR-PROFIT USERS](#for-profit-users)

<!-- TOC end -->

[![PyPI version](https://badge.fury.io/py/SAM-SPOT.svg)](https://badge.fury.io/py/SAM-SPOT)
[![Downloads](https://pepy.tech/badge/SAM-SPOT)](https://pepy.tech/project/SAM-SPOT)
[![Downloads](https://pepy.tech/badge/SAM-SPOT/month)](https://pepy.tech/project/SAM-SPOT)
[![Python version](https://img.shields.io/pypi/pyversions/SAM-SPOT)](https://pypistats.org/packages/SAM-SPOT)
[![GitHub stars](https://img.shields.io/github/stars/fyz11/SPOT?style=social)](https://github.com/fyz11/SPOT/)
[![GitHub forks](https://img.shields.io/github/forks/fyz11/SPOT?style=social)](https://github.com/fyz11/SPOT/)

#### June 2026
- SPOT is available in PyPI (https://pypi.org/project/SAM-SPOT/0.1.1/)
- Install using `pip install SAM-SPOT`. Requires Python 3.9 or 3.10

#### Jan 2026
- Tutorial video for installing SPOT, [link](https://www.dropbox.com/scl/fi/yyebz7clz4m46791zigj6/SPOT-tutorial.mp4?rlkey=aio3zsr33l4szh8gk4290ku1a&e=1&st=sw9ist2l&dl=0)
- added optional dependency to install `pacmap`. Requires externally installing from conda-forge the `python-annoy` package. 


<!-- TOC --><a name="introduction-to-spot"></a>
## Introduction to SPOT
SPOT is a generalized and streamlined workflow for analysing object dynamics in movies. It is designed to suit high-content imaging applications where analytical tools should be push-and-go and require no prior knowledge of the expected behaviour of the objects to be studied. In other words, users should be able to run through all the steps in one go, then retrospectively interpret the produced results. This workflow is inspired by that for single-cell sequencing and is summarized in the figure above. 

Three innovations drive SPOT for temporal image analysis:
1. a standardized Shape, Appearance and Motion phenome - a single feature set for all objects
2. a standardized temporal analysis of compiled SAM phenomes - minimal assumption, push-and-go
3. automated and standardized techniques to cluster related SAM features into SAM modules for interpreting discovered phenotypes

SPOT is provided here as a Python package to allow full flexibility. To get started, please check out exemplar scripts in the examples/ folder. We also include SPOTapp, a graphical user interface (GUI) to run SPOT stages 1 and 2. This app is currently in alpha development and is provided here as-is to support the paper publication. 

<p align="center">
  <img src="https://github.com/fyz11/SPOT/blob/main/docs/pictures/SAM_motivation2.jpg" width=100%/>
</p>

<!-- TOC --><a name="introduction-to-the-sam-phenome"></a>
## Introduction to the SAM phenome
Dynamic objects constantly change their behaviour. Motivated by observation of natural images like birds and cars, we hypothesize that three measurable properties; Shape, Appearance and Motion (SAM) provides complementary information necessary to characterize the instantaneous phenotypic state for any object.  

This led us to design a single generalized SAM feature set (depicted above) which can function similar to the single-cell transcriptome in single-cell sequencing analysis by considering in addition to shape, appearance and motion; global, (local) regional and (local) distribution features.

<!-- TOC --><a name="code-snippet-to-compute-phenome"></a>
### Code snippet to compute phenome
The SPOT library computes the shape, appearance and motion-associated features in the SAM phenome separately to allow separate usage. To compute, you need to have the video as an array (`vid`) of (n_frames, n_rows, n_columns) and tracked and segmented object boundaries given as an array (`boundaries`) of (n_objects, n_frames, n_contour_points, 2). The last dimension of the boundaries array specifies (y,x) image coordinate convention. The boundaries array should use `numpy.nan` for any timepoints where an object was not present. You also need to specify the pixel resolution in micrometers (`pixel_res`) and the time elapsed (`time_res`) between each video frame in hours

#### Computing Shape Phenome
```
import SPOT.Features.features as SPOT_SAM_features

metrics, metrics_labels, metrics_norm_bool = SPOT_SAM_features.compute_boundary_morphology_features(boundaries, 
                                                                                                    imshape,  
                                                                                                    pixel_xy_res = pixel_res)
```
#### Computing Appearance Phenome
```
import SPOT.Features.features as SPOT_SAM_features

metrics, metrics_labels, metrics_norm_bool = SPOT_SAM_features.compute_boundary_texture_features(vid, 
                                                                                                  boundaries)
```
#### Computing Motion Phenome
Unlike shape and appearance, motion needs to first compute the dense optical flow. Consequently motion features are not available for object instances in the last video frame. 
```
import SPOT.Features.features as SPOT_SAM_features
import SPOT.Tracking.optical_flow as SPOT_optical_flow
import skimage.exposure as skexposure

# compute optical flow first.
optical_flow_params = dict(pyr_scale=0.5, levels=5, winsize=15, iterations=5, poly_n=3, poly_sigma=1.2, flags=0)

vid_norm = np.array([skexposure.rescale_intensity(frame) for frame in vid])
flow = SPOT_optical_flow.extract_vid_optflow(vid_norm, 
                                             flow_params=optical_flow_params,
                                             rescale_intensity=True)

# then compute motion features. As the flow does not account for the last frame of the video, motion features also miss the last frame object instances.
metrics, metrics_labels, metrics_norm_bool = SPOT_SAM_features.compute_boundary_motion_features(flow 
                                                                                                boundaries)
```

<!-- TOC --><a name="associated-paper"></a>
## Associated Paper
SPOT is associated with the following papers (which are currently under journal revision), which you will be able to read after publication for more technical detail and get an idea of the many applications SPOT enables: 

1. (Methods paper, 2D fixed and live-cell imaging application) **Development of a universal imaging “phenome” using Shape, Appearance and Motion (SAM) features and the SAM Observation Tool (SPOT)**, (2025), written by Felix Y. Zhou, Adam Norton-Steele, Lewis Marsh, Helen M. Byrne, Heather A. Harrington and Xin Lu.

2. (3D Organoid Timelapse Application) **Identifying phenotype-genotype-function coupling in 3D organoid imaging using 
2 Shape, Appearance and Motion Phenotype Observation Tool (SPOT)**, 2025, written by Felix Y. Zhou, Brittany-Amber Jacobs, Adam Norton-Steele, Xiaoyue Han, Thomas M. Carroll, Carlos Ruiz Puig, Joseph Chadwick, Xiao Qin, Richard Lisle, Lewis Marsh, Helen M. Byrne, Heather A. Harrington, Linna Zhou and Xin Lu.

<!-- TOC --><a name="getting-started"></a>
## Getting Started
Detailed guide to setting up and installing SPOT, and running the exemplar scripts provided on example data is detailed in the [Getting Started word document]( https://github.com/fyz11/SPOT/blob/main/Getting%20Started%20with%20SPOT.docx) 

Exemplar scripts to run every step of the workflow are provided in the Examples folder to work with the provided example data located in the data/ folder. This includes:

**for SPOT Stage 1: video acquisition and object segmentation:**
1. SPOT_Stage1_Step0_translation-register-RGB-confocal_video.py
2. SPOT_Stage1_Step0_unmix-RGB-confocal_video.py
3. SPOT_Stage1_Step1_detect_bbox-RGB-confocal_video.py
4. SPOT_Stage1_Step2_track_detect_bbox-RGB-confocal_video.py
5. SPOT_Stage1_Step3_segment_tracked_bbox-RGB-confocal_video.py
6. SPOT_Stage1_Step4_postprocess_segment_tracked_bbox-RGB-confocal_video.py

**for SPOT Stage 2: computation of SAM (Shape, Appearance and Motion) phenome:**    

<u>based on the example fluorescent video:</u>
1. SPOT_Stage2_Step1_compute_SAM_phenomes.py
2. SPOT_Stage2_Step2_generate_metadata_table.py
3. SPOT_Stage2_Step3_compile_and_export_SAM_phenomes.py
4. SPOT_Stage2_Step4_compile_and_export_object_patch_imgs.py

<u>based on the single cell tracking challenge:</u>
1. SPOT_demo_single-cell-tracking-challenge_Step0_Prepare_Single_Cell_Tracking_Challenge_Dataset_All.py
2. SPOT_demo_single-cell-tracking-challenge_Step1_Compute_SAM_phenome_Single_Cell_Tracking_Challenge_Dataset_All.py

**for SPOT Stage 3: analysis of SAM (Shape, Appearance and Motion) phenome:**
1. SPOT_demo_single-cell-tracking-challenge_Step2_Compile_All_SAM_phenomes_cell-tracking-challenge.py
2. SPOT_demo_single-cell-tracking-challenge_Step3_SPOT_analyze_all_SAM_phenomes_cell_tracking_challenge.py

Each example should take around a few mins to run. The longest time is computing the SAM phenome which might take 10s of mins. 

<!-- TOC --><a name="spotapp"></a>
## SPOTapp
In parallel to the scripts, we provide an alpha version of a GUI to run SPOT called SPOTapp. You can find download instructions in the file SPOTapp.md. Instructions to use the software can also be found at the download link.

<!-- TOC --><a name="pretrained-neural-network-models-for-organoid-detection-and-segmentation"></a>
## Pretrained Neural Network Models for organoid detection and segmentation
We make available pretrained neural network organoid detection and segmentation models with this repo. The segmentation model is already provided in the repo. The detection model should be downloaded and copied to the models/detect_CNN_model/ folder of this repo if you are running the example scripts.

1. Organoid bounding box detector for brightfield/phase contrast and fluorescence microscopy. [Organoid YOLOv3 bounding box detection model weights](https://www.dropbox.com/scl/fi/qzowc9s9n30zh6qdyzeqw/keras_YOLOv3_organoid_detector2.h5?rlkey=6deiqemsmcz3yin9b5dnz0e6y&dl=0)
2. Organoid attention UNet segmentation model given a bounding box cropped image. (this repo, [models/segment_CNN_model/organoid-bbox_attn_seg-unet-master-v2_64x64-8feats_v2.h5](https://github.com/fyz11/SPOT/blob/main/models/segment_CNN_model/organoid-bbox_attn_seg-unet-master-v2_64x64-8feats_v2.h5)) 

<!-- TOC --><a name="documentation"></a>
## Documentation
Documented API of the functions provided in this library is available as a html in the docs/build/hmtl folder. You can build up-to-date docs by going into docs/ and executing:
```shell
make html
```

<!-- TOC --><a name="to-install"></a>
## To install
The package can be installed automatically using pip after cloning the repository. It has been tested and working for Python 3.9 and 3.10. Installation time in mins. We recommend installing into a new conda environment, and preinstalling the annoy package from conda-forge if you want to use pacmap instead of umap for dimensionality reduction:
```shell
conda create -n SAM-SPOT_env python=3.9
conda install -n SAM-SPOT_env -c conda-forge python-annoy
pip install .
```
You can also install directly from the github without cloning, but will still need to manually download the pretrained model weights:
```shell
pip install SPOT@git+https://github.com/fyz11/SPOT.git
```

<!-- TOC --><a name="copyright-information"></a>
### COPYRIGHT INFORMATION:

<!-- TOC --><a name="for-academic-and-non-profit-users"></a>
#### FOR ACADEMIC AND NON-PROFIT USERS
---
The software and scripts is made available as is under a Ludwig Software License for academic non-commercial research purposes. Please read the included software license agreement.

<!-- TOC --><a name="for-profit-users"></a>
#### FOR-PROFIT USERS
---
If you plan to use SPOT in any for-profit application, you are required to obtain a separate license. To do so, please contact Shayda Hemmati (shemmati@licr.org) or Par Olsson (polsson@licr.org) at the Ludwig Institute for  Cancer Research Ltd.
