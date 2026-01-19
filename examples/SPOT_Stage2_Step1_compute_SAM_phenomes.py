# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 01:51:28 2022

@author: fyz11
"""

if __name__=="__main__":
    
    """
    This example shows how to compute the SAM phenome using the postprocessed tracked organoid segmentations for a video
    
    """
    import numpy as np 
    import skimage.exposure as skexposure
    from tqdm import tqdm 
    import scipy.io as spio
    import os 
    import seaborn as sns
    import pylab as plt 
    import glob 
    
    import SPOT.Image.image as SPOT_image
    import SPOT.Utility_Functions.file_io as fio
    import SPOT.Tracking.track as SPOT_track
    import SPOT.Tracking.optical_flow as SPOT_optical_flow
    import SPOT.Features.features as SPOT_SAM_features
    
    

    imfile = r'./data/organoids/fluorescent_murine_colon/KRAS_G12D_EYFP_2.wmv'
    basename = os.path.split(imfile)[-1].split('.wmv')[0]
    
    vid = fio.read_video_cv2(imfile)
    
    """
    We need to specify the time sampling and spatial resolution 
    """
    pixel_res = 2.76 # um 
    time_res = 2 # h
    
    
    """
    Get the blue channel vid only 
    """
    n_channels = vid.shape[-1]
    desired_ch = 2 # blue channel only 
    vid_ch = vid[...,desired_ch].copy()
    
    
    """
    Define an output folder
    """
    # we will just use the same folder as the segmentations 
    outfolder = r'../test_outputs/test_segmentation_folder'
    fio.mkdir(outfolder)
    
    
    """
    Load the postprocessed segmentations
    """
    # this is just to show how to autodetect for multiple channels. 
    segmentation_folder = r'../test_outputs/test_segmentation_folder'
    
    savematfolder = os.path.join(outfolder, basename)
    savematfile = os.path.join(savematfolder, basename+'_boundaries_final_RGB.mat') 
    
    # load 
    boundaries_organoids_RGB = spio.loadmat(savematfile)['boundaries_smooth_final']
    
    if boundaries_organoids_RGB.shape[0] == 1:
        boundaries_organoids_RGB = boundaries_organoids_RGB[0]

    
    """
    Detect all the individual channel tracks 
    """
    imshape = vid.shape[1:-1]
    
    """
    0. Precompute optical flow for each image channel 
    """
    optical_flow_params = dict(pyr_scale=0.5, levels=5, winsize=15, iterations=5, poly_n=3, poly_sigma=1.2, flags=0)
    rev_channels = False # optionally specify if the channels should be reversed. 
    
    
    flow_channels = []
    
    for ch in np.arange(vid.shape[-1]):
        
        img_ch = vid[...,ch].copy()
        img_ch = np.array([skexposure.rescale_intensity(frame) for frame in img_ch])
        flow_ch = SPOT_optical_flow.extract_vid_optflow(img_ch, 
                                                         flow_params=optical_flow_params,
                                                         rescale_intensity=True)
        flow_channels.append(flow_ch)
        
    if rev_channels:
        flow_channels = flow_channels[::-1]
        boundaries_organoids_RGB = boundaries_organoids_RGB[::-1] # reverse. 
    

    """
    1. Compute and save the metrics into separate .mat files. (save as .pkl files using fio.write_pickle in the same way if individual file is > 4GB)
        a) shape features.
        b) image appearance features
        c) motion features.
    """
    
    # a) shape features. 
    all_metrics = []
      
    for boundaries_ii, boundaries in enumerate(boundaries_organoids_RGB):
    
        if len(boundaries) > 0 :
            
            print('computing shape metrics for Channel %d' %(boundaries_ii+1))
            metrics, metrics_labels, metrics_norm_bool = SPOT_SAM_features.compute_boundary_morphology_features(boundaries, 
                                                                                                                  imshape,  
                                                                                                                  all_feats_compute=True, 
                                                                                                                  contour_curve_feats=True, 
                                                                                                                  curve_order=4, 
                                                                                                                  curve_err=1., 
                                                                                                                  geom_features=True,
                                                                                                                  morph_features=True,
                                                                                                                  fourier_features=True, 
                                                                                                                  shape_context_features=True,
                                                                                                                  norm_scm=True,
                                                                                                                  n_ref_pts_scm=5,
                                                                                                                  pixel_xy_res = pixel_res)
        
            all_metrics.append(metrics)
            print(metrics.shape)
        else:
            all_metrics.append([])
        
        
    savemetricfile = os.path.join(savematfolder, basename+'_final_cont_RGB_shape-feats.mat')
    spio.savemat(savemetricfile, {'expt': basename, 
                                  'imgfile':imfile,
                                  'rgb_rev': rev_channels, 
                                  'metric_names': metrics_labels, 
                                  'metric_norm_bool': metrics_norm_bool, 
                                  'metrics': all_metrics,
                                  'pixel_res': pixel_res})


    # b) Image Appearance (textural) features. 
    all_metrics = []
      
    for boundaries_ii, boundaries in enumerate(boundaries_organoids_RGB):
    
        if len(boundaries) > 0 :
            print('computing motion metrics for Channel %d' %(boundaries_ii+1))
            metrics, metrics_labels, metrics_norm_bool = SPOT_SAM_features.compute_boundary_texture_features(vid[...,boundaries_ii], 
                                                                                                            boundaries, 
                                                                                                            use_gradient_vid=False,
                                                                                                            compute_all_feats =True,
                                                                                                            compute_intensity_feats=True,
                                                                                                            compute_contour_intensity_feats=True,
                                                                                                            n_contours=3,
                                                                                                            n_angles=1,
                                                                                                            angle_start=None,
                                                                                                            compute_sift=True,
                                                                                                            siftshape=(64,64),
                                                                                                            compute_haralick=True,
                                                                                                            haralick_distance=15)
            all_metrics.append(metrics)
        else:
            all_metrics.append([])
            
    savemetricfile = os.path.join(savematfolder, basename+'_final_cont_RGB_image-feats.mat')
    spio.savemat(savemetricfile, {'expt': basename, 
                                  'imgfile':imfile,
                                  'rgb_rev': rev_channels, 
                                  'metric_names': metrics_labels, 
                                  'metric_norm_bool': metrics_norm_bool, 
                                  'metrics': all_metrics})


    # c) Motion features. 
    all_metrics = []
      
    for boundaries_ii, boundaries in enumerate(boundaries_organoids_RGB):
    
        if len(boundaries) > 0 :
            print('computing motion metrics for Channel %d' %(boundaries_ii+1))
#                vid_flow = flow_channels[boundaries_ii].copy()
            metrics, metrics_labels, metrics_norm_bool = SPOT_SAM_features.compute_boundary_motion_features(flow_channels[boundaries_ii], 
                                                                                                              boundaries, 
                                                                                                              compute_all_feats =True,
                                                                                                              compute_global_feats=True,
                                                                                                              compute_contour_feats=True,
                                                                                                              n_contour_feat_bins=8,
                                                                                                              cnt_sigma=3., 
                                                                                                              n_contours=3,
                                                                                                              n_angles=1,
                                                                                                              angle_start=None,
                                                                                                              pixel_res=pixel_res,
                                                                                                              time_res=time_res,
                                                                                                              compute_sift_feats=True,
                                                                                                              siftshape=(64,64))
            
            all_metrics.append(metrics)
        else:
            all_metrics.append([])
            
    savemetricfile = os.path.join(savematfolder, basename+'_final_cont_RGB_motion-feats.mat')
    spio.savemat(savemetricfile, {'expt': basename, 
                                  'imgfile':imfile,
                                  'rgb_rev': rev_channels, 
                                  'metric_names': metrics_labels, 
                                  'metric_norm_bool': metrics_norm_bool, 
                                  'metrics': all_metrics,
                                  'pixel_res': pixel_res,
                                  'time_res': time_res})


