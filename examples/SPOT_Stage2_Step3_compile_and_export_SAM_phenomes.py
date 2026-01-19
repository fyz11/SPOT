# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 01:51:28 2022

@author: fyz11
"""

if __name__=="__main__":
    
    """
    In order to compile the SAM phenomes for every video, this example shows how to auto-generate and fill in a summary standard template metadata .csv for a single video set which can then be further populated by the user manually
    
    """
    import numpy as np 
    import scipy.io as spio
    import os 
    import glob 
    import pandas as pd 
    import skimage.io as skio 
    
    import SPOT.Utility_Functions.file_io as fio 

    """
    Specify the top level folder of the video dataset
    """
    masterimgfolder = r'./data/organoids/fluorescent_murine_colon'
    
    # specify whether we should inverse the channel order, in case the metrics were extracted for the inverse ordering i.e. BGR instead of RGB 
    rev_channels = False
    
    
    """
    specify the movie extension 
    """
    ext = '.wmv'
    
    """
    autofinds the video files
    """
    imgfiles = glob.glob(os.path.join(masterimgfolder, '*'+ext))
    n_files = len(imgfiles)
    
    
    """
    specify the save location of this master metadata file and read it 
    """
    csvsavefolder = masterimgfolder
    csvsavefile = os.path.join(csvsavefolder, 
                               'metadata_expt_filled.csv')
    
    expt_metadata_table = pd.read_csv(csvsavefile)
    
    
    print(expt_metadata_table)
    
    """
    specify the top level save location of the final features for each video
    """
    mastersavefolder = r'../test_outputs/test_segmentation_folder'
    
    
    """
    specify where we want to save the final compiled SAM phenomes over all the movies which will be used for analysis
    """
    final_compiled_savefolder = r'../test_outputs'
    fio.mkdir(final_compiled_savefolder)
    
    
    """
    Perform the compiling, iterating through each video in the experiment
    """
    # create the master metrics table for this dataset. 
    expt_shape_feats_table = []
    expt_image_feats_table = []
    expt_motion_feats_table = []
    
    # for the final save.
    expt_boundaries = []
    filenames_all = [] 
    patients_all = []
    conditions_all = [] 
    genetics_all = []
    img_channel_no_all = [] 
    org_id_all = []
    frame_no_all = []
    pixel_resolution_all = []
    frame_duration_all = []
    total_vid_frames = []
    
    
    # to avoid confusion
    for iiii in np.arange(n_files)[:]:
        
        imgfile = imgfiles[iiii]
        expt = os.path.split(imgfile)[-1].split(ext)[0]
        
        print(iiii, expt)
        
        if 'NO' in expt or '\xc2' in expt:
            print('passing, ', expt)
            pass
        else:
        
            """
            0. Load the original video.
            """
            include_flag = False
            
            if '.avi' in ext.lower() or 'wmv' in ext.lower():
                img = fio.read_video_cv2(imgfile)
                include_flag = True
            elif '.tif' in ext.lower():
                img = skio.imread(imgfile)
                include_flag = True
            else:
                print('not valid video extension')
                pass
            
            if include_flag == True:
                
                imshape = img.shape[1:-1]

                
                """
                0. Load all the boundaries (y,x) for the whole experiment to export to .mat 
                """
                savematfolder = os.path.join(mastersavefolder, expt)
                
                boundaryfile = os.path.join(savematfolder, expt+'_boundaries_final_RGB.mat')
                boundaryobj = spio.loadmat(boundaryfile) 
                
                boundaries_smooth = boundaryobj['boundaries_smooth_final'].ravel()
                
                if rev_channels:
                    boundaries_smooth = boundaries_smooth[::-1]
                
                print(boundaries_smooth[0].shape, boundaries_smooth[1].shape, boundaries_smooth[2].shape) # this is correct. 
                
                (filenames, conditions, genetics, patients, img_channel_no, org_id, frame_no_id, pix_res, frame_duration, total_vid_no), expt_boundary = fio.compile_boundaries_arrays(expt, 
                                                                                                                                                                                       org_expt_table=expt_metadata_table, 
                                                                                                                                                                                       boundaries=boundaries_smooth)
                
                expt_boundaries.append(expt_boundary)
                
                filenames_all.append(filenames)
                conditions_all.append(conditions)
                genetics_all.append(genetics)
                patients_all.append(patients)
                img_channel_no_all.append(img_channel_no)
                org_id_all.append(org_id)
                frame_no_all.append(frame_no_id)
                pixel_resolution_all.append(pix_res)
                frame_duration_all.append(frame_duration)
                total_vid_frames.append(total_vid_no)
                
                """
                1. Load all the computed metrics to export to csv.
                """
                        
                shapefeatsfile = os.path.join(savematfolder, expt+'_final_cont_RGB_shape-feats.mat')
                imagefeatsfile = os.path.join(savematfolder, expt+'_final_cont_RGB_image-feats.mat')
                motionfeatsfile = os.path.join(savematfolder, expt+'_final_cont_RGB_motion-feats.mat')
                
                
                # a) shape 
                shapefeatsobj = spio.loadmat(shapefeatsfile)
                shapefeatstable = fio.construct_metrics_table_csv(expt, 
                                                              org_expt_table=expt_metadata_table,  
                                                              metrics = shapefeatsobj['metrics'].ravel(), 
                                                              metricslabels = shapefeatsobj['metric_names'].ravel())   
                expt_shape_feats_table.append(shapefeatstable)
                
                # b) appearance
                imgfeatsobj = spio.loadmat(imagefeatsfile)
                imgfeatstable = fio.construct_metrics_table_csv(expt, 
                                                              org_expt_table=expt_metadata_table,  
                                                              metrics = imgfeatsobj['metrics'].ravel(), 
                                                              metricslabels = imgfeatsobj['metric_names'].ravel())   
                expt_image_feats_table.append(imgfeatstable)
                
                # c) motion
                motionfeatsobj = spio.loadmat(motionfeatsfile)
                motionfeatstable = fio.construct_metrics_table_csv(expt, 
                                                              org_expt_table=expt_metadata_table,  
                                                              metrics = motionfeatsobj['metrics'].ravel(), 
                                                              metricslabels = motionfeatsobj['metric_names'].ravel())  
                expt_motion_feats_table.append(motionfeatstable)
        
    """
    Merge individual tables into one big table for the whole dataset
    """
    expt_shape_feats_table = pd.concat(expt_shape_feats_table, ignore_index=True)
    expt_image_feats_table = pd.concat(expt_image_feats_table, ignore_index=True)
    expt_motion_feats_table = pd.concat(expt_motion_feats_table, ignore_index=True)
    
    
    """
    save the compiled tables all out to the top-level dataset folder 
    """
    csvsavefolder_out = final_compiled_savefolder
    master_expt_name = os.path.split(csvsavefolder_out)[-1]
    
    
    csvsavefile_shape = os.path.join(csvsavefolder_out, 
                                      master_expt_name+'_shape_features.csv')
    expt_shape_feats_table.to_csv(csvsavefile_shape, index=None)
    
    csvsavefile_image = os.path.join(csvsavefolder_out, 
                                      master_expt_name+'_image_features.csv')
    expt_image_feats_table.to_csv(csvsavefile_image, index=None)
    
    csvsavefile_motion = os.path.join(csvsavefolder_out, 
                                      master_expt_name+'_motion_features.csv')
    expt_motion_feats_table.to_csv(csvsavefile_motion, index=None) # motion would not be the same, the others should be !. 
    
    
    """
    Compile and save out as a .mat for usage. 
    """
    expt_boundaries = np.concatenate(expt_boundaries, axis=0) # not the same number? 
    
    filenames_all = np.hstack(filenames_all)
    conditions_all = np.hstack(conditions_all)
    genetics_all = np.hstack(genetics_all)
    patients_all = np.hstack(patients_all)
    img_channel_no_all = np.hstack(img_channel_no_all)
    org_id_all = np.hstack(org_id_all)
    frame_no_all = np.hstack(frame_no_all)
    pixel_resolution_all = np.hstack(pixel_resolution_all)
    frame_duration_all = np.hstack(frame_duration_all)
    total_vid_frames = np.hstack(total_vid_frames)
    
    expt_savematfile = os.path.join(csvsavefolder_out, 
                                      master_expt_name+'_boundaries_smooth_final.mat')
    # do no compression for now. 
    spio.savemat(expt_savematfile, 
                  {'filenames': filenames_all, 
                  'condition': conditions_all, 
                  'genetics': genetics_all, 
                  'patients': patients_all,
                  'img_channel_no': img_channel_no_all, 
                  'org_id': org_id_all, 
                  'frame_no': frame_no_all, 
                  'pixel_res':pixel_resolution_all, 
                  'frame_duration':frame_duration_all, 
                  'boundaries': expt_boundaries})