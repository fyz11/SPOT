# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 01:51:28 2022

@author: fyz11
"""

if __name__=="__main__":
    
    """
    This script shows how to use the compiled SAM phenomes and associated metadata to crop and export the corresponding image patches. This can then be used in analysis for visualization and is more memory-saving than the original sized images.
    
    """
    import numpy as np 
    import scipy.io as spio
    import os 
    import pandas as pd 
    import skimage.io as skio 
    from tqdm import tqdm 
    import skimage.transform as sktform
    import pylab as plt 
    
    
    import SPOT.Utility_Functions.file_io as fio 


    """
    Specify the top level folder of the video dataset
    """
    masterimgfolder = r'./data/organoids/fluorescent_murine_colon'
    
    # specify whether we should inverse the channel order, in case the metrics were extracted for the inverse ordering i.e. BGR instead of RGB 
    rev_channels = False
    debug_viz = False # if True, will also visualize the patches as it is being compiled for debugging

    
    """
    specify the movie extension 
    """
    ext = '.wmv'
    
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
    fio.mkdir(final_compiled_savefolder) # create if doesn't exist. 
    

    """
    specify the top-level name that the experiment was called 
    """
    master_expt_name = os.path.split(final_compiled_savefolder)[-1]
    
    
# =============================================================================
#     Load up the compiled boundary .mat file 
# =============================================================================

    expt_savematfile = os.path.join(final_compiled_savefolder, 
                                     master_expt_name+'_boundaries_smooth_final.mat') #5-17032021 3 patients -WNT again_boundaries_smooth_final_timesample
    ext_savematobj = spio.loadmat(expt_savematfile)
    
    filenames = np.hstack(ext_savematobj['filenames'])
    img_channel_no = np.hstack(ext_savematobj['img_channel_no'])
    frame_no = np.hstack(ext_savematobj['frame_no'])
    boundaries = ext_savematobj['boundaries']
    

    
# =============================================================================
#     Perform the patch extraction and compilation 
# =============================================================================

    bbox_sizes_all = []
    patches_all = []
    patches_all_size_original = []
    patch_size = (64,64) # this sets the standard size which here is (64,64)
     
    
    n_files = len(filenames)
    
    for ii in tqdm(np.arange(n_files)[:]):
        
        file_ii = os.path.join(masterimgfolder, 
                               filenames[ii].strip()+ ext )
        
        # 1. read the video. 
        if '.avi' in ext.lower() or 'wmv' in ext.lower():
            vid_ii = fio.read_video_cv2(file_ii)
            if rev_channels and len(vid_ii.shape)==4:
                vid_ii = vid_ii[...,::-1]
                
        elif '.tif' in ext.lower():
            vid_ii = skio.imread(file_ii)
            if rev_channels and len(vid_ii.shape)==4:
                vid_ii = vid_ii[...,::-1]
        else:
            print('not valid video extension')
            pass
            
        
        # 2. get the boundary
        boundary_ii = boundaries[ii]
        
        # 3. 
        frame_no_ii = frame_no[ii] - 1
        img_channel_no_ii = img_channel_no[ii] - 1
        
        # compute the boundary
        bbox = [int(np.min(boundary_ii[:,1])),
                int(np.min(boundary_ii[:,0])),
                int(np.ceil(np.max(boundary_ii[:,1]))),
                int(np.ceil(np.max(boundary_ii[:,0])))]
        bbox_sizes_all.append(np.hstack(bbox))
        
        patch = vid_ii[frame_no_ii, bbox[1]:bbox[3], bbox[0]:bbox[2], img_channel_no_ii] # cut this out. 
        patches_all_size_original.append(np.hstack(patch.shape))
        
        patch = np.uint8(sktform.resize(patch, 
                                        patch_size, 
                                        order=1,
                                        preserve_range=True))
        patches_all.append(patch)
        
        if debug_viz:
            plt.figure()
            plt.imshow(patch, cmap='gray')
            plt.show()
        
        
    bbox_sizes_all = np.vstack(bbox_sizes_all)
    patches_all = np.array(patches_all)
    patches_all_size_original = np.vstack(patches_all_size_original)
        
    # save out.... 
    savematfile_patches = os.path.join(final_compiled_savefolder, 
                                      master_expt_name+'_boundaries_smooth_final_img_patches.mat')
    
    
    spio.savemat(savematfile_patches, 
                  {'std_size': patch_size,
                   'patches_all': patches_all,
                   'bbox_size': bbox_sizes_all, 
                   'patch_size': np.hstack(patches_all_size_original)})
    
    
    