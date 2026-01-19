# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 01:51:28 2022

@author: fyz11
"""

if __name__=="__main__":
    
    """
    Multi-part acquisitions where media has been changed during timelapse filming tends to have stage drift. This extraneous movement needs be registered to reduce motion jitter.
    We demonstate this here using a phase-correlation method. 
    
    """
    import os
    import numpy as np 
    import SPOT.Image.image as SPOT_image
    import SPOT.Utility_Functions.file_io as fio
    import SPOT.Registration.registration_2D as SPOT_register
    import pylab as plt 
    from skimage.transform import SimilarityTransform
    import skimage.color as skcolor 
    import skimage.transform as sktform
    import skimage.filters as skfilters
    import scipy.ndimage as ndimage
    
    # set a seed
    np.random.seed(1242)
    
    
    imfile = r'./data/organoids/fluorescent_murine_colon/KRAS_G12D_EYFP_2.wmv'
    
    saveplots = True
    savedir = '../test_outputs/frame_registration'
    if saveplots:
        os.makedirs(savedir, exist_ok=True)
    
    vid = fio.read_video_cv2(imfile)
    
    """
    get the blue channel 
    """
    vid_gray = vid[...,2] / 255. 
    
    """
    we simulate some translational jitter by sampling uniform random displacements 
        - this is a much more harder case than reality where you experience a shift after some time.
    """
    x_disp_max = 10
    y_disp_max = 10
    
    x_disps_random = np.random.uniform(0, x_disp_max, len(vid_gray))
    y_disps_random = np.random.uniform(0, y_disp_max, len(vid_gray))
    yx_disps_random = np.vstack([y_disps_random, 
                                 x_disps_random]).T
    
    def apply_translation_to_vid(vid, yx_disps_random):
        
        vid_out = []
        
        for ii in np.arange(len(vid)):
            tform = SimilarityTransform(translation=yx_disps_random[ii])
            warped = sktform.warp(vid[ii], tform)
            vid_out.append(warped)
            
        return np.array(vid_out)
    
    vid_gray_jitter = apply_translation_to_vid(vid_gray, yx_disps_random)
            
    
    """
    visualize a movie of the jitter
    """
    for frame_no in np.arange(len(vid_gray_jitter)):
        plt.figure(figsize=(10,10))
        plt.title('Frame: ' + str(frame_no).zfill(5) + '_jittered')
        plt.imshow(vid_gray_jitter[frame_no], cmap='gray')
        if saveplots:
            plt.savefig(os.path.join(savedir, f'frame_{str(frame_no).zfill(5)}_jittered.png'))
        plt.show()
        
    
    # =============================================================================
    #   Perform the registration, and return the computed displacement at each timepoint.
    # =============================================================================
    
    stacked_video_translation, blurred_frames, disps = SPOT_register.translation_register_blurry_phase_contrast_videos(np.uint8(255*vid_gray_jitter), 
                                                                                                  shape=(512,512), 
                                                                                                  sub_pix_res = 1, 
                                                                                                  impute_pixels = True, 
                                                                                                  use_registered = False, 
                                                                                                  use_dog=True, 
                                                                                                  dog_sigma=3, 
                                                                                                  detect_blurred_frames = False, 
                                                                                                  blur_factor=3., 
                                                                                                  apply_crop_mask=False,
                                                                                                  return_crop=False)
                                            
    """
    visualize the registered movie
    """
    for frame_no in np.arange(len(vid_gray_jitter)):
        plt.figure(figsize=(10,10))
        plt.title('Frame: ' + str(frame_no).zfill(5) + '_registered')
        plt.imshow(stacked_video_translation[frame_no], cmap='gray')
        if saveplots:
            plt.savefig(os.path.join(savedir, f'frame_{str(frame_no).zfill(5)}_registered.png'))
        plt.show()
    

    
    