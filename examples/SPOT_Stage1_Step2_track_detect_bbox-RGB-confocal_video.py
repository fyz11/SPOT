# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 01:51:28 2022

@author: fyz11
"""

if __name__=="__main__":
    
    """
    This example shows how to track the frame-by-frame detected bounding boxes of organoids using our optical flow predictive LAP tracker
    
    """
    import numpy as np 
    import skimage.exposure as skexposure
    from tqdm import tqdm 
    import scipy.io as spio
    import os 
    
    import SPOT.Image.image as SPOT_image
    import SPOT.Utility_Functions.file_io as fio
    import SPOT.Tracking.track as SPOT_track
    import SPOT.Tracking.optical_flow as SPOT_optical_flow
    
    
    imfile = r'./data/organoids/fluorescent_murine_colon/KRAS_G12D_EYFP_2.wmv'
    basename = os.path.split(imfile)[-1].split('.wmv')[0]
    
    
    vid = fio.read_video_cv2(imfile)
    
    """
    Get the blue channel vid only 
    """
    desired_ch = 2 # blue channel only 
    vid_ch = vid[...,desired_ch].copy()
    
    """
    Precompute the optical flow for each channel we wish to track, here the blue channel which contains organoids. 
    """
    # setup some parameters for Farneback optical flow computation 
    optical_flow_params = dict(pyr_scale=0.5, levels=5, winsize=15, iterations=5, poly_n=3, poly_sigma=1.2, flags=0)
    
    
    # optical flow works better if we enhance the images / equalize intensities across frames.
    vid_flow = []
    for frame in tqdm(np.arange(len(vid)-1)):
        # frame0 = equalize_hist(rescale_intensity(vid_resize[frame]))*255.
        # frame1 = equalize_hist(rescale_intensity(vid_resize[frame+1]))*255.
        frame0 = skexposure.rescale_intensity(vid_ch[frame]) # pretty much the same. so we don't bother. 
        frame1 = skexposure.rescale_intensity(vid_ch[frame+1]) 
        flow01 = SPOT_optical_flow.Eval_dense_optic_flow(frame0, frame1, 
                                                          params=optical_flow_params)
        vid_flow.append(flow01)
        
    vid_flow = np.array(vid_flow).astype(np.float32) # to save some space. 
    print(vid_flow.shape)
    
    
    """
    Specify the detected bounding box folder 
    """
    bboxfolder = r'../test_outputs/test_detection_folder'
    
    # also specify the save output folder, here we will save to the same folder. 
    outfolder = bboxfolder
    fio.mkdir(outfolder)
    
    
    """
    Bounding box tracking
    """
    # setup parameters. 
    iou_match = 0.25 # this is the required iou match across frames.
    # ds_factor = 4
    ds_factor = 1
    min_aspect_ratio = 3 
    max_dense_bbox_cover = 8 # decreasing this to 4 reduces the length able to be tracked .... # should we include an appearance cost? 
    wait_time = 10
    
    
    vid_bboxes = fio.fetch_channel_boxes(imfile, bboxfolder, ch_no=desired_ch, ending='.wmv')
    
    print(vid_bboxes)
    print('============')
    
    # we now read and sort the boxes into chronological order
    vid_bboxes = fio.read_detected_bboxes_from_file_for_tracking(vid_bboxes)
    
    
    vid_bbox_tracks_prob, vid_bbox_tracks, vid_bbox_match_checks, (vid_bbox_tracks_all_lens, vid_bbox_tracks_all_start_time, vid_bbox_tracks_lifetime_ratios) = SPOT_track.track_organoid_bbox(vid_flow,
                                                                                                                                                                                                vid_bboxes, # bboxes are already loaded in yolo format.   
                                                                                                                                                                                                vid = vid_ch,
                                                                                                                                                                                                iou_match=iou_match,
                                                                                                                                                                                                ds_factor = ds_factor,
                                                                                                                                                                                                wait_time = wait_time,
                                                                                                                                                                                                min_aspect_ratio=min_aspect_ratio,
                                                                                                                                                                                                max_dense_bbox_cover=max_dense_bbox_cover,
                                                                                                                                                                                                to_viz=True,
                                                                                                                                                                                                saveanalysisfolder=None)
    """
    Save the tracking 
    """
    spio.savemat(os.path.join(outfolder, basename, 'flow_bbox_tracks_Ch-%d.mat' %(desired_ch+1)), 
                  {'tracks':vid_bbox_tracks, 
                  'prob_tracks': vid_bbox_tracks_prob, 
                  'track_quality': vid_bbox_match_checks, 
                  'track_lens': vid_bbox_tracks_all_lens,
                  'track_start_time': vid_bbox_tracks_all_start_time,
                  'track_lifetime_ratio': vid_bbox_tracks_lifetime_ratios})


