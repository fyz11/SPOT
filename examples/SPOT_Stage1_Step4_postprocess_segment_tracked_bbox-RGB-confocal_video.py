# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 01:51:28 2022

@author: fyz11
"""

if __name__=="__main__":
    
    """
    This example shows how to postprocess the tracked organoid segmentations to remove unreliable/unstable tracks prior to SAM phenome computation
    
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
    
    

    imfile = r'./data/organoids/fluorescent_murine_colon/KRAS_G12D_EYFP_2.wmv'
    basename = os.path.split(imfile)[-1].split('.wmv')[0]
    
    
    vid = fio.read_video_cv2(imfile)
    
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
    Load the segmentations
    """
    # this is just to show how to autodetect for multiple channels. 
    segmentation_folder = r'../test_outputs/test_segmentation_folder'
    savematfiles = glob.glob(os.path.join(segmentation_folder, basename, 'org_boundaries-'+'Channel-'+'*.mat')) 
    
    
    """
    Detect all the individual channel tracks 
    """
    boundary_samples=200 # the target number of boundary samples, used originally in the tracking. (Stage 1, step 3)
    
    channels_files = np.hstack([int(os.path.split(ff)[-1].split('Channel-')[1].split('.mat')[0]) for ff in savematfiles])
        
    # this is to ensure that the tracks can be operated on as a single numpy array, by padding no detection with np.nan
    boundaries_all = [SPOT_track.pad_tracks(spio.loadmat(savematfile)['boundaries'], boundary_samples=boundary_samples) for savematfile in savematfiles] # load in all the produced tracks into a list and pad....  
    
    
    
    """
    Run some quality control by filtering out any channels that had zero valid tracks. 
    """
    boundaries_all_nan = [SPOT_track.filter_nan_tracks(tra) for tra in boundaries_all]
    
    # reconstitute for plotting and for easy access. 
    boundaries_all_start = []
    
    file_counter = 0 
    for ch_no in range(n_channels):
        if ch_no+1 in channels_files:
            boundaries_all_start.append(boundaries_all_nan[file_counter])
            file_counter += 1 
        else:
            boundaries_all_start.append([])
    
    
    # detect any all nan tracksets 
    remove_index = np.hstack([len(bb) == 0 for bb in boundaries_all_nan])
    
    channels_files = [ channels_files[ii] for ii in np.arange(len(channels_files)) if remove_index[ii]==False]
    boundaries_all_nan = [ boundaries_all_nan[ii] for ii in np.arange(len(boundaries_all_nan)) if remove_index[ii]==False]
    
    
    """
    If there is at least one channel with valid tracks then proceed to do the following postprocessing of tracks
    """
    
    # setup some parameters. 
    border_pixel_size = 20 # implement larger. 
    border_pixel_frac = .1
    temporal_smooth_win_size = 3 # window size of 3 frames -> CNN seg requires less temporal smoothing. 
    saveplots = True # we will save. 
        
    split_tracks_bool = True
    min_split_track_len = 5 # minimum number of frame points. 
    iou_consistency_thresh = .25 # minimum threshold for consistent shape tracks. #-> this is a bit too much for the spreading. 
    rev_channels_for_plot = False # set this to true to reverse RGB channels when generating the final visualization. 
    ds = 1 # no downsampling 

    if len(boundaries_all_nan) > 0: 
            
        # else don't generate anything.
        """
        1. Concatenate remaining sets of tracks (should not be all nan) and apply non-stable filter to suppress potential of tracking the same organoid across channels and within the same video.
        """
        # =============================================================================
        #     Non stable track suppression and reassignment filter
        # =============================================================================
        boundaries_all_filter = SPOT_track.non_stable_track_suppression_filter(vid, 
                                                                                boundaries_all_nan,
                                                                                track_overlap_thresh=0.5, 
                                                                                weight_nan=0., weight_smooth=0.1, 
                                                                                max_obj_frames=10,
                                                                                obj_mean_func=np.nanmean,
                                                                                smoothness_mean_func=np.nanmean,
                                                                                debug_viz=False)
                   
        # repeat the above removal of index. 
        remove_index = np.hstack([len(bb) == 0 for bb in boundaries_all_filter])
        
        channels_files = [ channels_files[ii] for ii in np.arange(len(channels_files)) if remove_index[ii]==False]
        boundaries_all_filter = [ boundaries_all_filter[ii] for ii in np.arange(len(boundaries_all_filter)) if remove_index[ii]==False]
        

        # =============================================================================
        #     Marking of border organoids to output directly things here (i.e. no temporal smoothing etc.)
        # =============================================================================
        border_organoids_all_filter_bool = [SPOT_track.detect_image_border_organoid_tracks( boundaries_all_filter[jj], 
                                                                                            img_shape=vid[0,...,0].shape, 
                                                                                            border_pad = border_pixel_size, 
                                                                                            out_percent = border_pixel_frac, # this is the upper fraction of boundary points located at the image borders.
                                                                                            prepadded_tracks=True,
                                                                                            pad_track_len=boundary_samples,
                                                                                            apply_mask=False) for jj in range(len(boundaries_all_filter))] 
                    
        
        # =============================================================================
        #     Reconstruction for saving out, these should now be the proper tracks.  
        # =============================================================================
        boundaries_all_filter_all_ch = []
        border_organoids_all_filter_bool_all_ch = []
        
        file_counter = 0 
        for ch_no in range(n_channels):
            if ch_no+1 in channels_files:
                boundaries_all_filter_all_ch.append(boundaries_all_filter[file_counter])
                border_organoids_all_filter_bool_all_ch.append(border_organoids_all_filter_bool[file_counter])
                file_counter += 1 
            else:
                boundaries_all_filter_all_ch.append([])
                border_organoids_all_filter_bool_all_ch.append([])
        
        # =============================================================================
        # =============================================================================
        # =============================================================================
        # # #   Extended postprocessing for analysis which includes temporal smoothing, splitting of and construction of more stable iou tracks. 
        # =============================================================================
        # =============================================================================
        # =============================================================================
        
        remove_index = np.hstack([len(bb) == 0 for bb in boundaries_all_filter])
        
        channels_files = [ channels_files[ii] for ii in np.arange(len(channels_files)) if remove_index[ii]==False]
        boundaries_list_post_filter = [ boundaries_all_filter[ii] for ii in np.arange(len(boundaries_all_filter)) if remove_index[ii]==False]
        
        
        """
        2. Temporal smoothing  
        """
        # a) temporal track smoothing with moving average 
        boundaries_list_filter_smooth = [SPOT_track.temporal_smooth_tracks( tra_set, 
                                                                            method='ma', 
                                                                            win_size=temporal_smooth_win_size, 
                                                                            prepadded_tracks=True,
                                                                            pad_track_len=boundary_samples, 
                                                                            ma_avg_func=np.nanmean,
                                                                            ma_pad_mode='edge')  for tra_set in boundaries_list_post_filter] 
                
        """
        3. ensure that successive frames have overlap in appearance as measured by intersection-over-union (IoU), Where inconsistent, the tracks are broken up into tracklets.
        """
        # b) frame by frame check of iou to produce continuous tracks
        boundaries_list_filter_smooth_continuous = [SPOT_track.detect_iou_breaks_trackset(track_set, 
            																	    use_bbox=True, 
                                                                                    iou_thresh=iou_consistency_thresh,
                                                                                    prepadded_tracks=True, 
                                                                                    pad_track_len=boundary_samples, 
                                                                                    split_tracks=split_tracks_bool, 
                                                                                    min_split_track_len=min_split_track_len) for track_set in boundaries_list_filter_smooth]
       
        
        boundaries_list_filter_smooth_breaks = [bb[0] for bb in boundaries_list_filter_smooth_continuous] 
        boundaries_list_filter_smooth_cont_boundaries = [bb[1] for bb in boundaries_list_filter_smooth_continuous] 
        
        
        """
        4. remove objects close to the border, which means their shape is out of the field-of-view but we would measure reduced area / morphologies etc. that could skew the analysis
        """
        # c) redetect border organoids and remove 
        border_organoids_list = [SPOT_track.detect_image_border_organoid_tracks( bb, 
                                                                                img_shape=vid[0,...,0].shape, 
                                                                                border_pad = border_pixel_size, # smoothing will need a larger border. 
                                                                                out_percent = border_pixel_frac, # this is good. 
                                                                                prepadded_tracks=True,
                                                                                pad_track_len=boundary_samples,
                                                                                apply_mask=True) for bb in boundaries_list_filter_smooth_cont_boundaries]
    
        border_organoids_list_masks_post = [bb[0] for bb in border_organoids_list]
        border_organoids_list_boundaries_post = [bb[1] for bb in border_organoids_list]
        
        
        """
        5. final reconstruction of tracks for all channels. 
        """
        boundaries_all_filter_all_ch_final = []
        border_organoids_all_border_bool_all_ch_final = []
        border_organoids_all_breakpts_all_ch_final = []
        
        file_counter = 0 
        for ch_no in range(n_channels):
            if ch_no+1 in channels_files:
                boundaries_all_filter_all_ch_final.append(border_organoids_list_boundaries_post[file_counter])
                border_organoids_all_border_bool_all_ch_final.append(border_organoids_list_masks_post[file_counter])
                border_organoids_all_breakpts_all_ch_final.append(boundaries_list_filter_smooth_breaks[file_counter])
                file_counter += 1 
            else:
                boundaries_all_filter_all_ch_final.append([])
                border_organoids_all_border_bool_all_ch_final.append([])
                border_organoids_all_breakpts_all_ch_final.append([])

    
	# =============================================================================
	# =============================================================================
	# =============================================================================
	# # #         Saving all the computations and parameters used to generate this final set of tracks. 
	# =============================================================================
	# =============================================================================
	# =============================================================================
    
    """
    Final saving
    """
    savematfolder = os.path.join(outfolder, basename)
    savematfile_out = os.path.join(savematfolder, basename+'_boundaries_final_RGB.mat') # final combined trackset for all channels. 
    
    # save the final set of boundaries + intermediate applied masks. 
    spio.savemat(savematfile_out, { 'expt': basename,
                                   'boundaries_raw' : boundaries_all_start, 
                                   'boundaries_filter' : boundaries_all_filter_all_ch,
                                   'border_organoids_filter_bool' : border_organoids_all_filter_bool_all_ch,
                                   'boundaries_smooth_final' :  boundaries_all_filter_all_ch_final, 
                                   'border_organoids_smooth_final_bool' : border_organoids_all_border_bool_all_ch_final,
                                   'boundaries_organoids_smooth_final_breaks' : border_organoids_all_breakpts_all_ch_final,
									'ds': ds, 
									'boundary_samples':boundary_samples, 
									'rev_channels_for_plot': rev_channels_for_plot, 
									'border_pixel_size': border_pixel_size, 
									'border_pixel_frac': border_pixel_frac, 
									'temporal_smooth_win_size':temporal_smooth_win_size, 
									'split_tracks_bool': split_tracks_bool, 
									'min_split_track_len': min_split_track_len, 
									'iou_consistency_thresh': iou_consistency_thresh }) # to do: incorporate this into the saving. for future. 
            
    """
    Final visualization
    """
    if saveplots:
        
        plotfolderoriginal = os.path.join(outfolder, basename, 'raw_RGB'); fio.mkdir(plotfolderoriginal)
        plotfolderfilter = os.path.join(outfolder, basename, 'filter_RGB'); fio.mkdir(plotfolderfilter)
        plotfolderfinal = os.path.join(outfolder, basename, 'filter_smooth_final_RGB'); fio.mkdir(plotfolderfinal)

        img_shape = vid.shape[1:-1]

        plot_colors = ['r', 'g', 'b']
        if rev_channels_for_plot:
            plot_colors = plot_colors[::-1]

        for frame_no in range(len(vid)):
            
            
            # original 
            fig, ax = plt.subplots(figsize=(10,10))
            plt.title('Frame: %d' %(frame_no+1))
            vid_overlay = vid[frame_no].copy()
            
            if rev_channels_for_plot:
                vid_overlay = vid_overlay[...,::-1] 
            ax.imshow(vid_overlay, alpha=.5)
            
            for bb_i, bb in enumerate(boundaries_all_start[:]): 
                for bbb in bb:
                    ax.plot(bbb[frame_no][:,1], 
                            bbb[frame_no][:,0], color=plot_colors[bb_i], lw=3)
            
            ax.set_xlim([0, img_shape[1]-1])
            ax.set_ylim([img_shape[0]-1, 0])
            plt.axis('off')
            plt.grid('off')
            
            fig.savefig(os.path.join(plotfolderoriginal, 
                                     '%s-Frame-%s.png' %(basename, str(frame_no).zfill(3))), bbox_inches='tight')
            plt.show()
            plt.close(fig)
            
            
            # filtered 
            fig, ax = plt.subplots(figsize=(10,10))
            plt.title('Frame: %d' %(frame_no+1))
            vid_overlay = vid[frame_no].copy()
            
            if rev_channels_for_plot:
                vid_overlay = vid_overlay[...,::-1] 
            ax.imshow(vid_overlay, alpha=.5)
            
            for bb_i, bb in enumerate(boundaries_all_filter_all_ch[:]): 
                for bbb in bb:
                    ax.plot(bbb[frame_no][:,1], 
                            bbb[frame_no][:,0], color=plot_colors[bb_i], lw=3)
            
            ax.set_xlim([0, img_shape[1]-1])
            ax.set_ylim([img_shape[0]-1, 0])
            plt.axis('off')
            plt.grid('off')
            
            fig.savefig(os.path.join(plotfolderfilter, 
                                     '%s-Frame-%s.png' %(basename, str(frame_no).zfill(3))), bbox_inches='tight')
            plt.show()
            plt.close(fig)
            
            
            # final 
            fig, ax = plt.subplots(figsize=(10,10))
            plt.title('Frame: %d' %(frame_no+1))
            vid_overlay = vid[frame_no].copy()
            
            if rev_channels_for_plot:
                vid_overlay = vid_overlay[...,::-1] 
            ax.imshow(vid_overlay, alpha=.5)
            
            for bb_i, bb in enumerate(boundaries_all_filter_all_ch_final[:]): 
                for bbb in bb:
                    ax.plot(bbb[frame_no][:,1], 
                            bbb[frame_no][:,0], color=plot_colors[bb_i], lw=3)
            
            ax.set_xlim([0, img_shape[1]-1])
            ax.set_ylim([img_shape[0]-1, 0])
            plt.axis('off')
            plt.grid('off')
            
            fig.savefig(os.path.join(plotfolderfinal, 
                                     '%s-Frame-%s.png' %(basename, str(frame_no).zfill(3))), bbox_inches='tight')
            plt.show()
            plt.close(fig)


