#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 23:47:44 2020

@author: felix
"""
import numpy as np 

def _mkdir(directory):
    """ check if directory exists and create it through Python if it does not yet.

    Parameters
    ----------
    directory : str
        the directory path (absolute or relative) you wish to create (or check that it exists)

    Returns
    -------
        void function, no return
    """
    import os 
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return []

def _unique_pts(a):
    import numpy as np 
    # returns the unique rows of an array. 
    return np.vstack(list({tuple(row) for row in a}))


def _resample_curve(x,y, k=1, s=0, n_samples=10, per=True):
    
    import scipy.interpolate
    import numpy as np 

    if s is None:
        tck, u = scipy.interpolate.splprep([x,y], k=k, per=per)
    else:
        tck, u = scipy.interpolate.splprep([x,y], k=k, s=s, per=per)
    unew = np.linspace(0, 1.00, n_samples)
    out = scipy.interpolate.splev(unew, tck) 
    
    return np.vstack(out).T

def _corner_cutting_smoothing(x,y, n_iters=1):
    
    poly = np.vstack([x,y]).T
    poly = poly[:-1].copy()

    for ii in range(n_iters):
        
        m,n = poly.shape
        poly_ = np.zeros((2*m,n), dtype=np.float32)

        poly_close = np.vstack([poly, poly[0]])
        poly_0 = poly_close[:-1].copy()
        poly_1 = poly_close[1:].copy()
        
        poly_[::2] = poly_0 * .75 + poly_1 * .25
        poly_[1::2] = poly_0 * .25 + poly_1 * .75
        
        poly = poly_.copy()
        
    return np.vstack([poly, poly[0]])


# def iou_mask(mask1, mask2):
    
#     intersection = np.sum(np.abs(mask1*mask2))
#     union = np.sum(mask1) + np.sum(mask2) - intersection # this is the union area. 
    
#     overlap = intersection / float(union + 1e-8)
#     area1 = np.sum(mask1)
#     area2 = np.sum(mask2)
    
#     return intersection / float(area1 + 1e-8), intersection / float(area2 + 1e-8), overlap 

# def draw_poly_mask(pts,shape):
    
#     from skimage.draw import polygon
#     mask = np.zeros(shape, dtype=np.bool)
#     rr,cc = polygon(pts[:,0], pts[:,1])
#     mask[rr,cc] = 1
    
#     return mask > 0
    
# def iou_polygon_area(pts0, pts1,shape):
    
#     poly1 = draw_poly_mask(pts0, shape)
#     poly2 = draw_poly_mask(pts1, shape)
    
#     return iou_mask(poly1, poly2)

def get_iou(bb1, bb2):
    """ Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    iou : float
        the intersecton over union metric in [0, 1]

    """
    if np.sum(bb1) < 1 and np.sum(bb2) < 1:
        iou = 0
    else:
        bb1 = {'x1': bb1[0], 
               'y1': bb1[1], 
               'x2': bb1[2],
               'y2': bb1[3]}
        bb2 = {'x1': bb2[0], 
               'y1': bb2[1], 
               'x2': bb2[2],
               'y2': bb2[3]}
        
        # allow for collapsed boxes.           
        assert bb1['x1'] <= bb1['x2']
        assert bb1['y1'] <= bb1['y2']
        assert bb2['x1'] <= bb2['x2']
        assert bb2['y1'] <= bb2['y2']
    
        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])
    
        if x_right < x_left or y_bottom < y_top:
            return 0.0
    
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
    
    return iou


def _load_bbox_frame_voc( img, bbox_obj_frame):

    import numpy as np
    # this function takes the yolo boxes and converts them into voc format. At the same time clips non-plausible boxes (given the image limits)
    nrows, ncols = img.shape
    
    frame_No, bboxes = bbox_obj_frame
    probs = bboxes[:,1].astype(np.float32)
    boxes = bboxes[:,2:]

    x = boxes[:,0].astype(np.int32)
    y = boxes[:,1].astype(np.int32)
    w = boxes[:,2].astype(np.int32)
    h = boxes[:,3].astype(np.int32)
    
    
    x1 = x - w//2 ; x1 = np.clip(x1, 0, ncols-1)
    x2 = x1 + w ; x2 = np.clip(x2, 0, ncols-1)
    y1 = y - h//2 ; y1 = np.clip(y1, 0, nrows-1)
    y2 = y1 + h ; y2 = np.clip(y2, 0, nrows-1)
    
    
    boxes_voc = np.hstack([x1[:,None],y1[:,None],x2[:,None],y2[:,None]])
        
    return probs, boxes_voc


def predict_new_boxes_flow_tf(boxes, flow):
    """ predict new bounding box position in the next frame based on an affine transformation, given the optical flow. 
    
    Parameters
    ----------
    boxes : list or array of box
        each box in voc format, (x1,y1,x2,y2)
    flow : (n_rows, n_cols, 2) array
        image of x- (1st axis), y-(2nd axis) optical flow displacements 
    
    Returns
    -------
    tfs : list of transforms for each box 
        skimage.transform 'similarity' transformation object for each bounding box. If a transform could not be fitted, then then transform was empty i.e. []
    new_boxes_tf : list or array of box
        the predicted new bounding boxes, each box in voc format, (x1,y1,x2,y2)
        
    """
    from skimage.transform import estimate_transform, matrix_transform, SimilarityTransform
    from skimage.measure import ransac
    import numpy as np 
    
    flow_x = flow[:,:,0]
    flow_y = flow[:,:,1]
      
    new_boxes_tf = []
    tfs = []
    
    for box in boxes:
        x1,y1,x2,y2 = box
        
        x1 = int(x1); y1 =int(y1); x2=int(x2); y2 =int(y2);  # added, weirdly ...
        
        # how to take into account the change in size (scale + translation ? very difficult. )
        flow_x_patch = flow_x[y1:y2, x1:x2].copy()
        flow_y_patch = flow_y[y1:y2, x1:x2].copy()
        
        nrows_, ncols_ = flow_x_patch.shape
        
        # threshold_the mag
        mag_patch = np.sqrt(flow_x_patch ** 2 + flow_y_patch ** 2)
        select = mag_patch > 0
#        select = np.ones(mag_patch.shape) > 0
        pix_X, pix_Y = np.meshgrid(range(ncols_), range(nrows_))
        
        if np.sum(select) == 0:
            # if we cannot record movement in the box just append the original ?
            tfs.append([])
            new_boxes_tf.append([x1,y1,x2,y2])
        else:
            src_x = pix_X[select].ravel(); dst_x = src_x + flow_x_patch[select]
            src_y = pix_Y[select].ravel(); dst_y = src_y + flow_y_patch[select]
            src = np.hstack([src_x[:,None], src_y[:,None]])
            dst = np.hstack([dst_x[:,None], dst_y[:,None]])
            
            # estimate the transformation. 
            matrix = estimate_transform('similarity', src[:,[0,1]], dst[:,[0,1]])
            tf_scale = matrix.scale; tf_offset = matrix.translation
            
#            print tf_scale, tf_offset
            if np.isnan(tf_scale):
                tfs.append(([]))
                new_boxes_tf.append([x1,y1,x2,y2])
            else:
                x = .5*(x1+x2); w = x2-x1
                y = .5*(y1+y2); h = y2-y1
                
                x1_tf_new = x + tf_offset[0] - w*tf_scale/2.
                y1_tf_new = y + tf_offset[1] - h*tf_scale/2.
                x2_tf_new = x1_tf_new + w*tf_scale
                y2_tf_new = y1_tf_new + h*tf_scale
                
        #        print x1_tf_new
                tfs.append(matrix)
                new_boxes_tf.append([x1_tf_new, y1_tf_new, x2_tf_new, y2_tf_new])
        
    new_boxes_tf = np.array(new_boxes_tf)
    
    return tfs, new_boxes_tf


def bbox_tracks_2_array(vid_bbox_tracks, nframes, ndim):
    """ Convenience function to take a list of bounding box tracks and compile them into a regular numpy array where missing timepoints are replaced with np.nan
    
    Parameters
    ----------
    vid_bbox_tracks : list of N_tracks SAM bounding box tracks
        each bounding box track is a list of [timepoint, prob, box] where timepoint is the frame that the bounding box was detected at, prob is the score of the box and box is a list or array of 4 numbers, (x1,y1,x2,y2) describing the lower and upper points.
    nframes : int
        the total number of frames in the corresponding video. 
    ndim : int
        the dimension of the bounding box coordinates. This will be 4. 

    Returns
    -------
    vid_bbox_tracks_prob_all_array : (N_tracks, N_frames) array
        the probability score of each bounding box in every track at every potential timepoint 
    vid_bbox_tracks_all_array : (N_tracks, N_frames, 4) array 
        the compiled regular numpy array of bounding box tracks for easy postprocessing such as taking the average. 

    """
    import numpy as np 
    N_tracks = len(vid_bbox_tracks)

    vid_bbox_tracks_all_array = np.zeros((N_tracks, nframes, ndim))
    vid_bbox_tracks_all_array[:] = np.nan

    vid_bbox_tracks_prob_all_array = np.zeros((N_tracks, nframes))
    vid_bbox_tracks_prob_all_array[:] = np.nan

    for ii in np.arange(N_tracks):
        tra_ii = vid_bbox_tracks[ii]
        tra_ii_times = np.array([tra_iii[0] for tra_iii in tra_ii])
        tra_ii_boxes = np.array([tra_iii[-1] for tra_iii in tra_ii]) # boxes is the last array. 
        tra_ii_prob = np.array([tra_iii[1] for tra_iii in tra_ii])

        vid_bbox_tracks_all_array[ii,tra_ii_times] = tra_ii_boxes.copy()
        vid_bbox_tracks_prob_all_array[ii,tra_ii_times] = tra_ii_prob.copy()

    return vid_bbox_tracks_prob_all_array, vid_bbox_tracks_all_array


def bbox_scalar_2_array(vid_scalar_tracks, nframes):
    """  Convenience function to take a list of bounding box track associated scalar measurements made at specific timepoints and compile them into a regular numpy array where missing timepoints are replaced with np.nan
    
    Parameters
    ----------
    vid_scalar_tracks : list of N_tracks of scalar measurements from different timepoints
        each scalar measurements is a list of [timepoint, scalar] where timepoint is the frame that the measurement was measured at.
    nframes : int
        the total number of frames in the corresponding video. 

    Returns
    -------
    vid_scalar_tracks_all_array : (N_tracks, N_frames) array
        the compiled regular array of a scalar measurement for every track at every potential timepoint 
    
    """
    import numpy as np 
    N_tracks = len(vid_scalar_tracks)

    vid_scalar_tracks_all_array = np.zeros((N_tracks, nframes))
    vid_scalar_tracks_all_array[:] = np.nan

    for ii in np.arange(N_tracks):
        tra_ii = vid_scalar_tracks[ii]
        tra_ii_times = np.array([tra_iii[0] for tra_iii in tra_ii])
        tra_ii_vals = np.array([tra_iii[1] for tra_iii in tra_ii])

        vid_scalar_tracks_all_array[ii,tra_ii_times] = tra_ii_vals.copy()

    return vid_scalar_tracks_all_array


# is there a quick way to compute the occupied bbox density. 
def remove_very_large_bbox(boxes, 
                           shape, 
                           thresh=0.5, 
                           aspect_ratio=None, 
                           method='density', 
                           max_density=4, 
                           mode='fast', 
                           return_ind=False):
    
    """ Filters bounding boxes in an image and remove bounding boxes by extreme aspect ratio and by the number of co-overlapping bounding boxes (density) contained within it. This is used to remove spurious bounding box detections as part of preprocessing
    
    Parameters
    ----------
    boxes : (N_boxes,4)
        array of bounding box. each box is given in VOC format, (x1,y1,x2,y2)
    shape : 2-tuple
        (n_rows,n_cols) size of the image 
    thresh : float [0-1]
        used if method = 'area'. removes all bounding boxes with area which occupies a fraction > thresh of total image area. 
    aspect_ratio : float
        Only if method='density'. maximum aspect ratio cutoff to filter bounding boxes. aspect ratio is the maximum of width and height divided by the minimum of height and width. if None, an aspect ratio is not applied before density filtering.
    method : 'density' or 'area'
        specifies the filtering method either 'area' which removes purely overlarge bounding boxes or 'density' which removes bounding boxes of (optionally bounding aspect ratio) which cover too many other bounding boxes. The idea is that with organoids, its unlikly an organoid should hide a lot of inner organoids even in 2D.
    max_density : int 
        cut-off. The maximum number of organoids a bounding box may cover before it is removed, if method='density'
    mode : 'fast' or anything else
        if method='density' and mode='fast' estimate the density, the number of bounding box it covers using fast k-nearest neighbor lookup based on the bounding box centroids. If mode is not fast i.e. any other string, then the more accurate estimate of bounding box coverage is used based on computing explicit segmentation masks for each bounding box. 
    return_ind : bool 
        if True, return the indices of the bounding boxes in the original input that is kept in the final filtered bounding boxes.
    
    Returns 
    -------
    boxes_ : (N_boxes_filtered,4) array
        array of the final filtered bounding box. each box is given in VOC format, (x1,y1,x2,y2)
    keep_ind : (N_boxes_filtered,) array
        if return_ind=True, return the indices of the bounding boxes in the original input that the final filtered bounding boxes correspond to.
    """
    
    from sklearn.metrics.pairwise import pairwise_distances
    
    if method == 'density':
        
        boxes_ = boxes.copy()
        keep_ind = np.arange(len(boxes_))
        
        if aspect_ratio is not None:
            w = boxes[:,2] - boxes[:,0]
            h = boxes[:,3] - boxes[:,1]
            wh = np.vstack([w,h])
            
            aspect_w_h = np.max(wh, axis=0) / (np.min(wh, axis=0) + .1)
            boxes_ = boxes_[aspect_w_h<=aspect_ratio]
            keep_ind = keep_ind[aspect_w_h<=aspect_ratio]
        
        box_density = np.zeros(shape)
        bbox_coverage = np.zeros(len(boxes_))
        box_centroids_x = np.clip((.5*(boxes_[:,0] + boxes_[:,2])).astype(np.int32), 0, shape[1]-1).astype(np.int32)
        box_centroids_y = np.clip((.5*(boxes_[:,1] + boxes_[:,3])).astype(np.int32), 0, shape[0]-1).astype(np.int32)
        
        if mode == 'fast':
            box_centroids = np.vstack([box_centroids_x, box_centroids_y]).T
            box_centroids_r = np.sqrt((boxes_[:,2]-boxes_[:,0])*(boxes_[:,3]-boxes_[:,1])/np.pi +.1)
        
            box_centroids_distance = pairwise_distances(box_centroids)
            bbox_coverage = np.nansum(box_centroids_distance<=box_centroids_r[:,None], axis=1)
            
        else:
            # box_density[box_centroids_y, box_centroids_x] += 1
            for cent_ii in np.arange(len(box_centroids_x)):
                cent_x = box_centroids_x[cent_ii]
                cent_y = box_centroids_y[cent_ii]
                box_density[int(cent_y), int(cent_x)] += 1
            
            for box_ii, box in enumerate(boxes_):
                x1, y1, x2, y2 = box
                x1 = np.clip(int(x1), 0, shape[1]-1)
                y1 = np.clip(int(y1), 0, shape[0]-1)
                x2 = np.clip(int(x2), 0, shape[1]-1)
                y2 = np.clip(int(y2), 0, shape[0]-1)
                bbox_coverage[box_ii] = np.nansum(box_density[int(y1):int(y2), 
                                                              int(x1):int(x2)])        
        # print(bbox_coverage)
        boxes_ = boxes_[bbox_coverage<=max_density].copy()
        if return_ind == False:
            return boxes_
        else:
            keep_ind = keep_ind[bbox_coverage<=max_density].copy()
            return boxes_, keep_ind

    if method == 'area':
        areas_box = []
        area_shape = float(np.prod(shape))
        
    #    print(area_shape)
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            
            area_box = (y2-y1)*(x2-x1)
            areas_box.append(area_box)
            
        areas_box = np.hstack(areas_box)
        areas_box_frac = areas_box / float(area_shape)
        return boxes[areas_box_frac<=thresh]



def bbox_iou_corner_xy(bboxes1, bboxes2):
    """ Parallelised computation of the distance matrix between two sets of bounding boxes.
    
    Parameters
    ----------
    bboxes1 : shape (total_bboxes1, 4)
             with x1, y1, x2, y2 point order.
    bboxes2 : shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.

    Returns
    -------
    iou : shape (total_bboxes1, total_bboxes2)
        Tensor with shape (total_bboxes1, total_bboxes2) with the IoU (intersection over union) of bboxes1[i] and bboxes2[j] in [i, j].
    
    """
    import numpy as np 
    x11, y11, x12, y12 = bboxes1[:,0], bboxes1[:,1], bboxes1[:,2], bboxes1[:,3]
    x21, y21, x22, y22 = bboxes2[:,0], bboxes2[:,1], bboxes2[:,2], bboxes2[:,3]


    x11 = x11[:,None]; y11 = y11[:,None]; x12=x12[:,None]; y12=y12[:,None]
    x21 = x21[:,None]; y21 = y21[:,None]; x22=x22[:,None]; y22=y22[:,None]

    xI1 = np.maximum(x11, np.transpose(x21))
    xI2 = np.minimum(x12, np.transpose(x22))

    yI1 = np.maximum(y11, np.transpose(y21))
    yI2 = np.minimum(y12, np.transpose(y22))

    inter_area = np.maximum((xI2 - xI1), 0.) * np.maximum((yI2 - yI1), 0.)

    bboxes1_area = (x12 - x11) * (y12 - y11)
    bboxes2_area = (x22 - x21) * (y22 - y21)

    union = (bboxes1_area + np.transpose(bboxes2_area)) - inter_area

    # some invalid boxes should have iou of 0 instead of NaN
    # If inter_area is 0, then this result will be 0; if inter_area is
    # not 0, then union is not too, therefore adding a epsilon is OK.
    iou = inter_area / (union+0.0001)
    return iou


def track_organoid_bbox(vid_flow,
                        vid_bboxes, # bboxes are already loaded in yolo format.   
                        vid=None, 
                        iou_match=.25,
                        ds_factor = 1,
                        wait_time=10,
                        min_aspect_ratio=3,
                        max_dense_bbox_cover=8,
                        min_tra_len_filter=5,
                        min_tra_lifetime_ratio = .1, 
                        to_viz=True,
                        saveanalysisfolder=None):

    """ Master function to track bounding box detection of organoids using optical flow instead of Kalman filter as a frame-to-frame predictor to track robustly organoid morphological deformation.
    
    Parameters
    ----------
    vid_flow : (n_frames-1, n_rows, n_cols, 2) array
        precomputed array of the frame-to-frame optical flow displacements 
    vid_bboxes : list of bounding boxes 
        list of the bounding boxes detected in each frame by YOLO in yolo format i.e. (label, score, x, y, w, h)
    vid : (n_frames, n_rows, n_cols) or None
        optionally the frames of the video to display if to_viz=True to visualize and check the tracking frame to frame.  
    iou_match : float in [0,1]
        the minimal cutoff for a successful bounding box match between frames 
    ds_factor : float
        downsampling factor if the bounding box were computed on a downsampled version of the video and now one wishes to track and compute the bounding box at the orignal resolution.
    wait_time : int 
        specifies the wait time in # of frame before termination of a track. During the waiting time, the bounding box is predicted by optical flow. If no match is found at the end of the waiting time, the imputed bounding boxes are all removed. If within a detection is found, the wait time is reset to 0. 
    min_aspect_ratio : float
        specifies the maximum bounding box aspect_ratio for a true organoid detectiton. 
    max_dense_bbox_cover : int 
        specifies the maximum number of organoids an organoid bounding box can cover. Above this, we assume the bounding box is not reliable as we don't expect an organoid even with transparent lumen to cover so many organoids.
    min_tra_len_filter : int 
        return only bounding box tracks that last the specified minimum # of frames. 
    min_tra_lifetime_ratio : float in [0,1]
        the track lifetime ratio is defined as the # of frames tracked divided by the total number of frames it could have been tracked for. Assuming organoids do not merge or become occluded, the maximal number of frames an organoid can be tracked for is the number of video frames - the starting frame.  
    to_viz : bool
        if True, plot the tracked bounding boxes per frame. Boxes belonging to the same track over time will be colored with the same unique color. 
    saveanalysisfolder : str or None
        if the folder path is given, the individual frames with colored tracked bounding boxes are saved. This is useful for making a video of the tracking. 

    Returns 
    -------
    vid_bbox_tracks_prob_all_array : (N_tracks, N_frames) array
        the compiled regular array of bounding box scores for each track. np.nan is used if the bounding box did not exist at a timepoint. 
    vid_bbox_tracks_all_array : (N_tracks, N_frames, 4) array
        the compiled regular array of bounding boxes where each bounding boxe is specified in VOC format as (x1,y1,x2,y2). np.nan is used if the bounding box did not exist at a timepoint. 
    vid_match_checks_all_array : (N_tracks, N_frames) array
        the compiled regular array of match checks 0 or 1 for each track. a 1 indicates the bounding box in a track was not imputed by optical flow but was matched from bounding box detections. np.nan is used if the bounding box did not exist at a timepoint. 
    (vid_bbox_tracks_all_lens, vid_bbox_tracks_all_start_time, vid_bbox_tracks_lifetime_ratios) : ((N_tracks,), (N_tracks,), (N_tracks,)) list of arrays
        the length(# of frames), start frame and lifetime ratio (length/(total_video_frames-start_frame)) for each bounding box track.

    """
    import numpy as np 
    import seaborn as sns
    import pylab as plt 
    import os 
    from tqdm import tqdm 
    from scipy.optimize import linear_sum_assignment

    """
    initial setting up. 
    """
    im_shape = vid_flow[0].shape[:2]
    nframes = len(vid_flow)+1 # the number of frames in the video.

    if saveanalysisfolder is not None:
        print('saving graphics in folder: ', saveanalysisfolder)
        # saveanalysisfolder_movie = os.path.join(saveanalysisfolder, 'track_boundaries'); mkdir(saveanalysisfolder_movie); 
        saveanalysisfolder_movie_bbox = os.path.join(saveanalysisfolder, 'track_bbox'); 
        _mkdir(saveanalysisfolder_movie_bbox);     

    # =============================================================================
    #     BBox Tracks
    # =============================================================================

    # terminated_vid_cell_tracks = [] 
    terminated_vid_bbox_tracks = []
    terminated_check_match_bbox_tracks = []

    vid_bbox_tracks = [] # this is to keep track of the actual bbox we use, including inferred. 
    vid_bbox_tracks_last_time = [] # record when the last time a proper match was made. 
    vid_match_check_bbox_tracks = []

    # =============================================================================
    #     build up the tracks dynamically now, frame by frame dynamically , with a track waiting time before termination. 
    # ============================================================================
    for ii in tqdm(np.arange(nframes-1)[:]):
        """
        if working set is empty or otherwise, then add to working set. 
        """
        # add these to the working tracks. 
        if ii == 0 or len(vid_bbox_tracks)==0: 
            bboxfile_ii = vid_bboxes[ii]
            prob_ii, boxes_ii = _load_bbox_frame_voc( vid[0], bboxfile_ii)
            boxes_ii = boxes_ii / float(ds_factor)
            
            # clip
            boxes_ii[:,0] = np.clip(boxes_ii[:,0], 0, im_shape[1]-1)
            boxes_ii[:,1] = np.clip(boxes_ii[:,1], 0, im_shape[0]-1)
            boxes_ii[:,2] = np.clip(boxes_ii[:,2], 0, im_shape[1]-1)
            boxes_ii[:,3] = np.clip(boxes_ii[:,3], 0, im_shape[0]-1)
        
            # remove all boxes that are have an area 1 pixels or less.               
            boxes_ii_w = boxes_ii[:,2] - boxes_ii[:,0]
            boxes_ii_h = boxes_ii[:,3] - boxes_ii[:,1]
            boxes_ii = boxes_ii[boxes_ii_w*boxes_ii_h>0]

            # suppress implausible. 
            boxes_ii, keep_ind_ii = remove_very_large_bbox(boxes_ii, 
                                                          im_shape, 
                                                          thresh=0.5, 
                                                          aspect_ratio = 3, # we don't expect it to be very long. 
                                                          method='density', 
                                                          max_density=max_dense_bbox_cover, 
                                                          mode='fast', 
                                                          return_ind=True)
            assert(len(boxes_ii) == len(keep_ind_ii)) # keep a check here.       
            prob_ii = prob_ii[keep_ind_ii]

            for jj in np.arange(len(boxes_ii)):
                vid_bbox_tracks.append([[ii, prob_ii[jj], boxes_ii[jj]]]) # add the prob in as another entry here. 
                vid_bbox_tracks_last_time.append(ii) # update with current time. 
                vid_match_check_bbox_tracks.append([[ii, 1]]) # update with the current time., this is just to select whether this box was propagated or found in the original detections. 
                
        """
        load the current working set of tracks. 
        """
        # 1) check for track termination 
        # get the last timepoint. 
        boxes_ii_track_time = np.hstack(vid_bbox_tracks_last_time)
        
        # check if any of the box tracks need to be terminated. 
        track_terminate =  boxes_ii_track_time < ii - wait_time
        track_terminate_id = np.arange(len(track_terminate))[track_terminate]

        if len(track_terminate_id)>0:
            # update the relevant info.
            for idd in track_terminate_id:
                #update
                terminated_vid_bbox_tracks.append(vid_bbox_tracks[idd][:-wait_time-1])
                terminated_check_match_bbox_tracks.append(vid_match_check_bbox_tracks[idd][:-wait_time-1])
                # terminated_vid_cell_tracks_properties.append(vid_cell_tracks_properties[idd][:-wait_time-1])

            # # reform the working set. 
            # vid_cell_tracks = [vid_cell_tracks[jjj] for jjj in np.arange(len(vid_cell_tracks)) if jjj not in track_terminate_id] # i guess this is to keep track of actual cell ids that we segmented.
            vid_bbox_tracks = [vid_bbox_tracks[jjj] for jjj in np.arange(len(vid_bbox_tracks)) if jjj not in track_terminate_id] # this is to keep track of the actual bbox we use, including inferred. 
            # vid_cell_tracks_properties = [vid_cell_tracks_properties[jjj] for jjj in np.arange(len(vid_cell_tracks_properties)) if jjj not in track_terminate_id]
            vid_bbox_tracks_last_time = [vid_bbox_tracks_last_time[jjj] for jjj in np.arange(len(vid_bbox_tracks_last_time)) if jjj not in track_terminate_id]
            vid_match_check_bbox_tracks = [vid_match_check_bbox_tracks[jjj] for jjj in np.arange(len(vid_match_check_bbox_tracks)) if jjj not in track_terminate_id]

        # load the current version of the boxes. 
        boxes_ii_track = np.array([bb[-1][-1] for bb in vid_bbox_tracks]) # the bboxes to consider.
        boxes_ii_track_prob = np.array([bb[-1][1] for bb in vid_bbox_tracks]) 
        boxes_ii_track_time = np.array([bb[-1][0] for bb in vid_bbox_tracks]) # the time of the last track. 

        """
        Infer the next frame boxes from current boxes using optical flow.    
        """
        boxes_ii_track_pred = []
        
        for jjj in np.arange(len(boxes_ii_track)):
            """
            Predict using the flow the likely position of boxes. 
            """
            new_tfs, boxes_ii_pred = predict_new_boxes_flow_tf(boxes_ii_track[jjj][None,:], 
                                                               vid_flow[boxes_ii_track_time[jjj]])
            # clip  
            boxes_ii_pred[:,0] = np.clip(boxes_ii_pred[:,0], 0, im_shape[1]-1)
            boxes_ii_pred[:,1] = np.clip(boxes_ii_pred[:,1], 0, im_shape[0]-1)
            boxes_ii_pred[:,2] = np.clip(boxes_ii_pred[:,2], 0, im_shape[1]-1)
            boxes_ii_pred[:,3] = np.clip(boxes_ii_pred[:,3], 0, im_shape[0]-1)
            boxes_ii_track_pred.append(boxes_ii_pred[0])
  
        boxes_ii_track_pred = np.array(boxes_ii_track_pred)
            
        """
        load the next frame boxes. which are the candidates.  
        """
        bboxfile_jj = vid_bboxes[ii+1]
        prob_jj, boxes_jj = _load_bbox_frame_voc( vid[ii+1], 
                                                 bboxfile_jj)
        boxes_jj = boxes_jj/float(ds_factor)
        # clip
        boxes_jj[:,0] = np.clip(boxes_jj[:,0], 0, im_shape[1]-1)
        boxes_jj[:,1] = np.clip(boxes_jj[:,1], 0, im_shape[0]-1)
        boxes_jj[:,2] = np.clip(boxes_jj[:,2], 0, im_shape[1]-1)
        boxes_jj[:,3] = np.clip(boxes_jj[:,3], 0, im_shape[0]-1) 
        # remove all boxes that are have an area 1 pixels or less.               
        boxes_jj_w = boxes_jj[:,2] - boxes_jj[:,0]
        boxes_jj_h = boxes_jj[:,3] - boxes_jj[:,1]
        boxes_jj = boxes_jj[boxes_jj_w*boxes_jj_h>0]

        # suppress implausible. 
        boxes_jj, keep_ind_jj = remove_very_large_bbox(boxes_jj, 
                                                      im_shape, 
                                                      thresh=0.5, 
                                                      aspect_ratio = 3, # we don't expect it to be very long. 
                                                      method='density', 
                                                      max_density=max_dense_bbox_cover, 
                                                      mode='fast', return_ind=True)
        prob_jj = prob_jj[keep_ind_jj]
        
        """
        build the association matrix and match boxes based on iou. 
        """
        iou_matrix = bbox_iou_corner_xy(boxes_ii_track_pred, 
                                        boxes_jj); 
        iou_matrix = np.clip(1.-iou_matrix, 0, 1) # to make it a dissimilarity matrix. 
        
        # solve the pairing problem.
        ind_ii, ind_jj = linear_sum_assignment(iou_matrix)
        
        # threshold as the matching is maximal. 
        iou_ii_jj = iou_matrix[ind_ii, ind_jj].copy()
        keep = iou_ii_jj <= (1-iou_match)
        # keep = iou_ii_jj <= dist_thresh
        ind_ii = ind_ii[keep>0]; 
        ind_jj = ind_jj[keep>0]
        
        """
        Update the trajectories. 
        """
        # update first the matched.
        for match_ii in np.arange(len(ind_ii)):
            # vid_cell_tracks[ind_ii[match_ii]].append([ii+1, ind_jj[match_ii]]) # i guess this is to keep track of actual cell ids that we segmented.
            vid_bbox_tracks[ind_ii[match_ii]].append([ii+1, prob_jj[ind_jj[match_ii]], boxes_jj[ind_jj[match_ii]]]) # this is to keep track of the actual bbox we use, including inferred. 
            # vid_cell_tracks_properties[ind_ii[match_ii]].append([ii+1, masks_jj_properties[ind_jj[match_ii]]]) # append the properties of the next time point. 
            vid_bbox_tracks_last_time[ind_ii[match_ii]] = ii+1 # let this be a record of the last time a 'real' segmentation was matched, not one inferred from optical flow. 
            # vid_mask_tracks_last[ind_ii[match_ii]] = masks_jj[...,ind_jj[match_ii]] # retain just the last masks thats relevant for us. 
            # vid_mask_tracks_last[ind_ii[match_ii]] = boxes_jj[ind_jj[match_ii]]
            vid_match_check_bbox_tracks[ind_ii[match_ii]].append([ii+1, 1]) # success, append 0 
            
        no_match_ind_ii = np.setdiff1d(np.arange(len(boxes_ii_track_pred)), ind_ii)
        no_match_ind_jj = np.setdiff1d(np.arange(len(boxes_jj)), ind_jj)
        
        for idd in no_match_ind_ii:
            # these tracks already exist so we just add to the existant tracks.         
            # vid_cell_tracks[idd].append([ii+1, -1]) # i guess this is to keep track of actual cell ids that we segmented.
            vid_bbox_tracks[idd].append([ii+1, boxes_ii_track_prob[idd], boxes_ii_track_pred[idd]]) # this is to keep track of the actual bbox we use, including inferred. 
            # vid_cell_tracks_properties[idd].append([ii+1, properties_ii_track_pred[idd]])
            # vid_bbox_tracks_last_time[idd] = ii+1 # let this be a record of the last time a 'real' segmentation was matched, not one inferred from optical flow. 
            # vid_mask_tracks_last[idd] = boxes_ii_track_pred[idd] # retain just the last masks thats relevant for us.
            vid_match_check_bbox_tracks[idd].append([ii+1, 0]) # no success, append 0  
    
        for idd in no_match_ind_jj:
            # these tracks don't exsit yet so we need to create new tracks. 
            # vid_cell_tracks.append([[ii+1, idd]]) # i guess this is to keep track of actual cell ids that we segmented.
            vid_bbox_tracks.append([[ii+1, prob_jj[idd], boxes_jj[idd]]]) # this is to keep track of the actual bbox we use, including inferred. 
            # vid_cell_tracks_properties.append([[ii+1, masks_jj_properties[idd]]])
            vid_bbox_tracks_last_time.append(ii+1) # let this be a record of the last time a 'real' segmentation was matched, not one inferred from optical flow. 
            # vid_mask_tracks_last.append(masks_jj[...,idd]) # retain just the last masks thats relevant for us. 
            # vid_mask_tracks_last.append(boxes_jj[idd])
            vid_match_check_bbox_tracks.append([[ii+1, 1]])
                
    # =============================================================================
    #     Combine the tracks togther
    # =============================================================================
    vid_bbox_tracks_all = terminated_vid_bbox_tracks + vid_bbox_tracks # combine
    vid_match_checks_all = terminated_check_match_bbox_tracks + vid_match_check_bbox_tracks


    # =============================================================================
    #     Compute some basic track properties for later filtering. 
    # =============================================================================    
    vid_bbox_tracks_all_lens = np.hstack([len(tra) for tra in vid_bbox_tracks_all])
    vid_bbox_tracks_all_start_time = np.hstack([tra[0][0] for tra in vid_bbox_tracks_all])
    vid_bbox_tracks_lifetime_ratios = vid_bbox_tracks_all_lens / (len(vid) - vid_bbox_tracks_all_start_time).astype(np.float32)
    
    # =============================================================================
    #     Turn into a proper array of n_tracks x n_time x 4 or 5.... 
    # =============================================================================
    vid_bbox_tracks_prob_all_array, vid_bbox_tracks_all_array = bbox_tracks_2_array(vid_bbox_tracks_all, nframes=nframes, ndim=boxes_ii.shape[1])
    vid_match_checks_all_array = bbox_scalar_2_array(vid_match_checks_all, nframes=nframes)

    # =============================================================================
    #   Apply the given filtering parameters for visualization else it will be messy.        
    # =============================================================================

    if to_viz:

        fig, ax = plt.subplots()
        ax.scatter(vid_bbox_tracks_all_start_time, 
                   vid_bbox_tracks_all_lens, 
                   c=vid_bbox_tracks_lifetime_ratios, 
                   vmin=0, 
                   vmax=1, cmap='coolwarm')
        plt.show()


        # keep filter
        keep_filter = np.logical_and(vid_bbox_tracks_all_lens>=min_tra_len_filter, 
                                     vid_bbox_tracks_lifetime_ratios>=min_tra_lifetime_ratio)
        keep_ids = np.arange(len(vid_bbox_tracks_all_lens))[keep_filter]

        plot_vid_bbox_tracks_all = vid_bbox_tracks_all_array[keep_ids].copy()
        plot_colors = sns.color_palette('hls', len(plot_vid_bbox_tracks_all))

        # print(len(plot_vid_bbox_tracks_all), len(vid_bbox_tracks_all_array))
        # iterate over time. 
        for frame_no in np.arange(nframes):

            fig, ax = plt.subplots(figsize=(5,5))
            plt.title('Frame: %d' %(frame_no+1))
            
            if vid is not None:
                vid_overlay = vid[frame_no].copy()
            else:
                vid_overlay = np.zeros(im_shape)

            ax.imshow(vid_overlay, cmap='gray')
            
            for ii in np.arange(len(plot_vid_bbox_tracks_all))[:]:
                bbox_tra_ii = plot_vid_bbox_tracks_all[ii][frame_no] # fetch it at this point in time. 

                if ~np.isnan(bbox_tra_ii[0]):
                    # then there is a valid bounding box. 
                    x1,y1,x2,y2 = bbox_tra_ii 
                    ax.plot( [x1,x2,x2,x1,x1], 
                             [y1,y1,y2,y2,y1], lw=1, color = plot_colors[ii])

            ax.set_xlim([0, im_shape[1]-1])
            ax.set_ylim([im_shape[0]-1, 0])
            plt.axis('off')
            plt.grid('off')
            
            if saveanalysisfolder is not None:
                fig.savefig(os.path.join(saveanalysisfolder_movie_bbox, 'Frame-%s.png' %(str(frame_no).zfill(3))), bbox_inches='tight')
            plt.show()
            plt.close(fig)
        
    # spio.savemat(savematfile, 
    #                 {'boundaries':organoid_boundaries, 
    #                 'initial_seg': seg0_grab,
    #                 'motion_source_map': spixels_B_motion_sources,
    #                 'seg_pts': organoid_segs_pts})

    # return organoid_boundaries, organoid_segs_pts
    return vid_bbox_tracks_prob_all_array, vid_bbox_tracks_all_array, vid_match_checks_all_array, (vid_bbox_tracks_all_lens, vid_bbox_tracks_all_start_time, vid_bbox_tracks_lifetime_ratios)



def segment_organoid_bbox_track(vid,
                                bbox_track_array, # a densified track array 
                                segment_model, # bboxes are already loaded in yolo format.   
                                segment_size, 
                                min_I=None,
                                segment_thresh=.5, # what value to extract contours at after the segmentation 
                                clip_boundary_border=10, # this is to constrain the segmentation contour to the initial detected bounding box. 
                                smooth_boundary_iter = 0,
                                boundary_sample_pts = 200, # 100 doesn't quite capture fully complex geometries. # another way would be considering fourier coefficients. 
                                scale_method='real',
                                invert_intensity=False):

    """ Master function to segment tracked bounding boxes of organoids using a neural network trained segmentation which operates on the bounding box detections. Segmentation are returned both as masks and boundary contour

    Parameters
    ----------
    vid : (n_frames, n_rows, n_cols) array
        the video for which the intensities will be used by the pretrained CNN model to segment organoids from. 
    bbox_track_array : (n_tracks, n_frames, 4) array 
        the compiled regular array of bounding boxes where each bounding boxe is specified in VOC format as (x1,y1,x2,y2). np.nan is used if the bounding box did not exist at a timepoint. 
    segment_model : keras.model instance
        the trained CNN model which takes an image patch of (segment_size[0], segment_size[1]) and predicts an organoid probability map (segment_size[0], segment_size[1]) with intensities [0-1]
    segment_size : 2-tuple
        the resized patch size to run the CNN model at. All bounding box image crops are resized to this size before feeding into the CNN model.   
    min_I : None or uint8
        If specified, the minimum average intensity in [0-255] that the segmented positive binary region from the CNN_model prediction must satisfy for a valid segmentation.  
    segment_thresh : float in [0-1]
        The cutoff threshold for segmenting the probability map from CNN model and converting it to a binary segmentation.  
    clip_boundary_border : int 
        This is a uniform padding to constrain the segmentation boundary within the original bounding box and within the video frame size
    smooth_boundary_iter : int 
        If smooth_boundary_iter>0, the segmentation boundaries are smoothed 
    boundary_sample_pts : int 
        The number of coordinates to describe each segmentation boundary
    scale_method : 'real' or str 
        If scale_method='real', the probability map from the CNN model is resized to the actual bounding box resolution before parsing the binary segmentation - This is slow but potentially more accurate. Alternatively, we do not resize the probability map but multiply the segmented contour coordinates by the appropriate scaling factors. 
    invert_intensity : bool
        If True, invert the intensities of the input video intensities before input to the CNN model. This should be applied for phase contrast videos but not for confocal fluorescent videos.

    Returns 
    -------
    all_bbox_tra_patches : (n_tracks, n_frames, segment_size[0], segment_size[1]) array
        the input bounding box cropped and resized image patch as for CNN organoid segmentation. The patch image is all zeros if the bounding box did not exist at a timepoint. 
    all_bbox_tra_patches_contours : (n _tracks, n_frames, boundary_sample_pts, 2) array
        the compiled regular array of segmented organoid contours with the last 2 coordinates given in xy- geometric coordinate convention instead of image coordinate convention. np.nan is used if the bounding box did not exist or a segmentation was not predicted at a timepoint. 
   
    """
    from tqdm import tqdm 
    import numpy as np 
    import pylab as plt 
    import skimage.transform as sktform 
    from skimage.exposure import equalize_hist, rescale_intensity, adjust_gamma
    from skimage.measure import find_contours
    from scipy.ndimage import binary_fill_holes
    from skimage.filters import gaussian 
    
# =============================================================================
#   Iterate over the tracks. try parsing the boundaries directly on the crop ... and then just use appropriate multiplication to 'resize'
# =============================================================================
    
    vid_m, vid_n = vid[0].shape[:2]
    n_tracks, n_time, n_dim = bbox_track_array.shape

    # float all non-float?     
    all_bbox_tra_patches = np.zeros((n_tracks, n_time, segment_size[0], segment_size[1]), dtype=np.float32) # parse out the predicted bbox segmentations
    all_bbox_tra_patches[:] = np.nan
    all_bbox_tra_patches_contours = np.zeros((n_tracks, n_time, boundary_sample_pts, 2), dtype=np.float32)
    all_bbox_tra_patches_contours[:] = np.nan 


    for tra_ii in np.arange(n_tracks)[:]:
        
        bbox_tra_ii = bbox_track_array[tra_ii] # get the full track.  
        
        # this is here to crop and batch ? 
        bbox_tra_patches = []
        bbox_tra_patches_raw = []
        bbox_tra_patches_times = []
        bbox_tra_patches_sizes = []
        bbox_tra_patches_boxes = []
        
        for bb_frame_tt, bb_frame in enumerate(bbox_tra_ii):
            if ~np.isnan(bb_frame[0]): 
                bb_frame = bb_frame.astype(np.int32) # cast to int for image indexing. 
                x1,y1,x2,y2 = bb_frame.ravel()
            
#                # not an empty box. 
#                print(x1,y1,x2,y2)
                assert(x2>x1) # we need stricter,,.,,,, 
                assert(y2>y1)
            
                bbox_vid_frame = vid[int(bb_frame_tt), y1:y2, x1:x2].copy()
                # m, n = bbox_vid_frame.shape[:2] 
                bbox_tra_patches_sizes.append(np.hstack(bbox_vid_frame.shape))
            
                # factor_m = m/float(segment_size[0]);
                # factor_n = n/float(segment_size[1]);

                # bbox_frame = sktform.resize(bbox_frame, (32,32), preserve_range=True)/255.
                bbox_frame = sktform.resize(bbox_vid_frame, (segment_size[0],segment_size[1]), preserve_range=True)/255. # normalize the intensity 
                bbox_tra_patches_raw.append(bbox_frame)

                # do we do this step before or after? 
                if invert_intensity:
                    # switch to gamma correct?
                    bbox_frame = equalize_hist(bbox_frame.max() - bbox_frame).astype(np.float32)
                    bbox_frame = rescale_intensity(bbox_frame)
                else:
                    bbox_frame = equalize_hist(bbox_frame).astype(np.float32)
#                    bbox_frame = adjust_gamma(bbox_frame, .8).astype(np.float32)
#                    print(bbox_frame.max())
                    bbox_frame = rescale_intensity(bbox_frame)
                
                # model accepts a 3D image path. 
                bbox_tra_patches.append(np.dstack([bbox_frame, 
                                                   bbox_frame,
                                                   bbox_frame]))
                bbox_tra_patches_times.append(int(bb_frame_tt))    
                bbox_tra_patches_boxes.append([x1,y1,x2,y2])        
                
        bbox_tra_patches = np.array(bbox_tra_patches)
        # print(bbox_tra_patches.shape)
        # bbox_tra_patches = bbox_tra_patches.transpose(0,3,1,2) # to be permissible in unet. 
        bbox_tra_patches_raw = np.array(bbox_tra_patches_raw)
        bbox_tra_patches_times = np.hstack(bbox_tra_patches_times)
        bbox_tra_patches_sizes = np.array(bbox_tra_patches_sizes)
        bbox_tra_patches_boxes = np.array(bbox_tra_patches_boxes)
        
        bbox_tra_patches_seg = segment_model.predict(bbox_tra_patches)

        # bbox_tra_patches_seg = bbox_tra_patches_seg[:,0] # flatten the array. 
        bbox_tra_patches_seg = bbox_tra_patches_seg[...,0]
        bbox_tra_patches_seg = np.clip(bbox_tra_patches_seg, 0, 1)

# #        print(bbox_tra_patches_seg.max(), bbox_tra_patches_seg.min())
#         if segment_thresh is None:
# #            print(np.std(bbox_tra_patches_seg))
#             segment_thresh = np.mean(bbox_tra_patches_seg) #+ 1*np.std(bbox_tra_patches_seg)
             
#         print(segment_thresh)
        # else:
        #     seg_thresh = np.mean(bbox_tra_patches_seg)
        # print(seg_thresh)

        # save the probability predictions. 
        all_bbox_tra_patches[tra_ii, bbox_tra_patches_times] = bbox_tra_patches_seg.copy()
        
        """
        Parse the contour lines direct on the downsized images. 
        """
        for bb_ii in np.arange(len(bbox_tra_patches)):
            
            # patch_bb_ii = bbox_tra_patches_raw[bb_ii] # just duplicated.. 
            # patch_bb_ii = bbox_tra_patches[bb_ii][0] # just duplicated.. # equalize_hist version 
            patch_bb_ii = bbox_tra_patches[bb_ii][...,0]

            seg_bb_ii = bbox_tra_patches_seg[bb_ii]
            m, n = bbox_tra_patches_sizes[bb_ii]
            x1,y1,x2,y2 = bbox_tra_patches_boxes[bb_ii]
            bb_ii_time = bbox_tra_patches_times[bb_ii]

            if scale_method == 'real':
                seg_bb_ii_binary = np.zeros((vid_m, vid_n))
                seg_bb_ii_binary[y1:y2,x1:x2] = sktform.resize(seg_bb_ii, (m,n), order=1, preserve_range=True) # linear rescaling
                seg_bb_ii_binary = seg_bb_ii_binary>=segment_thresh; 
            else:
                factor_m = m / float(segment_size[0])
                factor_n = n / float(segment_size[1])

                seg_bb_ii = np.pad(seg_bb_ii, [[15,15], [15,15]]); 
                patch_bb_ii = np.pad(patch_bb_ii, [[15,15], [15,15]]) 
                seg_bb_ii = gaussian(seg_bb_ii, sigma=2, preserve_range=True); # this could changeeeeeeeeeee
                seg_bb_ii = seg_bb_ii / (float(seg_bb_ii.max())+1e-8)
                seg_bb_ii_binary = seg_bb_ii>=segment_thresh; 
                # seg_bb_ii_binary = np.pad(seg_bb_ii_binary, [[10,10], [10,10]]) # this padding does seem to fix a lot of instability. 
            seg_bb_ii_binary = binary_fill_holes(seg_bb_ii_binary)

            # minimum intensity threshold filter -> most useful for fluoresence imaging ..... 
            if min_I is not None:
                # print(seg_bb_ii_binary.shape, patch_bb_ii.shape, np.nanmean(patch_bb_ii[seg_bb_ii_binary>0]))
                if np.nanmean(patch_bb_ii[seg_bb_ii_binary>0]) < min_I/255.:
                    seg_bb_ii_binary = np.zeros_like(seg_bb_ii_binary)

            # contour = find_contours(seg_bb_ii, segment_thresh) # isosurface finding. 
            contour = find_contours(seg_bb_ii_binary, 0)

            if len(contour)>0:
                contour = contour[np.argmax([len(c) for c in contour])]
                # smooth this basic contour with corner cutting

                # plt.figure()
                # plt.imshow(seg_bb_ii_binary)
                # plt.plot(contour[:,1], 
                #          contour[:,0])
                # plt.show()
                if scale_method != 'real':
                    # print('hello')
                    contour[:,0] = contour[:,0] - 15
                    contour[:,1] = contour[:,1] - 15

                if contour[0,0] != contour[-1,0] or contour[0,1] != contour[-1,1]:
                    contour = np.vstack([contour, contour[-1]]) #force close contour.

                contour2 = contour.copy()

                if smooth_boundary_iter > 0:
                    # print('smoothing')
                    # # # regular sampling of boundary using spline fit. # check smoothing.
                    contour = _resample_curve(contour[:,0],contour[:,1], k=1, 
                                             s=smooth_boundary_iter, n_samples=2*boundary_sample_pts) # resample in linear fashion
                    # contour1 = contour.copy()
                    # corner cutting refinement. 

                    # # this function is bad!. 
                    # contour = corner_cutting_smoothing(contour[:,0], 
                    #                                    contour[:,1], 
                    #                                    n_iters=smooth_boundary_iter) # why does this relate to less smooth? 
                    
                    # plt.figure()
                    # plt.plot(contour2[:,0], contour2[:,1], 'k.')
                    # # plt.plot(contour1[:,0], contour1[:,1], 'r-')
                    # plt.plot(contour[:,0], contour[:,1], 'g.', alpha=.5)
                    # plt.show()
                    # print(contour1.shape, contour.shape)

                try:
                    # resample with linear splines:
                    contour = _resample_curve(contour[:,0],
                                             contour[:,1], 
                                                  k=1, 
                                                  # s=1, 
                                                  n_samples=boundary_sample_pts) 
                    # scale
                    if scale_method!='real':
                        # print('yo')
                        contour[:,0] = factor_m * contour[:,0] + y1
                        contour[:,1] = factor_n * contour[:,1] + x1
    
                    # # already square here? 
                    # plt.figure()
                    # plt.imshow(seg_bb_ii_binary)
                    # plt.plot(contour[:,1], 
                    #          contour[:,0])
                    # plt.show()
    
                    # constrain in bounding box and image limits.
                    x_min = np.maximum(0, x1-clip_boundary_border)
                    x_max = np.minimum(vid_n-1, x2+clip_boundary_border) 
                    y_min = np.maximum(0, y1-clip_boundary_border)
                    y_max = np.minimum(vid_m-1, y2+clip_boundary_border) 
    
                    contour[:,0] = np.clip(contour[:,0], y_min, y_max)
                    contour[:,1] = np.clip(contour[:,1], x_min, x_max) 
                
                except:
                    
                    # why failing?
                    # if just a line -> not a contour!. 
                    plt.figure()
                    plt.plot(contour[:,0],contour[:,1])
                    plt.show()
                    # failed. 
                    contour = np.ones((boundary_sample_pts,2))
                    contour[:] = np.nan

                all_bbox_tra_patches_contours[tra_ii, bb_ii_time] = contour.copy() # save this in the array. 

    return all_bbox_tra_patches, all_bbox_tra_patches_contours


def _get_id_cliques(id_list):
    """ given a list of identities, merges the common ids into a cluster/clique
    
    """
    N = len(id_list)
    cliques = [id_list[0]]
    
    for ii in range(1, N):
        
        id_list_ii = id_list[ii]
        
        add_clique = True
        for cc_i, cc in enumerate(cliques):
            if len(np.intersect1d(id_list_ii, cc)) > 0:
                cliques[cc_i] = np.unique(np.hstack([cc, id_list_ii]))
                add_clique = False
                break
            
        if add_clique:
            cliques.append(id_list_ii)
            
    return cliques

def iou_boundary_tracks(tra_1, tra_2):
    """ Compute the frame-by-frame intersection over union between 2 organoid contour tracks. 

    Parameters
    ----------
    tra_1 : (n_frames, n_boundary_pts, 2) array 
        the boundary x,y coordinates at every timepoint for a tracked organoid 1. Any frames for which the organoid did not exist will have coordinates of np.nan
    tra_2 : (n_frames, n_boundary_pts, 2) array 
        the boundary x,y coordinates at every timepoint for a tracked organoid 2. Any frames for which the organoid did not exist will have coordinates of np.nan

    Returns
    -------
    iou_time : (n_frames,) array
        the intersection-over-union, IoU from [0-1] at every timepoint. A value of np.nan for a timepoint will be returned if one or both of the organoids do not exist. 

    """
    import numpy as np 

    n_pts = len(tra_1) 
    iou_time = np.zeros(n_pts)

    for ii in range(n_pts):
        tra_1_ii = tra_1[ii]
        tra_2_ii = tra_2[ii]

        if np.isnan(tra_1_ii[0,0]) or np.isnan(tra_2_ii[0,0]):
            iou_time[ii] = np.nan
        else:
            x1, x2 = np.min(tra_1_ii[...,1]), np.max(tra_1_ii[...,1])
            y1, y2 = np.min(tra_1_ii[...,0]), np.max(tra_1_ii[...,0])
            
            x1_, x2_ = np.min(tra_2_ii[...,1]), np.max(tra_2_ii[...,1])
            y1_, y2_ = np.min(tra_2_ii[...,0]), np.max(tra_2_ii[...,0])
            
            bbox1 = np.hstack([x1,y1,x2,y2])
            bbox2 = np.hstack([x1_,y1_,x2_,y2_])
            
            # print(bbox1, bbox2)
            iou_12 = get_iou(bbox1, bbox2)
            iou_time[ii] = iou_12

    return iou_time

def pairwise_iou_tracks(boundaries_list):
    """ Compute the pairwise mean intersection-over-union between all potential pairs given a list of segmented organoid boundary tracks 

    Parameters
    ----------
    boundaries_list : list of (n_frames, n_boundary_pts, 2) array 
        list of segmented organoid boundary tracks where each organoid boundary track specifies the boundary x,y coordinates at every timepoint for a tracked organoid. Any frames for which the organoid did not exist will have coordinates of np.nan
    
    Returns
    -------
    ind_ids : (N_combinations,) str array
        Flattened array of the indices of all pairwise comparisons represented as a string e.g 'XX_YY' means comparing boundaries_list[XX] with boundaries_list[YY]
    (sim_matrix, shared_time_matrix) : ((N_combinations, N_combinations), (N_combinations, N_combinations)) tuple
        The mean intersection-over-union, IoU in [0-1] computed over the shared frames of each track pair. If the tracks do not share any frames, then the IoU is 0. The second matrix gives the total number of frames shared between a pair.
    
    """
    import itertools
    import numpy as np 
    
    Ns = np.hstack([len(bb) for bb in boundaries_list]) # this is for dissecting IDs.
    
    ind_ids = np.hstack([[str(jj)+'_'+str(ii) for jj in range(Ns[ii])] for ii in range(len(Ns))])
    # stack all of them together. 
    all_boundaries = np.vstack(boundaries_list)
    
    # print(all_boundaries.shape)
    sim_matrix = np.zeros((len(all_boundaries), len(all_boundaries)))
    shared_time_matrix = np.zeros((len(all_boundaries), len(all_boundaries)))
    
    for i, j in itertools.combinations(range(len(all_boundaries)),2):
        
        boundary_i = all_boundaries[i]
        boundary_j = all_boundaries[j]
        
        iou_track = iou_boundary_tracks(boundary_i, boundary_j)
        
        sim_matrix[i,j] = np.nanmean(iou_track)
        sim_matrix[j,i] = np.nanmean(iou_track)
        
        shared_time_matrix[i,j] = ~np.isnan(iou_track).sum()
        shared_time_matrix[j,i] = ~np.isnan(iou_track).sum()
        
    return ind_ids, (sim_matrix, shared_time_matrix)


def calculate_iou_matrix_time(box_arr1, box_arr2, eps=1e-9):
    """ Compute the frame-by-frame intersection over union between 2 sets of bounding box tracks. 

    Parameters
    ----------
    box_arr1 : (n_tracks, n_frames, 4) array 
        the bbox coordinates (voc format, (x1,y1,x2,y2)) at every timepoint for a tracked organoid for every organoid in set 1
    box_arr2 : (n_tracks, n_frames, 4) array 
        the bbox coordinates (voc format, (x1,y1,x2,y2)) at every timepoint for a tracked organoid for every organoid in set 2

    Returns
    -------
    iou : (n_frames, n_tracks, n_tracks) array
        the pairwise intersection-over-union, IoU from [0-1] between every timepoint and between every bounding box track. A value of np.nan for a timepoint will be returned if one or both of the organoids do not exist. 

    """
    import numpy as np 
    x11 = box_arr1[...,0]; y11 = box_arr1[...,1]; x12 = box_arr1[...,2]; y12 = box_arr1[...,3]
    x21 = box_arr2[...,0]; y21 = box_arr2[...,1]; x22 = box_arr2[...,2]; y22 = box_arr2[...,3]
    m,n = x11.shape
    # # n_tracks x n_time. 
    # flip this.
    x11 = x11.T; y11 = y11.T; x12 = x12.T; y12 = y12.T
    x21 = x21.T; y21 = y21.T; x22 = x22.T; y22 = y22.T 

    xA = np.maximum(x11[...,None], x21[:,None,:])
    yA = np.maximum(y11[...,None], y21[:,None,:])
    xB = np.minimum(x12[...,None], x22[:,None,:])
    yB = np.minimum(y12[...,None], y22[:,None,:])
    
    interArea = np.maximum((xB - xA + eps), 0) * np.maximum((yB - yA + eps), 0)
    boxAArea = (x12 - x11 + eps) * (y12 - y11 + eps)
    boxBArea = (x22 - x21 + eps) * (y22 - y21 + eps)

    # iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    iou = interArea / (boxAArea[...,None] + boxBArea[:,None,:] - interArea)
    # # iou = iou.reshape((m,n))
    return iou

def pairwise_iou_tracks_fast(boundaries_list, eps=1e-9, return_bbox=False, avg_func=np.nanmean):
    """ Compute the pairwise mean intersection-over-union between all potential pairs given a list of detected organoid segmented contour tracks. This is an expedited computation based on turning the contours into bounding boxes. 

    Parameters
    ----------
    boundaries_list : list of (n_tracks, n_frames, n_boundary_pts, 2) array 
        list of segmented organoid boundary tracks where each organoid boundary track specifies the boundary x,y coordinates at every timepoint for a tracked organoid. Any frames for which the organoid did not exist will have coordinates of np.nan
    eps : float
        small float to prevent division by zero internally 
    return_bbox : bool
        if True, return the bounding box tracks equivalent of the input organoid boundary contour tracks
    avg_func : function
        the average function to average the IoU over time for each pair of tracks. 
    
    Returns
    -------
    ind_ids : (N_combinations,) str array
        Flattened array of the indices of all pairwise comparisons represented as a string e.g 'XX_YY' means comparing boundaries_list[XX] with boundaries_list[YY]
    (sim_matrix, shared_time_matrix) : ((N_combinations, N_combinations), (N_combinations, N_combinations)) tuple
        The mean intersection-over-union, IoU in [0-1] computed over the shared frames of each track pair. If the tracks do not share any frames, then the IoU is 0. The second matrix gives the total number of frames shared between a pair.
    (all_boundaries_bbox, sim_matrix) : ((N_combinations, n_frames, 4), (N_combinations, N_combinations))
        only returned if return_bbox=True. Return the bounding box tracks equivalent of the input organoid boundary contour tracks
        
    """
    import numpy as np 
    import time 
    
    Ns = np.hstack([len(bb) for bb in boundaries_list]) # this is for dissecting IDs.
    
    ind_ids = np.hstack([[str(jj)+'_'+str(ii) for jj in range(Ns[ii])] for ii in range(len(Ns))])
    # stack all of them together. 
    all_boundaries = np.vstack(boundaries_list) # flatten all. 
    
    # turn the boundaries into bbox. 
    all_boundaries_bbox_xmin = np.min(all_boundaries[...,1], axis=-1)
    all_boundaries_bbox_xmax = np.max(all_boundaries[...,1], axis=-1)
    all_boundaries_bbox_ymin = np.min(all_boundaries[...,0], axis=-1)
    all_boundaries_bbox_ymax = np.max(all_boundaries[...,0], axis=-1)
    
    all_boundaries_bbox = np.concatenate([all_boundaries_bbox_xmin[...,None], 
                                          all_boundaries_bbox_ymin[...,None], 
                                          all_boundaries_bbox_xmax[...,None], 
                                          all_boundaries_bbox_ymax[...,None]], axis=-1)
    
    all_sim_matrix = calculate_iou_matrix_time(all_boundaries_bbox, 
                                               all_boundaries_bbox, 
                                            eps=eps)
    
    sim_matrix = avg_func(all_sim_matrix, axis=0)
    shared_time_matrix = np.sum(~np.isnan(all_sim_matrix), axis=0)
     
    if return_bbox:
        return ind_ids, (sim_matrix, shared_time_matrix), (all_boundaries_bbox, sim_matrix)
    else:
        return ind_ids, (sim_matrix, shared_time_matrix)

#    # 1. apply iou calculations pairwise between tracks to score overlap across channels
#    ind_ids, (sim_matrix, shared_time_matrix) = pairwise_iou_tracks(org_tracks_list) # this concatenates all the tracks etc together, resolving all inter-, intra- overlaps
#
#
#    print(ind_ids)
#
#    sim_matrix_ = sim_matrix.copy()
#    sim_matrix_[np.isnan(sim_matrix)] = 0 # replace any nan values which will not be useful. 
#
#    # 2. detect overlaps and cliques (clusters of tracks that correspond to one dominant organoid)
#    tracks_overlap = np.where(sim_matrix_ >= track_overlap_thresh)
        

def objectness_score_tracks(track, img_score, mean_func=np.nanmean, timepoint=0): 
    """ Given the objectness score over the image for the desired timepoint, this function will average this over the area covered by the given organoid contour track and get the mean score. 

    Parameters
    ----------
    track : (n_frames, n_boundary_pts, 2) array
        the organoid contour track with xy coordinate convention
    img_score : (n_rows, n_cols) array
        the objectness score as an image. score reflects a probability of a pixel being an object e.g. the probability map produced from a CNN segmenter. 
    mean_func : Python function
        the averaging function to average the score over the area encapsulated by a contour
    timepoint : int 
        desired timepoint to compute the mean objectness score over the bounded area of the organoid contour. Default is the first timepoint of the video. 
    
    Returns
    -------
    score : float
        the mean objectness score of the organoid contour

    """
    # give an image and a timepoint which the image corresponds to, to get the objectness score. default uses the first timepoint of the video.  
    from skimage.draw import polygon 
    
    if ~np.isnan(track[timepoint][0,0]):
        coords = track[timepoint].copy()
        rr,cc = polygon(coords[:,0], 
                        coords[:,1], shape=img_score.shape)
        
        score = mean_func(img_score[rr,cc])
    else:
        score = 0
    
    return score

# of the two versions, this is more useful because it ensures more temporal continuity.
def objectness_score_tracks_time(track, vid_score, mean_func=np.nanmean): 
    """ Given the objectness score over the full video, this function will average this over the area covered by the given organoid contour track and get the mean object score for every timepoint and return the mean score over the evaluated frames to give an objectnesss score for the full track.

    Parameters
    ----------
    track : (n_frames, n_boundary_pts, 2) array
        the organoid contour track with xy coordinate convention
    vid_score : (n_frames, n_rows, n_cols) array
        the objectness score at every timepoint as a video. score reflects a probability of a pixel being an object in a given frame e.g. the probability map produced from a CNN segmenter. 
    mean_func : Python function
        the averaging function to average the score over the area encapsulated by a contour
    
    Returns
    -------
    score : float
        the mean objectness score of the organoid contour for the track 

    """
    # give an image and a timepoint which the image corresponds to, to get the objectness score. default uses the first timepoint of the video.  
    from skimage.draw import polygon 
    
    img_shape = vid_score[0].shape
    nframes = len(track)
    scores = []
    
    for frame in range(nframes):
        coords = track[frame].copy()
        if ~np.isnan(coords[0,0]):
            rr,cc = polygon(coords[:,0], 
                            coords[:,1], shape=img_shape)
            score = mean_func(vid_score[frame][rr,cc])
            scores.append(score)
        else:
            scores.append(0)
    
    score = np.nanmean(scores)
    
    return score


def track_iou_time(track, use_bbox=True): 
    """ measures the intersection-over-union (IoU) between consecutive tracked organoid contours within a single track

    Parameters
    ----------
    track : (n_frames, n_boundary_pts, 2) array
        the organoid contour track with xy coordinate convention
    use_bbox : bool 
        only True is currently implemented. If True, the bounding box of the organoid contour is used to evaluate IoU

    Returns
    -------
    first_diff_iou : (n_frames-1,) array
        the timeseries of the IoU between the segmented and tracked organoid contours between successive timepoints.

    """    
    nframes = len(track)
    first_diff_iou = []
    
    for frame_no in np.arange(nframes-1):
        
        tra_1 = track[frame_no]
        tra_2 = track[frame_no+1]
        
        if np.isnan(tra_1[0][0]) or np.isnan(tra_2[0][0]):
            first_diff_iou.append(np.nan)
        else:
            
            if use_bbox:
                x1, x2 = np.min(tra_1[...,1]), np.max(tra_1[...,1])
                y1, y2 = np.min(tra_1[...,0]), np.max(tra_1[...,0])
                
                x1_, x2_ = np.min(tra_2[...,1]), np.max(tra_2[...,1])
                y1_, y2_ = np.min(tra_2[...,0]), np.max(tra_2[...,0])
                
                bbox1 = np.hstack([x1,y1,x2,y2])
                bbox2 = np.hstack([x1_,y1_,x2_,y2_])
                
                # print(bbox1, bbox2)
                iou_12 = get_iou(bbox1, bbox2)
                first_diff_iou.append(iou_12)
    
    first_diff_iou = np.hstack(first_diff_iou)
    
    return first_diff_iou
    

def track_set_iou_time(track_set, use_bbox=True): 
    """ measures the intersection-over-union (IoU) between consecutive tracked organoid contours for every single track in a given set of tracks

    Parameters
    ----------
    track_set : list of (n_frames, n_boundary_pts, 2) array
        list or array of organoid contour tracks with xy coordinate convention
    use_bbox : bool 
        only True is currently implemented. If True, the bounding box of the organoid contour is used to evaluate IoU

    Returns
    -------
    track_set_diff_iou : N_tracks of (n_frames-1,) array
        array of the timeseries of the IoU between the segmented and tracked organoid contours between successive timepoints for every track 

    """
    track_set_diff_iou = np.array([track_iou_time(track, use_bbox=use_bbox) for track in track_set])
    # number organoid tracks x n_time 
    
    return track_set_diff_iou
    

def detect_iou_breaks_trackset(trackset, 
                               use_bbox=True, 
                               iou_thresh=.5,
                               prepadded_tracks=False, 
                               pad_track_len=100,
                               split_tracks=False,
                               min_split_track_len=1):
    """ Find all timepoints for each track at which IoU drops below a given threshold. These timepoints constitute temporal 'breaks' which may signal a switch in the identity of organoids being tracked. If specified, the breakpoints are implemented and new tracks are generated from the resulting tracklets.  
    
    Parameters
    ----------
    trackset : (n_tracks, n_frames, n_contour_pts, 2) array or a list of list of (n_contour_pts, 2) arrays
        an array of a number of organoid contour tracks. 
    use_bbox : bool
        if True, IoU is computed on  
    iou_thresh : float
        IoU cutoff, below which we call a temporal breakpoint. 
    prepadded_tracks : bool
        if True, input trackset have already been appropriately padded with np.nan to be a regular (n_tracks, n_frames, n_contour_pts, 2) array and does not need to be padded. If False, padding will be applied internally.
    pad_track_len : int 
        this specifies the number of expected boundary points to describe the contour.  
    split_tracks : bool
        if True, split the organoid contour tracks and retain on the track up to the break time. If False, we retain all segments created from breaks and form new tracks each with guaranteed minimal frame-to-frame IoU consistency if these segments also satisfy the minimal length requirement
    min_split_track_len : int 
        the minimal number of frames for a valid track

    Returns
    -------
    breaks : list of (n_breaks,) array
        List is n_tracks long. Each entry lists all the break times identified for each track
    trackset_breaks : (n_tracks_new, n_frames, n_contour_pts, 2) array
        New array of the tracks where each track has a frame-to-frame IoU difference >= iou_thresh. 

    """
    iou_diffs = track_set_iou_time(trackset, use_bbox=True)
    iou_diffs_valid = iou_diffs < iou_thresh
    
    # find the break points for each trackset. 
    breaks = []
    
    for iou_diffs_valid_ii in iou_diffs_valid:
        breaks.append(np.arange(len(iou_diffs_valid_ii))[iou_diffs_valid_ii]+1)
        
    # apply the break points to the trackset, using the first. 
    if prepadded_tracks:
        trackset_breaks = trackset.copy()
    else:
        trackset_breaks = pad_tracks( trackset, boundary_samples=pad_track_len)

    if split_tracks == False:
        for ii in range(len(breaks)):
            break_ii = breaks[ii]
            if len(break_ii) > 0:
                break_time = break_ii[0]
                trackset_breaks[ii, break_time:] = np.nan # delete all future.
    else:
        # we generate a new trackset
        trackset_new = []
        n_time, n_pts, n_dim = trackset_breaks.shape[1:]

        # iterate
        for ii in range(len(breaks)):
            break_ii = breaks[ii]
            break_ii = np.hstack([0, break_ii, n_time])
            for jj in np.arange(len(break_ii)-1):
                trackset_new_ii = np.zeros((n_time, n_pts, n_dim))
                trackset_new_ii[:] = np.nan
                trackset_new_ii[break_ii[jj]:break_ii[jj+1]] = trackset_breaks[ii, break_ii[jj]:break_ii[jj+1]].copy()
                
                N_nonnan = np.sum(~np.isnan(trackset_new_ii[:,0,0]))

                if N_nonnan >= min_split_track_len:
                    trackset_new.append(trackset_new_ii)

        trackset_new = np.array(trackset_new)
        # print(trackset_new.shape)
        trackset_breaks = trackset_new.copy()

    return breaks, trackset_breaks
  

# this requires the boundaries at each timepoint to be actually aligned ... which we may be able to check? 
# should we check this?, may better allow us to infer the movement directly from the boundaries? 
def smoothness_score_tracks(track, mean_func=np.nanmean, second_order=True): 
    """ Compute a score of the smoothness of the tracked organoid boundaries for a track 

    Parameters
    ----------
    track : (n_frames, n_boundary_pts, 2) array
        the organoid contour track with xy coordinate convention
    mean_func : Python function 
        the averaging function to average the score over the area encapsulated by a contour
    second_order : bool
        If True, uses the second order differences as a measure of smoothness. If False, the 1st order differences or gradient is used as a measure of smoothness. 

    Returns
    -------
    score : float
        the mean smoothness score of the organoid contour for the track 

    """
    # get the second order track differences
    # from skimage.draw import polygon 
    import numpy as np 
    
    first_diff = np.gradient(track, axis=0) # find the gradient over time. 
    
    if second_order:
        second_diff = np.gradient(first_diff, axis=0)
        second_diff_norm = np.linalg.norm(second_diff, axis =-1)
    else:
        second_diff_norm = np.linalg.norm(first_diff, axis =-1)
    # second_diff_norm = np.nanmean(second_diff_norm, axis=1) # obtain the mean over the boundaries., should we use median?
    
    # print(mean_func(second_diff_norm, axis = 0).shape)
    # score = -mean_func(second_diff_norm, axis = 0)
    score = -np.nanmax(mean_func(second_diff_norm, axis = 0))
    return score


# the following bypasses the need for point alignment along the tracks? by using area. 
def smoothness_score_tracks_iou(track, mean_func=np.nanmean, second_order=False, use_bbox=True): 
    """ Compute a score of the smoothness of the tracked organoid boundaries for a track by measuring the abruptness of changes to the frame-to-frame IoU

    Parameters
    ----------
    track : (n_frames, n_boundary_pts, 2) array
        the organoid contour track with xy coordinate convention
    mean_func : Python function 
        the averaging function to average the score over the area encapsulated by a contour
    second_order : bool
        If True, uses the second order IoU differences as a measure of smoothness. If False, the 1st order IoU differences or gradient is used as a measure of smoothness. 
    use_bbox : bool 
        If True, uses the bounding box of the contour to estimate IoU. False is not implemented currently. 

    Returns
    -------
    score : float
        the mean smoothness score of the organoid contour for the track 

    """
    # get the second order track differences
    # from skimage.draw import polygon 
    import numpy as np 
        
    first_diff_iou = track_iou_time(track, use_bbox=use_bbox)
        
    if second_order:
        second_diff_norm = np.gradient(first_diff_iou)
    else:
        second_diff_norm = first_diff_iou.copy()
    # second_diff_norm = np.nanmean(second_diff_norm, axis=1) # obtain the mean over the boundaries., should we use median?

    if second_order:
        score = -mean_func(second_diff_norm) # this should be maxed for iou.
    else:
        score = mean_func(second_diff_norm)
    return score


def nan_stability_score_tracks(track): 
    """ measures the fraction of frames for which the tracked object existed for i.e. had coordinates that were not np.nan 
    
    Parameters
    ----------
    track : (n_frames, n_boundary_pts, 2) array 
        the organoid contour track with xy coordinate convention
    
    Returns
    -------
    score : float
        the # of frames for which the organoid track had coordinates that were not np.nan divided by the total frames in the video i.e. n_frames.

    """     
    len_track = float(len(track))
    valid_track_times = np.sum(~np.isnan(track[:,0,0]))

    score = valid_track_times / len_track 
    
    return score

# preprocessing for tracks (where should i place this function?)
def pad_tracks( boundary, boundary_samples=100):
    """ pad array of segmented organoid tracks to a regular array. For timepoints where a contour does not exist a np.nan is used.    
    
    Parameters
    ----------
    boundary : (n_organoids, n_frames) array of (boundary_samples,2) organoid contours. 
        The output of segment organoids is such that the boundary_samples=1 for no segmentation i.e. the contour is ([np.nan,np.nan])
    boundary_samples : int
        The number of boundary points to describe each segmentation boundary as specified in the original CNN organoid boundary segmentation 
    
    Returns
    -------
    boundary_ : (n_organoids, n_frames, boundary_samples, 2) array
        The regular padded array of segmented organoid contour tracks. For timepoints where a contour does not exist a np.nan is used.    

    """
    import numpy as np 
    
    n_org, n_time = boundary.shape[:2]
    boundary_ = np.zeros((n_org, n_time, boundary_samples,2))
    boundary_[:] = np.nan
    
    for ii in range(n_org):
        for tt in range(n_time):
            tra = boundary[ii,tt]
            if np.isnan(tra[0,0]):
                pass
            else:
                boundary_[ii,tt] = tra.copy()
                
    return boundary_


def filter_nan_tracks( boundary ):
    """ Removes all tracks where for the entire duration has only nan values.

    Parameters
    ----------
    boundary : (n_organoids, n_frames, boundary_samples, 2) array
        an array of a number of organoid contour tracks specified with xy coordinate convention. 

    Returns 
    -------
    boundaries_out : (n_organoids_out, n_frames, boundary_samples, 2) array
        an array of a number of organoid contour tracks specified with xy coordinate convention with any organoid tracks that only had np.nan values removed. 

    """
    import numpy as np 
    
    boundaries_out = []
    
    for ii in range(len(boundary)):
        
        tra = boundary[ii]
        tra_len = len(tra)
        n_nans = 0
        
        for tra_tt in tra:
            if np.isnan(tra_tt[0,0]):
                n_nans+=1
        if n_nans < tra_len:
            boundaries_out.append(tra) # append the whole track
            
    if len(boundaries_out) > 0:
        boundaries_out = np.array(boundaries_out)
        
    return boundaries_out


def non_stable_track_suppression_filter(obj_vid, 
                                            org_tracks_list, 
                                            track_overlap_thresh=0.25, 
                                            weight_nan=1., 
                                            weight_smooth=0.1, 
                                            max_obj_frames=5,
                                            obj_mean_func=np.nanmean,
                                            smoothness_mean_func=np.nanmean,
                                            fast_comp=True,
                                            debug_viz=False):
    """ Implements a variation of non-maximum suppression on entire organoid tracks, but using measures of stability. The idea is that several tracks may be tracking the same physical organoid. In these cases we may only want to keep the most 'stable' / confident track and delete the rest. Stability here is defined as a combination of the length of the track (nan stability), smoothness of the track and the objectness of the track
    
    Parameters
    ----------
    obj_vid : (n_frames, n_rows, n_cols, 1 or 3) array
        input grayscale or RGB video
    org_tracks_list : list of (n_organoids, n_frames, boundary_samples, 2) array
        list of an array of organoid contour tracks. e.g. list of 3 for all organoid tracksets generated from all 3 RGB channels.  
    track_overlap_thresh : float [0-1]
        The cutoff threshold whereby if the mean IoU between a pair of organoid tracks is greater than this, we consider them duplicated and only one organoid track should be kept.  
    weight_nan : float [0-1]
        The weighting of the nan stability score - the proportion of the total video, for which the track exists
    weight_smooth : float [0-1]
        The weighting of the smoothness score - how jumpy is the tracked organoid contour. We assume the more stable, the more confident was the tracking and reflects higher SNR in the original video.
    max_obj_frames : int
        The first XXX number of frames to evaluate the objectness score for the track. This assumes as a minimum we want to maximally track the initial organoids in the field of view in the first frame as these were used to setup the timelapse acquisition experiment.   
    obj_mean_func : Python function 
        The mean function used to get the temporally averaged objectness score per track. 
    smoothness_mean_func : Python function 
        The mean function used to get the temporally averaged smoothness score per track. 
    fast_comp : bool
        If True, uses the fast method for computing IoU overlap between organoid contour tracks. The fast method uses the bounding box of the contours to compute IoU in a vectorized manner.  
    debug_viz : bool
        If True, plots some debugging plots where for each overlapping clique of tracks, it is shown which is kept.  

    Returns 
    -------
    org_tracks_list_out : list of (n_organoids_new, n_frames, boundary_samples, 2) array
        list of an array of deduplicated organoid contour tracks in the same format as the input e.g. list of 3 for all organoid tracksets generated from all 3 RGB channels.  

    """
    if debug_viz:
        import pylab as plt 
    
    # 1. apply iou calculations pairwise between tracks to score overlap across channels
    if fast_comp:
        ind_ids, (sim_matrix, shared_time_matrix) = pairwise_iou_tracks_fast(org_tracks_list)
        # detrend the diagonals.(which is self connections)
        sim_matrix = sim_matrix - np.diag(np.diag(sim_matrix)) 
    else:
        ind_ids, (sim_matrix, shared_time_matrix) = pairwise_iou_tracks(org_tracks_list) # this concatenates all the tracks etc together, resolving all inter-, intra- overlaps

    sim_matrix_ = sim_matrix.copy()
    sim_matrix_[np.isnan(sim_matrix)] = 0 # replace any nan values which will not be useful. 

    # 2. detect overlaps and cliques (clusters of tracks that correspond to one dominant organoid)
    tracks_overlap = np.where(sim_matrix_ >= track_overlap_thresh)

#    print(tracks_overlap)
    if len(tracks_overlap[0]) > 0:

        # 2b. if there is any evidence of overlap! we collect this all together.  
        overlap_positive_inds = np.vstack(tracks_overlap).T
        overlap_positive_inds = np.sort(overlap_positive_inds, axis=1)
#        print(overlap_positive_inds)
        # overlap_positive_inds = np.unique(overlap_positive_inds, axis=0) #remove duplicate rows.
        overlap_positive_inds = _unique_pts(overlap_positive_inds)
        
        # merge these indices to identify unique cliques ... -> as those will likely be trying to track the same organoid. 
        cliq_ids = _get_id_cliques(overlap_positive_inds) 
  
        # 3. clique resolution -> determining which organoid is actually being tracked, and which track offers best tracking performance on average from all candidates in the clique. 
        assigned_cliq_track_ids = [] # stores which of the tracks we should use from the overlapped channels.
    
        for cc in cliq_ids[:]:

            # iterate, use objectness score provided by the input vid, to figure out which organoid is being tracked. 
            ind_ids_cc = ind_ids[cc] # what are the possible ids here. 

            obj_score_cc = []
            tra_stable_scores_cc = []
            
            # in the order of organoid id and channel. 
            if debug_viz:
                import seaborn as sns

                ch_colors = sns.color_palette('Set1', len(org_tracks_list))

                fig, ax = plt.subplots()
                ax.imshow(obj_vid[0], alpha=0.5) # just visualise the first frame is enough.
            
            for ind_ids_ccc in ind_ids_cc:
                org_id, org_ch = ind_ids_ccc.split('_')
                org_id = int(org_id)
                org_ch = int(org_ch)
                
                boundary_org = org_tracks_list[org_ch][org_id]

                # this is the problem? 
                # objectness score for deciding the dominant organoid. 
                obj_score = objectness_score_tracks_time(boundary_org[:max_obj_frames], 
                                                              obj_vid[:max_obj_frames,...,org_ch], 
                                                              mean_func=obj_mean_func)
                obj_score_cc.append(obj_score)
                
                # stability score which is weighted on 2 factors. to determine which track. 
                nan_score = nan_stability_score_tracks(boundary_org)
                
                # this should be a minimisation ..... ! 
                # smooth_score = smoothness_score_tracks(boundary_org, 
                #                                       mean_func=np.nanmean)
                smooth_score = smoothness_score_tracks_iou(boundary_org, 
                                                           mean_func=smoothness_mean_func)
                
                tra_stable_scores_cc.append(weight_nan*nan_score+weight_smooth*smooth_score)

            if debug_viz:
                ax.set_title( 'org: %s, stable: %s'  %(ind_ids_cc[np.argmax(obj_score_cc)], 
                                                ind_ids_cc[np.argmax(tra_stable_scores_cc)]))
                plt.show()
            
            # stack all the scores.     
            obj_score_cc = np.hstack(obj_score_cc)
            tra_stable_scores_cc = np.hstack(tra_stable_scores_cc)

            # decide on the organoid and track (argmax)
            cliq_org_id_keep = ind_ids_cc[np.argmax(obj_score_cc)]
            cliq_track_id_keep = ind_ids_cc[np.argmax(tra_stable_scores_cc)]
            
            # save this out for processing. 
            assigned_cliq_track_ids.append([cliq_org_id_keep, cliq_track_id_keep])

        # 4. new org_tracks_list production based on the filtered information. 
        org_tracks_list_out = []

        for list_ii in range(len(org_tracks_list)):

            org_tracks_list_ii = org_tracks_list[list_ii]
            org_tracks_list_ii_out = []

            for org_ii in range(len(org_tracks_list_ii)):
                tra_int_id = str(org_ii)+'_'+str(list_ii) # create the string id lookup. 
                
                include_track = True

                for cliq_ii in range(len(cliq_ids)):
                    ind_ids_cc = ind_ids[cliq_ids[cliq_ii]] # gets the clique members in string form -> is this organoid part of a clique. 
                    
                    if tra_int_id in ind_ids_cc:
                        include_track = False # do not automatically include.  

                        # test is this the dominant organoid in the clique. 
                        cliq_organoid_assign, cliq_organoid_assign_track = assigned_cliq_track_ids[cliq_ii] # get the assignment information of the clique. 
                        
                        if tra_int_id == cliq_organoid_assign:
                            # if this is the dominant organoid then we add the designated track. 
                            org_id_tra_assign, org_ch_tra_assign = cliq_organoid_assign_track.split('_')
                            org_id_tra_assign = int(org_id_tra_assign); org_ch_tra_assign=int(org_ch_tra_assign)
                            
                            org_tracks_list_ii_out.append(org_tracks_list[org_ch_tra_assign][org_id_tra_assign])

                        # do nothing otherwise -> exclude this organoid basically. 
                
                if include_track: 
                    # directly include. 
                    org_tracks_list_ii_out.append(org_tracks_list_ii[org_ii])

            if len(org_tracks_list_ii_out) > 0:
                # stack the tracks. 
                org_tracks_list_ii_out = np.array(org_tracks_list_ii_out)
            
            org_tracks_list_out.append(org_tracks_list_ii_out)
    else:
        org_tracks_list_out = list(org_tracks_list)

    # cleaned tracks, in the same input format as a list of numpy tracks for each channel. 
    return org_tracks_list_out

# def register_and_align_organoid_boundary_tracks(): # this we have got .... (save as .pkl?) how else to save?
    
# can we further clean up the organoid tracks? 
# def smooth_and_impute_organoid_boundary_tracks(): # parametric model? trained LSTM / ODE autoencoder
def temporal_smooth_tracks( track_set, 
                            method='ma', 
                            win_size=None, 
                            prepadded_tracks=False,
                            pad_track_len=100, 
                            ma_avg_func=np.nanmean,
                            ma_pad_mode='edge',
                            spline_smoothing=5):
    """ Temporally smooths an array of organoid contour tracks using one of two methods, 
        a) 'ma' - moving window average with a given averaging function (base model)
        b) 'spline' - pads data and conduct univariate data smoothing with B-splines.

    Parameters
    ----------
    trackset : (n_tracks, n_frames, n_contour_pts, 2) array or a list of list of (n_contour_pts, 2) arrays
        an array of a number of organoid contour tracks. 
    method : 'ma' or 'spline'
        the smoothing method. 'ma' - moving average with sliding windows or 'spline' - using Univariate polynomial order smoothing. 
    win_size : None or odd integer  
        the temporal sliding window size for smoothing tracks. The larger the window, the greater the smoothing 
    prepadded_tracks : bool 
        if True, the input trackset has been prepadded and is a regular numpy array. If False, the input trackset is padded to be regular before processing. 
    pad_track_len : int 
        this specifies the number of expected boundary points to describe the contour. This is equal to the length of boundary points using the organoid boundary segmentation. 
    ma_avg_func : Python function 
        the averaging function used to obtain the smoothed value over the sliding window when method='ma' 
    ma_pad_mode : str
        padding mode for border cases. One of the possible available in np.pad function. 
    spline_smoothing : int 
        the extent of smoothing, higher is more smoothing, used when method='spline' 

    Returns
    -------
    track_set_smooth_out : (n_tracks, n_frames, n_contour_pts, 2) array 
        an array of now temporally smoothed organoid contour tracks. 

    """
    if prepadded_tracks:
        track_set_ = track_set.copy()
    else:
        print('padding')
        track_set_ = pad_tracks( track_set, boundary_samples=pad_track_len)

    # detect all the nan_pos positions. 
    nan_pos = np.isnan(track_set_) # why is this empty? 
    tt_max = nan_pos.shape[1]
    # get the start end positions of nan in the full stack # smoothing will go into the nan, # this is no no... 
    nan_pos_start_end_times = []
    for ii in np.arange(nan_pos.shape[0]):
        non_vals = nan_pos[ii,:,0,0].copy();  
        index_vals = np.arange(len(non_vals)) # index over time.  
        tt_start = index_vals[non_vals==0][0] # first position # not quite
        tt_end = index_vals[non_vals==0][-1] # last position
        nan_pos_start_end_times.append([tt_start, tt_end])        
    nan_pos_start_end_times = np.vstack(nan_pos_start_end_times)
    
    # 1. find centroids at each time frame. 
    track_set_working = track_set_.copy(); 
    track_set_working[nan_pos==1] = 0 # first set to 0. # these apply to entire curves. (not missing points on the curve. )

    # these don't need to be normalized for 'moving average method. 
    centroids_yx = np.nanmean(track_set_working, axis=-2) # average over boundary contours.
    centroids_yx = np.tile(centroids_yx[:,:,None,:], reps=(1,1,pad_track_len,1))

    # 2. decentroidify 
    track_set_working = track_set_working - centroids_yx # these should be the same size now. 

    # 3. apply smoothing method. 

    track_set_smooth_out = np.zeros_like(track_set_)

    if method == 'ma':
        if win_size is None:
            win_size = track_set_working.shape[1]//2

        track_set_working_pad = np.pad(track_set_working, [[0,0], [win_size//2, win_size//2], [0,0], [0,0]], mode=ma_pad_mode) # padding in time.... 
        
        for frame in range(track_set_working.shape[1])[:]:
            track_set_smooth_out[:,frame] = ma_avg_func(track_set_working_pad[:,frame:frame+win_size], axis=1)

        track_set_smooth_out = track_set_smooth_out + centroids_yx
        
    if method == 'spline':
        
        from scipy.interpolate import UnivariateSpline
        if win_size is None:
            win_size = track_set_working.shape[1]//2
            
        # pad
        track_set_working_pad = np.pad(track_set_working, [[0,0], [win_size//2, win_size//2], [0,0], [0,0]], mode=ma_pad_mode)
        
        x_fit =  np.arange(track_set_working_pad.shape[1]) # just the time 
        # separate fits for x,y, coordinates. 
        for org_ii in range(len(track_set_working_pad)):
            for boundary_line_ii in np.arange(track_set_working_pad.shape[2]):
                # print(track_set_working_pad[:,:,boundary_line_ii,1].shape, len(x_fit))
                # print(track_set_working_pad[:,:,boundary_line_ii,1]).shape
                
                spl_x = UnivariateSpline(x_fit, track_set_working_pad[org_ii,:,boundary_line_ii,1])
                spl_y = UnivariateSpline(x_fit, track_set_working_pad[org_ii,:,boundary_line_ii,0])
                spl_x.set_smoothing_factor(np.nanvar(track_set_working_pad[org_ii,:,boundary_line_ii,1])*spline_smoothing)
                spl_y.set_smoothing_factor(np.nanvar(track_set_working_pad[org_ii,:,boundary_line_ii,0])*spline_smoothing)
                
                x_pred = spl_x(x_fit)[win_size//2:-win_size//2+1]
                y_pred = spl_y(x_fit)[win_size//2:-win_size//2+1]
                
                track_set_smooth_out[org_ii,:,boundary_line_ii,1] = x_pred.copy()
                track_set_smooth_out[org_ii,:,boundary_line_ii,0] = y_pred.copy()
            
        track_set_smooth_out = track_set_smooth_out + centroids_yx

    for ii in np.arange(len(nan_pos_start_end_times)):
        # enforce non-smoothing values in nan regions !. nan is nan!. 
        ind_start, ind_end = nan_pos_start_end_times[ii]
        if ind_start>0:
            # print(ii, ind_start, ind_end, 'erase_start')    
            track_set_smooth_out[ii,:ind_start] = np.nan 
        if ind_end < tt_max-1:
            # print(ii, ind_start, ind_end, 'erase_end')   
            track_set_smooth_out[ii, ind_end+1:] = np.nan
        
    # return this out now. 
    return track_set_smooth_out

    
# this is to find if any organoids are hitting the boundary. 
def detect_image_border_organoid_tracks( track_set, 
                                         img_shape, 
                                         border_pad = 5, 
                                         out_percent=.1,
                                         prepadded_tracks=False,
                                         pad_track_len=100,
                                         apply_mask=True):
    """ Detect and boolean flag the timepoints for each tracked organoid when the organoid boundary lies too close to the border of the video frame. These should be removed when computing metrics and conducting analysis as these organoids are likely not in full view and therefore artificially cropped.
    
    Parameters
    ----------
    track_set : (n_tracks, n_frames, n_contour_pts, 2) array or a list of list of (n_contour_pts, 2) arrays
        an array of a number of organoid contour tracks. 
    img_shape : (n_rows, n_cols) 2-tuple
        tuple specifying the size of the video frame 
    border_pad : int 
        the uniform padding from the edges of the image that constitutes the border
    out_percent : float [0-1]
        the fraction percentage of the total number of contour points within the border mask greater than which the organoid contour is flagged as being a 'border organoid' 
    prepadded_tracks : bool 
        if True, the input trackset has been prepadded and is a regular numpy array. If False, the input trackset is padded to be regular before processing. 
    pad_track_len : int
        this specifies the number of expected boundary points to describe the contour. This is equal to the length of boundary points using the organoid boundary segmentation. 
    apply_mask : bool
        If True, return as an optional second argument the new track_set, an (n_tracks, n_frames, n_contour_pts, 2) array that have all flagged border organoid contours masked out i.e. coordinates set to np.nan

    Returns
    -------
    border_orgs_time : (n_tracks, n_frames) array
        boolean array specifying which of the organoids are flagged as significantly being on the global image border 
    track_set_copy : (n_tracks, n_frames, n_contour_pts, 2) array
        an array of a number of organoid contour tracks with all flagged border organoid contours masked out i.e. coordinates set to np.nan

    """
    if prepadded_tracks:
        track_set_ = track_set.copy()
    else:
        track_set_ = pad_tracks( track_set, boundary_samples=pad_track_len)

    # replace all nan's to be 0's
    track_set_copy = track_set_.copy()
    track_set_[np.isnan(track_set_)] = 0

    border_x_min = border_pad
    border_x_max = img_shape[1] - border_pad
    border_y_min = border_pad
    border_y_max = img_shape[0] - border_pad

    border_positions_bool = np.logical_or(np.logical_or(track_set_[...,0] <= border_y_min, 
                                                        track_set_[...,0] >= border_y_max), 
                                          np.logical_or(track_set_[...,1] <= border_x_min, 
                                                        track_set_[...,1] >= border_x_max))

    border_positions_frac = np.sum(border_positions_bool, axis=-1) / float(border_positions_bool.shape[-1]) 

    border_orgs_time = border_positions_frac > out_percent # this is a bool

    if apply_mask:
        track_set_copy[border_orgs_time] = np.nan # set as all nans. 
        return border_orgs_time, track_set_copy
    else:
        return border_orgs_time


