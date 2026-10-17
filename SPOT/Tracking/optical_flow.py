# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:31:11 2014

@author: felix
@email: felixzhou1@gmail.com

This script and all the code within is licensed under the Ludwig License.

see README.md for details on usage.
    
"""

def get_default_optical_flow_params():
    
    optical_flow_params = dict(pyr_scale=0.5, levels=5, winsize=15, iterations=5, poly_n=3, poly_sigma=1.2, flags=0)
    
    return optical_flow_params


def Eval_dense_optic_flow(prev, present, params):
    r""" Computes the optical flow using Farnebacks Method

    Parameters
    ----------
    prev : numpy array
        previous frame, m x n image
    present :  numpy array
        current frame, m x n image
    params : Python dict
        a dict object to pass all algorithm parameters. Fields are the same as that in the opencv documentation, https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html. Our recommended starting values:
                
            * params['pyr_scale'] = 0.5
            * params['levels'] = 3
            * params['winsize'] = 15
            * params['iterations'] = 3
            * params['poly_n'] = 5
            * params['poly_sigma'] = 1.2
            * params['flags'] = 0
        
    Returns
    -------
    flow : (n_frames-1, n_rows, n_cols, 2)
        the xy displacement field between frames, prev and present such that :math:`\mathrm{prev}(y,x) = \mathrm{next}(y+\mathrm{flow}(y,x)[1], x+\mathrm{flow}(y,x)[0])` where (x,y) is the cartesian coordinates of the image.
    """
    
    import numpy as np 
    import warnings
    import cv2

    # Check version of opencv installed, if not 3.0.0 then issue alert.
#    if '3.0.0' in cv2.__version__ or '3.1.0' in cv2.__version__:
        # Make the image pixels into floats.
    prev = prev.astype(np.float32)
    present = present.astype(np.float32)

    if int(cv2.__version__.split('.')[0]) >= 3:
        flow = cv2.calcOpticalFlowFarneback(prev, present, None, params['pyr_scale'], params['levels'], params['winsize'], params['iterations'], params['poly_n'], params['poly_sigma'], params['flags']) 
    if cv2.__version__.split('.')[0] == '2':
        flow = cv2.calcOpticalFlowFarneback(prev, present, pyr_scale=params['pyr_scale'], levels=params['levels'], winsize=params['winsize'], iterations=params['iterations'], poly_n=params['poly_n'], poly_sigma=params['poly_sigma'], flags=params['flags']) 
#    print(flow.shape)
    return flow


def get_farneback_flow_params():

    r""" Generates default optical flow params for OpenCV dense Farneback optical flow

    Parameters
    ----------
    None : 

    Returns
    -------
    flow_params : Python dict
        a dict object with all the algorithm parameters to pass to SAM.Tracking.optical_flow.Eval_dense_optic_flow. Fields are the same as that in the opencv documentation, https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html. 
    
    """

    flow_params = dict(pyr_scale=0.5, 
                        levels=5, 
                        winsize=15, 
                        iterations=5, 
                        poly_n=3, 
                        poly_sigma=1.2, 
                        flags=0)

    return flow_params


def extract_vid_optflow(vid, flow_params, rescale_intensity=False):

    """ Given a grayscale movie as a numpy array, compute the optical flow using farneback with the given params. Returns the frame-to-frame optical flow field for the entire video.
    
    Parameters
    ----------
    vid : numpy array
        n_frames x n_rows x n_cols numpy array 
    flow params : python dict
        OpenCV opticalflowfarneback parameters (python dict) c.f. http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html 
    rescale_intensity : numpy bool
        if True, intensities of pairwise frames have intensities rescaled prior to computing flow. 

    Returns
    -------
    flow : nump array 
        (n_frames-1, n_rows x n_cols x 2) numpy array of the optical flow
    """

    import numpy as np 
    import skimage.exposure as skexposure
    from tqdm import tqdm # for a loading bar. 

    vid_flow = []

    for frame in tqdm(np.arange(len(vid)-1)):
        # frame0 = equalize_hist(rescale_intensity(vid_resize[frame]))*255.
        # frame1 = equalize_hist(rescale_intensity(vid_resize[frame+1]))*255.
        frame0 = vid[frame].copy()
        frame1 = vid[frame+1].copy()

        if rescale_intensity:
            frame0 = skexposure.rescale_intensity(frame0) # pretty much the same. so we don't bother. 
            frame1 = skexposure.rescale_intensity(frame1) 
        flow01 = Eval_dense_optic_flow(frame0, frame1, 
                                       params=flow_params)
        vid_flow.append(flow01)
        
    vid_flow = np.array(vid_flow).astype(np.float32) # to save some space. 
    
    return vid_flow
