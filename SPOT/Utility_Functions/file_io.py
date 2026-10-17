# -*- coding: utf-8 -*-
##########################################################################
# Created on Thu Jun 29 22:49:32 2017
#
# @author: felix
# @email: felixzhou1@gmail.com
#
# This script and all the code within is licensed under the Ludwig License.
#
# see README.md for details on usage.
###########################################################################
import numpy as np 


# read a single frame from a multi-page .tif file.
def read_PIL_frame(tiffile, frame):

    """ Use pillow library to read select .tif/.TIF files. (single frame)
    
    Parameters
    ----------
    tiffile : str
        input .tif file to read, can be multipage .tif (string)
    frame : int
        desired frame number given as C-style 0-indexing (int)
    
    Returns
    -------
    img : numpy array
        an image as a numpy array either (n_rows x n_cols) for grayscale or (n_rows x n_cols x 3) for RGB
    
    """
    from PIL import Image
    import numpy as np

    img = Image.open(tiffile)
    img.seek(frame)
    img = np.array(img)

    return img
    
def read_multiimg_PIL(tiffile):
    
    """ Use pillow library to read multipage .tif/.TIF files. 

    Parameters
    ----------
    tiffile : str
        input .tif file to read, can be multipage .tif (string)

    Returns
    -------
    imgs : numpy array
        either (n_frames x n_rows x n_cols) for grayscale or (n_frames x n_rows x n_cols x 3) for RGB

    """
    from PIL import Image
    import numpy as np

    img = Image.open(tiffile)

    imgs = []
    read = True

    frame = 0

    while read:
        try:
            img.seek(frame) # select this as the image
            imgs.append(np.array(img)[None,:,:])
            frame += 1
        except EOFError:
            # Not enough frames in img
            break

    imgs = np.concatenate(imgs, axis=0)

    return imgs

def read_multiimg_stack(tiffile, return_img=True):    
    """ Use the tifffile.py library through Scikit-Image to read multipage bioformat files such as .tif/.lsm files.

    Parameters
    ----------
    tiffile : str
        input .tif/.lsm file to read, can be multipage .tif (string)
    return_img : bool
        boolean True/False to specify if the image should be read as a numpy array or just the object be returned.

    Returns
    -------
    img_object : Python object
        A read image object containing the attributes: pages, series and micromanager_metadata.
    imgs : numpy array (only if return_img=True)
        an (n_frames x n_slices x n_channels x n_rows x n_cols) image.

    """
    from tifffile import TiffFile

    im_object = TiffFile(tiffile)

    if return_img:
        imgs = im_object.asarray()
        return im_object, imgs
    else:
        return im_object
    

def mkdir(directory):
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


def save_multipage_tiff(np_array, savename):
    
    """ save numpy array of images as a multipage .tiff file. Can also do this now through skimage.io.imsave 
    
    Parameters
    ----------
    np_array : numpy array
        (n_frames, n_rows, n_cols, n_channels) image 
    savename : str
        filepath to save the output .tif stack. 
    
    Returns
    -------
    void function
    
    """    
    from tifffile import imwrite
    import numpy as np 
    
    if np_array.max() < 1.1:
        imwrite(savename, np.uint8(255*np_array))
    else:
        imwrite(savename, np.uint8(np_array))
    
    return [] 
    

# more general save function that enables saving in imageJ format.
def save_imagej_vid(img, imgname):

    """ save a 2D+time grayscale in ImageJ format. Can also do this now through skimage.io.imsave 

    Parameters
    ----------
    img : numpy array 
        (n_frames, n_rows, n_cols) image str 
    imgname : str 
        filepath to save the output .tif stack. 

    Returns
    -------
    void function

    """
    
    from tifffile import imwrite

    imwrite(imgname, img[:,None,None,:,:,None], imagej=True)
        
    return []


def read_video_cv2(avifile, to_RGB=True):
    """ read a general movie file using opencv 

    Parameters
    ----------
    avifile : str 
        filepath to the input movie file

    Returns
    -------
    vid_array : numpy array 
        (n_frames, n_rows, n_cols, 1 or 3) numpy array  

    """
    import cv2
    import numpy as np 
    
    vidcap = cv2.VideoCapture(avifile)
    # success,image = vidcap.read()
    
    vid_array = []
    
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        if success:
            vid_array.append(image)
        count += 1
        
    vid_array = np.array(vid_array)
    
    if to_RGB:
        vid_array = vid_array[...,::-1].copy()
      
    return vid_array


def fetch_channel_boxes(imgfile, boxfolder, ch_no, ending='.avi', subfolder=''):
    """

    Parameters
    ----------
    imgfile : filepath
        file path of the video to load precomputed bounding box files for
    boxfolder : str 
        the full folder path location containing the individual bounding box .txt files
    ch_no : int
        which of the channel number in the video of imgfile to load bounding boxes. first channel is 0, second is 1 etc. 
    ending : str
        file extension of the imgfile. defaults to '.avi'
    subfolder : str
        optionally specifies subfolder within the default os.path.join(boxfolder, basename)

    """
    import os 
    import glob 
    basename = os.path.split(imgfile)[-1].split(ending)[0]
    # print(basename)
    bboxfolder = os.path.join(boxfolder, basename, subfolder, 'Channel-%s' %(str(ch_no+1).zfill(2)))
    # print(bboxfolder)
    boxfiles = glob.glob(os.path.join(bboxfolder, '*.txt'))

    return np.sort(boxfiles)


def read_bbox_from_file(bboxfile):
    """ reads a list of bounding box files, returning as a list of bounding box numpy arrays

    Parameters
    ----------
    bboxfile : filepath
        bounding boxes for a single image written in YOLO format as (label, score, x, y, w, h)
    
    Returns
    -------
    bboxes : numpy array 
        (n_boxes, 5) array with box in YOLO format as (label, score, x, y, w, h)

    """
    f = open(bboxfile,'r')
    bboxes = []
    
    for line in f:
        line = line.strip()
        label, score, box_x, box_y, box_w, box_h = line.split()
        
        bboxes.append([label, score, int(box_x), int(box_y), int(box_w), int(box_h)])

    if len(bboxes) > 0: 
        bboxes = np.array(bboxes)
        
    return bboxes


def read_detected_bboxes_from_file_for_tracking(bboxfiles):
    """ reads a list of bounding box files detected by our pretrained YOLOv3 algorithm, returning as a list of bounding box numpy arrays sorted by time
    
    Parameters
    ----------
    bboxfiles : list of filepaths
        list of bounding box files, each for a single image written in YOLO format as (label, score, x, y, w, h)
    
    Returns
    -------
    boxes : list of [frame_no, box]
        frame_no from which the box was predicted from and each box is a (n_boxes, 5) array with box in YOLO format as (label, score, x, y, w, h)
    
    """
    import os 
    
    boxes = []
    
    for f in bboxfiles:    
        box = read_bbox_from_file(f)
#        frame_no = int(((f.split('/')[-1]).split('_')[-1]).split('.txt')[0])
        frame_no = int(os.path.split(f)[-1].split('_')[2])
        boxes.append([frame_no, box])
        
    boxes = sorted(boxes, key=lambda x: x[0])
        
    return boxes


def write_pickle(savefile, a):
    r""" write pickle object given an array or dictionary

    Parameters
    ----------
    savefile : str
        filepath to write, this must be within an already created directory 
    a : array or dict
        array or dictionary of data

    Returns
    -------
    None
        
    """
    import pickle
    with open(savefile, 'wb') as handle:
        pickle.dump(a, handle)
    return []
    

def read_pickle(savefile):
    r""" read pickle saved file

    Parameters
    ----------
    savefile : str
        file to read

    Returns
    -------
    None
        
    """
    import pickle
    with open(savefile, 'rb') as handle:
        b = pickle.load(handle)
        
    return b



def read_bboxes_from_file(bboxfiles):
    """ reads a list of bounding box files with the frame number in the filename, returning as a frame number sorted list of bounding box numpy arrays

    Parameters
    ----------
    bboxfiles : list of filepaths
        list of all bounding box text files with bounding boxes written in YOLO format as (label, score, x, y, w, h). Frame number is included inside the filepath. 
    
    Returns
    -------
    boxes : list of numpy arrays
        frame number ascending sorted list of (n_boxes, 5) arrays with box in YOLO format as (label, score, x, y, w, h)
    
    """
    import os 
    boxes = []
    
    for f in bboxfiles:    
        box = read_bbox_from_file(f)
#        frame_no = int(((f.split('/')[-1]).split('_')[-1]).split('.txt')[0])
        frame_no = int(os.path.split(f)[-1].split('_')[2])
        boxes.append([frame_no, box])
        
    boxes = sorted(boxes, key=lambda x: x[0])
        
    return boxes


def load_bbox_frame_voc( img, bbox_obj_frame):
    """ Given a yolo format bounding boxes, convert to voc format given a target image. At the same time clips the bounding boxes to the size of the target image.
    
    Parameters
    ----------
    img : numpy array
        (n_rows, n_cols, X) image
    bbox_obj_frame : [frame_no, bboxes]

    """
    import numpy as np
    nrows, ncols = img.shape[:2]
    
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



def compile_boundaries_arrays(expt, 
                              org_expt_table, 
                              boundaries, 
                              channels_sep=',', 
                               patients_sep='+', 
                               conditions_sep='+', 
                              genetics_sep='+',
                              grayscale=False):
    
    r""" For the extracted object boundaries of a given video and the global metadata for the dataset this video was part of, this function uses the global metadata to annotate the object boundaries.
    
    Parameters
    ----------
    expt : str
        filename of the video. This should match with that specified in the global metadata as this acts like the key to pull out the correct information row. 
    org_expt_table : pd.DataFrame
        Pandas dataframe of the global metadata of the generated standard template for all the videos of a dataset
    boundaries : array
        a list of object boundaries, (n_objects, n_timepoints) for each image channel
    channels_sep : str
        videos may be multichannel. the Img_Channels column of the metadata table specifies objects from which channel is considered for analysis. More than one specified channel should be indicated with a delimiter e.g. 1,2 for first, second image channel respectively
    patients_sep : str
        videos may have objects from different patients/sources for multichannels. the Patients column of the metadata table is used to specify the source for each video channel and this should be indicated with a delimiter. the delimiter defaults to '+'
    conditions_sep : str
        videos may have objects subject to different treatments for multichannels. the Conditions column of the metadata table is used to specify the treatment condition for each video channel and this should be indicated with a delimiter. the delimiter defaults to '+'
    genetics_sep : str
        videos may have objects of different genotype for multichannels. the Genetics column of the metadata table is used to specify the genotype for each video channel and this should be indicated with a delimiter. the delimiter defaults to '+'
    grayscale : bool
        specify if video is single channel. 
    
    Returns
    -------
    (filenames_all, conditions_all, genetics_all, patients_all, img_channel_no_all, org_id_all, frame_no_all, pixel_resolution_all, frame_duration_all, total_vid_frames) : tuple
        the metadata of the video expanded to every measured object and all its timepoints. 
    boundaries_all_export : (n_object_instances x n_boundary_pts x 2)
        the flattened array of detected object boundaries.
    
    """
    import numpy as np 
    
    # get the corresponding info for the given experiment
    select_meta = np.arange(len(org_expt_table))[org_expt_table['Filename'].values == expt][0]
    meta_info = org_expt_table.loc[select_meta]
    
    ## parse channels
    ## parse the genotypes and channel information.
    #expt_channels = meta_info['Img_Channel']
    ## check it is integer or str
    #expt_channels_result = isinstance(expt_channels, np.int64)
    
    if not grayscale:
        # parse channels
        # parse the genotypes and channel information.
        expt_channels = meta_info['Img_Channel']
        # check it is integer or str
        expt_channels_result = isinstance(expt_channels, np.int64)
        
        # print(expt_channels_result)
        
        if expt_channels_result:
            expt_channels = np.hstack([int(expt_channels)])
        else:
            if channels_sep in expt_channels:
                expt_channels = expt_channels.split(channels_sep)
                expt_channels = np.hstack([int(ch) for ch in expt_channels])
            else:
                expt_channels = np.hstack([int(expt_channels)])
    else:
        expt_channels = np.hstack([1])
     
    # parse genetics 
    genotype_channels = meta_info['Genetics']
    try:
        if np.isnan(genotype_channels)==False:
            if genetics_sep in genotype_channels:
                genotype_channels = genotype_channels.split(genetics_sep)
                genotype_channels = np.hstack([ge.strip() for ge in genotype_channels])
            else:
                genotype_channels = np.hstack([genotype_channels.strip()])
        else:
            genotype_channels = np.hstack([np.nan])
    except:
        if genotype_channels:
            if genetics_sep in genotype_channels:
                genotype_channels = genotype_channels.split(genetics_sep)
                genotype_channels = np.hstack([ge.strip() for ge in genotype_channels])
            else:
                genotype_channels = np.hstack([genotype_channels.strip()])
        else:
            genotype_channels = np.hstack([np.nan])
        
    # parse conditions 
    condition_channels = meta_info['Conditions']
    try:
        if np.isnan(condition_channels)==False:
            if conditions_sep in condition_channels:
                condition_channels = condition_channels.split(conditions_sep)
                condition_channels = np.hstack([cond.strip() for cond in condition_channels])
            else:
                condition_channels = np.hstack([condition_channels.strip()])
        else:
            condition_channels = np.hstack([np.nan])
    except:
        if condition_channels:
            if conditions_sep in condition_channels:
                condition_channels = condition_channels.split(conditions_sep)
                condition_channels = np.hstack([cond.strip() for cond in condition_channels])
            else:
                condition_channels = np.hstack([condition_channels.strip()])
        else:
            condition_channels = np.hstack([np.nan])
        
        
    # parse patients
    patients_channels = meta_info['Patients']
    try:
        if np.isnan(patients_channels)==False:
            if patients_sep in patients_channels:
                patients_channels = patients_channels.split(patients_sep)
                patients_channels = np.hstack([patient.strip() for patient in patients_channels])
            else:
                patients_channels = np.hstack([patients_channels.strip()])
        else:
            patients_channels = np.hstack([np.nan])
    except:
        if patients_channels:
            if patients_sep in patients_channels:
                patients_channels = patients_channels.split(patients_sep)
                patients_channels = np.hstack([patient.strip() for patient in patients_channels])
            else:
                patients_channels = np.hstack([patients_channels.strip()])
        else:
            patients_channels = np.hstack([np.nan])
        
    pixel_res = meta_info['pixel_resolution[um]']
    time_res = meta_info['time_resolution[h]']

    total_vid_frames = meta_info['n_frames'] # not really used per say .... 
    
    """
    below is the compilation process. 
    """
#    which for some reason is super slow
#    # the below headers in the csv export out instead as arrays. 
#    headers = np.hstack(['Filename',
#                         'Condition',
#                         'Genetics',
#                         'Img_Channel_No', 
#                         'Org_ID', 
#                         'Frame_No', 
#                         'pixel_resolution[um]',
#                         'Frame_Duration[h]',
#                         'Total_Video_Frame_No',
    filenames_all = []
    conditions_all = []
    genetics_all = []
    patients_all = []
    img_channel_no_all = []
    org_id_all = []
    frame_no_all = []
    pixel_resolution_all = []
    frame_duration_all = []
    total_vid_frames_all = []
    
    boundaries_all_export = [] 
    
    if grayscale:
        boundaries = [boundaries]
    
    for channel_ii, boundaries_channel in enumerate(boundaries):
        # -> iterate over channels.
#        print(boundaries_channel.shape)
        
        if (channel_ii+1) in list(expt_channels):
            # print('hello', expt_channels)
            gene = genotype_channels[expt_channels==channel_ii+1][0]
            condition = condition_channels[expt_channels==channel_ii+1][0]
            patient = patients_channels[expt_channels==channel_ii+1][0]
            
            # -> iterate over organoids 
            # do nothing otherwise.
            for bb_ii, boundaries_channel_ii in enumerate(boundaries_channel):
                # -> iterate over timepoints. 
#                print(boundaries_channel_ii.shape)
                # bb_ii is the org_id. 
                for bb_ii_tt, boundaries_ii_tt in enumerate(boundaries_channel_ii):
#                    print(len(metrics_names))
                    # if features are nan do nothing
#                    print(boundaries_ii_tt.shape) # this can also be empty!.
                    # these are now coordinates.... 
#                    if len(boundaries_ii_tt) > 0:
                    if ~np.isnan(boundaries_ii_tt[0][0]):
                        
#                        print(channel_ii)
                        filenames_all.append(expt)
                        patients_all.append(patient)
                        # conditions_all.append(meta_info['Condition'].strip())
                        conditions_all.append(condition)
                        genetics_all.append(gene)
                        img_channel_no_all.append(channel_ii+1)
                        org_id_all.append(bb_ii+1)
                        frame_no_all.append(bb_ii_tt+1)
                        pixel_resolution_all.append(pixel_res)
                        frame_duration_all.append(time_res)
                        total_vid_frames_all.append(total_vid_frames) # this is causing the issue
                        
                        boundaries_all_export.append(boundaries_ii_tt)
                        
    
    filenames_all = np.hstack(filenames_all)
    patients_all = np.hstack(patients_all)
    conditions_all = np.hstack(conditions_all)
    genetics_all = np.hstack(genetics_all)
    img_channel_no_all = np.hstack(img_channel_no_all)
    org_id_all = np.hstack(org_id_all)
    frame_no_all = np.hstack(frame_no_all)
    pixel_resolution_all = np.hstack(pixel_resolution_all)
    frame_duration_all = np.hstack(frame_duration_all)
    total_vid_frames_all = np.hstack(total_vid_frames_all)
#    
##    print(filenames_all.shape)
##    print(len(boundaries_all_export))
#    # the creation of the array kills? 
    boundaries_all_export = np.array(boundaries_all_export)
#    print(boundaries_all_export.shape)
    
    return (filenames_all, conditions_all, genetics_all, patients_all, 
            img_channel_no_all, org_id_all, frame_no_all, 
            pixel_resolution_all, frame_duration_all, total_vid_frames) , boundaries_all_export


def construct_metrics_table_csv(expt, 
                                org_expt_table, 
                                metrics, 
                                metricslabels,
                                channels_sep=',', 
                                 patients_sep='+', 
                                 conditions_sep='+', 
                                genetics_sep='+',
                                grayscale=False):
    r""" For the SAM phenome of a given video and the global metadata for the dataset this video was part of, this function produces a table output using the global metadata to annotate this video, so it can be compiled into one large table.
    
    Parameters
    ----------
    expt : str
        filename of the video. This should match with that specified in the global metadata as this acts like the key to pull out the correct information row. 
    org_expt_table : pd.DataFrame
        Pandas dataframe of the global metadata of the generated standard template for all the videos of a dataset
    metrics : (n_objects, n_frames, n_features) array
        the features computed for every tracked organoid in the given video
    metricslabels : 
        the names of the features computed for every tracked organoid in the given video
    channels_sep : str
        videos may be multichannel. the Img_Channels column of the metadata table specifies objects from which channel is considered for analysis. More than one specified channel should be indicated with a delimiter e.g. 1,2 for first, second image channel respectively
    patients_sep : str
        videos may have objects from different patients/sources for multichannels. the Patients column of the metadata table is used to specify the source for each video channel and this should be indicated with a delimiter. the delimiter defaults to '+'
    conditions_sep : str
        videos may have objects subject to different treatments for multichannels. the Conditions column of the metadata table is used to specify the treatment condition for each video channel and this should be indicated with a delimiter. the delimiter defaults to '+'
    genetics_sep : str
        videos may have objects of different genotype for multichannels. the Genetics column of the metadata table is used to specify the genotype for each video channel and this should be indicated with a delimiter. the delimiter defaults to '+'
    grayscale : bool
        specify if video is single channel. 
    
    Returns
    -------
    all_data : pd.DataFrame
        the merged metadata and SAM features table
    
    """
    import pandas as pd 
    import numpy as np 
    
    # get the corresponding info for the given experiment
    select_meta = np.arange(len(org_expt_table))[org_expt_table['Filename'].values == expt][0]
    meta_info = org_expt_table.loc[select_meta]
    
    if not grayscale:
        # parse the genotypes and channel information.
        expt_channels = meta_info['Img_Channel']
        # check it is integer or str
        expt_channels_result = isinstance(expt_channels, np.int64)
        
        if expt_channels_result:
            expt_channels = np.hstack([int(expt_channels)])
        else:
            if channels_sep in expt_channels:
                expt_channels = expt_channels.split(channels_sep)
                expt_channels = np.hstack([int(ch) for ch in expt_channels])
            else:
                expt_channels = np.hstack([int(expt_channels)])
    else:
        expt_channels = np.hstack([1])
        
    # parse genetics 
    genotype_channels = meta_info['Genetics']
    try:
        if np.isnan(genotype_channels)==False:
            if genetics_sep in genotype_channels:
                genotype_channels = genotype_channels.split(genetics_sep)
                genotype_channels = np.hstack([ge.strip() for ge in genotype_channels])
            else:
                genotype_channels = np.hstack([genotype_channels.strip()])
        else:
            genotype_channels = np.hstack([np.nan])
    except:
        if genotype_channels:
            if genetics_sep in genotype_channels:
                genotype_channels = genotype_channels.split(genetics_sep)
                genotype_channels = np.hstack([ge.strip() for ge in genotype_channels])
            else:
                genotype_channels = np.hstack([genotype_channels.strip()])
        else:
            genotype_channels = np.hstack([np.nan])
        
    # parse conditions 
    condition_channels = meta_info['Conditions']
    try:
        if np.isnan(condition_channels)==False:
            if conditions_sep in condition_channels:
                condition_channels = condition_channels.split(conditions_sep)
                condition_channels = np.hstack([cond.strip() for cond in condition_channels])
            else:
                condition_channels = np.hstack([condition_channels.strip()])
        else:
            condition_channels = np.hstack([np.nan])
    except:
        if condition_channels:
            if conditions_sep in condition_channels:
                condition_channels = condition_channels.split(conditions_sep)
                condition_channels = np.hstack([cond.strip() for cond in condition_channels])
            else:
                condition_channels = np.hstack([condition_channels.strip()])
        else:
            condition_channels = np.hstack([np.nan])
        
        
    # parse patients
    patients_channels = meta_info['Patients']
    try:
        if np.isnan(patients_channels)==False:
            if patients_sep in patients_channels:
                patients_channels = patients_channels.split(patients_sep)
                patients_channels = np.hstack([patient.strip() for patient in patients_channels])
            else:
                patients_channels = np.hstack([patients_channels.strip()])
        else:
            patients_channels = np.hstack([np.nan])
    except:
        if patients_channels:
            if patients_sep in patients_channels:
                patients_channels = patients_channels.split(patients_sep)
                patients_channels = np.hstack([patient.strip() for patient in patients_channels])
            else:
                patients_channels = np.hstack([patients_channels.strip()])
        else:
            patients_channels = np.hstack([np.nan])
        
    pixel_res = meta_info['pixel_resolution[um]']
    time_res = meta_info['time_resolution[h]']

    total_vid_frames = meta_info['n_frames'] # not really used per say .... 
    
    
    all_data = []
    metrics_names = [name.strip() for name in metricslabels]
    
    if grayscale:

        metrics = [metrics]
    
    for channel_ii, metrics_channel in enumerate(metrics):
        
        if (channel_ii+1) in list(expt_channels):
            gene = genotype_channels[expt_channels==channel_ii+1][0]
            condition = condition_channels[expt_channels==channel_ii+1][0]
            patient = patients_channels[expt_channels==channel_ii+1][0]
            
            # do nothing otherwise.
            for bb_ii, metrics_ii in enumerate(metrics_channel):
                
                # bb_ii is the org_id. 
                for bb_ii_tt, metrics_ii_tt in enumerate(metrics_ii):
#                    print(len(metrics_names))
                    # if features are nan do nothing
                    if ~np.isnan(metrics_ii_tt[0]):
                        data = np.hstack([ expt, # filename
                                           condition, #experimental condition in the well.
                                           gene, # want the genetics of this orgnaoid
                                           patient,
                                           channel_ii+1, #want the image channel.
                                           bb_ii+1,  # want the oranoid id.
                                           bb_ii_tt+1, # want the frame_no.
                                           pixel_res,
                                           time_res, 
                                           total_vid_frames, 
                                           metrics_ii_tt, 
                                            ])
                        
                        all_data.append(data)
                        
    all_data = np.array(all_data, dtype=object)
    headers = np.hstack(['Filename',
                         'Condition',
                         'Genetics',
                         'Patients',
                         'Img_Channel_No', 
                         'Org_ID', 
                         'Frame_No', 
                         'pixel_resolution[um]',
                         'Frame_Duration[h]',
                         'Total_Video_Frame_No',
                         metrics_names])
                           
    all_data = pd.DataFrame(all_data, 
                            index=None,
                            columns=headers)
    
    return all_data


def load_SPOT_features_files(analysisfolders, 
                             expts,
                             boundaryfile_suffix='_boundaries_smooth_final_timesample.mat',
                             boundarykey='boundaries',
                             patchfile_suffix='_boundaries_smooth_final_img_patches_sampled.mat',
                             patches_key='patches_all',
                             patches_size_key='patch_size',
                             shapefeatsfile_suffix='_shape_features_timelapse.csv',
                             appearfeatsfile_suffix='_image_features_timelapse.csv', 
                             motionfeatsfile_suffix='_motion_features_timelapse.csv',
                             read_chunksize=2000,
                             test_feature_name='mean_speed_global_flow'):
    
    r""" Compiles the output of SPOT with a .csv of shape, appearance and motion features table and dictionary-like .mat or .pkl of object crops and object boundaries into one dictionary of metainformation and merged features table. 
    Takes into account if you only have one or two of shape, appearance or motion features computed, instead of the full set. 
    
    Parameters
    ----------
    analysisfolders : list of folderpaths
        top-level folders to analyse. Each corresponds to a video collection 
    expts : list of str
        list of the first part name to the feature files. The feature file assumes the naming convention of expt+suffix. 
    boundaryfile_suffix : str
        this is the second half of the compiled boundaries filename after the expt prefix part. Must be specified. 
    boundarykey : str
        key to access the boundaries from the dictionary
    patchfile_suffix : str
        this is the second half of the compiled cropped patches of individual objects filename after the expt prefix part. Must be specified. 
    patches_key : str
        key to access the object patches
    patches_size_key : str
        key to access the real object patch image dimensions 
    shapefeatsfile_suffix : str
        this is the second half of the compiled shape features table after the expt prefix part. Format is .csv. Optionally specified. Set as None if not using. 
    appearfeatsfile_suffix : str
        this is the second half of the compiled appearance features table after the expt prefix part. Format is .csv. Optionally specified. Set as None if not using. 
    motionfeatsfile_suffix : str
        this is the second half of the compiled motion features table after the expt prefix part. Format is .csv. Optionally specified. Set as None if not using. 
    read_chunksize : int
        the number of entries in the tables to read at once, this makes the pandas table reading more memory-efficient and faster.
    test_feature_name : str
        the name of the feature to use to test whether to keep an object instance. This feature should be computable and well-defined for every valid organoid instance. Default is 'mean_speed_global_flow'

    Returns
    -------
    all_object_feats : np.array
        Merged data only array of all combined numerical features
    metadict : dict 
        dictionary of the combined metadata for convenience. 
    
    """
    import numpy as np 
    import pandas as pd 
    import os 
    import scipy.io as spio
    
    object_ids = [] # keep a unique id number of all organoids. 
    all_object_uniq_row_ids = []
    all_object_feats = []
    all_object_boundaries = [] 
    all_object_TP = []
    all_object_Ts = []
    all_object_genetics = []
    all_object_patients = []
    all_object_condition = []
    all_object_expt = []
    
    all_object_patches = [] 
    all_object_patches_sizes = [] 

    for expt_ii in np.arange(len(expts)):

        expt_name = expts[expt_ii]
        server_path = analysisfolders[expt_ii]
        
        """
        get the boundary file (this must exist)
        """
        boundaryfile = os.path.join(server_path, 
                                        expt_name+boundaryfile_suffix)
        # get the boundary file so we can plot some contours ... after looking it back up. 
        if boundaryfile_suffix.endswith('.mat'):
            # print(spio.loadmat(boundaryfile).keys())
            boundaries = spio.loadmat(boundaryfile)[boundarykey]
        elif boundaryfile_suffix.endswith('.pkl') or boundaryfile_suffix.endswith('.pickle'):
            boundaries = read_pickle(boundaryfile)[boundarykey]
        else:
            print('invalid boundaries file format ...')
        
        """
        get the patches file (this must exist)
        """
        patchfile = os.path.join(server_path, 
                                 expt_name + patchfile_suffix)
        
        if patchfile_suffix.endswith('.mat'):
            # print(spio.loadmat(patchfile).keys())
            patches = spio.loadmat(patchfile)[patches_key]
            patches_sizes = spio.loadmat(patchfile)[patches_size_key]
        elif patchfile_suffix.endswith('.pkl') or patchfile_suffix.endswith('.pickle'):
            patches = read_pickle(patchfile)[patches_key]
            patches_sizes = read_pickle(patchfile)[patches_size_key]
        else:
            print('invalid patches file format ...')
           
        """
        SAM features parsing. 
        """
        SAM_feats = []
        SAM_feats_type = []
        
        """
        get the shape metrics file if it exists
        """
        if shapefeatsfile_suffix is not None:
            # get the shape file. 
            shapefeatsfile = os.path.join(server_path, 
                                          expt_name+shapefeatsfile_suffix)
    
            shapefeats = []
            for chunk in pd.read_csv(shapefeatsfile, chunksize=read_chunksize):
                shapefeats.append(chunk)
            shapefeats = pd.concat(shapefeats, ignore_index=True) # ignore index. 
            
            """
            create a unique id. to do merging.  
            """
            uniq_obj_id = np.hstack([shapefeats['Filename'][ii] + '_' + str(shapefeats['Img_Channel_No'][ii]) + '_' + str(shapefeats['Org_ID'][ii]) + '_'+ str(shapefeats['Frame_No'][ii]) for ii in range(len(shapefeats['Org_ID']))])  
            # organoid_ids.append(uniq_org_id)
            shapefeats['uniq_obj_id'] = uniq_obj_id
            shapefeats['uniq_row_id'] = np.arange(len(uniq_obj_id))
            
            SAM_feats.append(shapefeats)
            SAM_feats_type.append('S')
            
            print('---------------------------')
            print('loaded shape features table')
            print(shapefeats.shape)
            print('---------------------------')
            
            del shapefeats
            
        """
        get the appearance metrics file if it exists
        """
        if appearfeatsfile_suffix is not None:
            appearancefile = os.path.join(server_path, 
                                      expt_name+appearfeatsfile_suffix)
            
            appearancefeats = []
            for chunk in pd.read_csv(appearancefile, chunksize=read_chunksize):
                appearancefeats.append(chunk)
            appearancefeats = pd.concat(appearancefeats, ignore_index=True) # ignore index. 
            
            uniq_obj_id = np.hstack([appearancefeats['Filename'][ii] + '_' + str(appearancefeats['Img_Channel_No'][ii]) + '_' + str(appearancefeats['Org_ID'][ii]) + '_' + str(appearancefeats['Frame_No'][ii]) for ii in range(len(appearancefeats['Org_ID']))])  
            appearancefeats['uniq_obj_id'] = uniq_obj_id
            appearancefeats = appearancefeats.loc[:,'mean_intensity':]
            
            SAM_feats.append(appearancefeats)
            SAM_feats_type.append('A')
            
            print('---------------------------')
            print('loaded appearance features table')
            print(appearancefeats.shape)
            print('---------------------------')
            del appearancefeats
            
        """
        get the motion metrics file if it exists
        """
        if motionfeatsfile_suffix is not None:
            motionfile = os.path.join(server_path, 
                                      expt_name + motionfeatsfile_suffix) # we somehow miss the metadata? 
            motionfeats = []
            for chunk in pd.read_csv(motionfile, chunksize=read_chunksize):
                motionfeats.append(chunk)
            motionfeats = pd.concat(motionfeats, ignore_index=True) # ignore index. 
            
            uniq_obj_id = np.hstack([motionfeats['Filename'][ii] + '_' + str(motionfeats['Img_Channel_No'][ii]) + '_' + str(motionfeats['Org_ID'][ii]) + '_' + str(motionfeats['Frame_No'][ii]) for ii in range(len(motionfeats['Org_ID']))])  
            motionfeats['uniq_obj_id'] = uniq_obj_id
            motionfeats = motionfeats.loc[:,'mean_speed_global':] # get rid of the meta information here. 
            
            SAM_feats.append(motionfeats)
            SAM_feats_type.append('M')
            
            print('---------------------------')
            print('loaded motion features table')
            print(motionfeats.shape)
            print('---------------------------')
            
            del motionfeats
            
        """
        perform table merge based on the unique_id if there is more than one SAM features table. 
        """
        merge_table = []
        merge_table_type = None
        
        for ttt in np.arange(len(SAM_feats)):
            if ttt == 0:
                merge_table = SAM_feats[ttt]
                merge_table_type = SAM_feats_type[ttt]
            else:
                type1 = merge_table_type
                type2 = SAM_feats_type[ttt]

                merge_table_type = type1+type2
                
                if (type1 =='A' and type2=='M'):
                    # print('hello')
                    merge_table = pd.merge( left=merge_table,
                                right=SAM_feats[ttt],
                                how="inner",
                                on="uniq_obj_id",
                                left_on=None,
                                right_on=None,
                                left_index=False,
                                right_index=False,
                                sort=True,
                                suffixes=("_appear", "_motion"), # this handles the case when two columns have duplicate names. 
                                copy=True,
                                indicator=False,
                                validate="1:1",
                              )
                elif (type1 == 'SA' and type2 =='M'):
                    # print('hello')
                    merge_table = pd.merge( left=merge_table,
                                right=SAM_feats[ttt],
                                how="inner",
                                on="uniq_obj_id",
                                left_on=None,
                                right_on=None,
                                left_index=False,
                                right_index=False,
                                sort=True,
                                suffixes=("_appear", "_motion"), # this handles the case when two columns have duplicate names. 
                                copy=True,
                                indicator=False,
                                validate="1:1",
                              )
                else:
                    merge_table = pd.merge( left=merge_table,
                                            right=SAM_feats[ttt],
                                            how="inner",
                                            on="uniq_obj_id",
                                            left_on=None,
                                            right_on=None,
                                            left_index=False,
                                            right_index=False,
                                            sort=True,
                                            suffixes=("_x", "_y"),
                                            copy=True,
                                            indicator=False,
                                            validate="1:1",
                                          )
        
        
        """
        # drop incomplete rows.
        """        
        # keep_rows = np.arange(len(merge_table))[~np.isnan(merge_table['area'].values)] # any other valid perimeter. 
        # count_nans = (merge_table.isna().values).sum(axis=1)
        # keep_rows = np.arange(len(merge_table))[count_nans==0]
        keep_rows = np.arange(len(merge_table))[~np.isnan(merge_table[test_feature_name].values)] # any other valid perimeter.  # this finds more? 

        merge_table = merge_table.loc[keep_rows]
        merge_table.index = np.arange(len(merge_table))
        
        all_object_uniq_row_ids.append([expt_name+'_'+mm for mm in merge_table['uniq_obj_id'].values]) # required to reconstruct tracks 
        merge_table = merge_table.loc[:,np.logical_not( merge_table.columns=='uniq_obj_id')]
        
        keep_index = merge_table['uniq_row_id'].values
        all_object_boundaries.append(boundaries[keep_index])
        
        if len(patches.shape)>2:    
            all_object_patches.append(patches[keep_index])
        else:
            all_object_patches += [patches[0][ind] for ind in keep_index]
        all_object_patches_sizes.append(patches_sizes[keep_index])
        
        merge_table = merge_table.loc[:,np.logical_not( merge_table.columns=='uniq_row_id')]
        
# =============================================================================
#       parse the main statistics and meta as before. 
# =============================================================================
        # create a unique id for the organoid, that is unique for the organoid in the video.  
        try:
            condition = merge_table['Condition']
            all_object_condition.append(condition)
        except:
            pass
        
        try:
            genetics = merge_table['Genetics']
            all_object_genetics.append(genetics)
        except:
            pass
        
        try: 
            patients = merge_table['Patient'] # how to handle this ?
            all_object_patients.append(patients)
        except:
            pass
        
        obj_id = merge_table['Org_ID']
        
        # extract out only the relevant feats, the ones before this column will be metadata.  
        feats = merge_table.loc[:,'max_curvature':].values # get all the numerical features of interest. 
        all_object_feats.append(feats)
        all_object_TP.append(merge_table['Frame_No'].values)
        all_object_Ts.append(merge_table['Frame_Duration[h]'].values)
#        all_organoid_genetics.append(genetics)
        
        # all_object_patients.append(patients)
        # all_object_condition.append(condition)
        all_object_expt.append(np.hstack([expt_name]*len(feats)))
        
        # print(feats.shape)
        
        """
        create a expt id.  
        """
        try:
            uniq_obj_id = np.hstack([expt_name + '_' + condition[ii] + '_' + '_' + str(obj_id[ii]) for ii in range(len(feats))]) 
        except:
            uniq_obj_id = np.hstack([expt_name + '_' + '_' + '_' + '_' + str(obj_id[ii]) for ii in range(len(feats))]) 
            
        object_ids.append(uniq_obj_id)
        
    n_objs_per_expt = np.hstack([len(ee) for ee in all_object_expt])
    

    """
    stack. 
    """
    object_ids = np.hstack(object_ids) # keep a unique id number of all organoids. 
    all_object_uniq_row_ids = np.hstack(all_object_uniq_row_ids)
    all_object_feats = np.vstack(all_object_feats) 
    # print(all_object_feats.shape)
    all_object_feats = all_object_feats.astype(np.float32) # does this make a diff?
    # all_organoid_appear_feats = np.vstack(all_organoid_appear_feats)
    all_object_boundaries = np.vstack(all_object_boundaries)
    try:
        all_object_patches = np.vstack(all_object_patches)
    except:
        pass
    all_object_patches_sizes = np.vstack(all_object_patches_sizes)
    all_object_TP = np.hstack(all_object_TP)
    all_object_Ts = np.hstack(all_object_Ts)
    if len(all_object_genetics) > 0: 
        all_object_genetics = np.hstack(all_object_genetics)
    if len(all_object_patients) > 0: 
        all_object_patients = np.hstack(all_object_patients)
    if len(all_object_condition) > 0:
        all_object_condition = np.hstack(all_object_condition)
    all_object_expt = np.hstack(all_object_expt) # this tabulates which experiment each dataset comes from. 
    # all_object_patients = np.hstack(all_object_patients)
    
    """
    extract the final feature names.
    """
    start_index = np.arange(len(merge_table.columns))[merge_table.columns=='max_curvature']
    feature_names = merge_table.columns[start_index[0]:]
    
    assert(len(all_object_uniq_row_ids) == len(np.unique(all_object_uniq_row_ids))) # check 

    metadict = {'object_ids' : object_ids,
                'all_object_uniq_row_ids': all_object_uniq_row_ids,
                # 'all_object_shape_feats': all_object_shape_feats,
                'all_object_boundaries': all_object_boundaries,
                'all_object_patches': all_object_patches,
                'all_object_patches_sizes': all_object_patches_sizes,
                'all_object_TP': all_object_TP,
                'all_object_Ts': all_object_Ts,
                'all_object_genetics' : all_object_genetics,
                'all_object_patients': all_object_patients,
                'all_object_condition': all_object_condition,
                'all_object_expt': all_object_expt,
                'n_objs_per_expt': n_objs_per_expt,
                'merge_table_type': merge_table_type,
                'feature_names' : feature_names}

    return  all_object_feats, metadict   
