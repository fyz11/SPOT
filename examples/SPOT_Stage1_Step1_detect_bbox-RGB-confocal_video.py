# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 01:51:28 2022

@author: fyz11
"""

if __name__=="__main__":
    
    """
    This example shows how to use the weights of our pretrained YOLOv3 organoid model to detect frame-by-frame the organoids using Keras and Tensorflow
    
    """
    import numpy as np 
    import os 
    
    import SPOT.Image.image as SPOT_image
    import SPOT.Utility_Functions.file_io as fio
    import SPOT.Detection.detection as detection 
    
    imfile = r'./data/organoids/fluorescent_murine_colon/KRAS_G12D_EYFP_2.wmv'
    basename = os.path.split(imfile)[-1].split('.wmv')[0]
    
    vid = fio.read_video_cv2(imfile)
    
    """
    Specify the trained detection model - YOLOv3
    """
    outfolder = r'../test_outputs/test_detection_folder'
    outfolder = os.path.join(outfolder, basename)
    fio.mkdir(outfolder)

    """
    Specify the CNN detector weights location here. Our pretrained is available and can be downloaded from dropbox. Follow the link in the README.md to download then copy to models/detect_CNN_model/
    """
    # you should download this weights file and place it here. 
    weightsfile = r'./models/detect_CNN_model/keras_YOLOv3_organoid_detector2.h5'

    
    """
    Detect bounding boxes using YOLOv3 model and write straight to the output folder. 
    
    This function will detect all the bounding boxes for each channel. vid of dimensions (n_frames x n_rows x n_cols x n_channels)
    Results will be saved to outfolder, each channel with their own individual subfolder. 
    """
    detection.load_and_run_YOLOv3_weights_keras_detection_model_RGB(vid, 
                                                                    weightsfile,
                                                                    outfolder,  
                                                                    obj_thresh=0.001, 
                                                                    nms_thresh = 0.45,
                                                                    imsize=(512,512),  
                                                                    anchors=None,
                                                                    class_name_file=None,
                                                                    debug_viz=True)
