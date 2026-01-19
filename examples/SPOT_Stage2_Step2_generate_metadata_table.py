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
    specify basic info 
    """
    pixel_res = 2.76 # um 
    time_res = 2 # h
    
    """
    specify the save location of this master metadata file - usually the also in the top-level of the video dataset
    """
    csvsavefolder = masterimgfolder
    fio.mkdir(csvsavefolder)
    
    
    """
    metadata template table generation. 
    """
    data_table = []
    
    # to avoid confusion
    for iiii in np.arange(n_files)[:]:
        
        # basic filename 
        infile = imgfiles[iiii]
        fname = os.path.split(infile)[-1].split(ext)[0]
        
        print(iiii, fname)
        
        """
        generate a placeholder for specifying which image channel should be analyzed for each video, multiple channels should be separated by a '+', thus if we want first two channels this should be filled in manually as '1+2'
        """
        img_channel = np.nan        
        
        """
        generate some standard condition metafields, populate with NA. 
        """
        genetics = np.nan
        patient = np.nan
        condition = np.nan 
        
        
        if '.tif' in ext.lower() :
            vid = skio.imread(infile)
            n_frames = len(vid)
        elif '.avi' in ext.lower() or '.wmv' in ext.lower():
            vid = fio.read_video_cv2(infile)
            n_frames = len(vid)
        else:
            print('video cannot be read')
            n_frames = np.nan
            
        data_table_row = [fname, img_channel, genetics, patient, condition, pixel_res, time_res, n_frames]
        data_table.append(data_table_row)
        
    data_table = np.array(data_table, dtype=object)
    data_table = pd.DataFrame(data_table, 
                              index=None,
                              columns=['Filename',
                                       'Img_Channel',
                                       'Genetics',
                                       'Patients',
                                       'Conditions', 
                                       'pixel_resolution[um]',
                                       'time_resolution[h]',
                                       'n_frames'])
        
    csvsavefile = os.path.join(csvsavefolder, 
                               'metadata_expt.csv')
    data_table.to_csv(csvsavefile, 
                      index=None)

