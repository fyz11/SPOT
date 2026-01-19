# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 01:51:28 2022

@author: fyz11
"""

if __name__=="__main__":
    
    """
    For multi-channel fluorescent microscopy imaging there can be bleed-through. Spectral unmixing, here using the non-negative matrix factorization can help reduce this. This works if the algorithm converges otherwise it returns the original input. In this case you can try increasing alpha.
    
    We demo the usage of the functionality using an example video of a fluorescent organoid in the data folder
    
    """
    import numpy as np 
    import SPOT.Image.image as SPOT_image
    import SPOT.Utility_Functions.file_io as fio
    import pylab as plt 
    import os
    
    imfile = r'./data/organoids/fluorescent_murine_colon/KRAS_G12D_EYFP_2.wmv'

    saveplots = True
    savedir = '../test_outputs/RGB_unmixing'
    if saveplots:
        os.makedirs(savedir, exist_ok=True)
    
    vid = fio.read_video_cv2(imfile)
    
    """
    the input video is blue, simulate some bleed through (reduced intensity) in red channels. 
    """
    vid_bleedthrough_channel = np.uint8(np.clip(0.25*vid[...,2] + np.random.normal(0, scale=10, size=vid.shape[:3]), 0, 255))
    vid[...,1] = vid_bleedthrough_channel # copy to green 
    
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('raw max. proj channel 1')
    plt.imshow(vid[...,0].max(axis=0))
    plt.subplot(132)
    plt.title('raw max. proj channel 2')
    plt.imshow(vid[...,1].max(axis=0))
    plt.subplot(133)
    plt.title('raw max. proj channel 3')
    plt.imshow(vid[...,2].max(axis=0))
    if saveplots:
        plt.savefig(os.path.join(savedir, 'raw_max_projection.png'), dpi=150, bbox_inches='tight')
    plt.show()
    

    """
    spectral unmix to get only the blue component
    """
    vid_unmixed = SPOT_image.spectral_unmix_RGB_video(vid, alpha=5e-3, l1_ratio=.5)
    vid_unmixed = np.uint8(vid_unmixed)


    """
    visual comparison with the maximum projection image.
    """    
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('unmixed max. proj channel 1')
    plt.imshow(vid_unmixed[...,0].max(axis=0))
    plt.subplot(132)
    plt.title('unmixed max. proj channel 2')
    plt.imshow(vid_unmixed[...,1].max(axis=0))
    plt.subplot(133)
    plt.title('unmixed max. proj channel 3')
    plt.imshow(vid_unmixed[...,2].max(axis=0))
    if saveplots:
        plt.savefig(os.path.join(savedir, 'unmixed_max_projection.png'), dpi=150, bbox_inches='tight')
    plt.show()

