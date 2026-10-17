import numpy as np 

def spectral_unmix_RGB(img, n_components=3, alpha=1., l1_ratio=0.5):
    """ Spectrally unmixes an RGB image based on non-negative matrix factorization using the method of skimage.decomposition. Assumes uint8 RGB image.
    
    Parameters
    ----------
    img : numpy array
        (n_rows, n_cols, 3) image to unmix 
    n_components : int 
        the number of returned 'pure' components usually the number of channels of the original image. 
    alpha : float
        Constant that multiplies the regularization terms. Set it to zero to have no regularization. 
    l1_ratio : float 
        The regularization mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an elementwise L2 penalty (aka Frobenius Norm). For l1_ratio = 1 it is an elementwise L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    Returns
    -------
    img_vector_NMF_rgb : numpy array 
        (n_rows, n_cols, n_components) unmixed image. 
    color_model : scikit-learn model object
        the fitted non-negative matrix factorization model from sklearn.decomposition.NMF

    """
    from sklearn.decomposition import NMF
    from skimage.exposure import rescale_intensity
    import numpy as np 
    
    img_vector = img.reshape(-1,img.shape[-1]) / 255.
#    img_vector = img_vector_.copy()itakura-saito
#    img_vector[img_vector<0] = 0

    try:
        color_model = NMF(n_components=n_components, init='nndsvd', random_state=0, alpha=alpha, l1_ratio=l1_ratio) # Note ! we need a high alpha ->. 
    except:
        color_model = NMF(n_components=n_components, init='nndsvd', random_state=0, alpha_W=alpha, l1_ratio=l1_ratio)
    W = color_model.fit_transform(img_vector)
    
    img_vector_NMF_rgb = W.reshape((img.shape[0], img.shape[1], -1))
    img_vector_NMF_rgb = np.uint8(255*rescale_intensity(img_vector_NMF_rgb))
    
#    # get the same order of channels as previous using the model components.
#    channel_order = np.argmax(color_model.components_, axis=0); 
#    print(channel_order)
#    img_vector_NMF_rgb = img_vector_NMF_rgb[...,channel_order]
    
    # return the color model. 
    return img_vector_NMF_rgb, color_model
    

def apply_unmix_model(img, model):
    """ unmix a given image with a previously fitted scikit-learn unmixing model instance. Assumes uint8 RGB image.
    
    Parameters
    ----------
    img : numpy array
        (n_rows, n_cols, 3) image to unmix 
    model : scikit-learn model object
        a fitted non-negative matrix factorization model from sklearn.decomposition.NMF

    Returns
    -------
    img_proj_vector : numpy array
    """
    import numpy as np 
    import skimage.exposure as skexposure 

    img_vector = img.reshape(-1,img.shape[-1]) / 255.
    img_proj_vector = model.transform(img_vector)
    
    img_proj_vector = img_proj_vector.reshape((img.shape[0], img.shape[1], -1))
    img_proj_vector = np.uint8(255*skexposure.rescale_intensity(img_proj_vector*1.))
    
    return img_proj_vector


def spectral_unmix_RGB_video(vid, alpha=1, l1_ratio=.5):
    """ convenience wrapper function to unmix all frames of a given input uint8 RGB video based on learning a non-negative matrix factorization unmixing model automatically choosing the frame with the highest total RGB intensity.  
    
    Parameters
    ----------
    vid : numpy array
        (n_frames, n_rows, n_cols, 3) uint8 RGB to unmix 
    alpha : float
        Constant that multiplies the regularization terms. Set it to zero to have no regularization. 
    l1_ratio : float
        The regularization mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an elementwise L2 penalty (aka Frobenius Norm). For l1_ratio = 1 it is an elementwise L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    Returns
    -------
    unmixed_vid_out : numpy array
        (n_frames, n_rows, n_cols, 3) unmixed uint8 RGB 

    """
    import numpy as np 
    
    n_channels = vid.shape[-1] # how many channels are there in the video
    select_channels = np.arange(n_channels)
    
    # compute the mean RGB intensity of each frame 
    vid_channels_time = np.zeros((len(vid), vid.shape[-1]))
    for frame in range(len(vid)):
        vid_frame = vid[frame]
        for ch in range(vid.shape[-1]):
            vid_channels_time[frame, ch ] = np.mean(vid_frame[...,ch])
                
    # project RGB to grayscale curves. 
    # mean_curve = np.nanmedian(vid_channels_time, axis=1)
    mean_curve = np.nanmin(vid_channels_time, axis=1);  # should we smooth this? 
    ref_time = np.argmax(mean_curve)
    
    unmix_img, unmix_model = spectral_unmix_RGB(vid[ref_time], 
                                                n_components=n_channels, 
                                                alpha=alpha, 
                                                l1_ratio=l1_ratio)
    mix_components = unmix_model.components_.copy()
    
    mix_components_origin = np.argmax(mix_components, axis =1 )
    mix_components_origin_mag = np.max(mix_components, axis =1)
    
    mix_components_origin = mix_components_origin[mix_components_origin_mag>0]
    
    try:
        # use the select channels to properly select the ones. 
        NMF_channel_order = []
        NMF_select_channels = []
        
        for ch in select_channels:
            if ch in mix_components_origin:
                # find the order. 
                NMF_select_channels.append(ch)
                order = np.arange(len(mix_components_origin))[mix_components_origin==ch]
                NMF_channel_order.append(order)
        
        NMF_channel_order = np.hstack(NMF_channel_order)
        NMF_select_channels = np.hstack(NMF_select_channels)
        
        unmixed_vid = np.array([apply_unmix_model(frame, unmix_model) for frame in vid])
        
        # write this to a proper video. 
        unmixed_vid_out = np.zeros_like(vid)
        unmixed_vid_out[...,NMF_select_channels] = unmixed_vid[...,NMF_channel_order]
    except:
        unmixed_vid_out = vid.copy()

    return unmixed_vid_out


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
	r""" Linear image intensity rescaling based on the given minimum and maximum intensities, with optional clipping
	
	The new intensities will be 

	.. math::
		x_{new} = \frac{x-\text{mi}}{\text{ma}-\text{mi} + \text{eps}}
		
	Parameters
	----------
	x : nd array
		arbitrary n-dimensional image
	mi : scalar
		minimum intensity value 
	ma : scalar
		maximum intensity value 
	clip : bool
		whether to clip the transformed image intensities to [0,1] 
	eps : scalar
		small number for numerical stability 
	dtype : np.dtype
		if not None, casts the input x into the specified dtype 

	Returns
	-------
	x : nd array
		normalized output image of the same dimensionality as the input image

	"""
	if dtype is not None:
		x   = x.astype(dtype,copy=False)
		mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
		ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
		eps = dtype(eps)
	try:
		import numexpr
		x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
	except ImportError:
		x =                   (x - mi) / ( ma - mi + eps )
	if clip:
		x = np.clip(x,0,1)
	return x

def normalize(x, pmin=2, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
	r""" Percentile-based image normalization, same as :func:`unwrap3D.Image_Functions.image.imadust` but exposes more flexibility in terms of axis selection and optional clipping. 

	Parameters
	----------
	x : nd array
		arbitrary n-dimensional image
	pmin : scalar
		lower percentile of intensity, specified as a number in the range 0-100. All intensities in a lower percentile than p1 will be mapped to 0.
	pmax : scalar
		upper percentile of intensity, specified as a number in the range 0-100. All intensities in the upper percentile than p2 will be mapped to the maximum value of the data type.
	axis : int
		if not None, compute the percentiles restricted to the specified axis, otherwise they will be based on the statistics of the whole image.
	clip : bool
		whether to clip to the maximum and minimum of [0,1] for float32 or float64 data types.    
	eps : scalar
		small number for numerical stability 
	dtype : np.dtype
		if not None, casts the input x into the specified dtype 

	Returns
	-------
	x_out : nd array
		normalized output image of the same dimensionality as the input image

	"""
	mi = np.percentile(x,pmin,axis=axis,keepdims=True)
	ma = np.percentile(x,pmax,axis=axis,keepdims=True)

	x_out = normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)
	return x_out


def get_img_contours(img, thresh=100, n_boundary_pts=100, return_original=False): 
    
    from skimage.measure import find_contours
    from scipy.ndimage import binary_fill_holes
    from skimage.morphology import binary_dilation
    
    binary = img > thresh
    binary = binary_fill_holes(binary)
    binary = binary_dilation(binary, np.ones((3,3), dtype=bool)) # 3x3 square footprint
    binary_cont = find_contours(binary,0)
    binary_cont = binary_cont[np.argmax([len(l) for l in binary_cont])]
    
    binary_cont_resample = resample_curve(binary_cont[:,1], 
                                          binary_cont[:,0], k=1, s=0, n_samples=n_boundary_pts)
    
    if return_original:
        return binary_cont_resample, binary_cont[...,[1,0]]
    else:
        return binary_cont_resample
