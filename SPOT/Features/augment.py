# -*- coding: utf-8 -*-
"""
Rotation augmentation of the SAM features.

Several SAM features depend on the orientation of the object in the image (SIFT orientation histograms of appearance
and motion, the angular sectors of the regional windows, Haralick statistics on the pixel grid, the directions of the
Euler characteristic transform). compute_rotation_averaged_features rotates the video, the contours (and masks) by n
angles, computes the features at every angle with any of the SAM feature functions (original, *_fast or *_torch) and
averages them, so that the features become (approximately) invariant to the image orientation. For motion, the optical
flow is recomputed on every rotated video. rotation_sensitivity measures how much each feature changes with the angle.

Angles are in degrees, positive = the direction of np.rot90 (counter-clockwise as displayed). Multiples of 90 degrees
are exact (np.rot90, contours mapped exactly); other angles interpolate the frames (bilinear, masks nearest neighbour)
onto a canvas large enough to hold the whole rotated frame, which slightly smooths the image at those angles.

Coordinates are in (row, col) image convention, as the SAM feature functions.

@author: fyz11
"""

import inspect

import numpy as np


def _quarter_turns(angle):
    """ the number of quarter turns k (0..3) if angle is a multiple of 90 degrees, else None """
    q = float(angle) / 90.
    return int(round(q)) % 4 if np.isclose(q, round(q), rtol=0, atol=1e-9) else None


def _rotation_geometry(shape, angle):
    """ rotated canvas shape and the (row, col) -> (row, col) map p' = R (p - c) + c' of a rotation by angle degrees """
    H, W = shape
    k = _quarter_turns(angle)
    if k is not None:
        out_shape = (H, W) if k % 2 == 0 else (W, H)
        theta = np.deg2rad(90. * k)
    else:
        theta = np.deg2rad(float(angle))
        c, s = abs(np.cos(theta)), abs(np.sin(theta))
        out_shape = (int(np.ceil(H * c + W * s)), int(np.ceil(H * s + W * c)))
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    R[np.abs(R) < 1e-15] = 0.
    center = np.array([(H - 1) / 2., (W - 1) / 2.])
    center_out = np.array([(out_shape[0] - 1) / 2., (out_shape[1] - 1) / 2.])
    return out_shape, R, center, center_out


def rotation_transform(shape, angle):
    """ The map of (row, col) coordinates of an image of the given shape to the image rotated by angle degrees.

    Parameters
    ----------
    shape : (n_rows, n_cols)
        the image shape
    angle : float
        rotation angle in degrees (positive = np.rot90 direction)

    Returns
    -------
    transform : function
        maps an (..., 2) array of (row, col) coordinates to the rotated image (NaNs stay NaN)
    out_shape : (n_rows, n_cols)
        the shape of the rotated image
    """
    out_shape, R, center, center_out = _rotation_geometry(shape, angle)
    k = _quarter_turns(angle)

    def transform(coords):
        coords = np.asarray(coords, dtype=np.float64)
        if k is not None:
            # exact quarter turns: one turn maps (r, c) -> (W - 1 - c, r), as np.rot90
            out = coords.copy()
            h, w = shape
            for _ in range(k):
                out = np.stack([(w - 1) - out[..., 1], out[..., 0]], axis=-1)
                h, w = w, h
            return out
        return (coords - center) @ R.T + center_out

    return transform, out_shape


def rotate_video(vid, angle, order=1):
    """ Rotate every frame of a video by angle degrees.

    Parameters
    ----------
    vid : (n_frames, n_rows, n_cols) or (n_frames, n_rows, n_cols, n_channels) array
        the video (or masks)
    angle : float
        rotation angle in degrees (positive = np.rot90 direction)
    order : int
        interpolation order for angles that are not multiples of 90 degrees (1 = bilinear for images, 0 = nearest
        neighbour for masks / labels)

    Returns
    -------
    vid_rot : array
        the rotated video, same dtype. Multiples of 90 degrees are exact; otherwise the canvas is enlarged to hold the
        whole rotated frame and filled with 0 outside it.
    transform : function
        the (row, col) coordinate map, see rotation_transform
    """
    import scipy.ndimage as ndimage

    vid = np.asarray(vid)
    shape = vid.shape[1:3]
    transform, out_shape = rotation_transform(shape, angle)
    k = _quarter_turns(angle)
    if k is not None:
        return np.ascontiguousarray(np.rot90(vid, k, axes=(1, 2))), transform

    _, R, center, center_out = _rotation_geometry(shape, angle)
    # pull back: input = R^T (output - c') + c
    M = R.T
    offset = center - M @ center_out
    out = np.zeros((vid.shape[0],) + tuple(out_shape) + vid.shape[3:], dtype=vid.dtype)
    integer = np.issubdtype(vid.dtype, np.integer) or vid.dtype == bool
    for t in range(vid.shape[0]):
        frames = vid[t][..., None] if vid.ndim == 3 else vid[t]
        for ch in range(frames.shape[-1]):
            rot = ndimage.affine_transform(frames[..., ch].astype(np.float64), M, offset=offset, output_shape=out_shape,
                                           order=order, mode='constant', cval=0.)
            if integer:
                info = np.iinfo(vid.dtype) if vid.dtype != bool else None
                rot = np.round(rot)
                rot = rot > 0.5 if vid.dtype == bool else np.clip(rot, info.min, info.max)
            if vid.ndim == 3:
                out[t] = rot
            else:
                out[t, ..., ch] = rot
    return out, transform


def rotate_boundaries(boundaries, transform):
    """ Apply a rotation_transform / rotate_video coordinate map to (n_objects, n_frames, n_points, 2) (row, col)
    contours; NaN contours (absent objects) stay NaN. """
    return transform(np.asarray(boundaries, dtype=np.float64))


def _feature_family(feature_fn):
    params = inspect.signature(feature_fn).parameters
    if 'vid_flow' in params:
        return 'motion', params
    if 'vid' in params:
        return 'appearance', params
    if 'imshape' in params:
        return 'shape', params
    raise ValueError('feature_fn must be a SAM feature function (shape, appearance or motion)')


def compute_rotation_averaged_features(feature_fn, boundaries, vid=None, masks=None, imshape=None, n_rotations=4,
                                       angles=None, interp_order=3, flow_params=None, rescale_intensity=True,
                                       n_jobs=None, return_per_angle=False, verbose=True, **feature_kwargs):
    """ Rotation-averaged SAM features: rotate the inputs by n angles, compute the features at each angle with
    feature_fn and average them.

    Parameters
    ----------
    feature_fn : function
        a SAM feature function: compute_boundary_morphology_features, compute_boundary_texture_features,
        compute_boundary_motion_features or their *_fast / *_torch variants
    boundaries : (n_objects, n_frames, n_contour_pts, 2) array
        the (row, col) object contours
    vid : (n_frames, n_rows, n_cols) array
        the grayscale video; required for appearance and motion (the optical flow is recomputed at every angle)
    masks : None or (n_frames, n_rows, n_cols) array
        optional foreground masks (appearance / motion use_mask=True), rotated with nearest neighbour interpolation
    imshape : None or (n_rows, n_cols)
        the frame shape, required for shape features without vid
    n_rotations : int
        number of evenly spaced angles 360 * k / n_rotations, k = 0 .. n_rotations-1 (n_rotations=4 is exact)
    angles : None or list of float
        explicit angles in degrees instead of n_rotations
    interp_order : int
        spline order used to rotate the frames at angles that are not multiples of 90 degrees (3 = cubic, default; 1 = bilinear).
        Interpolation smooths the image, which changes intensity / texture features of high frequency textures by a
        noticeable amount; multiples of 90 degrees (e.g. n_rotations=4) are exact.
    flow_params : None or dict
        OpenCV Farneback parameters for the motion features, default SPOT.Tracking.optical_flow.get_farneback_flow_params()
    rescale_intensity : bool
        passed to SPOT.Tracking.optical_flow.extract_vid_optflow
    n_jobs : None or int
        passed to feature_fn if it takes n_jobs (the *_fast variants)
    return_per_angle : bool
        if True also return the (n_angles, n_objects, n_frames, n_features) features at every angle and the angles
    verbose : bool
        print the angle being computed
    feature_kwargs :
        any other arguments of feature_fn (e.g. use_mask=True, pixel_res, device)

    Returns
    -------
    out_array : (n_objects, n_frames, n_features) array
        the features averaged over the angles (NaN-mean)
    out_feat_labels : (n_features,) str array
    out_feat_labels_norm : (n_features,) bool array
    (per_angle, angles) : ((n_angles, n_objects, n_frames, n_features) array, (n_angles,) array), if return_per_angle
    """
    family, params = _feature_family(feature_fn)
    if angles is None:
        angles = [360. * k / n_rotations for k in range(n_rotations)]
    angles = np.asarray(angles, dtype=np.float64)
    if family in ('appearance', 'motion') and vid is None:
        raise ValueError('vid is required for %s features' % family)
    if family == 'shape' and imshape is None:
        if vid is None:
            raise ValueError('imshape (or vid) is required for shape features')
        imshape = np.asarray(vid).shape[1:3]
    if n_jobs is not None and 'n_jobs' in params:
        feature_kwargs['n_jobs'] = n_jobs
    if family == 'motion':
        from SPOT.Tracking.optical_flow import extract_vid_optflow, get_farneback_flow_params
        flow_params = flow_params or get_farneback_flow_params()

    per_angle, labels, norm = [], None, None
    for angle in angles:
        if verbose:
            print('rotation %.1f deg (%s features)' % (angle, family), flush=True)
        kw = dict(feature_kwargs)
        if family == 'shape':
            transform, out_shape = rotation_transform(tuple(imshape), angle)
            kw.update(boundaries=rotate_boundaries(boundaries, transform), imshape=out_shape)
        else:
            vid_rot, transform = rotate_video(vid, angle, order=interp_order)
            kw['boundaries'] = rotate_boundaries(boundaries, transform)
            if masks is not None:
                kw['masks'] = rotate_video(np.asarray(masks), angle, order=0)[0]
            if family == 'appearance':
                kw['vid'] = vid_rot
            else:
                kw['vid_flow'] = extract_vid_optflow(vid_rot, flow_params, rescale_intensity=rescale_intensity)
        feats, f_labels, f_norm = feature_fn(**kw)
        f_labels = np.asarray(f_labels).astype(str)
        if labels is None:
            labels, norm = f_labels, np.asarray(f_norm)
        elif list(f_labels) != list(labels):
            raise RuntimeError('feature names differ between angles')
        per_angle.append(np.asarray(feats, dtype=np.float64))
    per_angle = np.stack(per_angle)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN samples (absent objects)
        out = np.nanmean(per_angle, axis=0)
    if return_per_angle:
        return out, labels, norm, (per_angle, angles)
    return out, labels, norm


def rotation_sensitivity(per_angle, labels):
    """ How much each feature changes with the rotation angle.

    Parameters
    ----------
    per_angle : (n_angles, n_objects, n_frames, n_features) array
        the features at every angle (compute_rotation_averaged_features(..., return_per_angle=True)), angle 0 first
    labels : (n_features,) array
        the feature names

    Returns
    -------
    table : pandas DataFrame
        per feature: rot_std = std over the angles (mean over samples), sample_std = std over the samples at the first
        angle, sensitivity = rot_std / sample_std (0 = rotation invariant), and the SAM set / group of the feature
    """
    import pandas as pd
    from SPOT.Analysis.preprocessing import sam_feature_groups

    F = np.asarray(per_angle, dtype=np.float64)
    F = F.reshape(F.shape[0], -1, F.shape[-1])
    ok = np.isfinite(F).all(axis=(0, 2))
    F = F[:, ok]
    rot_std = F.std(axis=0).mean(axis=0)
    sample_std = F[0].std(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        sens = np.where(sample_std > 0, rot_std / sample_std, np.nan)
    sets, groups = sam_feature_groups(labels)
    return pd.DataFrame(dict(feature=np.asarray(labels).astype(str), feature_set=sets, group=groups,
                             rot_std=rot_std, sample_std=sample_std, sensitivity=sens))
