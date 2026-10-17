# -*- coding: utf-8 -*-
"""
Deterministic test inputs for the SAM feature functions.

Two cases, each giving a grayscale uint8 video, an optical flow (T-1, H, W, 2), foreground masks and boundaries
(n_obj, T, n_pts, 2) in (row, col) image convention:

    synthetic   4 textured objects (circle, ellipse, star, thin bar) moving and deforming over 5 frames in a 128 x 160
                frame, with a NaN gap and an object touching the image border
    u373        real cell contours of the cell tracking challenge PhC-C2DH-U373/01 (data/cell_tracking, first 6 cells x
                first 8 frames, NaN where a cell is absent) on a synthetic textured 520 x 696 video and smooth flow
"""

import os

import numpy as np
import scipy.ndimage as ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = ('synthetic', 'u373')


def _texture(rng, shape, sigma):
    return ndimage.gaussian_filter(rng.standard_normal(shape), sigma)


def _closed(cnt):
    return np.vstack([cnt, cnt[:1]])


def _polygon_mask(cnt, shape):
    from skimage.draw import polygon
    m = np.zeros(shape, dtype=bool)
    rr, cc = polygon(cnt[:, 0], cnt[:, 1], shape)
    m[rr, cc] = True
    return m


def _render(boundaries, shape, rng):
    """uint8 video: smooth background plus a distinct texture per object; masks are the union of the objects."""
    n_obj, T = boundaries.shape[:2]
    textures = [_texture(rng, shape, s) for s in (1., 1.5, 2., 3., 1.2, 2.5)]
    vid = np.zeros((T,) + shape)
    masks = np.zeros((T,) + shape, dtype=bool)
    for t in range(T):
        frame = 60 + 20 * _texture(rng, shape, 6.) / 0.05
        for i in range(n_obj):
            if np.isnan(boundaries[i, t, 0, 0]):
                continue
            m = _polygon_mask(boundaries[i, t], shape)
            tex = textures[i % len(textures)]
            frame[m] = 140 + 35 * (i % 3) + 60 * tex[m] / tex.std()
            masks[t] |= m
        vid[t] = frame
    return np.clip(vid, 0, 255).astype(np.uint8), masks


def _flow(rng, T, shape, scale):
    f = np.stack([ndimage.gaussian_filter(rng.standard_normal(shape + (2,)), (8, 8, 0)) for _ in range(T - 1)])
    return scale * f / f.std()


def synthetic(n_pts=200):
    rng = np.random.default_rng(0)
    H, W, T = 128, 160, 5
    t = np.linspace(0, 2 * np.pi, n_pts - 1, endpoint=False)
    objs = []
    for k in range(T):
        frame_objs = []
        # circle drifting right
        frame_objs.append(np.c_[35 + 14 * np.sin(t), 30 + 2 * k + 14 * np.cos(t)])
        # ellipse rotating
        a = 0.3 * k
        y, x = 9 * np.sin(t), 24 * np.cos(t)
        frame_objs.append(np.c_[90 + y * np.cos(a) + x * np.sin(a), 50 - y * np.sin(a) + x * np.cos(a)])
        # star deforming
        r = 18 * (1 + (0.25 + 0.03 * k) * np.cos(5 * t))
        frame_objs.append(np.c_[60 + r * np.sin(t), 110 + r * np.cos(t)])
        # thin bar touching the top border
        frame_objs.append(np.c_[1 + 10 * (1 + np.sin(t)), 140 + 3 * np.cos(t)])
        objs.append([_closed(o) for o in frame_objs])
    boundaries = np.array(objs).transpose(1, 0, 2, 3)  # (n_obj, T, n_pts, 2)
    boundaries[2, 2] = np.nan  # the star is missing in frame 2
    vid, masks = _render(boundaries, (H, W), rng)
    flow = _flow(rng, T, (H, W), 1.5)
    return dict(vid=vid, flow=flow, masks=masks, boundaries=boundaries, imshape=(H, W))


def u373(n_obj=6, T=8):
    import scipy.io as spio
    rng = np.random.default_rng(1)
    d = spio.loadmat(os.path.join(REPO, 'data', 'cell_tracking', 'PhC-C2DH-U373', '01', 'final_boundaries.mat'))
    org, frame, bnd = d['org_id'].ravel(), d['frame_no'].ravel(), d['boundary']
    orgs = np.unique(org)[:n_obj]
    frames = np.arange(frame.min(), frame.min() + T)
    boundaries = np.full((len(orgs), T, bnd.shape[1], 2), np.nan)
    for i, o in enumerate(orgs):
        for j, f in enumerate(frames):
            sel = np.nonzero((org == o) & (frame == f))[0]
            if len(sel):
                boundaries[i, j] = bnd[sel[0]]
    shape = (520, 696)
    vid, masks = _render(boundaries, shape, rng)
    flow = _flow(rng, T, shape, 2.)
    return dict(vid=vid, flow=flow, masks=masks, boundaries=boundaries, imshape=shape)


def load(case):
    return {'synthetic': synthetic, 'u373': u373}[case]()


# feature function calls covered by the golden outputs: name -> (function name, input builder)
def calls():
    """(key, function name, kwargs builder) for every configuration covered by the goldens."""
    c = [('morphology', 'compute_boundary_morphology_features',
          lambda d: dict(boundaries=d['boundaries'], imshape=d['imshape']))]
    c.append(('texture', 'compute_boundary_texture_features',
              lambda d: dict(vid=d['vid'], boundaries=d['boundaries'], use_mask=False)))
    for mode in ('cap', 'square', 'native'):
        c.append(('texture_masked_%s' % mode, 'compute_boundary_texture_features',
                  lambda d, mode=mode: dict(vid=d['vid'], boundaries=d['boundaries'], use_mask=True,
                                            masked_sift_resize=mode, masks=d['masks'])))
    c.append(('texture_masked_contour_only', 'compute_boundary_texture_features',
              lambda d: dict(vid=d['vid'], boundaries=d['boundaries'], use_mask=True)))
    c.append(('motion', 'compute_boundary_motion_features',
              lambda d: dict(vid_flow=d['flow'], boundaries=d['boundaries'], pixel_res=2., time_res=3., use_mask=False)))
    c.append(('motion_masked', 'compute_boundary_motion_features',
              lambda d: dict(vid_flow=d['flow'], boundaries=d['boundaries'], use_mask=True, masks=d['masks'])))
    return c
