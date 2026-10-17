# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 19:32:59 2020

@author: felix


Module to compute Shape, Appearance, Motion (SAM) features for objects.
"""

import numpy as np
import scipy.ndimage as ndimage
import scipy.special as special
import pandas as pd
import tifffile 
# from scipy.spatial.distance import cdist
# import pylab as plt
# from math import sqrt, acos, pi


# =============================================================================
# ported from demeter package (ECC features)
# =============================================================================

def _neighborhood_setup(dimension, downsample = 1):
    
    import itertools
    neighs = sorted(list(itertools.product(range(2), repeat=dimension)), key=np.sum)[1:]
    neighs = list(map(tuple, np.array(neighs)*downsample))
    subtuples = dict()
    for i in range(len(neighs)):
        subtup = [0]
        for j in range(len(neighs)):
            if np.all(np.subtract(neighs[i], neighs[j]) > -1):
                subtup.append(j+1)
        subtuples[neighs[i]] = subtup

    return neighs, subtuples


def _neighborhood(voxel, neighs, hood, dcoords):
    hood[0] = dcoords[voxel]
    neighbors = np.add(voxel, neighs)
    for j in range(1,len(hood)):
        key = tuple(neighbors[j-1,:])
        if key in dcoords:
            hood[j] = dcoords[key]
    return hood

def _centerVertices(verts):
    origin = -1*np.mean(verts, axis=0)
    verts = np.add(verts, origin)
    return verts


class CubicalComplex:
    
    def __init__(self, img):
        self.img = img

    def complexify(self, center=True, downsample=1):
        scoords = np.nonzero(self.img)
        scoords = np.vstack(scoords).T

        skip = np.zeros(self.img.ndim, dtype=int) + downsample
        coords = scoords[np.all(np.fmod(scoords, skip) == 0, axis=1), :]

        keys = [tuple(coords[i,:]) for i in range(len(coords))]
        dcoords = dict(zip(keys, range(len(coords))))
        neighs, subtuples = _neighborhood_setup(self.img.ndim, downsample)
        binom = [special.comb(self.img.ndim, k, exact=True) for k in range(self.img.ndim+1)]

        hood = np.zeros(len(neighs)+1, dtype=np.int32)-1
        cells = [[] for k in range(self.img.ndim+1)]

        for voxel in dcoords:
            hood.fill(-1)
            hood = _neighborhood(voxel, neighs, hood, dcoords)
            nhood = hood > -1
            c = 0
            if np.all(nhood[:-1]):
                for k in range(1, self.img.ndim):
                    for j in range(binom[k]):
                        cell = hood[subtuples[neighs[c]]]
                        cells[k].append(cell)
                        c += 1
                if nhood[-1]:
                    cells[self.img.ndim].append(hood.copy())
            else:
                for k in range(1, self.img.ndim):
                    for j in range(binom[k]):
                        cell = nhood[subtuples[neighs[c]]]
                        if np.all(cell):
                            cells[k].append(hood[subtuples[neighs[c]]])
                        c += 1

        dim = self.img.ndim
        for k in range(dim, -1, -1):
            if len(cells[k]) > 0:
                break

        self.ndim = dim
        self.cells = [np.array(cells[k]) for k in range(dim+1)]
        if center:
            self.cells[0] = _centerVertices(coords)
        else:
            self.cells[0] = coords

        return self

    def EC(self):
        chi = 0
        for i in range(len(self.cells)):
            chi += ((-1)**i)*len(self.cells[i])

        self.chi = chi
        return chi

    def summary(self):
        cellnames = ['vertices', 'edges', 'squares', 'cubes']
        for i in range(len(self.cells)):
            if i < len(cellnames):
                print('{}\t{}'.format(len(self.cells[i]), cellnames[i]))
            else:
                print('{}\t{:02d}hypercubes'.format(len(self.cells[i]), i))

        chi = self.EC()
        print('----\nEuler Characteristic: {}'.format(chi))
        return 0

    def ECC(self, filtration, T=32, bbox=None):

        if bbox is None:
            minh = np.min(filtration)
            maxh = np.max(filtration)
        else:
            minh,maxh = bbox

        buckets = [None for i in range(len(self.cells))]

        buckets[0], bins = np.histogram(filtration, bins=T, range=(minh, maxh))

        for i in range(1,len(buckets)):
            if len(self.cells[i]) > 0 :
                buckets[i], bins = np.histogram(np.max(filtration[self.cells[i]], axis=1), bins=T, range=(minh, maxh))

        ecc = np.zeros_like(buckets[0])
        for i in range(len(buckets)):
            if buckets[i] is not None:
                ecc = np.add(ecc, ((-1)**i)*buckets[i])

        return np.cumsum(ecc)

    def ECT(self, directions, T=32, verts=None, bbox=None):
        if verts is None:
            verts = self.cells[0]

        ect = np.zeros(T*directions.shape[0], dtype=int)

        for i in range(directions.shape[0]):
            heights = np.sum(verts*directions[i,:], axis=1)
            ecc = self.ECC(heights, T, bbox)
            ect[i*T : (i+1)*T] = ecc

        return ect

    def triangulate(self, center=True, downsample=1):
        scoords = np.nonzero(self.img)
        scoords = np.vstack(scoords).T

        skip = np.array([downsample,downsample,downsample])
        coords = scoords[np.all(np.fmod(scoords, skip) == 0, axis=1), :]

        keys = [tuple(coords[i,:]) for i in range(len(coords))]
        dcoords = dict(zip(keys, range(len(coords))))
        neighs, subtuples = _neighborhood_setup(self.img.ndim, downsample)
        binom = [special.comb(self.img.ndim, k, exact=True) for k in range(self.img.ndim+1)]

        hood = np.zeros(len(neighs)+1, dtype=int)-1
        cells = [[] for k in range(self.img.ndim+1)]

        for voxel in dcoords:
            hood.fill(-1)
            hood = _neighborhood(voxel, neighs, hood, dcoords)
            nhood = hood > -1
            c = 0
            if np.all(nhood[:-1]):
                for k in range(1, self.img.ndim):
                    for j in range(binom[k]):
                        cell = hood[subtuples[neighs[c]]]
                        cells[k].append(cell)
                        c += 1
                if nhood[-1]:
                    cells[self.img.ndim].append(hood.copy())
            else:
                for k in range(1, self.img.ndim):
                    for j in range(binom[k]):
                        cell = nhood[subtuples[neighs[c]]]
                        if np.all(cell):
                            cells[k].append(hood[subtuples[neighs[c]]])
                        c += 1

        dim = self.img.ndim
        for k in range(dim, -1, -1):
            if len(cells[k]) > 0:
                break

        self.ndim = dim
        self.cells = [np.array(cells[k]) for k in range(dim+1)]
        if center:
            self.cells[0] = _centerVertices(coords)
        else:
            self.cells[0] = coords

        return self


def _pole_directions(parallels, meridians, x=0, y=1, z=2, tol=1e-10):
    
    dirs = np.zeros((2*(meridians*parallels)-meridians+2, 3), dtype=np.float64)
    idx = 1

    dirs[0, :] = np.array([0,0,0])
    dirs[0, z] = 1

    for i in range(parallels):
        theta = (i+1)*np.pi/(2*parallels)
        for j in range(meridians):
            phi = j*2*np.pi/meridians
            dirs[idx,x] = np.cos(phi)*np.sin(theta)
            dirs[idx,y] = np.sin(phi)*np.sin(theta)
            dirs[idx,z] = np.cos(theta)
            idx += 1

    for i in range(parallels-1):
        theta = (i+1)*np.pi/(2*parallels) + 0.5*np.pi
        for j in range(meridians):
            phi = j*2*np.pi/meridians
            dirs[idx,x] = np.cos(phi)*np.sin(theta)
            dirs[idx,y] = np.sin(phi)*np.sin(theta)
            dirs[idx,z] = np.cos(theta)
            idx += 1


    dirs[-1, :] = np.array([0,0,0])
    dirs[-1, z] = -1
    dirs[np.abs(dirs) < tol] = 0

    return dirs

# //////////////////////////////////////////////////////////////////////
# For the 3D cases, the argument laid out by
# M Deserno "How to generate equidistributed points on
#            the surface of a sphere"
# https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
#
# was followed to define both
# uniformly random or regular direction choice.
# //////////////////////////////////////////////////////////////////////

def _random_directions(N=50, r=1, dims=3):
    rng = np.random.default_rng()
    phi = rng.uniform(0, 2*np.pi, N)

    if dims == 2:
        x = r*np.cos(phi)
        y = r*np.sin(phi)

        return np.column_stack((x,y))

    if dims == 3:
        z = rng.uniform(-r, r, N)
        x = np.sqrt(r**2 - z**2)*np.cos(phi)
        y = np.sqrt(r**2 - z**2)*np.sin(phi)

        return np.column_stack((x,y,z))

    else:
        print("Function implemented only for 2 and 3 dimensions")

def _regular_directions(N=50, r=1, dims=3):
    if dims==2:
        eq_angles = np.linspace(0, 2*np.pi, num=N, endpoint=False)
        return np.column_stack((np.cos(eq_angles), np.sin(eq_angles)))

    if dims==3:
        dirs = np.zeros((N, 3), dtype=np.float64)
        i = 0
        a = 4*np.pi*r**2/N
        d = np.sqrt(a)
        Mtheta = np.round(np.pi/d)
        dtheta = np.pi/Mtheta
        dphi = a/dtheta
        for m in range(int(Mtheta)):
            theta = np.pi*(m + 0.5)/Mtheta
            Mphi = np.round(2*np.pi*np.sin(theta)/dphi)
            for n in range(int(Mphi)):
                phi = 2*np.pi*n/Mphi
                # sometimes we get an error due to i == N for some choices of N
                if i < N:
                    dirs[i,:] = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
                    i += 1

        return dirs
    else:
        print("Function implemented only for 2 and 3 dimensions")


# Ported from original BCF implementation in MATLAB
def shape_context(cont, n_ref=5, n_dist=5, n_theta=12, b_tangent=1, close_contour=False):
    """ Compute the shape context feature descriptor of Malik using evenly sampled reference points. 
    see, 
    https://en.wikipedia.org/wiki/Shape_context, 
    https://proceedings.neurips.cc/paper/2000/file/c44799b04a1c72e3c8593a53e8000c78-Paper.pdf
        
    Code adapted from https://github.com/ChesleyTan/bcf/blob/master/shape_context.py
    
    Parameters
    ----------
    cont : (n_pts, 2) array
        the xy coordinate contour of a binary shape 
    n_ref : int
        the number of reference or landmark points. this is chosen by even sampling.
    n_dist : int
        the number of distance bins
    n_theta : int
        the number of angular bins 
    b_tangent : bool
        If True, normalize shape context with the tangent orientation
    close_contour : bool
        If True, the input contour has been closed such that the first = last coordinate. 
    
    Returns 
    -------
    sc : (n_theta * n_dist, n_ref) array
        the shape context descriptor for each reference point 
    V : (n_ref,2)
        the coordinates of the chosen reference points on the contour 
    dis_mat : (n_pts, n_ref)
        the distance (straight line) of all contour points with respect to the reference points
    ang_mat : (n_pts, n_ref)
        the angle (straight line) of all contour points with respect to the reference points
    """
    import numpy as np

    n_pt = cont.shape[0]
    X = np.array([cont[:, 0]]).transpose()
    Y = np.array([cont[:, 1]]).transpose()
    # Set reference point
    si = np.round(np.linspace(1, n_pt, n_ref)).astype(np.int32) - 1
    V = cont[si, :]
    vx = np.array([V[:, 0]])
    vy = np.array([V[:, 1]])
    # Orientations and geodesic distances between all landmark points
    Xs = np.tile(X, (1, n_ref))
    vxs = np.tile(vx, (n_pt, 1))
    dXs = Xs - vxs
    Ys = np.tile(Y, (1, n_ref))
    vys = np.tile(vy, (n_pt, 1))
    dYs = Ys - vys
    dis_mat = np.sqrt(dXs**2 + dYs**2) # distance (straight line) + angle wrt to the reference point.
    ang_mat = np.arctan2(dYs, dXs)

    # Normalize shape context with the tangent orientation
    if b_tangent:
#        print('tangent norm')

        # this seems wrong for contour fragment.
        Xs = np.append(np.append([X[-1]], X, axis=0), [X[0]], axis=0) # central difference.
        Ys = np.append(np.append([Y[-1]], Y, axis=0), [Y[0]], axis=0) # central difference.

        gX = np.gradient(Xs, axis=0)
        gY = np.gradient(Ys, axis=0)

        if not close_contour:
            gX[1] = X[1] - X[0]
            gX[-2] = X[-1] - X[-2]
            gY[1] = Y[1] - Y[0]
            gY[-2] = Y[-1] - Y[-2]

        thetas = np.arctan2(gY, gX)
        thetas = thetas[1:-1]
        thetas = np.tile(thetas, (1, n_ref)) # tangent vectors.

        ang_mat = ang_mat - thetas # relative displacements with respect to tangent vectors.
#        ang_mat = thetas - ang_mat
        idx = np.where(ang_mat > np.pi)
        ang_mat[idx] = ang_mat[idx] - 2 * np.pi
        idx = np.where(ang_mat < -np.pi)
        ang_mat[idx] = ang_mat[idx] + 2 * np.pi

    # Compute shape context
    sc = _shape_context_core(dis_mat, ang_mat, n_dist, n_theta)
    return sc, V, dis_mat, ang_mat

def _shape_context_core(dists, angles, n_dist=10, n_theta=16):
    
    import numpy as np 
    n_pts = dists.shape[1]
    # Using log distances
    logbase = 1.5
    mean_dis = np.mean(dists.flatten())
    b = 1
    a = (logbase ** (0.75 * n_dist) - b) / mean_dis # where is this from?

#    print(a)
    dists = np.floor(np.log(a * dists + b) / np.log(logbase)) # this is logdistance.
    dists = np.maximum(dists, np.resize(1, dists.shape))
    dists = np.minimum(dists, np.resize(n_dist, dists.shape))

#    print(dists.shape, dists.min(), dists.max())

    # Preprocessing angles
    delta_ang = 2 * np.pi / n_theta # angle bins.
    angles = np.ceil((angles + np.pi) / delta_ang) #[0,2pi] bins.
    angles = np.maximum(angles, np.resize(1, angles.shape))
    angles = np.minimum(angles, np.resize(n_theta, angles.shape))

    # Shape context
    sc_hist = np.zeros((n_theta * n_dist, n_pts))
    sctmp = np.zeros((n_dist, n_theta))

    for v in range(n_pts):
        for dis in range(n_dist):
            for ang in range(n_theta):
                sctmp[dis, ang] = np.count_nonzero(np.logical_and(dists[:, v] == dis + 1, angles[:, v] == ang + 1))
        sc_hist[:, v] = sctmp.flatten(order='F')
    sc_hist /= dists.shape[0]
    return sc_hist


# =============================================================================
#     geometrical manipulation helper functions
# =============================================================================

def _resample_curve(x,y,k=1, s=0, n_samples=10):
    
    import scipy.interpolate
    
    tck, u = scipy.interpolate.splprep([x,y], k=k, s=s)
    unew = np.linspace(0, 1.00, n_samples)
    out = scipy.interpolate.splev(unew, tck) 
    
    return np.vstack(out).T

def get_img_contours(img, thresh=0, n_boundary_pts=100, return_original=False): 
    
    from skimage.measure import find_contours
    from scipy.ndimage import binary_fill_holes
    from skimage.morphology import binary_dilation
    
    binary = img > thresh
    binary = binary_fill_holes(binary)
    binary = binary_dilation(binary, np.ones((3,3), dtype=bool)) # 3x3 square footprint
    binary_cont = find_contours(binary,0)
    binary_cont = binary_cont[np.argmax([len(l) for l in binary_cont])]
    
    binary_cont_resample = _resample_curve(binary_cont[:,0], 
                                           binary_cont[:,1], k=1, s=0, n_samples=n_boundary_pts)
    
    # must be (y,x) convention 
    if return_original:
        return binary_cont_resample, binary_cont#[...,[1,0]]
    else:
        return binary_cont_resample



def _rotate_pts(pts, angle=0, center=[0,0]):

    angle_rads = angle / 180. * np.pi
    rot_matrix = np.zeros((2,2))
    rot_matrix[0,0] = np.cos(angle_rads)
    rot_matrix[0,1] = -np.sin(angle_rads)
    rot_matrix[1,0] = np.sin(angle_rads)
    rot_matrix[1,1] = np.cos(angle_rads)

#    print(rot_matrix)
    center_ = np.hstack(center)[None,:]
    pts_ = rot_matrix.dot((pts-center_).T).T + center_

    return pts_

def reorient_contour(cnt, shape):
    r""" Given a contour, reindex the contour equivalent to rotational alignment with the principal eigenvector. 
    
    Parameters
    ----------
    cnt : (n_pts,2) array 
        the yx coordinate contour of a binary shape i.e. image convention. The contour is assumed closed i.e. first = last point
    shape : (n_rows, n_cols) tuple
        the size of the image that contains the contour.
        
    Returns
    -------
    cnt_rot : (n_pts,2) array 
        the reorganised yx coordinate contour of the input contour
        
    """
    from skimage.draw import polygon
    from skimage.measure import regionprops

    canvas = np.zeros(shape, dtype=np.int32)

    rr, cc = polygon(cnt[:,0],
                     cnt[:,1], shape=shape)

    canvas[rr,cc] = 1

#    plt.figure()
#    plt.imshow(canvas)
#    plt.show()

    # use image to compute orientation ... ( more stable ?)
    orientation = regionprops(canvas)[0].orientation

    center = cnt[:-1].mean(axis=0)
    # somehow introduces bad border pts?
    cnt_rot = _rotate_pts(cnt[:-1,::-1], angle=orientation/np.pi*180., center=center[::-1])[:,::-1]

    # relabel points i.e. sorting by angle .....
    cnt_angle = np.arctan2(cnt_rot[:, 0] - center[0], cnt_rot[:,1]- center[1])
    sort = np.argsort(cnt_angle)

    # reform a contour.
    cnt_rot_ = np.vstack([cnt_rot[sort[0]:],
                          cnt_rot[:sort[0]]])

    # close the contour.
    cnt_rot = np.vstack([cnt_rot_,
                         cnt_rot_[0]])

    return cnt_rot


# =============================================================================
#   Binary mask features
# =============================================================================
def contour_to_binary(cnt, shape=None, thresh=.5, return_bbox=False):
    """ given a closed contour, get the corresponding binary image 
    
    Parameters
    ----------
    cnt : (n_pts,2)
        the yx coordinate contour of a binary shape i.e. image convention. The contour is assumed closed i.e. first = last point
    shape : None or (n_rows, n_cols) tuple
        tuple specifying the shape of the canvas (must be greater than the contour coordinates) to draw the binary
    thresh : float [0-1]
        the threshold to rebinarise the image after resizing to the given shape if provided.
    return_bbox : bool
        if True, return the bounding box of the contour in VOC format, (x1,y1,x2,y2)
    
    Returns
    -------
    blank : (y2-y1+1,x2-x1+1) or shape array 
        the binary given by the bounding box bounds of the contour or of the desired shape.
    bbox : (4,) array
        (x1,y1,x2,y2) coordinate of the lower and upper corner of the bounding box of the contour 
    """
    from skimage.draw import polygon
    import skimage.transform as sktform

    min_x = int(np.min(cnt[:,1]))
    max_x = int(np.max(cnt[:,1]))
    min_y = int(np.min(cnt[:,0]))
    max_y = int(np.max(cnt[:,0]))

    m = max_y - min_y + 1
    n = max_x - min_x + 1
    blank = np.zeros((m,n))

    rr,cc = polygon(cnt[:,0] - min_y, cnt[:,1]-min_x, shape=(m,n))
    blank[rr,cc] = 1

    if shape is not None:
        blank = sktform.resize(blank, output_shape=shape, preserve_range=True)
    blank = blank > thresh

    if return_bbox:
        bbox = np.hstack([min_x,min_y,max_x,max_y])
        return blank, bbox
    else:
        return blank


def zernike_moments(im, radius=1., degree=8, diameter=None):
    """ Compute the Zernike features for a shape given as a binary image. If radius=1, this function computes the scale and rotation invariant features as it will map the shape to within the unit disc.
    
    Parameters
    ----------
    im : (n_rows, n_cols) array 
        binary shape image
    radius : float
        the fraction of the maximum radii defined as either maximum(n_rows,n_cols)/2. if diameter=None or diameter/2.
    degree : int
        the highest degree of the Zernike polynomial to obtain coefficients for. Higher degrees capture higher frequency, smaller spatial details.
    diameter : None or float
        the maximum diameter of the shape specified in pixels. Used to compute the maximum radius of the binary shape. 
        
    Returns
    -------
    zernike_feats : (n_feats,) array
        1d array of the Zernike coefficient features 

    """
    # this implementation is scale invariant and rotation invariant. # provided the radius is being mapped to the unit disc.
    # these are essentially spectral analysis on a unit disk, the mapping is accomplished by the radius.
    import mahotas as mh
    if diameter is None:
        max_radii = np.maximum(im.shape[0], im.shape[1]) /2. # ok... this is a function of the image. # we take whatever the largest dimension / by 2
    else:
        max_radii = diameter/2. # half this number # this should be able to better capture the enclosure of the shape.
    zernike_feats = mh.features.zernike_moments(im, radius=radius*max_radii, degree=degree)
    zernike_feats = zernike_feats.ravel()
    
    return zernike_feats


def _regionprop(regprop, name, old_name):
    """ regionprops attribute by its current scikit-image name, falling back to the old name for older scikit-image """
    try:
        return getattr(regprop, name)
    except AttributeError:
        return getattr(regprop, old_name)


def extract_binary_regionprops(binary, pixel_xy_res=1.):
    """ Convenience function to extract variance region based shape features of a binary image 

    In particular, the following are computed as non-normalised measures
        - area
        - convex area
        - perimeter
        - equivalent diameter
        - major axis length
        - minor axis length

    and the following as normalised i.e. dimensionless measures 
        - 4*np.pi*area / perimeter**2 (area_perim_aspect_ratio)
        - major_axis_length/minor_axis_length (major_minor_axis_ratio) 
        - moment_eccentricity (eccentricity, e of the equivalent ellipse)
        - solidity
        - extent
        - Hu moments
        - Zernike moments

    Parameters
    ----------
    binary : (n_rows, n_cols)
        binary shape image
    pixel_xy_res : float
        the pixel to physical units conversion. for organoids, the pixel to um conversion

    Returns
    -------
    (feats_nonnorm, feats_nonnorm_labels) : ((N_feats_nonnorm,) array, (N_feats_nonnorm,) array)
        the concatenated vector of non-normalised features with associated literal names
    (feats_norm, feats_norm_labels) : ((N_feats_norm,) array, (N_feats_norm,) array)
        the concatenated vector of normalised features with associated literal names

    """
    # should we prepad the binary to avoid edges on the boundary during quantification?
    from skimage.measure import regionprops

    # binary_pad = np.pad(binary, [[5,5], [5,5]], mode='constant')
    binary_pad = binary.copy()
    regprop = regionprops((binary_pad*1).astype(np.int32))[0]
    
    # regionprops names changed in scikit-image 0.19 (old names deprecated in 0.26); use the new ones when available.
    convex_area = _regionprop(regprop, 'area_convex', 'convex_area')
    equivalent_diameter = _regionprop(regprop, 'equivalent_diameter_area', 'equivalent_diameter')
    major_axis_length = _regionprop(regprop, 'axis_major_length', 'major_axis_length')
    minor_axis_length = _regionprop(regprop, 'axis_minor_length', 'minor_axis_length')

    feats_nonnorm = np.hstack([ regprop.area * (pixel_xy_res)**2.,
                                convex_area * (pixel_xy_res)**2., # area is squared.
                                regprop.perimeter * (pixel_xy_res),
                                # regprop.perimeter_crofton * (pixel_xy_res),
                                equivalent_diameter * pixel_xy_res,
                                major_axis_length * pixel_xy_res,
                                minor_axis_length * pixel_xy_res])

    feats_nonnorm_labels = np.hstack(['area',
                                      'convex_area',
                                      'perimeter',
                                      # 'crofton_perimeter', # does this exist?
                                      'equivalent_diameter',
                                      'major_axis_length',
                                      'minor_axis_length'])

    moments_hu = regprop.moments_hu

    # this doesn't require scaling, scale invariant measures. radius here is the multiplier to scale radius which by default is half the binary size.
    moments_zernike = zernike_moments(binary_pad, radius=1, degree=8, diameter=equivalent_diameter) # default uses the image centre., radius here is a multiplier of the image only.

    hu_moment_labels = np.hstack(['hu_moments_%s' %(str(h_ii+1)) for h_ii in range(len(moments_hu))])
    zernike_moment_labels = np.hstack(['zernike_moments_%s' %(str(h_ii+1)) for h_ii in range(len(moments_zernike))])

    feats_norm = np.hstack([4*np.pi*regprop.area/(regprop.perimeter**2.), # dimensionless
                            # np.abs(regprop.major_axis_length) / (np.abs(regprop.minor_axis_length)+1.), # dimensionless
                            major_axis_length / (minor_axis_length+1.), # dimensionless
                            regprop.eccentricity,
                            regprop.solidity,
                            regprop.extent,
                            moments_hu, # this is invariant and unitless.
                            moments_zernike]) # We need to find out how best to set this !!!

    feats_norm_labels = np.hstack(['area_perim_aspect_ratio',
                                    'major_minor_axis_ratio',
                                    'moment_eccentricity',
                                    'solidity',
                                    'extent',
                                    hu_moment_labels,
                                    zernike_moment_labels])

    return (feats_nonnorm, feats_nonnorm_labels), (feats_norm, feats_norm_labels)


# =============================================================================
#   Contour curvature features
# =============================================================================
def curvature_splines(x, y, k=4, error=0.1):
    """Calculate the signed curvature of a 2D curve at each point using interpolating splines.
    
    Parameters
    ----------
    x : (n_points,) array 
        x-coordinate of the contour 
    y : (n_points,) array 
        y-coordinate of the contour 
    k : int
        order of the interpolating spline
    error : float
        The admisible error when interpolating the splines
        
    Returns
    -------
    [x_, y_] : [(n_points,) array, (n_points,) array]
        the spline evaluated (and smoothened) x and y coordinate at each point of a 2D curve for which curvature is evaluated at
    [x_prime, y_prime] : [(n_points,) array, (n_points,) array]
        the 1st derivative of the x and y coordinate at each point of a 2D curve
    curvature : (n_points,) array 
        the signed curvature of a 2D curve at each point 
    """
    from scipy.interpolate import UnivariateSpline
    # handle list of complex case
    if y is None:
        x, y = x.real, x.imag

    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=k, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=k, w=1 / np.sqrt(std))

    x_ = fx(t)
    y_ = fy(t)

    x_prime = fx.derivative(1)(t)
    x_prime_prime = fx.derivative(2)(t)
    y_prime = fy.derivative(1)(t)
    y_prime_prime = fy.derivative(2)(t)
    curvature = (x_prime * y_prime_prime - y_prime* x_prime_prime) / np.power(x_prime** 2 + y_prime** 2, 3. / 2)
#    return [x_, y_], [xˈ, yˈ], curvature
    return [x_, y_], [x_prime, y_prime], curvature


# =============================================================================
#   Fourier elliptic feats
# =============================================================================
def compute_fft_boundary_feats(cont, pixel_xy_res=1., remove_closing_point=False):
    """ Compute the centroid distance Fourier feature descriptor of a contour

    Parameters
    ----------
    cont : (n_pts, 2) array
        the xy contour coordinates of the shape 
    pixel_xy_res : float
        the pixel to physical units conversion. for organoids, the pixel to um conversion
    remove_closing_point : bool
        if True and the contour is closed (first = last point), drop the repeated last point before the FFT. The
        magnitudes are then exactly invariant to rotation and to the start point of the contour; with the repeated point
        they change when the contour starts at a different point (e.g. after reorient_contour of a rotated object).

    Returns
    -------
    feats : (n_feats,) array
        the Fourier features descriptor

    """
    feats = []

    for ccc in cont:

        # if imshape is not None:
        #     scale = np.sqrt(np.product(imshape)) # gives a relative context in the image.
        # decenter.
        ccc = ccc - ccc[:-1].mean(axis=0)[None,:]
        # if imshape is not None:
        #     # idea is not to give scale invariance.
            # ccc = ccc / (float(scale) + 1.) # this is proportional to the image size..... but this is not an absolute scale..... -> should use a global scaling....
        ccc  = ccc * pixel_xy_res # translate to physical units.
        if remove_closing_point and len(ccc) > 1 and np.allclose(ccc[0], ccc[-1]):
            ccc = ccc[:-1] # each point once: a periodic sequence, so |FFT| is invariant to the start point
        ccc_xy = ccc[...,0] + 1j*ccc[...,1]
        ccc_xy = np.fft.fft(ccc_xy)

        fft_xy = np.abs(ccc_xy) # take the absolut values for rotation invariance.
#        fft_xy = fft_xy/np.max(fft_xy)
        feats.append(fft_xy)

    feats = np.vstack(feats)

    return feats


def compute_boundary_morphology_features(boundaries, imshape, 
                                         all_feats_compute=True,
                                         contour_curve_feats=True,
                                         curve_order=4, curve_err=1.,
                                         geom_features=True,
                                         morph_features=True,
                                         fourier_features=True,
                                         ect_features=True,
                                         shape_context_features=True,
                                         norm_scm=True,
                                         n_ref_pts_scm=5,
                                         pixel_xy_res=1.,
                                         timesample=None,
                                         fourier_remove_closing_point=False):
    """ Master function for extracting all shape-based features described in the SAM paper.
    
    Parameters
    ----------
    boundaries : (n_organoids, n_frames, n_contour_pts, 2) array
        the set of all tracked organoid contour tracks
    imshape : (n_rows, n_cols) tuple
        size of the video frame 
    all_feats_compute : bool
        if True, compute all the shape features described in the SAM paper. This bool flag overrides all other options. If False, then the specific feature sets are computed according to whether their particular boolean flag is set to True.
    contour_curve_feats : bool
        if True, compute the morphological features set based on the contour curvature 
    curve_order : int 
        the polynomial order of the interpolating spline used for computing line curvature
    curve_err : float 
        The allowed error in the spline fitting for computing 2D line curvature. Controls the smoothness of the fitted spline. The larger the error, the more smooth the fitting and curvature value variation. 
    geom_features : bool
        if True, compute the morphological features set based on the contour representation including centroidal distances and chordal distances histogram 
    morph_features : bool
        if True, compute the morphological features set which comprises all shape features derivable from the binary shape area including area, perimeter, Hu moments, Zernike features
    fourier_features : bool
        if True, compute the Fourier shape features
    ect_features : bool
        if True, compute the Euler characteristic curve features. 
    shape_context_features : bool
        if True, compute the shape context features 
    norm_scm : bool
        if True, apply L2 normalisation to the flattened shape context descriptor in the manner of SIFT.  
    n_ref_pts_scm : int
        the number of reference contour points to use for shape context descriptor computation 
    pixel_xy_res : float
        the pixel to physical units conversion. for organoids, the pixel to um conversion
    timesample : None or int
        if specified, compute the feature every timesample # of frames. This can be done when in successive frames, the variation does not vary a lot and one wishes to speed up computation or save on memory
    fourier_remove_closing_point : bool
        if True, the Fourier features drop the repeated closing point of the contour, which makes them exactly invariant
        to rotation and to the contour start point (there is then one Fourier feature less). False (default) keeps the
        original features.
    

    Returns 
    -------
    out_array : (n_organoids, n_frames, n_features) array
        the concatenated shape features for every organoid in every timepoint
    out_feat_labels : (n_features,) str array
        the names of each feature
    out_feat_labels_norm : (n_features,) str array
        a boolean vector where 1 or True means the feature is normalised i.e. dimensionless. 

    """
    # =============================================================================
    #   Given boundaries produced from the organoid tracker, compute shape morphology and boundary features of interest.
    #   give some optional extractions.
    # =============================================================================
    # boundaries should be at least (n_orgs, timepoints, discretisation, 2)

    import numpy as np
#    import time
    # import shape_context_module as scm # this has been integrated into here.
#    from sklearn.pairwise.metrics import pairwise_distances
    from sklearn.metrics.pairwise import pairwise_distances
    # import demeter.euler as euler
    # import demeter.directions as dirs
    from tqdm import tqdm

#    if imshape is not None:
#        scale_factor = np.sqrt(imshape[0] *imshape[1])
    out = []
    out_feat_labels = []
    out_feat_labels_norm = []

    if timesample is None:
        timesample = 1

#    print(scale_factor)
    for b_ii in tqdm(range(len(boundaries))):

        boundary_ii = boundaries[b_ii]
        boundary_ii_descriptors = []
#        boundary_ii_descriptors_labels = []
#        boundary_ii_descriptors_norm = []
        """
        no longer need the ect computation.
        """

        for tt in range(0,len(boundary_ii),timesample):

            boundary_ii_tt = boundary_ii[tt]

            if np.isnan(boundary_ii_tt[0][0]) == True:
                boundary_ii_descriptors.append([np.nan]) # don't compute anything.
            else:

                # what should be the standardised way to rotate this ...
                boundary_ii_tt = reorient_contour(boundary_ii_tt, imshape) # is this doing it correct?

                # placeholder to get all the relevant features.
                all_feats = []
                all_feats_labels = []
                all_feats_norm_bool = []

                """
                features 1: contour curve measurement features.
                """
                if all_feats_compute==True or (all_feats_compute==False and contour_curve_feats==True):

                    # this measures the line curvature, we pre-scale coordinates by the pixel resolution
                    smooth_curve, deriv_curve, curvature = curvature_splines(boundary_ii_tt[:,1]*pixel_xy_res,
                                                                             y=boundary_ii_tt[:,0]*pixel_xy_res,
                                                                             k=curve_order,
                                                                             error=curve_err*pixel_xy_res) # choose how much error. -> dictates how smooth the curvature computation.

                    # characterisation of the distribution of the curvature using the different moments. -> there must be better ways? (incorporate euler transform?)
                    curvature_feats = np.hstack([ np.max(curvature),
                                                  np.min(curvature),
                                                  np.mean(curvature),
                                                  np.mean(np.abs(curvature)),
                                                  np.std(curvature),
                                                  np.mean((curvature-np.mean(curvature))**3) / (np.std(curvature)**1.5 + 1e-8),
                                                  np.mean((curvature-np.mean(curvature))**4) / (np.std(curvature)**4 + 1e-8)])

                    curvature_feats_labels = ['max_curvature',
                                              'min_curvature',
                                              'mean_curvature',
                                              'mean_abs_curvature',
                                              'std_curvature',
                                              'mean_skew_curvature',
                                              'mean_kurtosis_curvature']
                    curvature_feats_labels = np.hstack(curvature_feats_labels)

                    all_feats.append(curvature_feats)
                    all_feats_labels.append(curvature_feats_labels)
                    all_feats_norm_bool.append(np.hstack([False]*len(curvature_feats)))

                """
                features 2: geometrical curve features
                """
                if all_feats_compute==True or (all_feats_compute==False and geom_features==True):

                    # scale by the given pixel resolution.
                    cnt_centroid_distances = np.linalg.norm(boundary_ii_tt[:-1]*pixel_xy_res - boundary_ii_tt[:-1].mean(axis=0)[None,:]*pixel_xy_res, axis=-1)

                    max_cell_radius = np.max(cnt_centroid_distances)
                    cell_radius_ratio = np.max(cnt_centroid_distances) / (np.min(cnt_centroid_distances) + 1)
                    mean_cell_radius = np.mean(cnt_centroid_distances)
                    cv_cell_radius = np.std(cnt_centroid_distances) / mean_cell_radius

                    chordal_distances = pairwise_distances(boundary_ii_tt) # euclidean norm in pixels. !
                    max_span = chordal_distances.max()

                    # chordal histogram distribution (only size of 8?)
                    chordal_hist, _ = np.histogram(chordal_distances[np.triu_indices(len(chordal_distances))]/max_span, range=(0,1), bins=8)
                    chordal_hist = chordal_hist/float(np.sum(chordal_hist)) # normalize the histogram.

                    geom_feats_nonnorm = np.hstack([max_cell_radius,
                                                    mean_cell_radius,
                                                    cv_cell_radius,
                                                    max_span])

                    geom_feats_nonnorm_labels = np.hstack(['max_centroid_distance',
                                                           'mean_centroid_distance',
                                                           'std_mean_centroid_distance_ratio',
                                                           'max_chordal_distance'])

                    all_feats.append(geom_feats_nonnorm)
                    all_feats_labels.append(geom_feats_nonnorm_labels)
                    all_feats_norm_bool.append(np.hstack([False]*len(geom_feats_nonnorm)))


                    geom_feats_norm = np.hstack([cell_radius_ratio,
                                                 chordal_hist])
                    geom_feats_norm_labels = np.hstack(['max_min_centroid_distance_ratio',
                                                        np.hstack(['chordal_hist_bin_%s'%(str(bin_i+1)) for bin_i in np.arange(len(chordal_hist))])])

                    all_feats.append(geom_feats_norm)
                    all_feats_labels.append(geom_feats_norm_labels)
                    all_feats_norm_bool.append(np.hstack([True]*len(geom_feats_norm)))

                """
                feature set 3: morphological features
                """
                if all_feats_compute==True or (all_feats_compute==False and morph_features==True):

                    cnt_binary = contour_to_binary(boundary_ii_tt, shape=None) # shape = None ensure tight boundary box.
                    cnt_binary_feats_nonnorm, cnt_binary_feats_norm = extract_binary_regionprops(cnt_binary, pixel_xy_res=pixel_xy_res) # don't scale the boundary but apply to the computations

                    cnt_binary_feats_nonnorm, cnt_binary_feats_nonnorm_labels = cnt_binary_feats_nonnorm
                    cnt_binary_feats_norm, cnt_binary_feats_norm_labels = cnt_binary_feats_norm

                    all_feats.append(cnt_binary_feats_nonnorm)
                    all_feats_labels.append(cnt_binary_feats_nonnorm_labels)
                    all_feats_norm_bool.append(np.hstack([False]*len(cnt_binary_feats_nonnorm)))

                    all_feats.append(cnt_binary_feats_norm)
                    all_feats_labels.append(cnt_binary_feats_norm_labels)
                    all_feats_norm_bool.append(np.hstack([True]*len(cnt_binary_feats_norm_labels)))

                """
                feature set 4: Fourier features.
                """
                if all_feats_compute==True or (all_feats_compute==False and fourier_features==True):

            #        cnt_fourier = compute_elliptic_feats([cnt], vid[0].shape, normalize=True).ravel()
                    cnt_fourier = compute_fft_boundary_feats([boundary_ii_tt], pixel_xy_res=pixel_xy_res,
                                                             remove_closing_point=fourier_remove_closing_point).ravel()
            #        cnt_fourier = cnt_fourier[1:len(cnt)//2]
                    cnt_fourier = np.abs(cnt_fourier[1:len(cnt_fourier)//2]) # 1st is typically a DC offset, keep only magnitude information!.
                    cnt_fourier = np.abs(cnt_fourier)**(0.2) #* np.sign(cnt_fourier) # power normalisation? of the fourier features
    #                cnt_fourier = cnt_fourier / (cnt_fourier.max() + 1e-8)
                    cnt_fourier = cnt_fourier / (np.linalg.norm(cnt_fourier) + 1e-8) # normalization provides scaling anyway...

                    cnt_fourier_labels = np.hstack(['Fourier_%s' %(str(f_ii+1)) for f_ii in range(len(cnt_fourier))])

                    all_feats.append(cnt_fourier)
                    all_feats_labels.append(cnt_fourier_labels)
                    all_feats_norm_bool.append(np.hstack([True]*len(cnt_fourier)))


                """
                feature set 5: Euler characteristic features. 
                """
                if all_feats_compute==True or (all_feats_compute==False and ect_features==True):

                    # curve_comp = euler.CubicalComplex(cnt_binary).complexify()
                    curve_comp = CubicalComplex(cnt_binary).complexify()
                    # specify directions for integration.
                    # circle_dirs = dirs.regular_directions(36, dims=curve_comp.ndim) # how many directions.
                    circle_dirs = _regular_directions(36, dims=curve_comp.ndim)

                    # derive the
                    n_thresholds = 32
                    ect_feat = curve_comp.ECT(circle_dirs, T=n_thresholds)
                    ect_feat_labels = np.hstack(['ECT_%s' %(str(f_ii+1)) for f_ii in range(len(ect_feat))])

                    all_feats.append(ect_feat)
                    all_feats_labels.append(ect_feat_labels)
                    all_feats_norm_bool.append(np.hstack([True]*len(ect_feat)))

                """
                feature set 6: Shape context features
                """
                if all_feats_compute==True or (all_feats_compute==False and shape_context_features==True):

                    # do we expose these settings?
                    scm_feats = shape_context(boundary_ii_tt*pixel_xy_res,
                                                n_ref=n_ref_pts_scm,
                                                n_dist=5,
                                                n_theta=12, b_tangent=1)[0].flatten(order='F')

                    if norm_scm:
                        scm_feats = scm_feats / (np.linalg.norm(scm_feats, ord=2) + 1e-8) # l-2 normalisation of features.

                    scm_feat_labels = np.hstack(['Shape_Context_%s' %(str(f_ii+1)) for f_ii in range(len(scm_feats))])

                    all_feats.append(scm_feats)
                    all_feats_labels.append(scm_feat_labels)
                    all_feats_norm_bool.append(np.hstack([True]*len(scm_feats)))


                all_feats = np.hstack(all_feats)
                all_feats_labels = np.hstack(all_feats_labels)
                all_feats_norm_bool = np.hstack(all_feats_norm_bool)


                boundary_ii_descriptors.append(all_feats)

                # overwrite the placeholder and we are done.
                out_feat_labels = all_feats_labels
                out_feat_labels_norm = all_feats_norm_bool


        out.append(boundary_ii_descriptors)
#        out_labels.append(boundary_ii_descriptors_labels) # to be fair this only needs to be computed once.
#        out_norm_bool.append(boundary_ii_descriptors_norm)

    """
    reprocessing to get this in a standard format.
    """
    out = np.array(out, dtype=object)

    # make this a regular array for analysis. 
    n_org, n_time = boundaries.shape[:2]

    if timesample is not None:
        n_time = len(np.arange(0, n_time, timesample)) # as we are subsampling... 
    n_feat_size = len(out_feat_labels)
    
    out_array = np.zeros((n_org, n_time, n_feat_size))

    for ii in range(n_org):
        for jj in range(n_time):
            val = out[ii,jj].copy()
            if len(val) == 1: 
                out_array[ii,jj] = np.nan
            else:
                out_array[ii,jj] = val

    return out_array, out_feat_labels, out_feat_labels_norm

# =============================================================================
#   Do Some textural analysis (intensity distribution pattern analysis.)
# =============================================================================


from scipy import signal

class DsiftExtractor:
    r"""
    The class that does dense sift feature extractor. See https://github.com/Yangqing/dsift-python
    Sample Usage:
        extractor = DsiftExtractor(gridSpacing,patchSize,[optional params])

        feaArr,positions = extractor.process_image(Image)
    
    Reference : 
        Y. Jia and T. Darrell. "Heavy-tailed Distances for Gradient Based Image Descriptors". NIPS 2011.
        
        Lowe, David G. "Object recognition from local scale-invariant features." Proceedings of the seventh IEEE international conference on computer vision. Vol. 2. Ieee, 1999.
    """
    def __init__(self, gridSpacing, patchSize,
                 nrml_thres = 1.0,\
                 sigma_edge = 0.8,\
                 sift_thres = 0.2,
                 Nangles = 8,
                 Nbins = 4,
                 alpha = 9.0):
        '''
        gridSpacing: 
            the spacing for sampling dense descriptors
        patchSize: int
            the size of each sift patch
        nrml_thres: scalar
            low contrast normalization threshold
        sigma_edge: scalar
            the standard deviation for the gaussian smoothing before computing the gradient
        sift_thres: scalar
            sift thresholding (0.2 works well based on Lowe's SIFT paper)
        '''
        self.Nangles = Nangles
        self.Nbins = Nbins
        self.alpha = alpha
        self.Nsamples = Nbins**2
        self.angles = np.array(range(Nangles))*2.0*np.pi/Nangles # the thresholds of the angle histogram [0,2pi]
        
        self.gS = gridSpacing
        self.pS = patchSize
        self.nrml_thres = nrml_thres
        self.sigma = sigma_edge
        self.sift_thres = sift_thres
        # compute the weight contribution map
        sample_res = self.pS / np.double(Nbins) # spatial resolution. 
        sample_p = np.array(range(self.pS))
        sample_ph, sample_pw = np.meshgrid(sample_p,sample_p) # this is 32 x 32 (image squared?)
        sample_ph.resize(sample_ph.size)
        sample_pw.resize(sample_pw.size)
        bincenter = np.array(range(1,Nbins*2,2)) / 2.0 / Nbins * self.pS - 0.5 
#        print(bincenter)
        bincenter_h, bincenter_w = np.meshgrid(bincenter,bincenter)
#        print(bincenter_h)
#        print(bincenter_w)
        bincenter_h.resize((bincenter_h.size,1))
        bincenter_w.resize((bincenter_w.size,1))
        dist_ph = abs(sample_ph - bincenter_h)
        dist_pw = abs(sample_pw - bincenter_w)
        weights_h = dist_ph / sample_res
        weights_w = dist_pw / sample_res
        weights_h = (1-weights_h) * (weights_h <= 1)
        weights_w = (1-weights_w) * (weights_w <= 1)
        # weights is the contribution of each pixel to the corresponding bin center
        self.weights = weights_h * weights_w

    def gen_dgauss(self,sigma):
        r'''
        generates a derivative of Gaussian filter with the same specified :math:`\sigma` in both the X and Y
        directions.
        '''
        fwid = np.int32(2*np.ceil(sigma))
        G = np.array(range(-fwid,fwid+1))**2
        G = G.reshape((G.size,1)) + G
        G = np.exp(- G / 2.0 / sigma / sigma)
        G /= np.sum(G)
        GH,GW = np.gradient(G)
        GH *= 2.0/np.sum(np.abs(GH))
        GW *= 2.0/np.sum(np.abs(GW))
        return GH,GW
        
    def process_image(self, image, positionNormalize = True,\
                       verbose = True):
        '''
        processes a single image, return the locations
        and the values of detected SIFT features.
        image: a M*N image which is a numpy 2D array. If you pass a color image, it will automatically be convertedto a grayscale image.
        positionNormalize: whether to normalize the positions to [0,1]. If False, the pixel-based positions of the top-right position of the patches is returned.

        Return values:
        feaArr: the feature array, each row is a feature
        positions: the positions of the features
        '''

        image = image.astype(np.float32)
        if image.ndim == 3:
            # we do not deal with color images.
            image = np.mean(image,axis=2)
        # compute the grids
        H,W = image.shape
        gS = self.gS
        pS = self.pS
        remH = np.mod(H-pS, gS)
        remW = np.mod(W-pS, gS)
        offsetH = remH//2
        offsetW = remW//2
        gridH,gridW = np.meshgrid(range(offsetH,H-pS+1,gS), range(offsetW,W-pS+1,gS))
        gridH = gridH.flatten()
        gridW = gridW.flatten()
        if verbose:
            print ('Image: w {}, h {}, gs {}, ps {}, nFea {}'.\
                    format(W,H,gS,pS,gridH.size))
        feaArr = self.calculate_sift_grid(image,gridH,gridW) # this is the heavy lifting.
        feaArr = self.normalize_sift(feaArr)
        if positionNormalize:
            positions = np.vstack((gridH / np.double(H), gridW / np.double(W)))
        else:
            positions = np.vstack((gridH, gridW))
        return feaArr, positions

    def calculate_sift_grid(self,image,gridH,gridW):
        '''
        This function calculates the unnormalized sift features at equidistantly spaced control points in the image as specified by the number in height (gridH) and in width (gridW)
        It is called by process_image().
        '''
        from scipy import signal
        
        H,W = image.shape
        Npatches = gridH.size
        feaArr = np.zeros((Npatches, self.Nsamples*self.Nangles)) # Nsamples is the number of grid positions of the image being taken. # number of angles 
        
        # calculate gradient
        GH,GW = self.gen_dgauss(self.sigma) # this is the gradient filter for the image. 
        
        IH = signal.convolve2d(image,GH,mode='same')
        IW = signal.convolve2d(image,GW,mode='same')
        Imag = np.sqrt(IH**2+IW**2)
        Itheta = np.arctan2(IH,IW)
        Iorient = np.zeros((self.Nangles,H,W))
        
        for i in range(self.Nangles):
            Iorient[i] = Imag * np.maximum(np.cos(Itheta - self.angles[i])**self.alpha,0) # either increment the count or not. 
            #pyplot.imshow(Iorient[i])
            #pyplot.show()
        for i in range(Npatches):
            currFeature = np.zeros((self.Nangles,self.Nsamples))
            for j in range(self.Nangles):
                # this is the gaussian spatial weights in each cell. 
                currFeature[j] = np.dot(self.weights,\
                        Iorient[j,gridH[i]:gridH[i]+self.pS, gridW[i]:gridW[i]+self.pS].flatten())
            feaArr[i] = currFeature.flatten()
        return feaArr

    def normalize_sift(self,feaArr):
        '''
        This function does sift feature normalization
        following David Lowe's definition (normalize length ->
        thresholding at 0.2 -> renormalize length)
        '''
        siftlen = np.sqrt(np.sum(feaArr**2,axis=1)) # this is the L2
        hcontrast = (siftlen >= self.nrml_thres)
        siftlen[siftlen < self.nrml_thres] = self.nrml_thres
        # normalize with contrast thresholding
        feaArr /= siftlen.reshape((siftlen.size,1))
        # suppress large gradients
        feaArr[feaArr>self.sift_thres] = self.sift_thres
        # renormalize high-contrast ones
        feaArr[hcontrast] /= np.sqrt(np.sum(feaArr[hcontrast]**2,axis=1)).\
                reshape((feaArr[hcontrast].shape[0],1))
        return feaArr

class SingleSiftExtractor(DsiftExtractor):
    '''
    The simple wrapper class that does feature extraction, treating
    the whole image as a local image patch.
    '''
    def __init__(self, patchSize,
                 nrml_thres = 1.0,\
                 sigma_edge = 0.8,\
                 sift_thres = 0.2,
                 Nangles = 8,
                 Nbins = 4,
                 alpha = 9.0):
        # simply call the super class __init__ with a large gridSpace
        DsiftExtractor.__init__(self, patchSize, patchSize, nrml_thres, sigma_edge, sift_thres)   
    
    def process_image(self, image):
        return DsiftExtractor.process_image(self, image, False, False)[0]


def sobel_frame(im):
    """ Compute the Sobel gradient image of a grayscale image

    Parameters
    ----------
    im : (n_rows, n_cols) image
        input grayscale image

    Returns
    -------
    sobel_im_ : (n_rows, n_cols) image
        output rescaled sobel gradient image with pixel intensity in [0,255]

    """
    from skimage.exposure import rescale_intensity
    from skimage.filters import sobel

    im_ = rescale_intensity(im)
    sobel_im_ = np.uint8(255.*rescale_intensity(sobel(im_)))

    return sobel_im_


def _fill_outside_mask_nearest(im, mask):
    """ Replace every pixel outside the binary mask with the value of its nearest pixel inside the mask. This removes the artificial step edge at the mask boundary so that gradient-based descriptors do not respond to the shape outline.

    Parameters
    ----------
    im : (n_rows, n_cols) image
        input image
    mask : (n_rows, n_cols) binary image
        pixels to keep

    Returns
    -------
    im_fill : (n_rows, n_cols) image
        image with outside-mask pixels filled by nearest-neighbour extrapolation
    """
    mask = mask > 0
    if mask.sum() == 0 or mask.all():
        return im.copy()
    inds = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return im[inds[0], inds[1]]


def _masked_sift_descriptor(im, mask, resize='cap', siftshape=(64,64), n_rings=4, n_sectors=4):
    """ Compute the global SIFT descriptor using only the image gradients within the binary mask, so the descriptor captures internal texture and not the shape outline.

    Outside-mask pixels are filled by nearest-neighbour extrapolation to remove the boundary step edge, and SIFT orientation histograms are pooled only over mask pixels. Instead of the fixed 4x4 square grid of standard SIFT (whose empty cells would encode the shape), the spatial cells adapt to the mask: n_rings equal-area rings of distance from the boundary x n_sectors equal-angle sectors about the mask centroid. Each cell is the mean (not sum) over its pixels, so cell area does not encode shape either.

    Parameters
    ----------
    im : (n_rows, n_cols) image
        cropped image patch
    mask : (n_rows, n_cols) binary image
        binary shape mask of the same size as im
    resize : 'cap', 'square' or 'native'
        how the patch is rescaled before computing gradients. 'cap': downscale, preserving the aspect ratio, so the longest axis is at most max(siftshape), smaller patches are not upscaled (fast, and texture scale only changes for objects larger than the cap). 'square': resize to siftshape as standard SIFT does (the texture is stretched by the bounding box aspect ratio and rescaled with object size, which leaks shape). 'native': no rescaling (texture scale fully preserved, slowest for large objects).
    siftshape : (size,size)
        target size for resize='square'; max(siftshape) is the cap for resize='cap'
    n_rings : int
        number of equal-area rings, from the boundary inwards
    n_sectors : int
        number of equal-angle sectors about the mask centroid, starting from the +x axis

    Returns
    -------
    sift_texture : (n_angles*n_rings*n_sectors,) array
        unit-normalised SIFT descriptor, 128 dimensional by default as for standard SIFT
    """
    from scipy import signal
    import skimage.transform as sktform

    mask_rs = mask > 0
    im_fill = _fill_outside_mask_nearest(im.astype(np.float64), mask_rs)

    if resize == 'square':
        out_shape = tuple(siftshape)
    elif resize == 'cap':
        scale = min(1., max(siftshape) / float(max(mask_rs.shape)))
        out_shape = tuple(max(1, int(round(s*scale))) for s in mask_rs.shape)
    elif resize == 'native':
        out_shape = mask_rs.shape
    else:
        raise ValueError("resize must be 'cap', 'square' or 'native'")
    if out_shape != mask_rs.shape:
        im_fill = sktform.resize(im_fill, out_shape, preserve_range=True)
        mask_rs = sktform.resize(mask_rs.astype(np.float64), out_shape, order=0, preserve_range=True, anti_aliasing=False) > 0

    siftextractor = SingleSiftExtractor(64) # for the standard SIFT gradient filter, orientation binning and threshold parameters.
    n_cells = n_rings * n_sectors
    if mask_rs.sum() == 0:
        return np.zeros(siftextractor.Nangles * n_cells)

    # oriented gradient energies, as in DsiftExtractor.calculate_sift_grid. Symmetric padding avoids spurious edges at the patch border.
    GH, GW = siftextractor.gen_dgauss(siftextractor.sigma)
    IH = signal.convolve2d(im_fill, GH, mode='same', boundary='symm')
    IW = signal.convolve2d(im_fill, GW, mode='same', boundary='symm')
    Imag = np.sqrt(IH**2+IW**2)[mask_rs]
    Itheta = np.arctan2(IH, IW)[mask_rs]
    Iorient = np.array([Imag * np.maximum(np.cos(Itheta - a), 0)**siftextractor.alpha for a in siftextractor.angles]) # (n_angles, n_mask_pixels)

    # shape-adapted spatial cells: equal-area depth rings x equal-angle sectors.
    depth = ndimage.distance_transform_edt(mask_rs)[mask_rs]
    ring = np.digitize(depth, np.quantile(depth, np.linspace(0, 1, n_rings+1)[1:-1]))
    yy, xx = np.nonzero(mask_rs)
    angle = np.arctan2(yy - yy.mean(), xx - xx.mean())
    sector = np.floor((angle + np.pi) / (2*np.pi) * n_sectors).astype(np.int64) % n_sectors
    cell = ring * n_sectors + sector

    counts = np.bincount(cell, minlength=n_cells).astype(np.float64)
    sift_texture = np.array([np.bincount(cell, weights=o, minlength=n_cells) for o in Iorient]) # (n_angles, n_cells)
    sift_texture = sift_texture / np.maximum(counts, 1)[None,:] # mean pooling, empty cells stay 0.
    sift_texture = sift_texture.ravel()

    # a flat interior has only floating point gradient noise, which normalisation would amplify to unit length.
    if Imag.max() <= 1e-6 * (np.abs(im_fill[mask_rs]).max() + 1e-12):
        return np.zeros_like(sift_texture)

    # Lowe normalisation: unit length -> clip large gradients -> unit length.
    sift_texture = sift_texture / (np.linalg.norm(sift_texture) + 1e-8)
    sift_texture = np.minimum(sift_texture, siftextractor.sift_thres)
    sift_texture = sift_texture / (np.linalg.norm(sift_texture) + 1e-8)

    return sift_texture


def _masked_haralick(im, mask, distance):
    """ Compute the 13 Haralick features from a gray-level co-occurrence matrix built only from pixel pairs where both pixels lie within the binary mask.

    Parameters
    ----------
    im : (n_rows, n_cols) integer image
        cropped image patch
    mask : (n_rows, n_cols) binary image
        binary shape mask of the same size as im
    distance : int
        co-occurrence distance in pixels. If the mask is too thin to have pixel pairs at this distance, the distance is halved until it does.

    Returns
    -------
    feats : (13,) array
        mean Haralick features over the 4 directions. NaN if no valid pixel pairs exist within the mask even at distance 1.
    """
    import mahotas as mh

    # shift gray levels up by 1 so 0 is reserved for outside-mask pixels and ignored by mahotas.
    im_q = im.astype(np.int32) + 1
    im_q[mask==0] = 0

    distance = int(distance)
    while distance >= 1:
        try:
            feats = mh.features.haralick(im_q,
                                         ignore_zeros=True,
                                         preserve_haralick_bug=False,
                                         compute_14th_feature=False,
                                         return_mean=True,
                                         return_mean_ptp=False,
                                         use_x_minus_y_variance=False,
                                         distance=distance)
            feats = np.asarray(feats, dtype=np.float64)
            feats[5] = feats[5] - 2 # undo the +1 gray-level shift in sum_average (sum of two gray levels).
            return feats
        except ValueError:
            # mahotas raises if a direction has an empty co-occurrence matrix i.e. the mask is too thin for this distance.
            distance = distance // 2

    return np.full(13, np.nan)


##### functions for creating a custom gridding scheme given the boundary masks. 
def _cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
    
def _pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def _draw_polygon_mask(xy, shape):
    
    from skimage.draw import polygon
    img = np.zeros(shape, dtype=bool)
    
    x_coords = np.clip(xy[:,0].astype(np.int32), 0, shape[1]-1)
    y_coords = np.clip(xy[:,1].astype(np.int32), 0, shape[0]-1)
    rr, cc = polygon(y_coords,
                     x_coords)
    img[rr, cc] = 1
    
    return img

def _tile_uniform_radial_windows_boundary(imsize, n_r, n_theta, boundary_line, 
                                             center=None, 
                                             bound_r=True, 
                                             zero_angle=None, 
                                             return_grid_pts=False):
    
    """
    given a contour we create radial masks of given radius away from the center.  the contour given must the furthest, the interpolated boundaries go 'inwards' from this towards the given centroid point
    
        contour is (y,x) image convention 
    """
    # import pylab as plt 
    import numpy as np 
    
    m, n = imsize
    
    if center is None:
        center = np.nanmean(boundary_line[:-1], axis=0)
    
    XX, YY = np.meshgrid(range(n), range(m))
    # r2 = (XX - center[1])**2  + (YY - center[0])**2
    # r = np.sqrt(r2)
    theta = np.arctan2(YY-center[0],XX-center[1])
    
    """
    construct contour lines that partition the space. 
    """
    edge_line_polar = np.vstack(_cart2pol(boundary_line[:,1] - center[1], boundary_line[:,0] - center[0])).T
    edge_line_central = np.vstack(_pol2cart(edge_line_polar[:,0], edge_line_polar[:,1])).T # back to cartesian. 
    
    # derive the other boundary lines
    contour_r_internal_polar = np.array([np.linspace(0, l, n_r+1) for l in edge_line_polar[:,0]]).T
    
    contour_r_internal_lines = [np.vstack(_pol2cart(l, edge_line_polar[:,1])).T for l in contour_r_internal_polar][:-1]
   
    all_dist_lines = contour_r_internal_lines + [edge_line_central] 
    all_dist_lines = [np.vstack([ll[:,0] + center[1], ll[:,1] + center[0]]).T  for ll in all_dist_lines]
    all_dist_masks = [_draw_polygon_mask(ll, imsize) for ll in all_dist_lines]
    
    # now construct binary masks. 
    all_dist_masks = [np.logical_xor(all_dist_masks[ii+1], all_dist_masks[ii]) for ii in range(len(all_dist_masks)-1)] # ascending distance. 
    
    """
    construct angle masks to partition the angle space. 
    """
    # 2. partition the angular direction
    angle_masks_list = [] # list of the binary masks in ascending angles.
    
    theta = theta + np.pi
    
    if zero_angle is None:
        theta_bounds = np.linspace(0, 2*np.pi, n_theta+1)
    else:
#        print(np.linspace(0, 2*np.pi, n_theta+1) + (180+360./n_theta/2.)/180.*np.pi + zero_angle/180.*np.pi)
#        theta_bounds = np.mod(np.linspace(0, 2*np.pi, n_theta+1), 2*np.pi) 
        theta_bounds = np.mod(np.linspace(0, 2*np.pi, n_theta+1) + (180.-360./n_theta/2.)/180.*np.pi - zero_angle/180.*np.pi, 2*np.pi)
        # print(theta_bounds)

    for ii in range(len(theta_bounds)-1):
        
        #### this works if all angles are within the 0 to 2 pi range. 
        if theta_bounds[ii+1] > theta_bounds[ii]:
            mask_theta = np.logical_and( theta>=theta_bounds[ii], theta <= theta_bounds[ii+1])
        else:
            mask_theta = np.logical_or(np.logical_and(theta>=theta_bounds[ii], theta<=2*np.pi), 
                                        np.logical_and(theta>=0, theta<=theta_bounds[ii+1]))
        angle_masks_list.append(mask_theta)
        
    """
    construct the final set of masks (which is for the angles. )  
    """  
    spixels = np.zeros((m,n), dtype=np.int32)
    
    counter = 1
    for ii in range(len(all_dist_masks)):
        for jj in range(len(angle_masks_list)):
            mask = np.logical_and(all_dist_masks[ii], angle_masks_list[jj])

            spixels[mask] = counter 
            counter += 1
        
    if return_grid_pts:
        return spixels, [all_dist_lines, theta_bounds]
    else:
        return spixels 


def compute_boundary_texture_features(vid, boundaries,
                            use_gradient_vid=False,
                            compute_all_feats =True,
                            compute_intensity_feats=True,
                            compute_contour_intensity_feats=True,
                            n_contours=3,
                            n_angles=1,
                            angle_start=None,
                            compute_sift=True,
                            siftshape=(64,64),
                            compute_haralick=True,
                            haralick_distance=15,
                            timesample=None,
                            use_mask=False,
                            masked_sift_resize='cap',
                            masks=None,
                            masked_haralick_levels=8,
                            masked_haralick_range=None):

    """ Compute the Appearance (or texture) features for given tracked organoid contour boundaries

    Parameters
    ----------
    vid : (n_frames, n_rows, n_cols) grayscale video
        input grayscale video intensities to extract appearance features
    boundaries : (n_organoids, n_frames, n_contour_pts, 2) array
        the set of all tracked organoid contour tracks
    use_gradient_video : bool
        if True, extract features on the sobel gradient image, not the given intensities
    compute_all_feats : bool
        if True, compute all the appearance features described in the SAM paper. This bool flag overrides all other options. If False, then the specific feature sets are computed according to whether their particular boolean flag is set to True.
    compute_contour_intensity_feats : bool 
        if True, compute the intensity features based on global and regional intensity statistics. 
    n_contours : int 
        specifies the number of equidistantly spaced concentric rings to divide the binary shape into to extract regional intensity statistics
    n_angles : int 
        specifies the number of angular segments to divide the binary shape into to extract regional intensity statistics
    angle_start : None or float in [0,360] degrees
        specifies the starting angle of the angular segments. If None, starting angle=0 and n_angles evenly divides the interval [0,360] degrees. 
    compute_sift : bool
        if True, compute the 128 dimensional SIFT feature descriptor using default parameters of original Lowe paper. 
    siftshape : (size,size)
        size of the image patch for computing SIFT. The cropped bounding box patch will be rescaled to this size to extract the global SIFT descriptor for the image
    compute_haralick : bool
        if True, compute the Haralick co-occurrencet statistics of the gray-level cooccurrence matrix (GLCM)
    haralick_distance : int
        The maximum distance in pixels to consider while computing the gray-level cooccurence matrix from which the Haralick features are derived from. 
    timesample : None or int
        if specified, compute the feature every timesample # of frames. This can be done when in successive frames, the variation does not vary a lot and one wishes to speed up computation or save on memory
    use_mask : bool
        if False (default, original SAM behaviour), Haralick and SIFT are computed on the full bounding box crop, so the background and the object outline contribute and these features partially encode shape. If True, Haralick uses only co-occurring pixel pairs inside the shape mask and SIFT uses only gradients inside the shape mask (with the boundary step edge removed), mean-pooled over shape-adapted ring x sector cells instead of a fixed square grid (see _masked_sift_descriptor), so the features describe internal appearance only. Intensity and regional intensity features are always masked.
    masked_sift_resize : 'cap', 'square' or 'native'
        only used if use_mask=True. How the masked patch is rescaled before SIFT. 'cap' (default): downscale preserving aspect ratio so the longest axis is at most max(siftshape). 'square': resize to siftshape as the unmasked SIFT. 'native': no rescaling.
    masks : None or (n_frames, n_rows, n_cols) binary array
        only used if use_mask=True. Optional foreground segmentation masks intersected with the contour mask, for when the boundary contour is coarser than the object and would include background pixels.
    masked_haralick_levels : None or int
        only used if use_mask=True. Number of grey levels the image is quantised to (over masked_haralick_range, or [0,255] if use_gradient_vid) before the masked Haralick features. Entropy-like Haralick statistics of a many-level co-occurrence matrix are biased by the number of pixel pairs, i.e. by object size, which leaks shape; a few levels largely removes this. None uses the raw grey levels.
    masked_haralick_range : None or (min, max)
        intensity range quantised into masked_haralick_levels. If None, the full range of the integer dtype of vid (e.g. [0,255] for uint8), or the min/max of vid for float images (which then depends on pixels outside the objects). For integer images using a small part of their dtype range (e.g. 12-bit data in uint16), pass the actual range.

    Returns
    -------
    out_array : (n_organoids, n_frames, n_features) array
        the concatenated appearance features for every organoid in every timepoint
    out_feat_labels : (n_features,) str array
        the names of each feature
    out_feat_labels_norm : (n_features,) str array
        a boolean vector where 1 or True means the feature is normalised i.e. dimensionless. 

    """
    # boundaries should be at least (n_orgs, timepoints, discretisation, 2)
    # from sklearn.metrics.pairwise import pairwise_distances
    import mahotas as mh  # requires mahotas.
    import skimage.transform as sktform
    from tqdm import tqdm

    # assume the first dimension is time!.
    imshape = vid.shape[1:] # must be grayscale!

#    if imshape is not None:
#        scale_factor = np.sqrt(imshape[0] *imshape[1])
    out = []
    out_feat_labels = [] # placeholder
    out_feat_labels_norm = [] # placeholder

    if timesample is None:
        timesample = 1

    if use_mask and masked_haralick_levels is not None:
        # fixed range for the Haralick grey-level quantisation, independent of the pixels outside the objects.
        if masked_haralick_range is not None:
            vid_min, vid_max = masked_haralick_range
        elif np.issubdtype(np.asarray(vid).dtype, np.integer):
            vid_min, vid_max = np.iinfo(np.asarray(vid).dtype).min, np.iinfo(np.asarray(vid).dtype).max
        else:
            vid_min, vid_max = float(np.nanmin(vid)), float(np.nanmax(vid))

#    print(scale_factor)
    for b_ii in tqdm(range(len(boundaries))):

        boundary_ii = boundaries[b_ii]
        boundary_ii_descriptors = []

        for tt in range(0,len(boundary_ii),timesample):

            vid_frame = vid[tt]
            boundary_ii_tt = boundary_ii[tt]

            if np.isnan(boundary_ii_tt[0][0]) == True:
                boundary_ii_descriptors.append([np.nan]) # don't compute.
            else:

                all_feats = []
                all_feats_labels = []
                all_feats_norm_bool = []

                """
                Get the binary mask specified by the boundary contours and crop the localised image out.
                """
                cnt_binary, bbox = contour_to_binary(boundary_ii_tt, shape=None, return_bbox=True)
                vid_cnt = vid_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1].copy()

                # test for size
                if vid_cnt.shape[0] ==0 or vid_cnt.shape[1]==0:
                    # then return nan.
                    boundary_ii_descriptors.append([np.nan]) # don't compute
                else:
                    cnt_binary = sktform.resize(cnt_binary, output_shape=vid_cnt.shape, order=0, preserve_range=True) > 0

                    fg_frame = None
                    if use_mask and masks is not None:
                        # restrict to the foreground segmentation, in case the contour is coarser than the object.
                        fg_frame = np.asarray(masks[tt]) > 0
                        cnt_binary = np.logical_and(cnt_binary, fg_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1])

    #                print(vid_cnt.shape)
    #                if vid_cnt.shape[0]>haralick_distance and vid_cnt.shape[1] > haralick_distance:
        #                vid_cnt_im = vid_cnt.copy()
                    max_dist = np.min(vid_cnt.shape)-1

                    if use_gradient_vid:
                        """
                        extract sobel.
                        """
                        vid_cnt = sobel_frame(vid_cnt) # also do the dog .

                    if compute_all_feats==True or (compute_all_feats==False and compute_intensity_feats==True):

    #                    print(vid_cnt.shape, cnt_binary.shape, bbox, vid_frame.shape)
                        mean_cnt_global = np.nanmean(vid_cnt[cnt_binary>0])
                        std_cnt_global = np.nanstd(vid_cnt[cnt_binary>0])

                        all_feats.append(np.hstack([mean_cnt_global,
                                                    std_cnt_global]))
                        all_feats_labels.append(np.hstack(['mean_intensity',
                                                           'stddev_intensity']))
                        all_feats_norm_bool.append(np.hstack([False]*2))

                    if compute_all_feats==True or (compute_all_feats==False and compute_contour_intensity_feats==True):

                        # this creates contour based on the given boundary line and operates on the global image coordinate.
                        boundary_grid = _tile_uniform_radial_windows_boundary(imsize=vid_frame.shape,
                                                                             n_r=n_contours, n_theta=n_angles,
                                                                             boundary_line = boundary_ii_tt,
                                                                             center=None,
                                                                             bound_r=True,
                                                                             zero_angle=angle_start,
                                                                             return_grid_pts=False)

                        uniq_regions = np.setdiff1d(np.unique(boundary_grid), 0)

                        try:
                            assert(len(uniq_regions)==3) # right
                        except:
                            if len(uniq_regions) < n_contours:
                                # pad.
                                uniq_regions = np.hstack([uniq_regions, [uniq_regions[-1]]*(n_contours-len(uniq_regions))])
                                
                        contour_img_feats = []
                        contour_img_feats_labels = []

                        for reg in uniq_regions:
                            reg_mask = boundary_grid==reg
                            if fg_frame is not None:
                                reg_mask = np.logical_and(reg_mask, fg_frame)

                            contour_img_feats.append(np.nanmean(vid_frame[reg_mask>0]))
                            contour_img_feats.append(np.nanstd(vid_frame[reg_mask>0]))

                            contour_img_feats_labels.append('mean_intensity_region_%s' %(str(reg).zfill(3)))
                            contour_img_feats_labels.append('stddev_intensity_region_%s' %(str(reg).zfill(3)))

                        contour_img_feats = np.hstack(contour_img_feats)
                        contour_img_feats_labels = np.hstack(contour_img_feats_labels)

                        all_feats.append(contour_img_feats)
                        all_feats_labels.append(contour_img_feats_labels)
                        all_feats_norm_bool.append(np.hstack([False]*len(contour_img_feats_labels)))

                    if compute_all_feats==True or (compute_all_feats==False and compute_haralick==True):

                        # this is absolutely not normalised .....
                        if use_mask:
                            vid_cnt_q = vid_cnt
                            if masked_haralick_levels is not None:
                                # quantise over a fixed intensity range so the levels mean the same for every object.
                                vmin, vmax = (0., 255.) if use_gradient_vid else (vid_min, vid_max)
                                vid_cnt_q = np.clip(np.floor((vid_cnt.astype(np.float64) - vmin) / (vmax - vmin + 1e-8) * masked_haralick_levels),
                                                    0, masked_haralick_levels-1).astype(np.int32)
                            cnt_texture = _masked_haralick(vid_cnt_q, cnt_binary,
                                                           distance=np.minimum(max_dist, haralick_distance))
                        else:
                            cnt_texture = mh.features.haralick(vid_cnt,
                                                               ignore_zeros=False,
                                                               preserve_haralick_bug=False,
                                                               compute_14th_feature=False,
                                                               return_mean=True, # mean of the different directions.
                                                               return_mean_ptp=False,
                                                               use_x_minus_y_variance=False,
                                                               distance=np.minimum(max_dist, haralick_distance)) # i see the distance is the limiting factor.

    #                    cnt_texture = cnt_texture/(np.linalg.norm(cnt_texture) + 1e-8)
                        cnt_texture_labels = np.hstack(['angular_second_moment',
                                                        'contrast',
                                                        'correlation',
                                                        'sum_squares_variance',
                                                        'inv_diff_moment',
                                                        'sum_average',
                                                        'sum_variance',
                                                        'sum_entropy',
                                                        'entropy',
                                                        'diff_variance',
                                                        'diff_entropy',
                                                        'info_measure_correlation_1',
                                                        'info_measure_correlation_2'])

                        all_feats.append(cnt_texture)
                        all_feats_labels.append(cnt_texture_labels)
                        all_feats_norm_bool.append(np.hstack([False]*len(cnt_texture)))

                    if compute_all_feats==True or (compute_all_feats==False and compute_sift==True):

                        if use_mask:
                            sift_texture = _masked_sift_descriptor(vid_cnt, cnt_binary, resize=masked_sift_resize, siftshape=siftshape)
                        else:
                            siftextractor = SingleSiftExtractor(siftshape[0]) # note this is already in the above.

                            vid_cnt_im = sktform.resize(vid_cnt, siftshape, preserve_range=True)
                            sift_texture = siftextractor.process_image(vid_cnt_im).ravel()
                            sift_texture = sift_texture / (np.linalg.norm(sift_texture) + 1e-8)
                        sift_texture_labels = np.hstack(['sift_appear_%s'%(str(iii+1)) for iii in np.arange(len(sift_texture))]) # using sift_appear to distinguish from motion 

                        all_feats.append(sift_texture)
                        all_feats_labels.append(sift_texture_labels)
                        all_feats_norm_bool.append(np.hstack([True]*len(sift_texture)))


                    all_feats = np.hstack(all_feats)
                    all_feats_labels = np.hstack(all_feats_labels)
                    all_feats_norm_bool = np.hstack(all_feats_norm_bool)

                    boundary_ii_descriptors.append(all_feats)

                    out_feat_labels = all_feats_labels
                    out_feat_labels_norm = all_feats_norm_bool

        out.append(boundary_ii_descriptors)

    out = np.array(out, dtype=object)

    # make this a regular array for analysis.
    n_org, n_time = boundaries.shape[:2]

    if timesample is not None:
        n_time = len(np.arange(0, n_time, timesample)) # as we are subsampling...
    n_feat_size = len(out_feat_labels)

    out_array = np.zeros((n_org, n_time, n_feat_size))
    # print(out_array.shape)

    for ii in range(n_org):
        for jj in range(n_time):
            val = out[ii,jj]
            if len(val) == 1:
                out_array[ii,jj] = np.nan
            else:
                out_array[ii,jj] = val

    return out_array, out_feat_labels, out_feat_labels_norm


# =============================================================================
#   Do Some Motion Distribution analysis (intensity distribution pattern analysis.)
#       Other than distribution also consider other types of movement!.
# =============================================================================

# this i haven't quite decided yet, but should be on the basis of motion histograms. -> not sure about the sift like thing, should be more like HoG? -> also preferably should be dense! rather than on the tracks -> summarise the dense flow in the region? -> and the boundary?
# the key here is to capture rotationally-invariant movement features on the global + local scale.
# proposed:
#   - mean speed of organoid entity (translational movement)
#   - histogram of speeds in different contours of the organoid to the centre.

# kde instead of a histogram to combat slighlty poor sample numbers.
def _kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    from scipy.stats import gaussian_kde
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


def _curl_vector_flow(im, *args):
    """ Computes the curl vector field given as a 2D or 3D vector data.
    
    Parameters
    ----------
    im : numpy array
        (nrows, ncols, 2) for 2d image or (n_z, nrows, ncols, 3) for 3d volumetric image, last channel is the (x,y) or (x,y,z) vectors.
    args : 
        optional arguments passed to np.gradient

    Returns
    -------
    curl : numpy array
        (nrows, ncols, 3) array representing the curl in i,j,k. For 2d image, the first two channels are 0.
    """

    dim = im.shape[-1] # number of dimensions of vector.

    if dim == 2:
        Fx = im[...,0]  
        Fy = im[...,1]
        dFx_dy, dFx_dx = np.gradient(Fx, *args) 
        dFy_dy, dFy_dx = np.gradient(Fy, *args) 
        
        curl = np.dstack([np.zeros(Fx.shape), np.zeros(Fy.shape), dFy_dx - dFx_dy])
               
    if dim == 3:
        # get the flow in image convention order? 
        Fx = im[..., 0]
        Fy = im[..., 1]
        Fz = im[..., 2]

        dFx_dz, dFx_dy, dFx_dx = np.gradient(Fx, *args)
        dFy_dz, dFy_dy, dFy_dx = np.gradient(Fy, *args)
        dFz_dz, dFz_dy, dFz_dx = np.gradient(Fz, *args)
        
        # hard-code the equations.
        curl_x = dFz_dy - dFy_dz
        curl_y = -(dFz_dx - dFx_dz)
        curl_z = dFy_dx - dFx_dy
        
        curl = np.stack([curl_x, curl_y, curl_z])

    return curl
    
    
def _div_vector_flow(im, *args):
    
    """Computes the divergence of a vector field given as a 2D or 3D vector data.
    
    Parameters
    ----------
    im : numpy array
        (nrows, ncols, 2) for 2d image or (n_z, nrows, ncols, 3) for 3d volumetric image, last channel is the (x,y) or (x,y,z) vectors.
    args : 
        optional arguments passed to np.gradient

    Returns
    -------
    div : numpy array
        (nrows, ncols) array representing the divergence
    
    """
    dim = im.shape[-1] # number of dimensions of vector.
    
    if dim == 2:
        Fx = im[...,0]  
        Fy = im[...,1]
        dFx_dy, dFx_dx = np.gradient(Fx, *args) 
        dFy_dy, dFy_dx = np.gradient(Fy, *args) 
        
        div = dFx_dx + dFy_dy

    if dim == 3:
        Fx = im[..., 0]
        Fy = im[..., 1]
        Fz = im[..., 2]

        dFx_dz, dFx_dy, dFx_dx = np.gradient(Fx, *args)
        dFy_dz, dFy_dy, dFy_dx = np.gradient(Fy, *args)
        dFz_dz, dFz_dy, dFz_dx = np.gradient(Fz, *args)
        
        # hard-code the equations.        
        div = dFx_dx + dFy_dy + dFz_dz
        
    return div



def compute_boundary_motion_features(vid_flow,
                                     boundaries,
                                     compute_all_feats =True,
                                     compute_global_feats=True,
                                     compute_contour_feats=True,
                                     n_contour_feat_bins=8,
                                     cnt_sigma=3.,
                                     n_contours=3,
                                     n_angles=1,
                                     angle_start=None,
                                     pixel_res=1.,
                                     time_res=1.,
                                     compute_sift_feats=True,
                                     siftshape=(64,64),
                                     timesample=None,
                                     use_mask=False,
                                     masked_sift_resize='cap',
                                     masks=None):
    """ Compute the Motion features for given tracked organoid contour boundaries. Can only computed for a video as opposed to shape and appearance. 

    Parameters
    ----------
    vid_flow : (n_frames-1, n_rows, n_cols) array 
        input grayscale frame-to-frame optical flow for the video specifying the pixel-to-pixel displacement
    boundaries : (n_organoids, n_frames, n_contour_pts, 2) array
        the set of all tracked organoid contour tracks
    compute_all_feats : bool
        if True, compute all the appearance features described in the SAM paper. This bool flag overrides all other options. If False, then the specific feature sets are computed according to whether their particular boolean flag is set to True.
    compute_global_feats : bool 
        if True, compute the global speed features such as global mean and standard deviation speed.
    compute_contour_feats : bool 
        if True, compute the regional speed statistics and speed histogram after partitioning spatially the enclosed region within the contour. 
    n_contour_feat_bins : int
        if compute_contour_feats=True, compute the histogram of the speed in each region, with speed bounds mean speed in region +/- cnt_sigma * standard deviation mean speed in region. 
    cnt_sigma : float
        specifies the speed range as a scalar multiple of the standard deviation of the speed around the mean speed in the region
    n_contours : int
        specifies the number of equidistantly spaced concentric rings to divide the binary shape into to extract regional speed statistics
    n_angles : int 
        specifies the number of angular segments to divide the binary shape into to extract regional speed statistics
    angle_start : None or float in [0,360] degrees
        specifies the starting angle of the angular segments. If None, starting angle=0 and n_angles evenly divides the interval [0,360] degrees. 
    pixel_res : float
        the pixel to physical units conversion. for organoids, the pixel to um conversion
    time_res : float
        the temporal sampling of the video, i.e. the time elapsed between two consecutive frames 
    compute_sift_feats : bool
        if True, compute the 128 dimensional SIFT feature descriptor using default parameters of original Lowe paper. 
    siftshape : (size,size)
        size of the image patch for computing SIFT. The cropped bounding box patch will be rescaled to this size to extract the global SIFT descriptor for the image
    timesample : None or int
        if specified, compute the feature every timesample # of frames. This can be done when in successive frames, the variation does not vary a lot and one wishes to speed up computation or save on memory
    use_mask : bool
        if False (default, original SAM behaviour), the speed, curl and divergence SIFT descriptors are computed on the bounding box crop with the pixels *inside* the shape set to 0, so they mainly encode the background flow and the object outline i.e. shape. If True, curl and divergence are recomputed from the flow inside the shape mask only (so the flow discontinuity at the boundary does not leak in) and the SIFT descriptors use only gradients of pixels inside the shape mask (with the boundary step edge removed), mean-pooled over shape-adapted ring x sector cells instead of a fixed square grid (see _masked_sift_descriptor), so they describe internal motion only. Global and regional speed statistics are always masked.
    masked_sift_resize : 'cap', 'square' or 'native'
        only used if use_mask=True. How the masked patch is rescaled before SIFT. 'cap' (default): downscale preserving aspect ratio so the longest axis is at most max(siftshape). 'square': resize to siftshape as the unmasked SIFT. 'native': no rescaling.
    masks : None or (n_frames, n_rows, n_cols) binary array
        only used if use_mask=True. Optional foreground segmentation masks intersected with the contour mask, for when the boundary contour is coarser than the object and would include background pixels.

    Returns
    -------
    out_array : (n_organoids, n_frames, n_features) array
        the concatenated motion features for every organoid in every timepoint
    out_feat_labels : (n_features,) str array
        the names of each feature
    out_feat_labels_norm : (n_features,) str array
        a boolean vector where 1 or True means the feature is normalised i.e. dimensionless. 

    """
    # boundaries should be at least (n_orgs, timepoints, discretisation, 2)
    # vid_flow is (timepoints-1,  nrows, ncols, 2) (in x,y) format.
    import skimage.transform as sktform
    from tqdm import tqdm

    # assume the first dimension is time!.
    imshape = vid_flow.shape[1:] # must be grayscale!

#    if imshape is not None:
#        scale_factor = np.sqrt(imshape[0] *imshape[1])
    out = []
    out_feat_labels = [] # placeholder
    out_feat_labels_norm = [] # placeholder

    if timesample is None:
        timesample = 1

#    print(scale_factor)
    for b_ii in tqdm(range(len(boundaries))):

        boundary_ii = boundaries[b_ii]
        boundary_ii_descriptors = []

        # iterating over time.
        for tt in range(0,len(boundary_ii)-1, timesample):

            flow_frame = vid_flow[tt].copy();
#            flow_frame = flow_frame * pixel_res / float(time_res)

            """
            compute the instantaneous divergence and curl measures.
            """
            # both are unitless except by /time.
            curl_vid_frame = _curl_vector_flow(flow_frame)[...,-1] / float(time_res) # in 2d only last channel # this is the correct unit.
            div_vid_frame = _div_vector_flow(flow_frame) / float(time_res) # already corrected?

            boundary_ii_tt = boundary_ii[tt]

            if np.isnan(boundary_ii_tt[0][0]) == True:
                # if current boundary is not a number don't compute.
                boundary_ii_descriptors.append([np.nan]) # don't compute.
            else:

                # test to see if the next boundary is not a number.
                boundary_ii_tt_plus_1 = boundary_ii[tt+1]
                if np.isnan(boundary_ii_tt_plus_1[0][0]) == True:
                    boundary_ii_descriptors.append([np.nan]) # don't compute.
                else:

                    # all placeholders.
                    all_feats = []
                    all_feats_labels = []
                    all_feats_norm_bool = []

                    """
                    Get the binary mask specified by the boundary contours and crop the localised image out.
                    """
                    cnt_binary, bbox = contour_to_binary(boundary_ii_tt, shape=None, return_bbox=True)
                    flow_vid_cnt = flow_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1].copy() # somewhat noisy but looks ok?

                    if flow_vid_cnt.shape[0]==0 or flow_vid_cnt.shape[1]==0 or cnt_binary.shape[0]==0 or cnt_binary.shape[1]==0:
                        boundary_ii_descriptors.append([np.nan])
                    else:
                        # get the cropped out divergence and curl magnitudes.
                        curl_vid_cnt = curl_vid_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1].copy() # somewhat noisy but looks ok?
                        div_vid_cnt = div_vid_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1].copy() # somewhat noisy but looks ok?

                        cnt_binary = cnt_binary = sktform.resize(cnt_binary, output_shape=flow_vid_cnt.shape[:-1], order=0, preserve_range=True) > 0

                        fg_frame = None
                        if use_mask and masks is not None:
                            # restrict to the foreground segmentation, in case the contour is coarser than the object.
                            fg_frame = np.asarray(masks[tt]) > 0
                            cnt_binary = np.logical_and(cnt_binary, fg_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1])

                        if use_mask:
                            # recompute curl and divergence from the flow inside the mask only (outside filled by nearest neighbour), so the flow discontinuity at the object boundary does not leak in.
                            flow_vid_cnt_fill = np.dstack([_fill_outside_mask_nearest(flow_vid_cnt[...,ch], cnt_binary) for ch in range(flow_vid_cnt.shape[-1])])
                            curl_vid_cnt = _curl_vector_flow(flow_vid_cnt_fill)[...,-1] / float(time_res)
                            div_vid_cnt = _div_vector_flow(flow_vid_cnt_fill) / float(time_res)

                        """
                        first get the central displacement between current and next timepoint.
                        """
                        if compute_all_feats==True or (compute_all_feats==False and compute_global_feats==True):

                            disp = (boundary_ii_tt_plus_1 - boundary_ii_tt) * pixel_res / float(time_res) # convert to physical units. typically um/min or um/h
    #                        mean_disp_speed = np.linalg.norm(np.nanmean(disp, axis=0))
                            # this is the centroid displacement.
                            mean_disp_speed = np.linalg.norm( np.nanmean(boundary_ii_tt_plus_1[:-1], axis=0) -
                                                              np.nanmean(boundary_ii_tt[:-1], axis=0), axis=-1) * pixel_res / float(time_res)
                            std_disp_speed = np.nanstd(np.linalg.norm(disp, axis=-1)) # get the variation.

                            all_feats.append(np.hstack([mean_disp_speed,std_disp_speed]))
                            all_feats_labels.append(np.hstack(['mean_speed_global',
                                                               'stddev_speed_global']))
                            all_feats_norm_bool.append(np.hstack([False]*2))


                            # compute also this from the optical flow.
                            flow_vid_cnt_ = flow_vid_cnt.copy() * pixel_res / float(time_res)
    #                        mean_disp_speed_flow = np.linalg.norm( np.nanmean(flow_vid_cnt[cnt_binary>0], axis=0) )
                            mean_disp_speed_flow = np.nanmean( np.linalg.norm(flow_vid_cnt_[cnt_binary>0], axis=-1))
                            std_disp_speed_flow = np.nanstd( np.linalg.norm(flow_vid_cnt_[cnt_binary>0], axis=-1) )

                            all_feats.append(np.hstack([mean_disp_speed_flow,
                                                        std_disp_speed_flow]))
                            all_feats_labels.append(np.hstack(['mean_speed_global_flow',
                                                               'stddev_speed_global_flow']))
                            all_feats_norm_bool.append(np.hstack([False]*2))


                            # =============================================================================
                            #    Add in now the divergence and curl features.
                            # =============================================================================
                            mean_curl_mag = np.nanmean( curl_vid_cnt[cnt_binary>0])
                            std_curl_mag = np.nanstd( curl_vid_cnt[cnt_binary>0])
                            all_feats.append(np.hstack([mean_curl_mag,
                                                        std_curl_mag]))
                            all_feats_labels.append(np.hstack(['mean_curl_global_flow',
                                                               'stddev_curl_global_flow']))
                            all_feats_norm_bool.append(np.hstack([False]*2))

                            mean_div_mag = np.nanmean( div_vid_cnt[cnt_binary>0])
                            std_div_mag = np.nanstd( div_vid_cnt[cnt_binary>0])
                            all_feats.append(np.hstack([mean_div_mag,
                                                        std_div_mag]))
                            all_feats_labels.append(np.hstack(['mean_divergence_global_flow',
                                                               'stddev_divergence_global_flow']))
                            all_feats_norm_bool.append(np.hstack([False]*2))

                        """
                        second get the local spatial variation of speed between current and next timepoint.
                            Get the binary mask specified by the boundary contours and crop the localised image out.
                        """
                        if compute_all_feats==True or (compute_all_feats==False and compute_contour_feats==True):

                            # build the spatial window schema.
                            boundary_grid = _tile_uniform_radial_windows_boundary(imsize=flow_frame.shape[:-1],
                                                                                n_r=n_contours, n_theta=n_angles,
                                                                                boundary_line = boundary_ii_tt,
                                                                                center=None,
                                                                                bound_r=True,
                                                                                zero_angle=angle_start,
                                                                                return_grid_pts=False)

                            uniq_regions = np.setdiff1d(np.unique(boundary_grid), 0)

                            try:
                                assert(len(uniq_regions)==3) # right
                            except:
                                if len(uniq_regions) < n_contours:
                                    # pad.
                                    uniq_regions = np.hstack([uniq_regions, [uniq_regions[-1]]*(n_contours-len(uniq_regions))])

                            for reg in uniq_regions:
                                reg_mask = boundary_grid==reg
                                if fg_frame is not None:
                                    reg_mask = np.logical_and(reg_mask, fg_frame)

                                flow_reg_mask = flow_frame[reg_mask>0] * pixel_res / float(time_res)

                                # histograms.
                                if len(flow_reg_mask) > n_contour_feat_bins: # requires a minimum number else we can't really estimate.
                                    flow_reg_mask_mag = np.linalg.norm(flow_reg_mask, axis=-1)
                                    lower_bin = np.nanmean(flow_reg_mask_mag) - np.nanstd(flow_reg_mask_mag) * cnt_sigma
                                    upper_bin = np.nanmean(flow_reg_mask_mag) + np.nanstd(flow_reg_mask_mag) * cnt_sigma
                                    # hist_flow, _ = np.histogram(flow_reg_mask.ravel(), bins=n_contour_feat_bins, range=(lower_bin, upper_bin))
                                    # do a gaussian kde estimation to smooth and combat low sample numbers
                                    x_grid = np.linspace(lower_bin, upper_bin, n_contour_feat_bins)
    #                                print(flow_reg_mask_mag)

    #                                print(flow_reg_mask_mag)
                                    if np.sum(flow_reg_mask_mag>1e-3) > 0 and np.nanstd(flow_reg_mask_mag) < 1e-8:
                                        # constant nonzero speed: gaussian_kde is singular. As std->0 the bins collapse within one kernel bandwidth so the normalised kde tends to uniform.
                                        hist_flow = np.ones(n_contour_feat_bins) / float(n_contour_feat_bins)
                                    elif np.sum(flow_reg_mask_mag>1e-3) > 0:
                                        hist_flow = _kde_scipy(flow_reg_mask_mag, x_grid, bandwidth=.2) # direct gaussian kde estimation.
                                        hist_flow = hist_flow / float(np.sum(hist_flow) + 1e-8) # normalisation ( we only care for the shape)
                                    else:
                                        hist_flow = np.zeros(n_contour_feat_bins)
                                else:
                                    hist_flow = np.zeros(n_contour_feat_bins)

                                all_feats.append(hist_flow)
                                all_feats_labels.append(['hist_bin_%d_region_%s' %(bbb+1, str(reg).zfill(3)) for bbb in range(len(hist_flow))])
                                all_feats_norm_bool.append(np.hstack([True]*len(hist_flow)))

                                # mean statistics.
                                mean_disp_speed_flow_region = np.nanmean( np.linalg.norm(flow_reg_mask, axis=-1))
                                std_disp_speed_flow_region = np.nanstd( np.linalg.norm(flow_reg_mask, axis=-1) )

                                all_feats.append(np.hstack([mean_disp_speed_flow_region,
                                                            std_disp_speed_flow_region]))
                                all_feats_labels.append(np.hstack(['mean_speed_flow_region_%d' %(reg),
                                                                   'stddev_speed_flow_region%d' %(reg)]))
                                all_feats_norm_bool.append(np.hstack([False]*2))


                        if compute_all_feats==True or (compute_all_feats==False and compute_sift_feats==True):

                            flow_vid_cnt_ = flow_vid_cnt * pixel_res / float(time_res)
                            flow_vid_cnt_ = np.linalg.norm(flow_vid_cnt_, axis=-1) # create the magnitude of the flow.

                            if use_mask:
                                sift_texture = _masked_sift_descriptor(flow_vid_cnt_, cnt_binary, resize=masked_sift_resize, siftshape=siftshape)
                            else:
                                flow_vid_cnt_[cnt_binary>0] = 0 # mask to be specific to within the boundary.

                                # create the sift extractor for this pseudo image.
                                siftextractor = SingleSiftExtractor(siftshape[0]) # note this is already in the above.

                                flow_vid_cnt_im = sktform.resize(flow_vid_cnt_, siftshape, preserve_range=True)
                                sift_texture = siftextractor.process_image(flow_vid_cnt_im).ravel()
                                sift_texture = sift_texture / (np.linalg.norm(sift_texture) + 1e-8)
                            sift_texture_labels = np.hstack(['sift_motion_%s'%(str(iii+1)) for iii in np.arange(len(sift_texture))])

                            all_feats.append(sift_texture)
                            all_feats_labels.append(sift_texture_labels)
                            all_feats_norm_bool.append(np.hstack([True]*len(sift_texture)))


                            # =============================================================================
                            #   Additionally add the texture to curl and divergence?
                            # =============================================================================
#                            curl_vid_cnt = curl_vid_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1].copy() # somewhat noisy but looks ok?
#                            div_vid_cnt = div_vid_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1].copy() # somewhat noisy but looks ok?
#
                            if use_mask:
                                sift_texture = _masked_sift_descriptor(curl_vid_cnt, cnt_binary, resize=masked_sift_resize, siftshape=siftshape)
                            else:
                                # create the sift extractor for this pseudo image.
                                siftextractor = SingleSiftExtractor(siftshape[0]) # note this is already in the above.

                                curl_vid_cnt_ = curl_vid_cnt.copy();
                                curl_vid_cnt_[cnt_binary>0] = 0
                                curl_vid_cnt_im = sktform.resize(curl_vid_cnt_, siftshape, preserve_range=True)
                                sift_texture = siftextractor.process_image(curl_vid_cnt_im).ravel()
                                sift_texture = sift_texture / (np.linalg.norm(sift_texture) + 1e-8)
                            sift_texture_labels = np.hstack(['curl_sift_%s'%(str(iii+1)) for iii in np.arange(len(sift_texture))])

                            all_feats.append(sift_texture)
                            all_feats_labels.append(sift_texture_labels)
                            all_feats_norm_bool.append(np.hstack([True]*len(sift_texture)))


                            if use_mask:
                                sift_texture = _masked_sift_descriptor(div_vid_cnt, cnt_binary, resize=masked_sift_resize, siftshape=siftshape)
                            else:
                                # create the sift extractor for this pseudo image.
                                siftextractor = SingleSiftExtractor(siftshape[0]) # note this is already in the above.

                                div_vid_cnt_ = div_vid_cnt.copy();
                                div_vid_cnt_[cnt_binary>0] = 0
                                div_vid_cnt_im = sktform.resize(div_vid_cnt_, siftshape, preserve_range=True)
                                sift_texture = siftextractor.process_image(div_vid_cnt_im).ravel()
                                sift_texture = sift_texture / (np.linalg.norm(sift_texture) + 1e-8)
                            sift_texture_labels = np.hstack(['div_sift_%s'%(str(iii+1)) for iii in np.arange(len(sift_texture))])

                            all_feats.append(sift_texture)
                            all_feats_labels.append(sift_texture_labels)
                            all_feats_norm_bool.append(np.hstack([True]*len(sift_texture)))

                        # contour_img_feats = np.hstack(contour_img_feats)
                        # contour_img_feats_labels = np.hstack(contour_img_feats_labels)

                        # all_feats.append(contour_img_feats)
                        # all_feats_labels.append(contour_img_feats_labels)
                        # all_feats_norm_bool.append(np.hstack([True]*len(contour_img_feats_labels)))

                    all_feats = np.hstack(all_feats)
                    all_feats_labels = np.hstack(all_feats_labels)
                    all_feats_norm_bool = np.hstack(all_feats_norm_bool)

                    boundary_ii_descriptors.append(all_feats)

                    out_feat_labels = all_feats_labels
                    out_feat_labels_norm = all_feats_norm_bool

        out.append(boundary_ii_descriptors)

    out = np.array(out, dtype=object)

    # make this a regular array for analysis.
    n_org, n_time = boundaries.shape[:2]; n_time = n_time-1 # since this is motion.

    if timesample is not None:
        n_time = len(np.arange(0, n_time, timesample)) # as we are subsampling...
    n_feat_size = len(out_feat_labels)

    out_array = np.zeros((n_org, n_time, n_feat_size))

    for ii in range(n_org):
        for jj in range(n_time):
            val = out[ii,jj]
            if len(val) == 1:
                out_array[ii,jj] = np.nan
            else:
                out_array[ii,jj] = val
                
    return out_array, out_feat_labels, out_feat_labels_norm









# =============================================================================
#   Fast variants of the SAM feature functions.
#       compute_boundary_*_features_fast return the same outputs as compute_boundary_*_features but
#           - avoid full-frame work per object (rasterisation / region masks are done in the object's bounding box,
#             curl and divergence once per frame instead of once per object and frame),
#           - vectorise the Euler characteristic transform and shape context histograms,
#           - optionally run in parallel over objects (n_jobs).
# =============================================================================

def _ect_fast(binary, directions, T=32):
    """ Euler characteristic transform of a binary image, identical to CubicalComplex(binary).complexify().ECT(directions, T) but with the vertices, edges and squares of the cubical complex built with array operations.

    Parameters
    ----------
    binary : (n_rows, n_cols) binary image
        the binary shape
    directions : (n_directions, 2) array
        the unit directions to compute the Euler characteristic curve for
    T : int
        number of thresholds of each Euler characteristic curve

    Returns
    -------
    ect : (n_directions*T,) int array
        the concatenated Euler characteristic curves
    """
    on = np.asarray(binary) > 0
    coords = np.vstack(np.nonzero(on)).T
    verts = _centerVertices(coords)

    index = np.full(on.shape, -1, dtype=np.int64)
    index[on] = np.arange(len(coords))
    # edges join 4-connected pixels, squares are 2x2 blocks of pixels (as CubicalComplex.complexify)
    horz = on[:,:-1] & on[:,1:]
    vert = on[:-1,:] & on[1:,:]
    edges = np.vstack([np.vstack([index[:,:-1][horz], index[:,1:][horz]]).T,
                       np.vstack([index[:-1,:][vert], index[1:,:][vert]]).T])
    sq = on[:-1,:-1] & on[:-1,1:] & on[1:,:-1] & on[1:,1:]
    squares = np.vstack([index[:-1,:-1][sq], index[:-1,1:][sq], index[1:,:-1][sq], index[1:,1:][sq]]).T

    heights = np.sum(verts[:,None,:]*directions[None,:,:], axis=-1) # (n_verts, n_directions), same arithmetic as CubicalComplex.ECT
    edge_heights = np.max(heights[edges], axis=1) if len(edges) > 0 else None
    square_heights = np.max(heights[squares], axis=1) if len(squares) > 0 else None

    ect = np.zeros(T*directions.shape[0], dtype=int)
    for i in range(directions.shape[0]):
        filtration = heights[:,i]
        minh = np.min(filtration)
        maxh = np.max(filtration)
        ecc, bins = np.histogram(filtration, bins=T, range=(minh, maxh))
        ecc = np.add(np.zeros_like(ecc), ecc)
        if edge_heights is not None:
            ecc = np.add(ecc, -1*np.histogram(edge_heights[:,i], bins=T, range=(minh, maxh))[0])
        if square_heights is not None:
            ecc = np.add(ecc, np.histogram(square_heights[:,i], bins=T, range=(minh, maxh))[0])
        ect[i*T : (i+1)*T] = np.cumsum(ecc)

    return ect


def _shape_context_core_fast(dists, angles, n_dist=10, n_theta=16):
    """ _shape_context_core with the per reference point 2D histogram computed by np.bincount. Same output. """
    import numpy as np

    n_pts = dists.shape[1]
    # Using log distances
    logbase = 1.5
    mean_dis = np.mean(dists.flatten())
    b = 1
    a = (logbase ** (0.75 * n_dist) - b) / mean_dis

    dists = np.floor(np.log(a * dists + b) / np.log(logbase))
    dists = np.maximum(dists, np.resize(1, dists.shape))
    dists = np.minimum(dists, np.resize(n_dist, dists.shape))

    # Preprocessing angles
    delta_ang = 2 * np.pi / n_theta
    angles = np.ceil((angles + np.pi) / delta_ang)
    angles = np.maximum(angles, np.resize(1, angles.shape))
    angles = np.minimum(angles, np.resize(n_theta, angles.shape))

    # Shape context, flattened in Fortran order of the (n_dist, n_theta) histogram
    sc_hist = np.zeros((n_theta * n_dist, n_pts))
    for v in range(n_pts):
        valid = np.logical_and(np.isfinite(dists[:,v]), np.isfinite(angles[:,v]))
        key = (dists[valid,v] - 1) + n_dist * (angles[valid,v] - 1)
        sc_hist[:, v] = np.bincount(key.astype(np.int64), minlength=n_theta * n_dist)
    sc_hist /= dists.shape[0]
    return sc_hist


def shape_context_fast(cont, n_ref=5, n_dist=5, n_theta=12, b_tangent=1, close_contour=False):
    """ shape_context using _shape_context_core_fast. Same parameters and output as shape_context. """
    import numpy as np

    n_pt = cont.shape[0]
    X = np.array([cont[:, 0]]).transpose()
    Y = np.array([cont[:, 1]]).transpose()
    # Set reference point
    si = np.round(np.linspace(1, n_pt, n_ref)).astype(np.int32) - 1
    V = cont[si, :]
    vx = np.array([V[:, 0]])
    vy = np.array([V[:, 1]])
    # Orientations and geodesic distances between all landmark points
    Xs = np.tile(X, (1, n_ref))
    vxs = np.tile(vx, (n_pt, 1))
    dXs = Xs - vxs
    Ys = np.tile(Y, (1, n_ref))
    vys = np.tile(vy, (n_pt, 1))
    dYs = Ys - vys
    dis_mat = np.sqrt(dXs**2 + dYs**2)
    ang_mat = np.arctan2(dYs, dXs)

    # Normalize shape context with the tangent orientation
    if b_tangent:
        Xs = np.append(np.append([X[-1]], X, axis=0), [X[0]], axis=0)
        Ys = np.append(np.append([Y[-1]], Y, axis=0), [Y[0]], axis=0)

        gX = np.gradient(Xs, axis=0)
        gY = np.gradient(Ys, axis=0)

        if not close_contour:
            gX[1] = X[1] - X[0]
            gX[-2] = X[-1] - X[-2]
            gY[1] = Y[1] - Y[0]
            gY[-2] = Y[-1] - Y[-2]

        thetas = np.arctan2(gY, gX)
        thetas = thetas[1:-1]
        thetas = np.tile(thetas, (1, n_ref))

        ang_mat = ang_mat - thetas
        idx = np.where(ang_mat > np.pi)
        ang_mat[idx] = ang_mat[idx] - 2 * np.pi
        idx = np.where(ang_mat < -np.pi)
        ang_mat[idx] = ang_mat[idx] + 2 * np.pi

    sc = _shape_context_core_fast(dis_mat, ang_mat, n_dist, n_theta)
    return sc, V, dis_mat, ang_mat


def reorient_contour_fast(cnt, shape):
    """ reorient_contour with the orientation computed from the rasterised shape in its bounding box instead of a full image canvas. Same output. """
    from skimage.draw import polygon
    from skimage.measure import regionprops

    rr, cc = polygon(cnt[:,0],
                     cnt[:,1], shape=shape)
    r0, c0 = rr.min(), cc.min()
    canvas = np.zeros((rr.max()-r0+1, cc.max()-c0+1), dtype=np.int32)
    canvas[rr-r0,cc-c0] = 1

    orientation = regionprops(canvas)[0].orientation

    center = cnt[:-1].mean(axis=0)
    cnt_rot = _rotate_pts(cnt[:-1,::-1], angle=orientation/np.pi*180., center=center[::-1])[:,::-1]

    # relabel points i.e. sorting by angle
    cnt_angle = np.arctan2(cnt_rot[:, 0] - center[0], cnt_rot[:,1]- center[1])
    sort = np.argsort(cnt_angle)

    cnt_rot_ = np.vstack([cnt_rot[sort[0]:],
                          cnt_rot[:sort[0]]])

    cnt_rot = np.vstack([cnt_rot_,
                         cnt_rot_[0]])

    return cnt_rot


def _linspace_rows(stop, num):
    """ np.array([np.linspace(0, l, num) for l in stop]) without the loop, with the same arithmetic as np.linspace. """
    stop = np.asarray(stop, dtype=np.float64)
    div = num - 1
    y = np.arange(0, num, dtype=np.float64)[None,:]
    step = stop / div
    zero = step == 0
    out = y * step[:,None]
    if zero.any():
        out[zero] = (y / div) * stop[zero][:,None]
    out += 0.
    out[:,-1] = stop
    return out


def _radial_window_int_lines(imsize, n_r, boundary_line, center=None):
    """ the integer polygon vertices (rows, cols) of the n_r+1 nested window boundaries of _tile_uniform_radial_windows_boundary, as rasterised by its _draw_polygon_mask, and the center used. """
    m, n = imsize

    if center is None:
        center = np.nanmean(boundary_line[:-1], axis=0)

    edge_line_polar = np.vstack(_cart2pol(boundary_line[:,1] - center[1], boundary_line[:,0] - center[0])).T
    edge_line_central = np.vstack(_pol2cart(edge_line_polar[:,0], edge_line_polar[:,1])).T

    contour_r_internal_polar = _linspace_rows(edge_line_polar[:,0], n_r+1).T
    contour_r_internal_lines = [np.vstack(_pol2cart(l, edge_line_polar[:,1])).T for l in contour_r_internal_polar][:-1]

    all_dist_lines = contour_r_internal_lines + [edge_line_central]
    all_dist_lines = [np.vstack([ll[:,0] + center[1], ll[:,1] + center[0]]).T  for ll in all_dist_lines]

    int_lines = [(np.clip(ll[:,1].astype(np.int32), 0, m-1), np.clip(ll[:,0].astype(np.int32), 0, n-1)) for ll in all_dist_lines]
    return int_lines, center


def _tile_uniform_radial_windows_boundary_crop(imsize, n_r, n_theta, boundary_line,
                                               center=None,
                                               zero_angle=None,
                                               line_pixels=None):
    """ _tile_uniform_radial_windows_boundary computed only within the bounding box of the windows.

    Parameters
    ----------
    (see _tile_uniform_radial_windows_boundary)
    line_pixels : None or list of (rr, cc)
        the rasterised polygons of _radial_window_int_lines(imsize, n_r, boundary_line, center), if already computed (e.g. on the GPU). If None they are rasterised with skimage.draw.polygon.

    Returns
    -------
    spixels : (n_rows_crop, n_cols_crop) int32 array
        the region labels (0 = background) in the bounding box, identical to the corresponding crop of _tile_uniform_radial_windows_boundary(imsize, ...)
    (r0, c0) : (int, int)
        the top left corner of the bounding box in the image
    """
    from skimage.draw import polygon

    int_lines, center = _radial_window_int_lines(imsize, n_r, boundary_line, center=center)
    r0 = int(min(yy.min() for yy, xx in int_lines)); r1 = int(max(yy.max() for yy, xx in int_lines))
    c0 = int(min(xx.min() for yy, xx in int_lines)); c1 = int(max(xx.max() for yy, xx in int_lines))
    crop_shape = (r1-r0+1, c1-c0+1)

    if line_pixels is None:
        line_pixels = [polygon(yy, xx) for yy, xx in int_lines]
    all_dist_masks = []
    for rr, cc in line_pixels:
        img = np.zeros(crop_shape, dtype=bool)
        img[rr-r0, cc-c0] = 1
        all_dist_masks.append(img)
    all_dist_masks = [np.logical_xor(all_dist_masks[ii+1], all_dist_masks[ii]) for ii in range(len(all_dist_masks)-1)]

    XX, YY = np.meshgrid(range(c0, c1+1), range(r0, r1+1))
    theta = np.arctan2(YY-center[0],XX-center[1])
    theta = theta + np.pi

    if zero_angle is None:
        theta_bounds = np.linspace(0, 2*np.pi, n_theta+1)
    else:
        theta_bounds = np.mod(np.linspace(0, 2*np.pi, n_theta+1) + (180.-360./n_theta/2.)/180.*np.pi - zero_angle/180.*np.pi, 2*np.pi)

    angle_masks_list = []
    for ii in range(len(theta_bounds)-1):
        if theta_bounds[ii+1] > theta_bounds[ii]:
            mask_theta = np.logical_and( theta>=theta_bounds[ii], theta <= theta_bounds[ii+1])
        else:
            mask_theta = np.logical_or(np.logical_and(theta>=theta_bounds[ii], theta<=2*np.pi),
                                        np.logical_and(theta>=0, theta<=theta_bounds[ii+1]))
        angle_masks_list.append(mask_theta)

    spixels = np.zeros(crop_shape, dtype=np.int32)
    counter = 1
    for ii in range(len(all_dist_masks)):
        for jj in range(len(angle_masks_list)):
            mask = np.logical_and(all_dist_masks[ii], angle_masks_list[jj])
            spixels[mask] = counter
            counter += 1

    return spixels, (r0, c0)


def _uniq_tile_regions(boundary_grid, n_contours):
    """ the region labels of a tiling, padded as in compute_boundary_texture_features / compute_boundary_motion_features """
    uniq_regions = np.setdiff1d(np.unique(boundary_grid), 0)
    if len(uniq_regions) != 3 and len(uniq_regions) < n_contours:
        uniq_regions = np.hstack([uniq_regions, [uniq_regions[-1]]*(n_contours-len(uniq_regions))])
    return uniq_regions


def _run_fast_feature_chunks(chunk_fn, n_obj, n_jobs, args):
    """ run chunk_fn(obj_index_array, *args) over chunks of objects and frames (in parallel with joblib if n_jobs != 1) and return the per object results in object order.

    args ends with (frames, opts). With few objects (e.g. one object tracked over a long video) the frames are also split into chunks, so all workers are used.
    """
    from tqdm import tqdm
    from joblib import Parallel, delayed, effective_n_jobs

    n_jobs = effective_n_jobs(n_jobs)
    if n_obj == 0:
        return []
    *head, frames, opts = args
    frames = np.asarray(frames)
    if n_jobs == 1:
        obj_chunks = [np.array([ii]) for ii in range(n_obj)]
        frame_chunks = [frames]
    else:
        n_tasks = 4*n_jobs
        obj_chunks = [c for c in np.array_split(np.arange(n_obj), min(n_obj, n_tasks)) if len(c) > 0]
        n_frame_chunks = max(1, min(len(frames), int(np.ceil(n_tasks / len(obj_chunks)))))
        frame_chunks = [f for f in np.array_split(frames, n_frame_chunks) if len(f) > 0]
    tasks = [(c, f) for c in obj_chunks for f in frame_chunks]
    if n_jobs == 1:
        results = [chunk_fn(c, *head, f, opts) for c, f in tqdm(tasks)]
    else:
        results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(chunk_fn)(c, *head, f, opts) for c, f in tqdm(tasks))
    # per object: concatenate the frame chunks in order
    per_obj = []
    pos = 0
    for c in obj_chunks:
        parts = results[pos:pos+len(frame_chunks)]
        pos += len(frame_chunks)
        for kk in range(len(c)):
            per_obj.append([val for part in parts for val in part[kk]])
    return per_obj


def _assemble_fast_features(per_obj, n_obj, n_time):
    """ (n_obj, n_time, n_feat) array, labels and normalised flags from per object lists of None or (feats, labels, norm) """
    out_feat_labels = []
    out_feat_labels_norm = []
    for obj in per_obj:
        for val in obj:
            if val is not None:
                out_feat_labels, out_feat_labels_norm = val[1], val[2]
                break
        if len(out_feat_labels) > 0:
            break
    out_array = np.zeros((n_obj, n_time, len(out_feat_labels)))
    for ii in range(n_obj):
        for jj in range(n_time):
            val = per_obj[ii][jj]
            if val is None:
                out_array[ii,jj] = np.nan
            else:
                out_array[ii,jj] = val[0]
    return out_array, out_feat_labels, out_feat_labels_norm


def _morphology_features_sample_fast(boundary_ii_tt, imshape, opts, pre=None):
    """ shape features of one contour, as in compute_boundary_morphology_features. pre: optional dict of precomputed 'reoriented' contour, contour 'binary' and 'ect' features (e.g. by the torch variant). """
    from sklearn.metrics.pairwise import pairwise_distances

    all_feats_compute = opts['all_feats_compute']
    pixel_xy_res = opts['pixel_xy_res']

    pre = {} if pre is None else pre
    boundary_ii_tt = pre['reoriented'] if 'reoriented' in pre else reorient_contour_fast(boundary_ii_tt, imshape)

    all_feats = []
    all_feats_labels = []
    all_feats_norm_bool = []

    if all_feats_compute==True or (all_feats_compute==False and opts['contour_curve_feats']==True):
        smooth_curve, deriv_curve, curvature = curvature_splines(boundary_ii_tt[:,1]*pixel_xy_res,
                                                                 y=boundary_ii_tt[:,0]*pixel_xy_res,
                                                                 k=opts['curve_order'],
                                                                 error=opts['curve_err']*pixel_xy_res)
        curvature_feats = np.hstack([ np.max(curvature),
                                      np.min(curvature),
                                      np.mean(curvature),
                                      np.mean(np.abs(curvature)),
                                      np.std(curvature),
                                      np.mean((curvature-np.mean(curvature))**3) / (np.std(curvature)**1.5 + 1e-8),
                                      np.mean((curvature-np.mean(curvature))**4) / (np.std(curvature)**4 + 1e-8)])
        curvature_feats_labels = np.hstack(['max_curvature',
                                            'min_curvature',
                                            'mean_curvature',
                                            'mean_abs_curvature',
                                            'std_curvature',
                                            'mean_skew_curvature',
                                            'mean_kurtosis_curvature'])
        all_feats.append(curvature_feats)
        all_feats_labels.append(curvature_feats_labels)
        all_feats_norm_bool.append(np.hstack([False]*len(curvature_feats)))

    if all_feats_compute==True or (all_feats_compute==False and opts['geom_features']==True):
        cnt_centroid_distances = np.linalg.norm(boundary_ii_tt[:-1]*pixel_xy_res - boundary_ii_tt[:-1].mean(axis=0)[None,:]*pixel_xy_res, axis=-1)

        max_cell_radius = np.max(cnt_centroid_distances)
        cell_radius_ratio = np.max(cnt_centroid_distances) / (np.min(cnt_centroid_distances) + 1)
        mean_cell_radius = np.mean(cnt_centroid_distances)
        cv_cell_radius = np.std(cnt_centroid_distances) / mean_cell_radius

        chordal_distances = pairwise_distances(boundary_ii_tt)
        max_span = chordal_distances.max()

        chordal_hist, _ = np.histogram(chordal_distances[np.triu_indices(len(chordal_distances))]/max_span, range=(0,1), bins=8)
        chordal_hist = chordal_hist/float(np.sum(chordal_hist))

        geom_feats_nonnorm = np.hstack([max_cell_radius,
                                        mean_cell_radius,
                                        cv_cell_radius,
                                        max_span])
        all_feats.append(geom_feats_nonnorm)
        all_feats_labels.append(np.hstack(['max_centroid_distance',
                                           'mean_centroid_distance',
                                           'std_mean_centroid_distance_ratio',
                                           'max_chordal_distance']))
        all_feats_norm_bool.append(np.hstack([False]*len(geom_feats_nonnorm)))

        geom_feats_norm = np.hstack([cell_radius_ratio,
                                     chordal_hist])
        all_feats.append(geom_feats_norm)
        all_feats_labels.append(np.hstack(['max_min_centroid_distance_ratio',
                                           np.hstack(['chordal_hist_bin_%s'%(str(bin_i+1)) for bin_i in np.arange(len(chordal_hist))])]))
        all_feats_norm_bool.append(np.hstack([True]*len(geom_feats_norm)))

    cnt_binary = None
    if all_feats_compute==True or (all_feats_compute==False and opts['morph_features']==True):
        cnt_binary = pre['binary'] if 'binary' in pre else contour_to_binary(boundary_ii_tt, shape=None)
        cnt_binary_feats_nonnorm, cnt_binary_feats_norm = extract_binary_regionprops(cnt_binary, pixel_xy_res=pixel_xy_res)

        cnt_binary_feats_nonnorm, cnt_binary_feats_nonnorm_labels = cnt_binary_feats_nonnorm
        cnt_binary_feats_norm, cnt_binary_feats_norm_labels = cnt_binary_feats_norm

        all_feats.append(cnt_binary_feats_nonnorm)
        all_feats_labels.append(cnt_binary_feats_nonnorm_labels)
        all_feats_norm_bool.append(np.hstack([False]*len(cnt_binary_feats_nonnorm)))

        all_feats.append(cnt_binary_feats_norm)
        all_feats_labels.append(cnt_binary_feats_norm_labels)
        all_feats_norm_bool.append(np.hstack([True]*len(cnt_binary_feats_norm_labels)))

    if all_feats_compute==True or (all_feats_compute==False and opts['fourier_features']==True):
        cnt_fourier = compute_fft_boundary_feats([boundary_ii_tt], pixel_xy_res=pixel_xy_res,
                                                 remove_closing_point=opts.get('fourier_remove_closing_point', False)).ravel()
        cnt_fourier = np.abs(cnt_fourier[1:len(cnt_fourier)//2])
        cnt_fourier = np.abs(cnt_fourier)**(0.2)
        cnt_fourier = cnt_fourier / (np.linalg.norm(cnt_fourier) + 1e-8)

        all_feats.append(cnt_fourier)
        all_feats_labels.append(np.hstack(['Fourier_%s' %(str(f_ii+1)) for f_ii in range(len(cnt_fourier))]))
        all_feats_norm_bool.append(np.hstack([True]*len(cnt_fourier)))

    if all_feats_compute==True or (all_feats_compute==False and opts['ect_features']==True):
        # as in the original, the binary comes from the morphological features block
        ect_feat = pre['ect'] if 'ect' in pre else _ect_fast(cnt_binary, _regular_directions(36, dims=2), T=32)

        all_feats.append(ect_feat)
        all_feats_labels.append(np.hstack(['ECT_%s' %(str(f_ii+1)) for f_ii in range(len(ect_feat))]))
        all_feats_norm_bool.append(np.hstack([True]*len(ect_feat)))

    if all_feats_compute==True or (all_feats_compute==False and opts['shape_context_features']==True):
        scm_feats = shape_context_fast(boundary_ii_tt*pixel_xy_res,
                                       n_ref=opts['n_ref_pts_scm'],
                                       n_dist=5,
                                       n_theta=12, b_tangent=1)[0].flatten(order='F')
        if opts['norm_scm']:
            scm_feats = scm_feats / (np.linalg.norm(scm_feats, ord=2) + 1e-8)

        all_feats.append(scm_feats)
        all_feats_labels.append(np.hstack(['Shape_Context_%s' %(str(f_ii+1)) for f_ii in range(len(scm_feats))]))
        all_feats_norm_bool.append(np.hstack([True]*len(scm_feats)))

    return np.hstack(all_feats), np.hstack(all_feats_labels), np.hstack(all_feats_norm_bool)


def _morphology_features_chunk_fast(obj_ids, boundaries, imshape, frames, opts):
    out = []
    for b_ii in obj_ids:
        descriptors = []
        for tt in frames:
            boundary_ii_tt = boundaries[b_ii][tt]
            if np.isnan(boundary_ii_tt[0][0]) == True:
                descriptors.append(None)
            else:
                descriptors.append(_morphology_features_sample_fast(boundary_ii_tt, imshape, opts))
        out.append(descriptors)
    return out


def compute_boundary_morphology_features_fast(boundaries, imshape,
                                              all_feats_compute=True,
                                              contour_curve_feats=True,
                                              curve_order=4, curve_err=1.,
                                              geom_features=True,
                                              morph_features=True,
                                              fourier_features=True,
                                              ect_features=True,
                                              shape_context_features=True,
                                              norm_scm=True,
                                              n_ref_pts_scm=5,
                                              pixel_xy_res=1.,
                                              timesample=None,
                                              n_jobs=1,
                                              fourier_remove_closing_point=False):
    """ Fast variant of compute_boundary_morphology_features with the same parameters and outputs.

    The contour reorientation rasterises the shape in its bounding box instead of a full image canvas, the Euler characteristic transform builds its cubical complex with array operations and the shape context histograms use np.bincount. Objects can be processed in parallel.

    Parameters
    ----------
    (see compute_boundary_morphology_features)
    n_jobs : int
        number of parallel processes over objects (joblib, -1 = all CPUs). With n_jobs=1 the output is identical to the original function; in parallel,
        the worker processes limit the BLAS threads, so BLAS-based statistics (the Haralick features) can differ in floating point round-off (~1e-15 relative), as the original function does under different BLAS thread settings,
        and last-bit differences of the vectorised log / arctan2 can rarely move a contour point across a shape context histogram bin edge.
        boundaries are used in C order; the original function's output can differ in rare tie-breaking cases (pixel-grid contours) when given a non C-contiguous array (e.g. from scipy.io.loadmat).
    fourier_remove_closing_point : bool
        if True, the Fourier features drop the repeated closing point of the contour, which makes them exactly invariant
        to rotation and to the contour start point (there is then one Fourier feature less). False (default) keeps the
        original features.

    Returns
    -------
    out_array : (n_organoids, n_frames, n_features) array
        the concatenated shape features for every organoid in every timepoint
    out_feat_labels : (n_features,) str array
        the names of each feature
    out_feat_labels_norm : (n_features,) str array
        a boolean vector where 1 or True means the feature is normalised i.e. dimensionless.
    """
    # C order: the features are sensitive at the last bit to the memory layout of the contours (e.g. reductions such as
    # the contour mean), which can flip ties on pixel-grid contours; a fixed layout makes all paths identical.
    boundaries = np.ascontiguousarray(boundaries)
    if timesample is None:
        timesample = 1
    if all_feats_compute==False and ect_features==True and morph_features==False:
        raise ValueError('ect_features requires morph_features=True, as in compute_boundary_morphology_features')
    frames = np.arange(0, boundaries.shape[1], timesample)
    opts = dict(all_feats_compute=all_feats_compute, contour_curve_feats=contour_curve_feats, curve_order=curve_order,
                curve_err=curve_err, geom_features=geom_features, morph_features=morph_features,
                fourier_features=fourier_features, ect_features=ect_features,
                shape_context_features=shape_context_features, norm_scm=norm_scm, n_ref_pts_scm=n_ref_pts_scm,
                pixel_xy_res=pixel_xy_res, fourier_remove_closing_point=fourier_remove_closing_point)
    per_obj = _run_fast_feature_chunks(_morphology_features_chunk_fast, len(boundaries), n_jobs,
                                       (boundaries, imshape, frames, opts))
    return _assemble_fast_features(per_obj, len(boundaries), len(frames))


def _texture_haralick_input(vid_cnt, cnt_binary, max_dist, opts):
    """ the integer image and distance passed to the (masked) Haralick features in compute_boundary_texture_features """
    distance = np.minimum(max_dist, opts['haralick_distance'])
    if opts['use_mask']:
        vid_cnt_q = vid_cnt
        if opts['masked_haralick_levels'] is not None:
            levels = opts['masked_haralick_levels']
            vmin, vmax = (0., 255.) if opts['use_gradient_vid'] else opts['vid_range']
            vid_cnt_q = np.clip(np.floor((vid_cnt.astype(np.float64) - vmin) / (vmax - vmin + 1e-8) * levels),
                                0, levels-1).astype(np.int32)
        return vid_cnt_q, distance
    return vid_cnt, distance


def _texture_crop(vid_frame, fg_frame, boundary_ii_tt, opts, binary_bbox=None):
    """ the cropped image, contour mask (intersected with the foreground) and bounding box of one sample, as in compute_boundary_texture_features. None if the crop is empty. """
    import skimage.transform as sktform
    cnt_binary, bbox = binary_bbox if binary_bbox is not None else contour_to_binary(boundary_ii_tt, shape=None, return_bbox=True)
    vid_cnt = vid_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1].copy()
    if vid_cnt.shape[0] ==0 or vid_cnt.shape[1]==0:
        return None
    cnt_binary = sktform.resize(cnt_binary, output_shape=vid_cnt.shape, order=0, preserve_range=True) > 0
    if fg_frame is not None:
        cnt_binary = np.logical_and(cnt_binary, fg_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1])
    max_dist = np.min(vid_cnt.shape)-1
    if opts['use_gradient_vid']:
        vid_cnt = sobel_frame(vid_cnt)
    return vid_cnt, cnt_binary, bbox, max_dist


def _texture_features_sample_fast(vid_frame, fg_frame, boundary_ii_tt, opts, siftextractor, pre=None):
    """ appearance features of one contour in one frame, as in compute_boundary_texture_features. Returns None where the original returns NaN. pre: optional dict of precomputed 'binary_bbox', regional 'tile', 'haralick' features and 'sift' descriptor (before L2 normalisation) (e.g. by the torch variant). """
    import mahotas as mh
    import skimage.transform as sktform

    compute_all_feats = opts['compute_all_feats']
    use_mask = opts['use_mask']
    pre = {} if pre is None else pre

    all_feats = []
    all_feats_labels = []
    all_feats_norm_bool = []

    crop = _texture_crop(vid_frame, fg_frame, boundary_ii_tt, opts, binary_bbox=pre.get('binary_bbox'))
    if crop is None:
        return None
    vid_cnt, cnt_binary, bbox, max_dist = crop

    if compute_all_feats==True or (compute_all_feats==False and opts['compute_intensity_feats']==True):
        mean_cnt_global = np.nanmean(vid_cnt[cnt_binary>0])
        std_cnt_global = np.nanstd(vid_cnt[cnt_binary>0])

        all_feats.append(np.hstack([mean_cnt_global,
                                    std_cnt_global]))
        all_feats_labels.append(np.hstack(['mean_intensity',
                                           'stddev_intensity']))
        all_feats_norm_bool.append(np.hstack([False]*2))

    if compute_all_feats==True or (compute_all_feats==False and opts['compute_contour_intensity_feats']==True):
        if 'tile' in pre:
            boundary_grid, (r0, c0) = pre['tile']
        else:
            boundary_grid, (r0, c0) = _tile_uniform_radial_windows_boundary_crop(imsize=vid_frame.shape,
                                                                                 n_r=opts['n_contours'], n_theta=opts['n_angles'],
                                                                                 boundary_line=boundary_ii_tt,
                                                                                 center=None,
                                                                                 zero_angle=opts['angle_start'])
        win = (slice(r0, r0+boundary_grid.shape[0]), slice(c0, c0+boundary_grid.shape[1]))
        vid_win = vid_frame[win]
        fg_win = fg_frame[win] if fg_frame is not None else None
        uniq_regions = _uniq_tile_regions(boundary_grid, opts['n_contours'])

        contour_img_feats = []
        contour_img_feats_labels = []
        for reg in uniq_regions:
            reg_mask = boundary_grid==reg
            if fg_win is not None:
                reg_mask = np.logical_and(reg_mask, fg_win)

            contour_img_feats.append(np.nanmean(vid_win[reg_mask>0]))
            contour_img_feats.append(np.nanstd(vid_win[reg_mask>0]))

            contour_img_feats_labels.append('mean_intensity_region_%s' %(str(reg).zfill(3)))
            contour_img_feats_labels.append('stddev_intensity_region_%s' %(str(reg).zfill(3)))

        all_feats.append(np.hstack(contour_img_feats))
        all_feats_labels.append(np.hstack(contour_img_feats_labels))
        all_feats_norm_bool.append(np.hstack([False]*len(contour_img_feats_labels)))

    if compute_all_feats==True or (compute_all_feats==False and opts['compute_haralick']==True):
        har_im, har_distance = _texture_haralick_input(vid_cnt, cnt_binary, max_dist, opts)
        if 'haralick' in pre:
            cnt_texture = pre['haralick']
        elif use_mask:
            cnt_texture = _masked_haralick(har_im, cnt_binary, distance=har_distance)
        else:
            cnt_texture = mh.features.haralick(vid_cnt,
                                               ignore_zeros=False,
                                               preserve_haralick_bug=False,
                                               compute_14th_feature=False,
                                               return_mean=True,
                                               return_mean_ptp=False,
                                               use_x_minus_y_variance=False,
                                               distance=np.minimum(max_dist, opts['haralick_distance']))
        all_feats.append(cnt_texture)
        all_feats_labels.append(np.hstack(['angular_second_moment',
                                           'contrast',
                                           'correlation',
                                           'sum_squares_variance',
                                           'inv_diff_moment',
                                           'sum_average',
                                           'sum_variance',
                                           'sum_entropy',
                                           'entropy',
                                           'diff_variance',
                                           'diff_entropy',
                                           'info_measure_correlation_1',
                                           'info_measure_correlation_2']))
        all_feats_norm_bool.append(np.hstack([False]*len(cnt_texture)))

    if compute_all_feats==True or (compute_all_feats==False and opts['compute_sift']==True):
        siftshape = opts['siftshape']
        if use_mask:
            sift_texture = _masked_sift_descriptor(vid_cnt, cnt_binary, resize=opts['masked_sift_resize'], siftshape=siftshape)
        else:
            if 'sift' in pre:
                sift_texture = pre['sift']
            else:
                vid_cnt_im = sktform.resize(vid_cnt, siftshape, preserve_range=True)
                sift_texture = siftextractor.process_image(vid_cnt_im).ravel()
            sift_texture = sift_texture / (np.linalg.norm(sift_texture) + 1e-8)
        all_feats.append(sift_texture)
        all_feats_labels.append(np.hstack(['sift_appear_%s'%(str(iii+1)) for iii in np.arange(len(sift_texture))]))
        all_feats_norm_bool.append(np.hstack([True]*len(sift_texture)))

    return np.hstack(all_feats), np.hstack(all_feats_labels), np.hstack(all_feats_norm_bool)


def _texture_features_chunk_fast(obj_ids, vid, boundaries, masks, frames, opts):
    siftextractor = SingleSiftExtractor(opts['siftshape'][0])
    out = [[None]*len(frames) for b_ii in obj_ids]
    for jj, tt in enumerate(frames):
        fg_frame = np.asarray(masks[tt]) > 0 if (opts['use_mask'] and masks is not None) else None
        for kk, b_ii in enumerate(obj_ids):
            boundary_ii_tt = boundaries[b_ii][tt]
            if np.isnan(boundary_ii_tt[0][0]) == False:
                out[kk][jj] = _texture_features_sample_fast(vid[tt], fg_frame, boundary_ii_tt, opts, siftextractor)
    return out


def compute_boundary_texture_features_fast(vid, boundaries,
                                           use_gradient_vid=False,
                                           compute_all_feats =True,
                                           compute_intensity_feats=True,
                                           compute_contour_intensity_feats=True,
                                           n_contours=3,
                                           n_angles=1,
                                           angle_start=None,
                                           compute_sift=True,
                                           siftshape=(64,64),
                                           compute_haralick=True,
                                           haralick_distance=15,
                                           timesample=None,
                                           use_mask=False,
                                           masked_sift_resize='cap',
                                           masks=None,
                                           masked_haralick_levels=8,
                                           masked_haralick_range=None,
                                           n_jobs=1):
    """ Fast variant of compute_boundary_texture_features with the same parameters and outputs.

    The regional intensity windows are built in the bounding box of the object instead of the full frame and the SIFT extractor is created once. Objects can be processed in parallel.

    Parameters
    ----------
    (see compute_boundary_texture_features)
    n_jobs : int
        number of parallel processes over objects (joblib, -1 = all CPUs). With n_jobs=1 the output is identical to the original function; in parallel,
        the worker processes limit the BLAS threads, so BLAS-based statistics (the Haralick features) can differ in floating point round-off (~1e-15 relative), as the original function does under different BLAS thread settings,
        and last-bit differences of the vectorised log / arctan2 can rarely move a contour point across a shape context histogram bin edge.
        boundaries are used in C order; the original function's output can differ in rare tie-breaking cases (pixel-grid contours) when given a non C-contiguous array (e.g. from scipy.io.loadmat).

    Returns
    -------
    out_array : (n_organoids, n_frames, n_features) array
        the concatenated appearance features for every organoid in every timepoint
    out_feat_labels : (n_features,) str array
        the names of each feature
    out_feat_labels_norm : (n_features,) str array
        a boolean vector where 1 or True means the feature is normalised i.e. dimensionless.
    """
    # C order: the features are sensitive at the last bit to the memory layout of the contours (e.g. reductions such as
    # the contour mean), which can flip ties on pixel-grid contours; a fixed layout makes all paths identical.
    boundaries = np.ascontiguousarray(boundaries)
    if timesample is None:
        timesample = 1
    vid_range = None
    if use_mask and masked_haralick_levels is not None:
        if masked_haralick_range is not None:
            vid_range = masked_haralick_range
        elif np.issubdtype(np.asarray(vid).dtype, np.integer):
            vid_range = (np.iinfo(np.asarray(vid).dtype).min, np.iinfo(np.asarray(vid).dtype).max)
        else:
            vid_range = (float(np.nanmin(vid)), float(np.nanmax(vid)))
    frames = np.arange(0, boundaries.shape[1], timesample)
    opts = dict(use_gradient_vid=use_gradient_vid, compute_all_feats=compute_all_feats,
                compute_intensity_feats=compute_intensity_feats,
                compute_contour_intensity_feats=compute_contour_intensity_feats, n_contours=n_contours,
                n_angles=n_angles, angle_start=angle_start, compute_sift=compute_sift, siftshape=siftshape,
                compute_haralick=compute_haralick, haralick_distance=haralick_distance, use_mask=use_mask,
                masked_sift_resize=masked_sift_resize, masked_haralick_levels=masked_haralick_levels,
                vid_range=vid_range)
    per_obj = _run_fast_feature_chunks(_texture_features_chunk_fast, len(boundaries), n_jobs,
                                       (vid, boundaries, masks, frames, opts))
    return _assemble_fast_features(per_obj, len(boundaries), len(frames))


def _motion_features_sample_fast(flow_frame, curl_vid_frame, div_vid_frame, fg_frame, boundary_ii_tt,
                                 boundary_ii_tt_plus_1, opts, siftextractor, pre=None, sift_inputs=None):
    """ motion features of one contour between two frames, as in compute_boundary_motion_features. Returns None where the original returns NaN.

    pre: optional dict of precomputed 'binary_bbox', regional 'tile' and 'sift' descriptors (list of 3, before L2 normalisation) (e.g. by the torch variant).
    sift_inputs: if a list is given, the 3 resized images whose (unmasked) SIFT descriptors are needed are appended to it and the function returns before computing any feature.
    """
    import skimage.transform as sktform

    compute_all_feats = opts['compute_all_feats']
    use_mask = opts['use_mask']
    pixel_res = opts['pixel_res']
    time_res = opts['time_res']
    n_contour_feat_bins = opts['n_contour_feat_bins']
    siftshape = opts['siftshape']

    pre = {} if pre is None else pre

    all_feats = []
    all_feats_labels = []
    all_feats_norm_bool = []

    cnt_binary, bbox = pre['binary_bbox'] if 'binary_bbox' in pre else contour_to_binary(boundary_ii_tt, shape=None, return_bbox=True)
    flow_vid_cnt = flow_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1].copy()

    if flow_vid_cnt.shape[0]==0 or flow_vid_cnt.shape[1]==0 or cnt_binary.shape[0]==0 or cnt_binary.shape[1]==0:
        return None

    curl_vid_cnt = curl_vid_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1].copy()
    div_vid_cnt = div_vid_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1].copy()

    cnt_binary = sktform.resize(cnt_binary, output_shape=flow_vid_cnt.shape[:-1], order=0, preserve_range=True) > 0
    if fg_frame is not None:
        cnt_binary = np.logical_and(cnt_binary, fg_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1])

    if use_mask:
        flow_vid_cnt_fill = np.dstack([_fill_outside_mask_nearest(flow_vid_cnt[...,ch], cnt_binary) for ch in range(flow_vid_cnt.shape[-1])])
        curl_vid_cnt = _curl_vector_flow(flow_vid_cnt_fill)[...,-1] / float(time_res)
        div_vid_cnt = _div_vector_flow(flow_vid_cnt_fill) / float(time_res)

    if sift_inputs is not None:
        # only collect the resized images for the unmasked SIFT descriptors
        flow_vid_cnt_ = np.linalg.norm(flow_vid_cnt * pixel_res / float(time_res), axis=-1)
        for sift_im in (flow_vid_cnt_, curl_vid_cnt, div_vid_cnt):
            sift_im = sift_im.copy()
            sift_im[cnt_binary>0] = 0
            sift_inputs.append(sktform.resize(sift_im, siftshape, preserve_range=True))
        return None

    if compute_all_feats==True or (compute_all_feats==False and opts['compute_global_feats']==True):
        disp = (boundary_ii_tt_plus_1 - boundary_ii_tt) * pixel_res / float(time_res)
        mean_disp_speed = np.linalg.norm( np.nanmean(boundary_ii_tt_plus_1[:-1], axis=0) -
                                          np.nanmean(boundary_ii_tt[:-1], axis=0), axis=-1) * pixel_res / float(time_res)
        std_disp_speed = np.nanstd(np.linalg.norm(disp, axis=-1))

        all_feats.append(np.hstack([mean_disp_speed,std_disp_speed]))
        all_feats_labels.append(np.hstack(['mean_speed_global',
                                           'stddev_speed_global']))
        all_feats_norm_bool.append(np.hstack([False]*2))

        flow_vid_cnt_ = flow_vid_cnt.copy() * pixel_res / float(time_res)
        mean_disp_speed_flow = np.nanmean( np.linalg.norm(flow_vid_cnt_[cnt_binary>0], axis=-1))
        std_disp_speed_flow = np.nanstd( np.linalg.norm(flow_vid_cnt_[cnt_binary>0], axis=-1) )

        all_feats.append(np.hstack([mean_disp_speed_flow,
                                    std_disp_speed_flow]))
        all_feats_labels.append(np.hstack(['mean_speed_global_flow',
                                           'stddev_speed_global_flow']))
        all_feats_norm_bool.append(np.hstack([False]*2))

        mean_curl_mag = np.nanmean( curl_vid_cnt[cnt_binary>0])
        std_curl_mag = np.nanstd( curl_vid_cnt[cnt_binary>0])
        all_feats.append(np.hstack([mean_curl_mag,
                                    std_curl_mag]))
        all_feats_labels.append(np.hstack(['mean_curl_global_flow',
                                           'stddev_curl_global_flow']))
        all_feats_norm_bool.append(np.hstack([False]*2))

        mean_div_mag = np.nanmean( div_vid_cnt[cnt_binary>0])
        std_div_mag = np.nanstd( div_vid_cnt[cnt_binary>0])
        all_feats.append(np.hstack([mean_div_mag,
                                    std_div_mag]))
        all_feats_labels.append(np.hstack(['mean_divergence_global_flow',
                                           'stddev_divergence_global_flow']))
        all_feats_norm_bool.append(np.hstack([False]*2))

    if compute_all_feats==True or (compute_all_feats==False and opts['compute_contour_feats']==True):
        if 'tile' in pre:
            boundary_grid, (r0, c0) = pre['tile']
        else:
            boundary_grid, (r0, c0) = _tile_uniform_radial_windows_boundary_crop(imsize=flow_frame.shape[:-1],
                                                                                 n_r=opts['n_contours'], n_theta=opts['n_angles'],
                                                                                 boundary_line=boundary_ii_tt,
                                                                                 center=None,
                                                                                 zero_angle=opts['angle_start'])
        win = (slice(r0, r0+boundary_grid.shape[0]), slice(c0, c0+boundary_grid.shape[1]))
        flow_win = flow_frame[win]
        fg_win = fg_frame[win] if fg_frame is not None else None
        uniq_regions = _uniq_tile_regions(boundary_grid, opts['n_contours'])

        for reg in uniq_regions:
            reg_mask = boundary_grid==reg
            if fg_win is not None:
                reg_mask = np.logical_and(reg_mask, fg_win)

            flow_reg_mask = flow_win[reg_mask>0] * pixel_res / float(time_res)

            if len(flow_reg_mask) > n_contour_feat_bins:
                flow_reg_mask_mag = np.linalg.norm(flow_reg_mask, axis=-1)
                lower_bin = np.nanmean(flow_reg_mask_mag) - np.nanstd(flow_reg_mask_mag) * opts['cnt_sigma']
                upper_bin = np.nanmean(flow_reg_mask_mag) + np.nanstd(flow_reg_mask_mag) * opts['cnt_sigma']
                x_grid = np.linspace(lower_bin, upper_bin, n_contour_feat_bins)
                if np.sum(flow_reg_mask_mag>1e-3) > 0 and np.nanstd(flow_reg_mask_mag) < 1e-8:
                    hist_flow = np.ones(n_contour_feat_bins) / float(n_contour_feat_bins)
                elif np.sum(flow_reg_mask_mag>1e-3) > 0:
                    hist_flow = _kde_scipy(flow_reg_mask_mag, x_grid, bandwidth=.2)
                    hist_flow = hist_flow / float(np.sum(hist_flow) + 1e-8)
                else:
                    hist_flow = np.zeros(n_contour_feat_bins)
            else:
                hist_flow = np.zeros(n_contour_feat_bins)

            all_feats.append(hist_flow)
            all_feats_labels.append(['hist_bin_%d_region_%s' %(bbb+1, str(reg).zfill(3)) for bbb in range(len(hist_flow))])
            all_feats_norm_bool.append(np.hstack([True]*len(hist_flow)))

            mean_disp_speed_flow_region = np.nanmean( np.linalg.norm(flow_reg_mask, axis=-1))
            std_disp_speed_flow_region = np.nanstd( np.linalg.norm(flow_reg_mask, axis=-1) )

            all_feats.append(np.hstack([mean_disp_speed_flow_region,
                                        std_disp_speed_flow_region]))
            all_feats_labels.append(np.hstack(['mean_speed_flow_region_%d' %(reg),
                                               'stddev_speed_flow_region%d' %(reg)]))
            all_feats_norm_bool.append(np.hstack([False]*2))

    if compute_all_feats==True or (compute_all_feats==False and opts['compute_sift_feats']==True):
        flow_vid_cnt_ = flow_vid_cnt * pixel_res / float(time_res)
        flow_vid_cnt_ = np.linalg.norm(flow_vid_cnt_, axis=-1)

        sift_images = [(flow_vid_cnt_, 'sift_motion_%s'),
                       (curl_vid_cnt, 'curl_sift_%s'),
                       (div_vid_cnt, 'div_sift_%s')]
        for sift_k, (sift_im, sift_label) in enumerate(sift_images):
            if use_mask:
                sift_texture = _masked_sift_descriptor(sift_im, cnt_binary, resize=opts['masked_sift_resize'], siftshape=siftshape)
            else:
                if 'sift' in pre:
                    sift_texture = pre['sift'][sift_k]
                else:
                    sift_im = sift_im.copy()
                    sift_im[cnt_binary>0] = 0
                    sift_im = sktform.resize(sift_im, siftshape, preserve_range=True)
                    sift_texture = siftextractor.process_image(sift_im).ravel()
                sift_texture = sift_texture / (np.linalg.norm(sift_texture) + 1e-8)
            all_feats.append(sift_texture)
            all_feats_labels.append(np.hstack([sift_label%(str(iii+1)) for iii in np.arange(len(sift_texture))]))
            all_feats_norm_bool.append(np.hstack([True]*len(sift_texture)))

    return np.hstack(all_feats), np.hstack(all_feats_labels), np.hstack(all_feats_norm_bool)


def _motion_features_chunk_fast(obj_ids, vid_flow, boundaries, masks, frames, opts):
    siftextractor = SingleSiftExtractor(opts['siftshape'][0])
    out = [[None]*len(frames) for b_ii in obj_ids]
    for jj, tt in enumerate(frames):
        todo = [kk for kk, b_ii in enumerate(obj_ids)
                if np.isnan(boundaries[b_ii][tt][0][0]) == False and np.isnan(boundaries[b_ii][tt+1][0][0]) == False]
        if len(todo) == 0:
            continue
        # curl and divergence once per frame
        flow_frame = np.asarray(vid_flow[tt]).copy()
        curl_vid_frame = _curl_vector_flow(flow_frame)[...,-1] / float(opts['time_res'])
        div_vid_frame = _div_vector_flow(flow_frame) / float(opts['time_res'])
        fg_frame = np.asarray(masks[tt]) > 0 if (opts['use_mask'] and masks is not None) else None
        for kk in todo:
            b_ii = obj_ids[kk]
            out[kk][jj] = _motion_features_sample_fast(flow_frame, curl_vid_frame, div_vid_frame, fg_frame,
                                                       boundaries[b_ii][tt], boundaries[b_ii][tt+1], opts,
                                                       siftextractor)
    return out


def compute_boundary_motion_features_fast(vid_flow,
                                          boundaries,
                                          compute_all_feats =True,
                                          compute_global_feats=True,
                                          compute_contour_feats=True,
                                          n_contour_feat_bins=8,
                                          cnt_sigma=3.,
                                          n_contours=3,
                                          n_angles=1,
                                          angle_start=None,
                                          pixel_res=1.,
                                          time_res=1.,
                                          compute_sift_feats=True,
                                          siftshape=(64,64),
                                          timesample=None,
                                          use_mask=False,
                                          masked_sift_resize='cap',
                                          masks=None,
                                          n_jobs=1):
    """ Fast variant of compute_boundary_motion_features with the same parameters and outputs.

    Curl and divergence are computed once per frame instead of once per object and frame, the regional speed windows are built in the bounding box of the object instead of the full frame and the SIFT extractor is created once. Objects can be processed in parallel.

    Parameters
    ----------
    (see compute_boundary_motion_features)
    n_jobs : int
        number of parallel processes over objects (joblib, -1 = all CPUs). With n_jobs=1 the output is identical to the original function; in parallel,
        the worker processes limit the BLAS threads, so BLAS-based statistics (the Haralick features) can differ in floating point round-off (~1e-15 relative), as the original function does under different BLAS thread settings,
        and last-bit differences of the vectorised log / arctan2 can rarely move a contour point across a shape context histogram bin edge.
        boundaries are used in C order; the original function's output can differ in rare tie-breaking cases (pixel-grid contours) when given a non C-contiguous array (e.g. from scipy.io.loadmat).

    Returns
    -------
    out_array : (n_organoids, n_frames-1, n_features) array
        the concatenated motion features for every organoid in every timepoint
    out_feat_labels : (n_features,) str array
        the names of each feature
    out_feat_labels_norm : (n_features,) str array
        a boolean vector where 1 or True means the feature is normalised i.e. dimensionless.
    """
    # C order: the features are sensitive at the last bit to the memory layout of the contours (e.g. reductions such as
    # the contour mean), which can flip ties on pixel-grid contours; a fixed layout makes all paths identical.
    boundaries = np.ascontiguousarray(boundaries)
    if timesample is None:
        timesample = 1
    frames = np.arange(0, boundaries.shape[1]-1, timesample)
    opts = dict(compute_all_feats=compute_all_feats, compute_global_feats=compute_global_feats,
                compute_contour_feats=compute_contour_feats, n_contour_feat_bins=n_contour_feat_bins,
                cnt_sigma=cnt_sigma, n_contours=n_contours, n_angles=n_angles, angle_start=angle_start,
                pixel_res=pixel_res, time_res=time_res, compute_sift_feats=compute_sift_feats, siftshape=siftshape,
                use_mask=use_mask, masked_sift_resize=masked_sift_resize)
    per_obj = _run_fast_feature_chunks(_motion_features_chunk_fast, len(boundaries), n_jobs,
                                       (vid_flow, boundaries, masks, frames, opts))
    return _assemble_fast_features(per_obj, len(boundaries), len(frames))
