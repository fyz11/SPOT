# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 18:30:30 2023

@author: S205272
"""
import numpy as np

def scale_normalize_curvature_features(all_feats, feature_names):
    r""" Curvature features are not invariant to size of the object. It is of units length^-1. In order to allow like-wise comparison across different objects, we multiply features by the equivalent diameter of the object. 
    
    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be analyzed. 
    feature_names : (n_feats,) np.array
        numpy array of the names of each computed feature
    
    Returns
    -------
    all_feats_corrected : (n_objects, n_feats) np.array
        numpy array of all features with size-corrected curvature features
    feature_names : (n_feats,) np.array
        numpy array of the names of each computed feature. It is identical to the input and is output for consistent api
    
    """
    
    all_feats_corrected = all_feats.copy()
    
    # correction for curvature features. 
    for feat_ii, feature_name in enumerate(feature_names):
        if 'curvature' in feature_name:
            all_feats_corrected[:,feat_ii] = all_feats[:,feat_ii] * all_feats[:,feature_names == 'equivalent_diameter'].ravel()

    return all_feats_corrected, feature_names
    
    
# remove entirely zero-valued features
def remove_zero_features(all_feats, feature_names, return_index=False):
    r""" remove any features that are zero-valued across all object instances as well as any that have zero standard deviation across all object instances as they are uninformative 
    
    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be analyzed. 
    feature_names : (n_feats,) np.array
        numpy array of the names of each computed feature
    
    Returns
    -------
    all_feats_corrected : (n_objects, n_feats) np.array
        numpy array of all features with zero-valued features removed
    feature_names_corrected : (n_feats,) np.array
        numpy array of the names of each retained feature
    
    """
    import numpy as np 
    
    zero_feats = np.nanmax(all_feats, axis=0) == 0    
        
    if return_index:
        select_index = np.arange(len(zero_feats))
        select_index = select_index[~zero_feats].copy()
        
    feature_names_corrected = feature_names[~zero_feats].copy()
    all_feats_corrected = all_feats[:, ~zero_feats].copy() 
    
    std_feats = np.std(all_feats_corrected, axis=0)
    zero_std_feats = std_feats == 0
    
    if return_index:
        select_index = select_index[~zero_std_feats].copy()
        
    feature_names_corrected = feature_names_corrected[~zero_std_feats]
    all_feats_corrected = all_feats_corrected[:, ~zero_std_feats].copy() # so we have 400 feats. 
    
    if return_index:
        return all_feats_corrected, feature_names_corrected, select_index
    else:
        return all_feats_corrected, feature_names_corrected


# remove entirely zero-valued features
def remove_nan_variance_features(all_feats, feature_names, return_index=False):
    r""" remove any features that have zero variation across all object instances as well as any that have zero standard deviation across all object instances as they are uninformative 
    
    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be analyzed. 
    feature_names : (n_feats,) np.array
        numpy array of the names of each computed feature
    
    Returns
    -------
    all_feats_corrected : (n_objects, n_feats) np.array
        numpy array of all features with zero-valued features removed
    feature_names_corrected : (n_feats,) np.array
        numpy array of the names of each retained feature
    
    """
    import numpy as np 
    
    zero_feats = np.isnan(np.std(all_feats, axis=0))     
        
    if return_index:
        select_index = np.arange(len(zero_feats))
        select_index = select_index[~zero_feats].copy()
    
    feature_names_corrected = feature_names[~zero_feats].copy()
    all_feats_corrected = all_feats[:, ~zero_feats].copy() 
    
    if return_index:
        return all_feats_corrected, feature_names_corrected, select_index
    else:
        return all_feats_corrected, feature_names_corrected

    
def remove_high_variance_features(all_feats, feature_names, variance_threshold_sigma=3, return_index=False):
    r""" remove features that are highly variable and likely to represent outliers
    
    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be analyzed. 
    feature_names : (n_feats,) np.array
        numpy array of the names of each computed feature
    
    Returns
    -------
    all_feats_corrected : (n_objects, n_feats) np.array
        numpy array of all features with high variance features removed
    feature_names_corrected : (n_feats,) np.array
        numpy array of the names of each retained feature. 
        
    """
    import numpy as np 
    
    high_var_feats_thresh = np.mean(np.std(all_feats, axis=0)) + variance_threshold_sigma*np.std(np.std(all_feats, axis=0))
    keep_feats = np.std(all_feats, axis=0)<=high_var_feats_thresh
    
    if return_index:
        select_index = np.arange(len(keep_feats))
        select_index = select_index[keep_feats].copy()
    # high_var_feats_thresh = np.mean(np.std(all_feats, axis=0)) + 2*np.std(np.std(all_feats, axis=0))
    # feature_names = feature_names[np.std(all_feats, axis=0)<=high_var_feats_thresh]
    
    feature_names_corrected = feature_names[keep_feats].copy()
    all_feats_corrected = all_feats[:, keep_feats].copy() 
        
    if return_index:
        return all_feats_corrected, feature_names_corrected, select_index
    else:
        return all_feats_corrected, feature_names_corrected


def select_time_varying_features(all_feats, feature_names, all_time, ridge_alpha=1., norm_time=True, cutoff_method='mean', offset_threshold=None):
    
    r""" Selects features that are most strongly correlated with time using linear ridge regression. Features are kept if the absolute value of their regression coefficient is greater than the mean across all features
    
    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be analyzed. 
    feature_names : (n_feats,) np.array
        numpy array of the names of each computed feature
    all_time : (n_objects, n_feats) np.array
        numpy array of the timepoint each object instance was extracted from. 
    ridge_alpha : non-negative float
        Constant that multiplies the L2 term, controlling regularization strength. alpha must be a non-negative float i.e. in [0, inf).
    norm_true : bool
        if True, assumes time is given as frame number and performs (all_time-1.)/(np.max(all_time-1.)) to rescale to 0-1 float
    cutoff_method : str
        one of 'mean' or 'median' for computing the coeff cutoff threshold
    offset_threshold : float
        optional modification of the inferred cutoff: cutoff_new = cutoff + offset_threshold * np.std( coeffs )
                                                                                                                               
    Returns
    -------
    all_feats_corrected : (n_objects, n_feats) np.array
        numpy array of all features with high variance features removed
    feature_names_corrected : (n_feats,) np.array
        numpy array of the names of each retained feature. 
    coeffs : (n_feats,) np.array
        fitted Ridge regression coefficients. The larger the absolute magnitude, the more the feature covaries with time.
    """
    
    # what about straightforward regression of features with time?
    from sklearn import linear_model
    import numpy as np 
    
    reg = linear_model.Ridge(alpha=1.)

    if norm_time:
        all_time_score = all_time-1.
        all_time_score = all_time_score/all_time_score.max() # to normalize. 
    else:
        all_time_score = all_time.copy()
        
    reg.fit(all_feats, all_time_score)
    coeffs = reg.coef_.copy()

    if cutoff_method=='mean':
        coef_thresh = np.mean(np.abs(coeffs)) # thresh by the mean 
    elif cutoff_method == 'median':
        coef_thresh = np.median(np.abs(coeffs)) # thresh by the mean 
    else:
        coef_thresh = np.mean(np.abs(coeffs)) # thresh by the mean 
    
    if offset_threshold is not None:
        coef_thresh = coef_thresh + offset_threshold*np.std(np.abs(coeffs))

    keep_feats = np.abs(reg.coef_)>=coef_thresh # create a binary indicator 
    
    feature_names_corrected = feature_names[keep_feats].copy()
    all_feats_corrected = all_feats[:, keep_feats].copy() 
    
    return all_feats_corrected, feature_names_corrected, coeffs


def kernel_dim_reduction_ECC_features(all_feats, feature_names, n_dim=100, gamma=None, random_state=1, return_index=False):
    r""" finds Euler Characteristic Curve features and using Nystroem with RBF kernel to perform dimensional reduction to the specified dimension size. 
    
    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be analyzed. 
    feature_names : (n_feats,) np.array
        numpy array of the names of each computed feature
    
    Returns
    -------
    all_feats_corrected : (n_objects, n_feats) np.array
        numpy array of all features with transformed kernel ECC features
    feature_names_corrected : (n_feats,) np.array
        numpy array of the names of each retained feature. 
    ECT_rbf_feature_tformer : Scikit-learn Nystroem transformer object
        the fitted sklearn.kernel_approximation.Nystroem object
        
    """
    from sklearn.kernel_approximation import Nystroem
    import numpy as np 
    
    # drop all ECC/ECT feats
    ECT_feats_index = np.arange(all_feats.shape[1])[np.hstack(['ECT_' in name for name in feature_names])]
    non_ECT_feats_index = np.setdiff1d(np.arange(all_feats.shape[1]), ECT_feats_index)

    
    # set up the transformer
    ECT_rbf_feature_tformer = Nystroem(n_components=n_dim, 
                                       gamma=gamma,
                                       random_state=random_state) # how many features? -> say 100 for now anyway. 
    ECT_rbf_feats = ECT_rbf_feature_tformer.fit_transform(all_feats[:,ECT_feats_index])
    
    # now modify 
    all_feats_corrected = all_feats[:,non_ECT_feats_index].copy() # keep as before. 
    feature_names_corrected = feature_names[non_ECT_feats_index].copy()
    
    # add in the transformed kernel features.
    all_feats_corrected = np.hstack([all_feats_corrected, 
                                     ECT_rbf_feats])
    feature_names_corrected = np.hstack([feature_names_corrected,
                                         ['ECT_rbf_%d'%(dd+1) for dd in np.arange(ECT_rbf_feats.shape[1])]])
    
    if return_index:
        return all_feats_corrected, feature_names_corrected, ECT_rbf_feature_tformer, (non_ECT_feats_index, ECT_feats_index)    
    else:
        return all_feats_corrected, feature_names_corrected, ECT_rbf_feature_tformer
    
    
def kernel_dim_reduction_ECC_features_with_transform(all_feats, indices, ECT_rbf_feature_tformer):
    r""" finds Euler Characteristic Curve features and using Nystroem with RBF kernel to perform dimensional reduction to the specified dimension size. 
    
    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be analyzed. 
    indices : ((M,), (N,)) np.array
        non ECC features index, ECC features index
    ECT_rbf_feature_tformer : Scikit-learn Nystroem transformer object
        the fitted sklearn.kernel_approximation.Nystroem object
    
    Returns
    -------
    all_feats_corrected : (n_objects, n_feats) np.array
        numpy array of all features with transformed kernel ECC features
        
    """
    import numpy as np 
    
    non_ECT_feats_index, ECT_feats_index = indices
    
    ECT_rbf_feats = ECT_rbf_feature_tformer.transform(all_feats[:,ECT_feats_index])
    
    # now modify 
    all_feats_corrected = all_feats[:,non_ECT_feats_index].copy() # keep as before. 
    
    # add in the transformed kernel features.
    all_feats_corrected = np.hstack([all_feats_corrected, 
                                     ECT_rbf_feats])
    
    return all_feats_corrected

    
    
# individual feature normalization. 
def power_transform_and_scale_features(all_feats, feature_names, all_conditions):
    r""" Applies standard scaling (zscore) followed by power transform using Yeo-Johnson to normalize, whiten features and make more Gaussian-like for each unique condition to control 
    
    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be analyzed. 
    feature_names : (n_feats,) np.array
        numpy array of the names of each computed feature
    all_conditions : integer or categorical np.array
        each unique integer corresponds to a unique condition. Within each condition we fit a different power transformer and standard scale transformer
        
    Returns
    -------
    all_feats_corrected : (n_objects, n_feats) np.array
        numpy array of all features with transformed kernel ECC features
    feature_names_corrected : (n_feats,) np.array
        numpy array of the names of each retained feature. 
    pow_tformer_dict : dict
        * 'all_obj_trajectories': list
                This is a list of all the reconstructed trajectories. Each trajectory is given as a list of the input indices that make it up. 
        * 'all_obj_trajectories_times': list
                This is a list of all the reconstructed trajectories. Each trajectory is a list of the frame number that the object instances occurred at 
    std_tformer_dict : dict
        * str(condition) : sklearn.preprocessing StandardScaler object
                Fitted StandardScaler object for the given unique condition 
        * 'all_obj_trajectories_times': list
                This is a list of all the reconstructed trajectories. Each trajectory is a list of the frame number that the object instances occurred at 
    """
    # =============================================================================
    #   Now we do individual feature normalisation -> modify in place. 
    # =============================================================================
    import numpy as np 
    from sklearn.preprocessing import PowerTransformer, StandardScaler
    
    all_feats_corrected = np.zeros(all_feats.shape)
    feature_names_corrected = feature_names.copy()
    
    pow_tformer_dict = {}
    std_tformer_dict = {}
    
    # normalizing over each condition 
    for uniq_condition in np.unique(all_conditions):
        
        select = all_conditions == uniq_condition
        
        pow_tformer = StandardScaler()
        std_tformer = PowerTransformer()
        
        feats_norm = pow_tformer.fit_transform(std_tformer.fit_transform(all_feats[select])) 
                            
        pow_tformer_dict[str(uniq_condition)] = pow_tformer
        std_tformer_dict[str(uniq_condition)] = std_tformer
        pow_tformer_dict[str(uniq_condition)+'_data_select'] = select
        std_tformer_dict[str(uniq_condition)+'_data_select'] = select
        
        all_feats_corrected[select] = feats_norm.copy()
        
    return all_feats_corrected, feature_names_corrected, pow_tformer_dict, std_tformer_dict 
    
    
def preprocess_feats(X, std_tformer, pow_tformer):
    r""" Applies already fitted scikit-learn standard scaler and powertransformer objects to the input matrix. The power transform is applied first then standard scale. 

    Parameters
    ----------
    X : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be transformed 
    std_tformer : sklearn.preprocessing.StandardScaler object
    pow_tformer : sklearn.preprocessing.power_transform object
        
    Returns
    -------
    X_tfm : (n_objects, n_feats) np.array
        numpy array of all features with transformed features
    """
    # apply learnt transformations to input features X.
    X_tfm = std_tformer.transform(pow_tformer.transform(X))

    return X_tfm 


# def map_intensity_interp2(query_pts, grid_shape, I_ref, method='spline', cast_uint8=False, s=0):

#     import numpy as np 
#     from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator 
    
#     if method == 'spline':
#         spl = RectBivariateSpline(np.arange(grid_shape[0]), 
#                                   np.arange(grid_shape[1]), 
#                                   I_ref,
#                                   s=s)
#         I_query = spl.ev(query_pts[...,0], 
#                          query_pts[...,1])
#     else:
#         spl = RegularGridInterpolator((np.arange(grid_shape[0]), 
#                                        np.arange(grid_shape[1])), 
#                                        I_ref, method=method, bounds_error=False, fill_value=0)
#         I_query = spl((query_pts[...,0], 
#                        query_pts[...,1]))

#     if cast_uint8:
#         I_query = np.uint8(I_query)
    
#     return I_query


# =============================================================================
#   Equal weighting of the SAM feature sets and of the feature groups within them
# =============================================================================
_HARALICK_NAMES = ('angular_second_moment', 'contrast', 'correlation', 'sum_squares_variance', 'inv_diff_moment',
                   'sum_average', 'sum_variance', 'sum_entropy', 'entropy', 'diff_variance', 'diff_entropy',
                   'info_measure_correlation_1', 'info_measure_correlation_2')

# (SAM set, feature group, rule on the feature name), checked in order; the groups are the feature families computed
# by compute_boundary_morphology_features, compute_boundary_texture_features and compute_boundary_motion_features
SAM_FEATURE_GROUPS = [
    # shape
    ('shape', 'curvature', lambda n: n.endswith('_curvature')),
    ('shape', 'contour geometry', lambda n: 'centroid_distance' in n or 'chordal' in n),
    ('shape', 'Fourier', lambda n: n.startswith('Fourier_')),
    ('shape', 'Euler characteristic transform', lambda n: n.startswith('ECT_')),
    ('shape', 'shape context', lambda n: n.startswith('Shape_Context_')),
    ('shape', 'binary morphology', lambda n: n in ('area', 'convex_area', 'perimeter', 'equivalent_diameter',
                                                   'major_axis_length', 'minor_axis_length', 'area_perim_aspect_ratio',
                                                   'major_minor_axis_ratio', 'moment_eccentricity', 'solidity', 'extent')
                                         or n.startswith('hu_moments_') or n.startswith('zernike_moments_')),
    # appearance
    ('appearance', 'regional intensity', lambda n: n.startswith('mean_intensity_region_') or n.startswith('stddev_intensity_region_')),
    ('appearance', 'global intensity', lambda n: n in ('mean_intensity', 'stddev_intensity')),
    ('appearance', 'Haralick texture', lambda n: n in _HARALICK_NAMES),
    ('appearance', 'SIFT', lambda n: n.startswith('sift_appear_')),
    # motion
    ('motion', 'global motion', lambda n: n.endswith('_global') or n.endswith('_global_flow')),
    ('motion', 'regional speed', lambda n: n.startswith('mean_speed_flow_region') or n.startswith('stddev_speed_flow_region')),
    ('motion', 'regional speed distribution', lambda n: n.startswith('hist_bin_')),
    ('motion', 'speed SIFT', lambda n: n.startswith('sift_motion_')),
    ('motion', 'curl SIFT', lambda n: n.startswith('curl_sift_')),
    ('motion', 'divergence SIFT', lambda n: n.startswith('div_sift_')),
]


# names of older SPOT exports, resolved within a known SAM set (see sam_feature_groups(feature_sets=...))
_LEGACY_GROUPS = {
    'appearance': [('SIFT', lambda n: n.startswith('sift_'))],
    'motion': [('global motion', lambda n: n.startswith('mean_disp_global')),
               ('speed SIFT', lambda n: n.startswith('sift_'))],
}


def _name_prefix(name):
    """ the name without a trailing _<number> index, e.g. 'thresh_adj_stat_3' -> 'thresh_adj_stat' """
    import re
    return re.sub(r'_?\d+$', '', name) or name


def sam_feature_groups(feature_names, feature_sets=None):
    r""" The SAM feature set (shape, appearance, motion) and feature group of each feature, from its name.

    The groups are the feature families of the SAM phenome (see SAM_FEATURE_GROUPS):

        shape       curvature, contour geometry, binary morphology (regionprops, Hu, Zernike moments), Fourier,
                    Euler characteristic transform (raw 'ECT_' or the kernel reduced 'ECT_rbf_' features), shape context
        appearance  global intensity, regional intensity, Haralick texture, SIFT
        motion      global motion (speed, flow speed, curl, divergence), regional speed, regional speed distribution,
                    speed SIFT, curl SIFT, divergence SIFT

    Parameters
    ----------
    feature_names : (n_feats,) array of str
        the feature names returned by the SAM feature functions
    feature_sets : None or (n_feats,) array of str
        the SAM set ('shape', 'appearance', 'motion') of each feature, if known (e.g. from the file the feature was read
        from). Needed for older exports where appearance and motion SIFT features were both named 'sift_<n>'. Names that
        are not recognised within their set form their own group(s) of that set, by name prefix.

    Returns
    -------
    feature_sets : (n_feats,) str array
        'shape', 'appearance', 'motion', or 'other' for unrecognised names when feature_sets is not given
    feature_groups : (n_feats,) str array
        the feature group; unrecognised names are grouped by their name prefix
    """
    import numpy as np
    names = np.asarray(feature_names).astype(str)
    given = None if feature_sets is None else np.asarray(feature_sets).astype(str)
    sets, groups = [], []
    for ii, name in enumerate(names):
        hint = None if given is None else given[ii]
        # SPOT.Utility_Functions.file_io suffixes appearance / motion columns with duplicate names when merging tables
        for suffix, s in (('_appear', 'appearance'), ('_motion', 'motion')):
            if name.endswith(suffix) and not name.startswith('sift_motion'):
                name, hint = name[:-len(suffix)], hint or s
        if hint is None:
            rules = SAM_FEATURE_GROUPS + [('motion', g, rule) for g, rule in _LEGACY_GROUPS['motion'] if g == 'global motion']
        else:
            rules = [r for r in SAM_FEATURE_GROUPS if r[0] == hint] +                     [(hint, g, rule) for g, rule in _LEGACY_GROUPS.get(hint, [])]
        for s, g, rule in rules:
            if rule(name):
                sets.append(s); groups.append(g)
                break
        else:
            sets.append('other' if hint is None else hint)
            groups.append(_name_prefix(name))
    return np.array(sets), np.array(groups)


def weight_sam_features(all_feats, feature_names, group_weighting=True, set_weighting=True, feature_sets=None,
                        return_weights=False):
    r""" Weight already normalised (e.g. standard scaled and power transformed) SAM features so that each feature set
    and each feature group contributes equally to distances (e.g. for UMAP, clustering), instead of in proportion to
    its number of features.

    The features are centred and scaled per group (group_weighting=True) or per set (group_weighting=False) such that
        - within each SAM set, every feature group has the same total variance (group_weighting=True), or the set is
          scaled as a whole, keeping the relative variances of its features (group_weighting=False; with standard scaled
          inputs every feature then counts equally, so large groups dominate), and
        - every SAM set (shape, appearance, motion) has total variance 1 (set_weighting=True), or keeps its total
          variance (set_weighting=False).
    Features that are not recognised as SAM features (see sam_feature_groups) are grouped by name prefix, in their set if
    feature_sets is given, else in an extra 'other' set (with a warning).

    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        normalised features
    feature_names : (n_feats,) np.array
        the names of the features
    group_weighting : bool
        give every feature group within a SAM set the same total variance (e.g. the 6 curvature features count as much
        as the ~300 shape context features)
    set_weighting : bool
        give every SAM set the same total variance
    feature_sets : None or (n_feats,) array of str
        the SAM set of each feature if known, see sam_feature_groups
    return_weights : bool
        if True also return (mean, weight) such that the output is (all_feats - mean) * weight, to apply the same
        weighting to new data

    Returns
    -------
    all_feats_weighted : (n_objects, n_feats) np.array
    (mean, weight) : ((n_feats,) np.array, (n_feats,) np.array), if return_weights
    """
    import numpy as np
    X = np.asarray(all_feats, dtype=np.float64)
    mean = np.nanmean(X, axis=0)
    var = np.nanvar(X, axis=0)
    sets, groups = sam_feature_groups(feature_names, feature_sets=feature_sets)
    if (sets == 'other').any():
        import warnings
        warnings.warn('weight_sam_features: features not recognised as SAM features are weighted as an extra set: %s'
                      % sorted(set(groups[sets == 'other'])))
    weight = np.zeros(X.shape[1])
    for s in dict.fromkeys(sets):
        in_set = sets == s
        set_total = 1. if set_weighting else var[in_set].sum()
        units = list(dict.fromkeys(groups[in_set])) if group_weighting else [None]
        for g in units:
            cols = in_set & (groups == g) if g is not None else in_set
            total = var[cols].sum()
            if total > 0:
                weight[cols] = np.sqrt(set_total / len(units) / total)
    out = (X - mean[None, :]) * weight[None, :]
    if return_weights:
        return out, (mean, weight)
    return out
