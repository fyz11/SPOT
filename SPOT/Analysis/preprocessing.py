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
    ('shape', 'elliptic Fourier', lambda n: n.startswith('EFD_')),
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
    ('motion', 'shear SIFT', lambda n: n.startswith('shear_sift_')),
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



# =============================================================================
#   Regressing out confounding covariates (object size, intensity, batch / plate)
# =============================================================================
def _covariate_dict(cov, n, prefix):
    """ covariates given as None, a dict name -> (n,) array, a pandas DataFrame / Series or an (n,) / (n, k) array, as
    an ordered dict name -> (n,) array """
    import numpy as np
    if cov is None:
        return {}
    if hasattr(cov, 'columns'):  # pandas DataFrame
        cov = {str(c): cov[c].values for c in cov.columns}
    elif hasattr(cov, 'name') and hasattr(cov, 'values'):  # pandas Series
        cov = {str(cov.name) if cov.name is not None else prefix + '0': cov.values}
    elif not isinstance(cov, dict):
        cov = np.asarray(cov)
        if cov.ndim == 1:
            cov = cov[:, None]
        cov = {'%s%d' % (prefix, j): cov[:, j] for j in range(cov.shape[1])}
    out = {}
    for name, v in cov.items():
        v = np.asarray(v).ravel()
        if len(v) != n:
            raise ValueError('covariate %r has %d values, expected %d (one per object)' % (name, len(v), n))
        out[name] = v
    return out


class ConfoundRegressor(object):
    r""" Regress confounding / undesired effects out of (normalised) features, e.g. object size, intensity and
    batch / plate / experiment variation.

    Every feature is modelled as

    .. math:: x = \beta_0 + \sum_c f_c(\mathrm{continuous}_c) + \sum_b \gamma_b[\mathrm{level}_b] + \epsilon

    where each continuous covariate is mapped to [0, 1] by its rank (the empirical CDF of the fit data) and expanded in
    a cubic B-spline basis, so that arbitrary (non)monotone nonlinear effects are captured robustly to outliers, and
    each categorical covariate contributes one mean shift per level. All features are fitted at once by a single least
    squares solve, so thousands of features are fast. The fitted effects are subtracted and the per-feature mean of the
    reference rows is added back; optionally the per-level residual spread is equalised (location + scale batch
    correction) and the result is re-standardised.

    This is meant to be applied after the per-feature normalisation (StandardScaler + power transform, see
    power_transform_and_scale_features) and before SPOT.Analysis.preprocessing.weight_sam_features / UMAP.

    Parameters
    ----------
    spline : bool
        expand each continuous covariate in a cubic B-spline basis (nonlinear effect). If False the (rank of the)
        covariate enters linearly.
    rank : bool
        use the rank of each continuous covariate (empirical CDF of the reference rows of the fit data, in (0, 1))
        instead of its raw value. Ranks are robust to outliers and skew; other values are interpolated against the
        sorted reference values and clipped to their range. If False the raw (standardised) value is used.
    n_knots : int
        number of knots of the spline basis (sklearn.preprocessing.SplineTransformer, degree 3)
    batch_scale : bool
        after removing the mean effects, rescale the residuals of each level of each categorical covariate so that their
        per-feature standard deviation equals the pooled residual standard deviation of the reference rows
    restandardize : bool
        z-score every output feature with the mean / standard deviation of the corrected fit data

    Attributes
    ----------
    coef_ : (n_design, n_feats) np.array
        least squares coefficients of the design matrix (intercept, spline bases, one-hot levels)
    continuous_names_, categorical_names_ : list of str
        names of the covariates seen in fit, transform must be given the same ones
    levels_ : dict
        name -> np.array of the levels of each categorical covariate seen in the reference rows
    ref_mean_ : (n_feats,) np.array
        per-feature mean of the reference rows, added back to the residuals
    scale_ : dict
        name -> (n_levels, n_feats) np.array of per-level residual scale factors, if batch_scale
    mean_, std_ : (n_feats,) np.array
        statistics of the corrected fit data used to re-standardise, if restandardize

    Notes
    -----
    X must be finite: impute or remove NaN / inf values beforehand. If a reference mask is given (e.g. control objects)
    the coefficients are estimated on the reference rows only and applied to every row, so differences between
    conditions that are not explained by the covariates are preserved, even if the covariates are correlated with the
    condition. Without a reference all rows are used and any condition effect that is correlated with a covariate is
    (partly) removed too.

    Examples
    --------
    >>> reg = ConfoundRegressor(batch_scale=True)
    >>> X_corr = reg.fit_transform(X, continuous={'size': size, 'intensity': intensity},
    ...                            categorical={'plate': plate}, reference=is_control)
    >>> X_new_corr = reg.transform(X_new, continuous={'size': size_new, 'intensity': intensity_new},
    ...                            categorical={'plate': plate_new})
    """

    def __init__(self, spline=True, rank=True, n_knots=6, batch_scale=False, restandardize=True):
        self.spline = spline
        self.rank = rank
        self.n_knots = n_knots
        self.batch_scale = batch_scale
        self.restandardize = restandardize

    # ---- covariate encoding ----
    @staticmethod
    def _finite(name, v):
        import numpy as np
        v = np.asarray(v, dtype=np.float64)
        if not np.all(np.isfinite(v)):
            raise ValueError('continuous covariate %r contains NaN / inf values' % name)
        return v

    def _fit_continuous(self, name, v):
        import numpy as np
        from scipy.stats import rankdata
        from sklearn.preprocessing import SplineTransformer
        v = self._finite(name, v)
        # empirical CDF at the (unique) fit values, mid-ranks for ties, in (0, 1)
        u_all = (rankdata(v) - 0.5) / len(v)
        cdf_x, first = np.unique(v, return_index=True)
        center, scale = v.mean(), v.std()
        scale = scale if scale > 0 else 1.
        spl = None
        if self.spline:
            u = u_all if self.rank else (v - center) / scale
            spl = SplineTransformer(n_knots=self.n_knots, degree=3,
                                    extrapolation='constant' if self.rank else 'linear').fit(u[:, None])
        self._cont_[name] = (cdf_x, u_all[first], center, scale, spl)

    def _encode_continuous(self, name, v):
        import numpy as np
        v = self._finite(name, v)
        cdf_x, cdf_u, center, scale, spl = self._cont_[name]
        u = np.interp(v, cdf_x, cdf_u) if self.rank else (v - center) / scale
        if self.spline:
            return spl.transform(u[:, None])
        return u[:, None]

    def _level_index(self, name, v):
        import numpy as np
        levels = self.levels_[name]
        v = np.asarray(v)
        unseen = set(np.unique(v).tolist()) - set(levels.tolist())
        if unseen:
            raise ValueError('categorical covariate %r has levels %s that were not seen in the %s data used to fit the '
                             'confound model, their effect cannot be estimated' %
                             (name, sorted(unseen, key=str), 'reference' if self.has_reference_ else 'fit'))
        return np.searchsorted(levels, v)

    def _covariates(self, n, cont, cat, check=True):
        cont = _covariate_dict(cont, n, 'continuous')
        cat = _covariate_dict(cat, n, 'categorical')
        if check:
            for kind, given, expected in (('continuous', cont, self.continuous_names_),
                                          ('categorical', cat, self.categorical_names_)):
                if sorted(given) != sorted(expected):
                    raise ValueError('%s covariates %s do not match those used in fit %s' %
                                     (kind, sorted(given), sorted(expected)))
        return cont, cat

    def _design(self, n, cont, cat):
        """ design matrix [intercept, spline bases, one-hot levels without the first] and the level index of each row """
        import numpy as np
        blocks = [np.ones((n, 1))]
        for name in self.continuous_names_:
            blocks.append(self._encode_continuous(name, cont[name]))
        idx = {}
        for name in self.categorical_names_:
            idx[name] = self._level_index(name, cat[name])
            onehot = np.zeros((n, len(self.levels_[name])))
            onehot[np.arange(n), idx[name]] = 1.
            blocks.append(onehot[:, 1:])  # the first level is absorbed in the intercept
        return np.hstack(blocks), idx

    @staticmethod
    def _check_X(X):
        import numpy as np
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError('X must be a 2D (n_objects, n_feats) array')
        if not np.all(np.isfinite(X)):
            raise ValueError('X contains NaN / inf values, impute or remove them before regressing out confounders')
        return X

    # ---- public api ----
    def fit(self, X, continuous=None, categorical=None, reference=None):
        r""" Fit the confound model.

        Parameters
        ----------
        X : (n_objects, n_feats) np.array
            normalised features, must be finite
        continuous : None, (n_objects,) or (n_objects, k) np.array, dict or pandas DataFrame
            continuous covariates, e.g. {'size': equivalent_diameter, 'intensity': mean_intensity}
        categorical : None, (n_objects,) or (n_objects, k) np.array, dict or pandas DataFrame
            categorical covariates, e.g. {'batch': batch_id, 'plate': plate_id}
        reference : None or (n_objects,) bool np.array
            rows used to estimate the coefficients (e.g. control objects); all rows if None

        Returns
        -------
        self : ConfoundRegressor
        """
        import numpy as np
        X = self._check_X(X)
        n = len(X)
        cont, cat = self._covariates(n, continuous, categorical, check=False)
        ref = np.ones(n, dtype=bool) if reference is None else np.asarray(reference, dtype=bool).ravel()
        if len(ref) != n:
            raise ValueError('reference has %d values, expected %d' % (len(ref), n))
        if ref.sum() < 2:
            raise ValueError('reference must select at least 2 rows')
        self.has_reference_ = reference is not None

        self.continuous_names_ = list(cont)
        self.categorical_names_ = list(cat)
        self._cont_ = {}
        for name, v in cont.items():  # rank and spline knots from the reference rows, where the effect is estimated
            self._fit_continuous(name, np.asarray(v)[ref])
        self.levels_ = {name: np.unique(np.asarray(v)[ref]) for name, v in cat.items()}

        D, idx = self._design(n, cont, cat)
        self.coef_ = np.linalg.lstsq(D[ref], X[ref], rcond=None)[0]
        self.ref_mean_ = X[ref].mean(axis=0)
        resid = X - D @ self.coef_

        self.scale_ = {}
        if self.batch_scale:
            for name in self.categorical_names_:
                pooled = resid[ref].std(axis=0)
                factors = np.ones((len(self.levels_[name]), X.shape[1]))
                for li in range(len(self.levels_[name])):
                    sel = ref & (idx[name] == li)
                    if sel.sum() > 1:
                        sd = resid[sel].std(axis=0)
                        ok = sd > 0
                        factors[li, ok] = pooled[ok] / sd[ok]
                self.scale_[name] = factors
                resid = resid * factors[idx[name]]

        out = resid + self.ref_mean_[None, :]
        self.mean_ = out.mean(axis=0)
        std = out.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.)
        return self

    def transform(self, X, continuous=None, categorical=None):
        r""" Remove the fitted confound effects from X.

        Parameters
        ----------
        X : (n_objects, n_feats) np.array
            normalised features with the same columns as in fit, must be finite
        continuous, categorical : see fit
            the covariates of the rows of X, with the same names as in fit. Categorical levels must have been seen in
            the (reference rows of the) fit data, else a ValueError is raised.

        Returns
        -------
        X_corrected : (n_objects, n_feats) np.array
        """
        X = self._check_X(X)
        if X.shape[1] != self.coef_.shape[1]:
            raise ValueError('X must have %d features, got %d' % (self.coef_.shape[1], X.shape[1]))
        cont, cat = self._covariates(len(X), continuous, categorical)
        D, idx = self._design(len(X), cont, cat)
        resid = X - D @ self.coef_
        for name, factors in self.scale_.items():
            resid = resid * factors[idx[name]]
        out = resid + self.ref_mean_[None, :]
        if self.restandardize:
            out = (out - self.mean_[None, :]) / self.std_[None, :]
        return out

    def fit_transform(self, X, continuous=None, categorical=None, reference=None):
        r""" fit then transform X, see fit and transform """
        return self.fit(X, continuous=continuous, categorical=categorical,
                        reference=reference).transform(X, continuous=continuous, categorical=categorical)


def regress_out_confounders(X, continuous=None, categorical=None, feature_names=None, covariate_features=None,
                            keep_covariate_features=True, reference=None, drop_associated=True, max_abs_rho=0.3,
                            max_eta2=0.1, return_model=False, **regressor_kwargs):
    r""" Regress confounding effects (object size, intensity, batch / plate) out of normalised features with a
    ConfoundRegressor (rank -> cubic spline for continuous covariates, mean (and optionally scale) shift per level for
    categorical covariates, optionally estimated on reference rows only), then drop the features that remain strongly
    associated with a confound (e.g. features that depend on size in a way a smooth curve cannot remove, such as
    indicator-like features). Generalises regressing the features on a cubic spline of the rank of a single covariate.

    Meant to be applied after the per-feature normalisation (StandardScaler + power transform) and before
    SPOT.Analysis.preprocessing.weight_sam_features / UMAP.

    Parameters
    ----------
    X : (n_objects, n_feats) np.array
        normalised features, must be finite
    continuous : None, array, dict or pandas DataFrame
        extra continuous covariates not in X, see ConfoundRegressor.fit
    categorical : None, array, dict or pandas DataFrame
        categorical covariates, e.g. {'batch': batch_id}
    feature_names : None or (n_feats,) array of str
        names of the columns of X, required with covariate_features. If None, the column indices are used.
    covariate_features : None or list of str
        names of columns of X used as continuous covariates, e.g. ['equivalent_diameter', 'mean_intensity']. These
        columns are not residualised (they would become ~0) but returned unchanged, or dropped if
        keep_covariate_features=False.
    keep_covariate_features : bool
        keep the covariate_features columns (unchanged) in the output, else drop them
    reference : None or (n_objects,) bool np.array
        rows used to estimate the confound effects (e.g. controls), so condition differences are not regressed away.
        The remaining association (drop_associated) is then also measured on these rows only, so features that differ
        between conditions are not dropped because a confound (e.g. size) also differs between conditions.
    drop_associated : bool
        drop the corrected features whose remaining association with any confound exceeds max_abs_rho (|Spearman rho|,
        continuous covariates) or max_eta2 (between-level variance share, categorical covariates)
    max_abs_rho, max_eta2 : float
        the thresholds of drop_associated
    return_model : bool
        also return the fitted ConfoundRegressor, to correct new data with model.transform. The model acts on the
        residualised columns only (X[:, ~np.isin(feature_names, covariate_features)]) and takes the covariate_features
        columns as continuous covariates named by their feature name. model.keep_ is the boolean mask of the output
        columns among the input columns of X and model.dropped_features_ the names of the dropped features.
    **regressor_kwargs :
        spline, rank, n_knots, batch_scale, restandardize of ConfoundRegressor

    Returns
    -------
    X_corrected : (n_objects, n_feats_out) np.array
    feature_names_out : (n_feats_out,) np.array
        the names (or column indices) of the output features
    model : ConfoundRegressor
        only if return_model
    """
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    names = np.arange(X.shape[1]) if feature_names is None else np.asarray(feature_names).astype(str)
    cont = _covariate_dict(continuous, len(X), 'continuous')
    is_cov = np.zeros(X.shape[1], dtype=bool)
    if covariate_features:
        if feature_names is None:
            raise ValueError('feature_names is required to take covariate_features from the columns of X')
        for name in covariate_features:
            hits = np.flatnonzero(names == name)
            if len(hits) == 0:
                raise ValueError('covariate feature %r is not in feature_names' % name)
            if name in cont:
                raise ValueError('covariate %r given both as continuous and as a covariate feature' % name)
            cont[name] = X[:, hits[0]]
            is_cov[hits] = True

    model = ConfoundRegressor(**regressor_kwargs)
    corrected = model.fit_transform(X[:, ~is_cov], continuous=cont or None, categorical=categorical,
                                    reference=reference)

    # features still associated with a confound after the correction
    keep_resid = np.ones(corrected.shape[1], dtype=bool)
    if drop_associated:
        cat = _covariate_dict(categorical, len(X), 'categorical')
        rows = np.ones(len(X), dtype=bool) if reference is None else np.asarray(reference, dtype=bool).ravel()
        assoc = confound_association(corrected[rows], np.arange(corrected.shape[1]),
                                     continuous={k: np.asarray(v)[rows] for k, v in cont.items()} or None,
                                     categorical={k: np.asarray(v)[rows] for k, v in cat.items()} or None)
        too_high = ((assoc['kind'] == 'continuous') & (assoc['association'] > max_abs_rho)) | \
                   ((assoc['kind'] == 'categorical') & (assoc['association'] > max_eta2))
        keep_resid[np.unique(assoc.loc[too_high, 'feature'].astype(int))] = False

    keep = np.ones(X.shape[1], dtype=bool)
    keep[np.flatnonzero(~is_cov)[~keep_resid]] = False
    if not keep_covariate_features:
        keep &= ~is_cov
    out = X.copy()
    out[:, ~is_cov] = corrected
    model.keep_ = keep
    model.dropped_features_ = names[~keep & ~is_cov]

    result = (out[:, keep], names[keep])
    if return_model:
        result = result + (model,)
    return result


def confound_association(X, feature_names, continuous=None, categorical=None):
    r""" Strength of association of every feature with every covariate, for before / after confound regression
    diagnostics: absolute Spearman rho for continuous covariates and eta^2 (the share of the feature variance explained
    by the level means) for categorical covariates. Both are in [0, 1], 0 = no association.

    Parameters
    ----------
    X : (n_objects, n_feats) np.array
        features, must be finite
    feature_names : (n_feats,) array of str
        names of the columns of X
    continuous, categorical : None, array, dict or pandas DataFrame
        covariates, see ConfoundRegressor.fit

    Returns
    -------
    assoc : pandas DataFrame
        one row per (feature, covariate) with columns 'feature', 'covariate', 'kind' ('continuous' / 'categorical'),
        'measure' ('abs_spearman' / 'eta2') and 'association'
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    feature_names = np.asarray(feature_names).astype(str)
    cont = _covariate_dict(continuous, n, 'continuous')
    cat = _covariate_dict(categorical, n, 'categorical')

    rows = []
    with np.errstate(invalid='ignore', divide='ignore'):
        if cont:
            R = rankdata(X, axis=0)
            R = R - R.mean(axis=0)
            R_norm = np.sqrt((R ** 2).sum(axis=0))
        for name, v in cont.items():
            r = rankdata(np.asarray(v, dtype=np.float64))
            r = r - r.mean()
            rho = (R.T @ r) / (R_norm * np.sqrt((r ** 2).sum()))
            rows.append(pd.DataFrame({'feature': feature_names, 'covariate': name, 'kind': 'continuous',
                                      'measure': 'abs_spearman', 'association': np.abs(rho)}))
        Xc = X - X.mean(axis=0)
        ss_tot = (Xc ** 2).sum(axis=0)
        for name, v in cat.items():
            _, inv = np.unique(np.asarray(v), return_inverse=True)
            inv = inv.ravel()
            counts = np.bincount(inv)
            sums = np.zeros((len(counts), X.shape[1]))
            np.add.at(sums, inv, Xc)
            ss_between = (sums ** 2 / counts[:, None]).sum(axis=0)
            rows.append(pd.DataFrame({'feature': feature_names, 'covariate': name, 'kind': 'categorical',
                                      'measure': 'eta2', 'association': ss_between / ss_tot}))
    if not rows:
        return pd.DataFrame(columns=['feature', 'covariate', 'kind', 'measure', 'association'])
    return pd.concat(rows, ignore_index=True)
