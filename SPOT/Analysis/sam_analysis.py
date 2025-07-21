# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:30:22 2023

@author: S205272
"""

import numpy as np 

def compute_SAM_features_and_scope_contribution(all_features, 
                                                feature_names,
                                                random_state=0):
    
    r""" Given the feature names, categorises the computed features into type: shape, appearance and motion and scope: global, local-regional, local-distribution and compute the contribution of these to explaining the data variation using PCA.
    
    Parameters
    ----------
    all_features : (n_objects, n_features) array
        feature matrix of all objects 
    feature_names : (n_features,) array
        names of every included computed feature
    random_state : int
        random seed used in PCA for reproducibility 
        
    Returns 
    -------
    (feature_type, feature_scope) : ( (n_features,) array, (n_features,) array ) tuple
        type of feature as an integer where 0 = shape, 1 = appearance and 2 = motion. scope of feature as an integer where 0 = global, 1 = local-regional and 2 = local-distribution
    (SAM_expr, Scope_expr) : ( (n_objects, 3) array, (n_objects, 3) array )    
        the concatenated PC1 expression of individually subsetting by shape, appearance and motion features and of subsetting by global, local-regional and local-distribution respectively
    (SAM_contribution, scope_contribution) : ((3,) array, (3,) array) 
        The absolute value of the first principal axes in feature space, representing the contribution of shape, appearance and motion and global, local-regional and local-distribution to the maximum variance in the data. 
    """
    # import sam_analysis as SAM_tools
    from sklearn.decomposition import PCA
    
    # feature_type, feature_scope = SAM_tools.get_SAM_feature_type(feature_names)
    feature_type, feature_scope = get_SAM_feature_type(feature_names)
    
    # eval SAM contribution ( best to do a variance decomposition. )
    SAM_expr = []
    
    for ii in np.arange(3): # 3 for shape appearance and motion 
        select = feature_type == ii
        
        if np.sum(select) > 0: 
            select_feats = all_features[:,select].copy()
            pca_fit = PCA(n_components=1, random_state=random_state) # we are going to use the principal variation 
            pca_expr = pca_fit.fit_transform(select_feats)
            pca_expr = np.hstack(pca_expr)
            SAM_expr.append(pca_expr)
        else:
            SAM_expr.append(np.zeros(len(all_features)))

    SAM_expr = np.vstack(SAM_expr).T

    # do PCA
    pca_SAM = PCA(n_components=3, random_state=random_state)
    pca_SAM_coords = pca_SAM.fit_transform(SAM_expr) # this is for fitting purposes. 
    
    SAM_contribution = np.abs(pca_SAM.components_[0])
    
    # eval scope contribution 
    scope_expr = []
    for ii in np.arange(3):
        select = feature_scope == ii
        
        if np.sum(select) > 0: 
            select_feats = all_features[:,select].copy()
            pca_fit = PCA(n_components=1, random_state=random_state)
            pca_expr = pca_fit.fit_transform(select_feats)
            pca_expr = np.hstack(pca_expr)
            scope_expr.append(pca_expr)
        else:
            scope_expr.append(np.zeros(len(all_features))) # all zeros. 

    Scope_expr = np.vstack(scope_expr).T
    
    # do PCA
    pca_scope = PCA(n_components=3, random_state=random_state)
    pca_scope_coords = pca_scope.fit_transform(Scope_expr)
    
    scope_contribution = np.abs(pca_scope.components_[0])
        
    return (feature_type, feature_scope), (SAM_expr, Scope_expr), (SAM_contribution, scope_contribution)


def _baseline_als(y, lam, p, niter=10):
        r""" Estimates a baseline signal using asymmetric least squares. It can also be used for generic applications where a 1D signal requires smoothing.
        Specifically the baseline signal, :math:`z` is the solution to the following optimization problem 

        .. math::
            z = arg\,min_z \{w(y-z)^2 + \lambda\sum(\Delta z)^2\}

        where :math:`y` is the input signal, :math:`\Delta z` is the 2nd derivative, :math:`\lambda` is the smoothing regularizer and :math:`w` is an asymmetric weighting

        .. math::
            w = 
            \Biggl \lbrace 
            { 
            p ,\text{ if } 
              {y>z}
            \atop 
            1-p, \text{ otherwise } 
            }

        Parameters
        ----------
        signal : 1D numpy array
            The 1D signal to estimate a baseline signal. 
        p :  scalar
            Controls the degree of asymmetry in the weighting. p=0.5 is the same as smoothness regularized least mean squares.
        lam : scalar
            Controls the degree of smoothness in the baseline
        niter: int
            The number of iterations to run the algorithm. Only a few iterations is required generally. 

        Returns
        -------
        z : 1D numpy array
            the estimated 1D baseline signal

        """
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        import numpy as np 
        
        L = len(y)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z
    
    

def hierarchical_cluster_features_into_SAM_modules(all_feats, 
                                                   feature_names, 
                                                   feature_type=None,
                                                   feature_scope=None,
                                                   hcluster_heatmap_color='vlag',
                                                   hcluster_method='average', 
                                                   hcluster_metric='euclidean',
                                                   lam_smooth=1e1, 
                                                   p_smooth=0.5, 
                                                   niter_smooth=10,
                                                   min_peak_distance=5,
                                                   max_clust_len = 25,
                                                   buffer=30,
                                                   debugviz=False,
                                                   savefile=None):

    r""" Use automatic hierarchical clustering to group individual features into modules based on their covariation within the dataset
    
    Parameters
    ----------
    all_feats : (n_objects, n_features) array
        feature matrix of all objects 
    feature_names : (n_features,) array
        names of every included computed feature
    feature_type : (n_features,) array 
        type of feature as an integer where 0 = shape, 1 = appearance and 2 = motion.
    feature_scope : (n_features,) array 
        scope of feature as an integer where 0 = global, 1 = local-regional and 2 = local-distribution
    hcluster_heatmap_color : str
        heatmap color name
    hcluster_method : str
        hierarchical clustering linkage method, any method available to seaborn clustermap
    hcluster_metric : str
        metric used in hierarchical clustering, any valid metric available to seaborn clustermap
    lam_smooth : float
        controls the extent of smoothing in asymmetric least square smoothing used to get the number of SAM modules
    p_smooth : float (0-1)
        controls the asymmetrical bias when smoothing a curve. 0.5 is equivalent to ordinary least squares method
    niter_smooth : int
        controls the number of smoothing iterations. The higher the number, the greater the smoothing
    min_peak_distance : int
        the minimum spacing in terms of number of cluster between peaks in the davies_bouldin_score used to evaluate the number of SAM modules 
    debugviz=False
        if True, plots the graph of #SAM modules vs davies_bouldin_score
    savefile : str 
        save path 
    
    Returns
    -------
    g : seaborn clustermap object
        the seaborn clustermap object, where features have been clustered by their pairwise Pearsons correlation matrix 
    final_feature_clusters : (n_features,) array
        an integer array where each feature has been assigned to a distinct SAM module. Each unique integer is a distinct SAM module
    features_clusters_colorbar : (n_features, 3) array 
        colorbar, in the order of clustermap g, to help draw the cluster boundaries. 
    final_feature_names_clusters : list
        the names of each feature assigned to each SAM module. SAM module is ordered in terms of increasing integer id. 
    final_feature_names_ids : list
        the index of the features assigned to each SAM module. SAM module is ordered in terms of increasing integer id.    
    
    """
    import seaborn as sns 
    import pylab as plt 
    import pandas as pd 
    from scipy.cluster.hierarchy import fcluster # this is used to find clusters from the dendrogram tree
    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.metrics import davies_bouldin_score
    from scipy.signal import find_peaks
    from tqdm import tqdm 
    import time 
    

    if feature_type is None or feature_scope is None:
        feature_type, feature_scope = get_SAM_feature_type(feature_names)    
    
    # =============================================================================
    #   We should do a heirarchical clustering here and labelling by shape appearance or motion?          
    # =============================================================================    
    # Create a categorical palette to identify the networks
    # SAM_pal = sns.husl_palette(3, s=1)
    SAM_pal = sns.color_palette('Spectral', n_colors=3)
    SAM_lut = dict(zip(map(int, [0,1,2]), SAM_pal))
    
    #### creating a spatial lut
    spatial_pal = sns.color_palette('coolwarm_r', n_colors=3)
    spatial_lut = dict(zip(map(int, [0,1,2]), spatial_pal))
    
    
    # use pandas hcluster, so we first convert to a table. 
    df = pd.DataFrame(all_feats,
                      columns = feature_names)
    SAM_colors = pd.Series(feature_type, index=df.columns).map(SAM_lut)
    spatial_colors = pd.Series(feature_scope, index=df.columns).map(spatial_lut)


    df_corr = pd.DataFrame(1.-pairwise_distances(all_feats.T, metric='correlation', n_jobs=-1),
                           index = feature_names,
                           columns = feature_names)
    df_corr = df_corr.fillna(value=0) # in case any potential entries evaluated to nan 
    # t2 = time.time()
    # print('finished correlation', t2-t1)
    
    # hcluster. 
    g=sns.clustermap(#df.corr(), # this is too slow
                     df_corr,
                      cmap=hcluster_heatmap_color, 
                      method=hcluster_method, # use average or maybe complete. 
                      metric=hcluster_metric,
                      vmin=-1, 
                      vmax=1, 
                      col_colors=[SAM_colors, spatial_colors])
    g.ax_row_dendrogram.remove()
    
    # # g.ax_cbar.remove()
    # # g.ax_heatmap.xaxis.set_ticks(np.arange(len(g.dendrogram_col.reordered_ind))+.5)
    # # g.ax_heatmap.xaxis.set_ticklabels(g.dendrogram_col.reordered_ind)
    
    if savefile is not None:
        # g.savefig(os.path.join(saveplotsfolder_cells, 'UMAP_SAM_modules_dendrogram_cluster-average.svg'), 
                  # bbox_inches='tight', pad_inches=0, dpi=300)
        g.savefig(savefile, 
                  bbox_inches='tight',
                  pad_inches=0, 
                  dpi=300)
    # # # # ax.set_aspect(1)
    
    # # get the correlation matrix. 
    # feature_corr_matrix = df.corr().values.copy()
    
    cluster_formation_cols = np.sort(g.dendrogram_col.calculated_linkage[:,-1]) # grab the clustered linkage from pandas. 
    cluster_formation_cols_diff = np.diff(cluster_formation_cols)
    cluster_formation_cols_break_pts = np.arange(len(cluster_formation_cols_diff))[cluster_formation_cols_diff>0]
    cluster_formation_cols_break_pts_len = np.diff(cluster_formation_cols_break_pts)
    
    # set the min detected length ...  --- we shouldn't scan to the max i assume..... 
    min_cluster_len = 1
    
    max_clust_len_ref = 2 + np.arange(len(cluster_formation_cols_break_pts_len))[cluster_formation_cols_break_pts_len==min_cluster_len][1] # tolerate the 2nd ? 

    # print(max_clust_len_ref)
    if max_clust_len is None:
        max_clust_len = 2 + np.arange(len(cluster_formation_cols_break_pts_len))[cluster_formation_cols_break_pts_len==min_cluster_len][1] # tolerate the 2nd ? 
    # else:
        
    """
    Automated clustering 
    """
    # print('maximum # clusters', max_clust_len)
    # homogeneity = []
    labels_all = []
    # corr_all = []
    scores_all = [] # clustering scores. 
    # sil_scores_all = []
    
    # then use this to scan the max.... -> we know each step adds a cluster. 
    for link_ii in tqdm(np.arange(2, len(g.dendrogram_col.calculated_linkage))[:max_clust_len+buffer]): # give a smaller buffer?
        lab = fcluster(g.dendrogram_col.calculated_linkage, 
                        t=g.dendrogram_col.calculated_linkage[-link_ii-1,2], 
                        criterion='distance', depth=2, R=None, monocrit=None)
        labels_all.append(lab)
        # print(link_ii, len(np.unique(lab)))
        
        # # print()
        # # # lab gives the unique cluster labellings.
        # # # now we go through the partitioning and get the mean intra-cluster correlations.
        # label_intra_vals = [df.values[:,lab==lab_uniq].T for lab_uniq  in np.unique(lab)]
        # # label_intra_corr = [1.-pairwise_distances(val, metric='correlation') for val in label_intra_vals] # is this val.T? # should be the inverse (double check)
        # # label_intra_corr = [np.nanmean(val - np.diag(np.ones(len(val)))) for val in label_intra_corr]
        
        # # mean_score = np.nanmean(label_intra_corr)
        # # corr_all.append(mean_score)   ##### this will of course always decrease... we need a better clustering. 
        
        # # sil_scores_all.append(calinski_harabasz_score(df.T, lab))
        # scores_all.append(davies_bouldin_score(df.corr(), lab)) # do not recompute! 
        scores_all.append(davies_bouldin_score(df_corr.values, lab))
        # sil_scores_all.append(silhouette_score(df.corr(), lab))
        # print([np.arange(len(cluster_ids))[lab==lab_uniq] for lab_uniq in np.unique(lab)])
        
        # max_p = [np.mean(df_max_col[lab==lab_uniq]) for lab_uniq in np.unique(lab)]
        # homogeneity_all.append(max_p)
        # print(max_p)
        # homogeneity.append(np.min(max_p))
        # print('===')
        
    # # try and separate out the clustering based on this. 
    # labels_all = np.vstack(labels_all)
    # corr_all = np.hstack(corr_all)
    scores_all = np.hstack(scores_all)
    # sil_scores_all = np.hstack(sil_scores_all)
    
    """
    smooth this and find the inflection point. ----> does this always hold? 
    """
    scores_all_smooth = _baseline_als(scores_all, 
                                      lam=lam_smooth, 
                                      p=p_smooth, 
                                      niter=niter_smooth)
    
    
    # plot this. 
    if debugviz:
        plt.figure(figsize=(5,5))
        plt.plot(np.arange(len(labels_all))+2, scores_all,'k', lw=2)
        plt.plot(np.arange(len(labels_all))+2, scores_all_smooth,'g--', lw=2)
        plt.tick_params(length=10, right=True)
        plt.ylabel('Mean Score')
        plt.xlabel('# Modules')
        plt.show()
    
    """
    Use peaks
    """
    min_peak, min_peak_props = find_peaks(scores_all_smooth, distance=min_peak_distance)
    stop_module = min_peak[0]
    
    if debugviz:
        plt.figure(figsize=(5,5))
        plt.plot(np.arange(len(labels_all))+2, scores_all,'k', lw=2)
        plt.plot(np.arange(len(labels_all))+2, scores_all_smooth,'g--', lw=2)
        plt.tick_params(length=10, right=True)
        plt.ylabel('Mean Score')
        plt.xlabel('# Modules')
        plt.vlines(stop_module+2,
                    np.min(scores_all)-0.01, 
                    np.max(scores_all)+0.01, 
                    color='g', lw=2)
        plt.show()
    
    """
    find the clustering and output into a separate csv.
    """
    final_feature_clusters = labels_all[stop_module].copy()
    final_feature_names_clusters = [feature_names[final_feature_clusters==lab] for lab in np.unique(final_feature_clusters)]
    final_feature_names_ids = [np.arange(len(feature_names))[final_feature_clusters==lab] for lab in np.unique(final_feature_clusters)]
    
    # """
    # what is this again ? 
    # """
    # N_features_clusters = [len(ffff) for ffff in final_feature_names_clusters]
    # N_features_clusters_max = np.max(N_features_clusters)
    
    # features_clusters_table = np.zeros((N_features_clusters_max, len(N_features_clusters)), dtype=object)
    # features_clusters_table[:] = np.nan
    
    # for kkk in np.arange(len(N_features_clusters)):
    #     features_clusters_table[:len(final_feature_names_clusters[kkk]),kkk] = final_feature_names_clusters[kkk].copy()
    
    # # save out the table.
    # features_clusters_table = pd.DataFrame(features_clusters_table, 
    #                                        columns=['Module_%s' %(str(kkk).zfill(3)) for kkk in np.arange(len(N_features_clusters))])
    
    
    """
    reparse clusters to be in the same order as the dendrogram ordering. 
    """
    dendrogram_clusters_order = final_feature_clusters[g.dendrogram_col.reordered_ind]
    dendrogram_order = []
    
    for ii in np.arange(len(dendrogram_clusters_order)):
        if len(dendrogram_order)>0:
            if dendrogram_clusters_order[ii]!=dendrogram_order[-1]:
                dendrogram_order.append(dendrogram_clusters_order[ii])
        else:
            dendrogram_order.append(dendrogram_clusters_order[ii])
    
    # the above is the present order.. which should not be reordered.
    final_feature_clusters_new = np.zeros_like(final_feature_clusters);
    final_feature_names_clusters_new = []
    final_feature_names_ids_new = []
    
    for ii, clust_ii in enumerate(dendrogram_order):
    
        final_feature_clusters_new[final_feature_clusters==clust_ii] = ii+1
        final_feature_names_clusters_new.append(final_feature_names_clusters[clust_ii-1])
        final_feature_names_ids_new.append(final_feature_names_ids[clust_ii-1])
        
    final_feature_clusters = final_feature_clusters_new.copy()
    final_feature_names_clusters = list(final_feature_names_clusters_new)
    final_feature_names_ids = list(final_feature_names_ids_new)
    
    """
    Get the clusters! ( by colorbar ) in dendrogram order... 
    """
    features_clusters_colors = sns.color_palette('Spectral', len(final_feature_names_ids))
    features_clusters_colors = np.vstack(features_clusters_colors)
    features_clusters_colorbar = np.zeros((df.shape[-1], 3))
    len_features_clusters = [len(cc) for cc in final_feature_names_ids]
    len_features_clusters_cumsum = np.hstack([0,np.cumsum(len_features_clusters)])
    # features_clusters_dendrogram_order = final_feature_clusters[g.dendrogram_col.reordered_ind]
    
    for kk in np.arange(len(final_feature_names_ids)):    
        # # data = final_feature_names_ids[kk]
        features_clusters_colorbar[len_features_clusters_cumsum[kk]:
                                    len_features_clusters_cumsum[kk+1]] = features_clusters_colors[kk][None,:].copy()
        # features_clusters_colorbar[features_clusters_dendrogram_order==kk+1] = features_clusters_colors[kk][None,:].copy()
    
    
    return g, final_feature_clusters, features_clusters_colorbar, final_feature_names_clusters, final_feature_names_ids  #, features_clusters_table
    
    # return []

    # features_clusters_table.to_csv(os.path.join(saveplotsfolder_cells, 
                                                # 'SAM_hcluster_modules.csv'), index=None)
    
    # """
    # Get the clusters! ( by colorbar )
    # """
    # features_clusters_colors = sns.color_palette('Spectral', len(final_feature_names_ids))
    # features_clusters_colors = np.vstack(features_clusters_colors)
    # features_clusters_colorbar = np.zeros((df.shape[-1], 3))
    # len_features_clusters = [len(cc) for cc in final_feature_names_ids]
    # len_features_clusters_cumsum = np.hstack([0,np.cumsum(len_features_clusters)])
    
    # for kk in np.arange(len(final_feature_names_ids)):    
    #     # data = final_feature_names_ids[kk]
    #     features_clusters_colorbar[len_features_clusters_cumsum[kk]:
    #                                len_features_clusters_cumsum[kk+1]] = features_clusters_colors[kk][None,:].copy()
    
    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.imshow(features_clusters_colorbar[None,...])
    # ax.set_aspect('auto')
    # plt.savefig(os.path.join(saveplotsfolder_cells, 
    #                          'feature_cluster_colorbar.svg'), dpi=300, bbox_inches='tight')
    # plt.show()


def compute_individual_feature_contributions_in_SAM_modules(all_feats, 
                                                            SAM_module_indexes,
                                                            random_state=0):
    r"""
    
    Parameters
    ----------
    all_feats : (n_objects, n_features) array
        feature matrix of all objects 
    SAM_module_indexes : list
        list of the individual feature index in each module 
    random_state : int 
        random seed for PCA

    Returns
    -------
    feature_module_expr : (n_objects, n_modules) array 
        the per module PC1 expression of each object
    feature_module_expr_contrib : list
        contribution of individual features in a module to the modules' PC1 expression
    
    """
    from sklearn.decomposition import PCA
    
    feature_module_expr = []
    feature_module_expr_contrib = []
    
    for mod_ii in np.arange(len(SAM_module_indexes)):
        
        index_mod_ii = SAM_module_indexes[mod_ii]
        pca_mod = PCA(n_components=1, 
                      random_state=random_state, 
                      whiten=False)
        tform_all_feats = pca_mod.fit_transform(all_feats[:,index_mod_ii])
        # tform_all_feats = tform_all_feats[:,-1].copy() # least variation. 
        feature_module_expr.append(np.squeeze(tform_all_feats))
        # feature_module_expr.append(pca_mod) # save the module. 
        
        """
        grab the contribution 
        """
        pc1 = pca_mod.components_[0].copy()
        feature_module_expr_contrib.append(pc1)
        # sort = np.argsort(np.abs(pc1))[::-1]
        
        # fig, ax = plt.subplots(figsize=(15,5))
        # plt.title('module_%s' %(str(mod_ii+1).zfill(3)))
        # ax.bar(np.arange(len(pc1)), pc1[sort], width=0.5)
        # plt.xticks(np.arange(len(pc1)), 
        #             feature_names_new[final_feature_names_ids[mod_ii]][sort],
        #             rotation=90,
        #             fontsize=6)
        # plt.xlim([-0.5, len(pc1)-0.5])
        # plt.savefig(os.path.join(saveplotsfolder_cells, 
        #                           'PC1_module_%s.pdf' %(str(mod_ii+1).zfill(3))), dpi=300, bbox_inches='tight')
        # plt.show()
                   
    feature_module_expr = np.vstack(feature_module_expr).T # have a look at samples in this way. 
    
    return feature_module_expr, feature_module_expr_contrib


def compute_most_representative_image_patches( object_feature_module_expr,
                                                all_patches, 
                                                n_rows = 3, 
                                                n_cols = 3,
                                                rescale_intensity=True):
    
    r""" Compute the purity/representativeness score of each object instance for each SAM module and select the top most representative for each module. 
    
    Parameters
    ----------
    object_feature_module_expr : (n_objects, n_modules) array
        feature matrix of all objects 
    all_patches : (n_features,) array
        names of every included computed feature
    n_rows : int
        the number of images per row in the mosaic panel of each module 
    n_cols : int 
        the number of images per column in the mosaic panel of each module 
    rescale_intensity : bool
        if True, rescales each patch in the mosaic individually before assembling
    
    Returns
    -------
    object_SAM_module_purity_scores : (n_features,n_modules) array
        the enrichment/representativeness of this object instance for each module. The higher, the more unique the object is at representing the module. 
    sample_images_modules : list
        list of the n_rows x n_cols mosaic image panel of most representative objects per module 
    sample_object_index_modules : 
        list of the flattened index of the objects constituting the n_rows x n_cols mosaic image panel of most representative objects per module
        
    """
    # =============================================================================
    #     grab the top patches that are enriched in each modules. 
    # =============================================================================
    import skimage.exposure as skexposure
    import numpy as np 
    
    N = n_rows * n_cols
    
    # random sample a selection ? 
    sample_images_modules = []
    sample_object_index_modules = []
    object_SAM_module_purity_scores = []
    
    N_features_clusters = object_feature_module_expr.shape[-1]
    
    for lab_ii in np.arange(N_features_clusters): 
        # rand_index = np.argsort(samples_feature_module_expr[:,lab_ii])[::-1]
        purity_index = object_feature_module_expr[:,lab_ii] - np.max(object_feature_module_expr[:,np.setdiff1d(np.arange(N_features_clusters), lab_ii)], axis=1)
        object_SAM_module_purity_scores.append(purity_index)
        
        select_index = np.argsort(purity_index)[::-1]
        select_index = select_index[:N]
        
        sample_object_index_modules.append(select_index)
        
        # montage.... 
        all_patches_clust = [] 
        all_patches_sizes = []
        mean_I = []
        
        for ind in select_index:
            img_panel_r_ind = all_patches[ind]
            if rescale_intensity:
                img_panel_r_ind = skexposure.rescale_intensity(img_panel_r_ind)
                
            mean_I.append(np.min(img_panel_r_ind))
            all_patches_clust.append(img_panel_r_ind)
            all_patches_sizes.append(img_panel_r_ind.shape)
        all_patches_sizes = np.vstack(all_patches_sizes)
        
        
        patch_m, patch_n = np.max(all_patches_sizes, axis=0)
        img_panel = np.percentile(mean_I,75)*np.ones((n_rows*patch_m, 
                                                      n_cols*patch_n)) #* np.nanmean(mean_I)
        
        for ii in np.arange(n_rows):
            for jj in np.arange(n_cols):
                kk = ii*n_cols + jj
                patch = all_patches_clust[kk]
                # center this.
                mm, nn = patch.shape[:2]
                
                offset_mm = (patch_m - mm)//2
                offset_nn = (patch_n - nn)//2
                
                img_panel[ii*patch_m+offset_mm:ii*patch_m+patch.shape[0]+offset_mm, 
                          jj*patch_n+offset_nn:jj*patch_n+patch.shape[1]+offset_nn] = patch.copy()
        
        # print(np.min(img_panel), np.max(img_panel))
        
        img_panel = np.uint8(255.*skexposure.rescale_intensity(img_panel*1.)) # this rescales globally. 
        # # img_panel = np.vstack([np.hstack(img_panel[:4]), 
        # #                         np.hstack(img_panel[4:8]),
        # #                         np.hstack(img_panel[8:12]),
        # #                         np.hstack(img_panel[12:16])])
        # plt.figure()
        # plt.title('SAM Module %d' %(lab_ii+1))
        # plt.imshow(img_panel, cmap='gray')
        # plt.axis('off')
        # plt.grid('off')
        # plt.savefig(os.path.join(saveplotsfolder_cells, 
        #                          'SAM Module %d_highest_differential3x3.svg' %(lab_ii+1)), dpi=300, bbox_inches='tight')
        # plt.show()
        sample_images_modules.append(img_panel)
        
    object_SAM_module_purity_scores = np.vstack(object_SAM_module_purity_scores).T 
    sample_images_modules = np.array(sample_images_modules, dtype=object) # create a standard array. 
        
    return object_SAM_module_purity_scores, sample_images_modules, sample_object_index_modules
    


def get_SAM_feature_type(feature_names):
    
    r""" Given SPOT's SAM phenome's feature names, return whether the feature is classified as shape, appearance or motion and if its scope is global, local-regional or local-distribution 
    
    Parameters
    ----------
    feature_names : (n_features,) array
        names of the individual SAM features to get the type of from SPOT's SAM phenome
    
    Returns
    -------
    feature_SAM_type : (n_features,) array
        each feature with its type labelled as either 0: shape, 1: appearance, 2: motion 
    feature_scope : (n_features,) array
        each feature with its spatial scope labelled as either 0: global, 1: (local) regional, 2: (local) distribution 

    """
    import numpy as np 
    
    # 0-shape, 1-appearance, 2-motion
    feature_SAM_type = np.zeros(len(feature_names), dtype=np.int32)
    
    """
    we need to generate auto labels. 
    """
    haralick_names = np.hstack(['angular_second_moment', 
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
    
    
    # let them be 0 - shape, 1-appearance, 2-motion 
    for name_ii in np.arange(len(feature_names)):
        name = feature_names[name_ii]
        
        if 'intensity' in name:
            feature_SAM_type[name_ii] = 1
        if 'appear' in name:
            feature_SAM_type[name_ii] = 1
        if 'flow' in name:
            feature_SAM_type[name_ii] = 2
        if 'speed' in name:
            feature_SAM_type[name_ii] = 2
        if 'motion' in name:
            feature_SAM_type[name_ii] = 2
        if 'div' in name:
            feature_SAM_type[name_ii] = 2
        if 'curl' in name:
            feature_SAM_type[name_ii] = 2
        if 'hist_bin' in name and 'region_' in name:
            feature_SAM_type[name_ii] = 2 # change this for now. # intensity is not called this? 
        
        # haralick needs to be in appearance
        if name in haralick_names:
            feature_SAM_type[name_ii] = 1
    
        if 'sift_appear' in name:
            feature_SAM_type[name_ii] = 1
        if 'sift_motion' in name:
            feature_SAM_type[name_ii] = 2
    
    """
    We further generate global/regional features/distributional features
    """
    # let them be 0 - macro, 1-regional, 2-distributional 
    feature_scope = np.zeros(len(feature_names), dtype=np.int32)
    
    for name_ii in np.arange(len(feature_names)):
        name = feature_names[name_ii]
        
        if 'hu_moments' in name:
            feature_scope[name_ii] = 1 
        if 'zernike_moments' in name:
            feature_scope[name_ii] = 1
        if 'Fourier' in name:
            feature_scope[name_ii] = 1
        if 'Shape_Context' in name:
            feature_scope[name_ii] = 2
        if 'region' in name:
            feature_scope[name_ii] = 1
        if name in haralick_names: # i guess this is distributional. 
            feature_scope[name_ii] = 2
        if 'sift_' in name:
            feature_scope[name_ii] = 2
        if 'hist' in name:
            feature_scope[name_ii] = 2
        if 'ECT' in name:
            feature_scope[name_ii] = 2
            
    return feature_SAM_type, feature_scope



def map_intensity_interp2(query_pts, grid_shape, I_ref, method='spline', cast_uint8=False, s=0, fill_value=0):
    r""" Interpolate a 2D image at specified coordinate points. 

    Parameters
    ----------
    query_pts : (n_points,2) array
        array of (y,x) image coordinates to interpolate intensites at from ``I_ref``
    grid_shape : (M,N) tuple
        shape of the image to get intensities from 
    I_ref : (M,N) array
        2D image to interpolate intensities from 
    method : str
        Interpolating algorithm. Either 'spline' to use scipy.interpolate.RectBivariateSpline or any other e.g. 'linear' from scipy.interpolate.RegularGridInterpolator
    cast_uint8 : bool
        Whether to return the result as a uint8 image. 
    s : scalar
        if method='spline', s=0 is a interpolating spline else s>0 is a smoothing spline.
    fill_value : scalar
        specifies the imputation value if any of query_pts needs to extrapolated. 
    
    Returns
    -------
    I_query : (n_points,) array
        the intensities of ``I_ref`` interpolated at the specified ``query_pts`` 

    """
    import numpy as np 
    from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator 
    
    if method == 'spline':
        spl = RectBivariateSpline(np.arange(grid_shape[0]), 
                                  np.arange(grid_shape[1]), 
                                  I_ref,
                                  s=s)
        I_query = spl.ev(query_pts[...,0], 
                         query_pts[...,1])
    else:
        spl = RegularGridInterpolator((np.arange(grid_shape[0]), 
                                       np.arange(grid_shape[1])), 
                                       I_ref, method=method, bounds_error=False, fill_value=fill_value)
        I_query = spl((query_pts[...,0], 
                       query_pts[...,1]))

    if cast_uint8:
        I_query = np.uint8(I_query)
    
    return I_query
    
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects

@delayed
@wrap_non_picklable_objects
def kMeansRes(scaled_data, k, alpha_k=0.02):
    r''' Compute the kmeans scaled inertia for use in elbow method number of cluster choosing
    
    Parameters 
    ----------
    scaled_data: matrix 
        scaled data. rows are samples and columns are features for clustering
    k: int
        current k for applying KMeans
    alpha_k: float
        manually tuned factor that gives penalty to the number of clusters
    
    Returns 
    -------
    scaled_inertia: float
        scaled inertia value for current k           
    '''
    import numpy as np 
    from sklearn.cluster import KMeans
    
    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    # fit k-means
    kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia

def chooseBestKforKMeansParallel(scaled_data, k_range):
    ''' Parameter scan the number of Kmeans cluster and choose that which minimizes the scaled inertia metric
    
    Parameters 
    ----------
    scaled_data: matrix 
        scaled data. rows are samples and columns are features for clustering
    k_range: list of integers
        k range for applying KMeans
    Returns 
    -------
    best_k: int
        chosen value of k out of the given k range.
        chosen k is k with the minimum scaled inertia value.
    results: pandas DataFrame
        adjusted inertia value for each k in k_range
    '''
    # from joblib import Parallel, delayed # this has problems with serialization 
    import pandas as pd 
    
    # ans = Parallel(n_jobs=-1,verbose=10)(delayed(kMeansRes)(scaled_data, k) for k in range(k_range))
    ans = Parallel(n_jobs=-1,verbose=10)(kMeansRes(scaled_data, k) for k in k_range)
    # ans = [kMeansRes(scaled_data, k) for k in range(k_range)]
    # ans = [kMeansRes(scaled_data, k) for k in range(k_range)]
    ans = list(zip(k_range,ans))
    results = pd.DataFrame(ans, columns = ['k','Scaled Inertia']).set_index('k')
    best_k = results.idxmin()[0]
    
    return best_k, results


def KMeans_cluster_low_dimensional_embedding(low_dimensional_feats,
                                             k=2,
                                             random_state=0):
    ''' Use Kmeans cluster to cluster the low-dimensional embedding such as that from UMAP with the desired number of clusters, k
    
    Parameters 
    ----------
    low_dimensional_feats: (N,n_dim) array 
        feature matrix, the result of applying dimensional reduction to SPOT SAM features.
    k: int
        the desired number of final clusters
        
    Returns 
    -------
    cluster_labels : (N,) array 
        integer label array where each unique integer corresponds to a unique KMeans cluster id 
    clusterer : sklearn.cluster.KMeans object
        this is the final fitted KMeans clusterer using the value found for best_k
        
    '''
    from sklearn.cluster import KMeans

    # recluster with the best k 
    clusterer = KMeans(n_clusters=k, random_state=random_state) # how many clusters? 
    clusterer.fit(low_dimensional_feats)

    cluster_labels = clusterer.predict(low_dimensional_feats)
    
    return cluster_labels, clusterer


def KMeans_cluster_low_dimensional_embedding_with_scan(low_dimensional_feats,
                                                         k_range=(2,20),
                                                         ploton=True):
    ''' Use Kmeans cluster to cluster the low-dimensional embedding such as that from UMAP and choose the number of clusters, k which minimizes the scaled inertia metric in the given range.
    
    Parameters 
    ----------
    low_dimensional_feats: (N,n_dim) array 
        feature matrix, the result of applying dimensional reduction to SPOT SAM features.
    k_range: 2-tuple or list 
        the minimum and maximum number of clusters, the function will scan every integer in between. 
    ploton: bool
        if True, will plot the number of clusters, k vs the scaled inertia metric
        
    Returns 
    -------
    cluster_labels : (N,) array 
        integer label array where each unique integer corresponds to a unique KMeans cluster id 
    clusterer : sklearn.cluster.KMeans object
        this is the final fitted KMeans clusterer using the value found for best_k
    best_k: int
        chosen value of k out of the given k range.
        chosen k is k with the minimum scaled inertia value.
    scan_results: pandas DataFrame
        adjusted inertia value for each k in k_range
    '''
    from sklearn.cluster import KMeans
    
    best_k, scan_results = chooseBestKforKMeansParallel(low_dimensional_feats, k_range=range(k_range[0], k_range[1]))

    if ploton:
        import pylab as plt 

        # plot the results
        plt.figure(figsize=(8,4))
        plt.plot(scan_results,'o')
        plt.title('Adjusted Inertia for each K')
        plt.xlabel('K')
        plt.ylabel('Adjusted Inertia')
        plt.xticks(range(k_range[0],k_range[1],1))
        plt.show()

    # recluster with the best k 
    clusterer = KMeans(n_clusters=best_k, random_state=0) # how many clusters? 
    clusterer.fit(low_dimensional_feats)

    cluster_labels = clusterer.predict(low_dimensional_feats)
    
    return cluster_labels, clusterer, best_k, scan_results
    

def reorder_and_relabel_clusters(clusters, 
                                 score, 
                                 order='ascending'):
    
    r""" Relabel cluster ids according to ascending or descending mean values of a score in the cluster and return the new id mappings
    
    Parameters 
    ----------
    clusters: (N,) array 
        input clustering, unique integers are taken as unique cluster ids
    score: (N,) array
        the scalar scoring function which we will sort clusters according to 
    order: 'ascending' or 'descending'
        sort clusters so that going from smallest to largest cluster id either corresponds to 'ascending' small to large or 'descending' large to small of score 
        
    Returns 
    -------
    relabeled_clusters : (N,) array 
        same size as the input clusters but now relabelled so that cluster ids have been sorted by the mean score 
    cluster_mapping_dict: dict
        dictionary such that given the old id returns the new cluster id, cluster_mapping_dict[old_label_id] = new_label_id
    
    """
    import numpy as np 
    # get the average statistic per cluster label
    clust_labels_mean_score = []
    
    for lab in np.unique(clusters):
        select_lab = clusters == lab 
        clust_labels_mean_score.append(np.nanmean(score[select_lab]))
        
    clust_labels_mean_score = np.hstack(clust_labels_mean_score)
    
    if order == 'ascending':
        ordering = np.argsort(clust_labels_mean_score) # sort by increasing eccentricity
    elif order == 'descending' :
        ordering = np.argsort(clust_labels_mean_score)[::-1] # sort by increasing eccentricity        
    else:
        print('invalid ordering specified')
        
    cluster_mapping_dict = {}
    relabeled_clusters = np.zeros_like(clusters)
    
    for ii, order_ii in enumerate(ordering):
        
        lab = np.unique(clusters)[order_ii]
        select_lab = clusters == lab 
        
        relabeled_clusters[select_lab] = ii
        
        # update the mapping 
        cluster_mapping_dict[lab] = ii
    
    return relabeled_clusters, cluster_mapping_dict


def get_subset_features(all_feats, featnames, featnames_select):

    r""" Given the names of the features, retrieve the corresponding subset of the numerical feature matrix 
    
    Parameters 
    ----------
    all_feats: (N_objects, N_feats) array 
        table of numerical features we wish to index
    featnames: (N_feats,) array
        the corresponding full feature names
    featnames_select: (N_feats_select,) array
        list or array of the names of the features desired to be selected
        
    Returns 
    -------
    all_feats_select : (N_objects, N_feats_select) array 
        table of selected numerical features, all column entries will be set to np.nan if the feature name could not be matched exactly to an entry in featnames 
    
    """
    import numpy as np 
    
    # construct the selectors 
    feat_select_inds = [] 
    for ff in featnames_select:
        select = featnames == ff
        if np.sum(select) > 0: # it exists
            feat_select_ind = np.arange(all_feats.shape[1])[select][0]
            feat_select_inds.append(feat_select_ind)
        else:
            feat_select_inds.append(-1)            
    feat_select_inds = np.hstack(feat_select_inds)

    # those that are -1 replace with zeros. 
    all_feats_select = all_feats[:,feat_select_inds].copy()
    all_feats_select[:,feat_select_inds==-1] = np.nan # replace with nans.

    return all_feats_select


def compute_mean_score_features_cluster(clusters, all_feats, avg_func=np.nanmean):
    r""" Compute the mean of the selected features for each unique cluster id 
    
    Parameters 
    ----------
    clusters: (N,) array 
        input clustering, unique integers are taken as unique cluster ids
    score: (N,) array
        the scalar scoring function which we will sort clusters according to 
    order: 'ascending' or 'descending'
        sort clusters so that going from smallest to largest cluster id either corresponds to 'ascending' small to large or 'descending' large to small of score 
        
    Returns 
    -------
    unique_clust_labels : (N_clusters,) array 
        unique cluster ids
    cluster_mean_scores: (N_clusters,N_feats) array
        mean value of each feature per cluster
    cluster_std_scores: (N_clusters,N_feats) array
        standard deviation of each feature per cluster
    
    """
    import numpy as np 
    
    unique_clust_labels = np.unique(clusters)
    
    cluster_mean_scores = []
    cluster_std_scores = []

    for lab in unique_clust_labels:
        
        select_lab = clusters == lab     
        cluster_mean_scores.append(avg_func(all_feats[select_lab, :], axis=0))
        cluster_std_scores.append(np.std(all_feats[select_lab, :], axis=0))
        
    cluster_mean_scores = np.vstack(cluster_mean_scores)
    cluster_std_scores = np.vstack(cluster_std_scores)
    
    return unique_clust_labels, cluster_mean_scores, cluster_std_scores


def compute_heatmap_density_image_Embedding(lower_dimension_pts_2D, 
                                            all_conditions=None, 
                                            unique_conditions=None,
                                            cmap='coolwarm',
                                            grid_scale_factor=500, 
                                            sigma_factor=0.25, 
                                            contour_sigma_levels=[1,2,3,3.5,4, 4.5, 5],
                                            saveplotsfolder=None):
    
    r""" Given the coordinates of the lower dimensional embedding, estimate the local point density via image-based Gaussian smoothing method
    
    Parameters
    ----------
    lower_dimension_pts_2D : (N_objects, n_dim) array 
        dimensional reduced coordinates of SPOT SAM features. Can be of any dimension but the 1st two will be taken as the 2D phenomic landscape coordinates 
    all_conditions : (N_objects,) array 
        the condition of each object instance. If not specified, all object instances are assumed to come from a single condition - 'na' class    
    unique_conditions : list or array
        the unique conditions in the experiment. If not specified, the unique conditions will be obtained by np.unique(all_conditions) 
    cmap : str
        the color to show the point density if debugviz=True
    grid_scale_factor : int 
        the image size to compute the point density via image means. The larger the size, the finer the higher the resolution but slower the computation as filtering will take longer.      
    sigma_factor : 
        the scale correction factor for estimating the bandwidth of the smoothing gaussian filter. The sigma = sigma_factor * mean(pairwise_distances of points per condition per time interval)
    contour_sigma_levels : list of floats
        will plot isocontour lines for each of the contour_sigma_levels of the density i.e. at heatmap mean(density) + contour_sigma_level * std(density). 
    saveplotsfolder : str    
        if specified, will save the generated heatmap figures to this folder 
        
    Returns
    -------
    all_pts_density : (n_conditions,) list
        for each condition, for each object in each condition, gives the local point density around its position in the phenomic landscape.
    all_pts_density_select : (n_conditions, N) array
        boolean array specifying which of the original array correspond to each unique condition
    
    """
    import numpy as np 
    import pylab as plt 
    from sklearn.metrics import pairwise_distances
    import skimage.filters as skfilters
    from skimage.measure import find_contours
    import os 
    
    # convert the points to images to estimate the point densities. 
    pad = 1
    min_pt = np.min(lower_dimension_pts_2D) - pad
    max_pt = np.max(lower_dimension_pts_2D) + pad
    
    uu_tform = (lower_dimension_pts_2D - min_pt) / (max_pt - min_pt) * grid_scale_factor
    
        
    grid_pts_image = np.zeros((grid_scale_factor+1, grid_scale_factor+1), dtype=np.float32)
    
    all_pts_density = []
    all_pts_density_select = []
    
    if all_conditions is None:
        all_conditions = np.ones(len(lower_dimension_pts_2D), dtype=np.int64) # everything is one
        
    if unique_conditions is None:
        unique_conditions = np.unique(all_conditions)
        
    for cond_ii in np.arange(len(unique_conditions))[:]:
    
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title(unique_conditions[cond_ii])
        ax.scatter(lower_dimension_pts_2D[:,0], 
                   lower_dimension_pts_2D[:,1], c='lightgrey')
        
        if all_conditions is not None:
            select = all_conditions==unique_conditions[cond_ii]
        else:
            select = np.ones(len(lower_dimension_pts_2D), dtype=bool)
            
        all_pts_density_select.append(select)
        
        xy = uu_tform[select].copy()
        sigma = sigma_factor*np.mean(pairwise_distances(uu_tform[select])) # how to get a better estimate... 
        
        grid_pts_xy = grid_pts_image.copy().astype(np.float32)
        grid_pts_xy[xy[:,0].astype(np.int32), xy[:,1].astype(np.int32)] += 1. # have to keep adding.
        grid_pts_xy = skfilters.gaussian(grid_pts_xy, 
                                         sigma=sigma, 
                                         mode='reflect', 
                                         preserve_range=True) # depends on the sigma. 
        
        # interpolate to the points.
        z = map_intensity_interp2(uu_tform[select,:], 
                                  grid_shape=grid_pts_xy.shape, 
                                  I_ref=grid_pts_xy, method='linear', 
                                  cast_uint8=False, 
                                  s=0)
        all_pts_density.append(z)
        
        z_id = np.argsort(z) # derive for the full coordinates.
        ax.scatter(lower_dimension_pts_2D[select][z_id,0], 
                   lower_dimension_pts_2D[select][z_id,1], 
                   c=z[z_id], 
                   cmap=cmap)
        

        for level in contour_sigma_levels:
            contour_0 = find_contours(grid_pts_xy, 
                                      np.mean(grid_pts_xy)+level*np.std(grid_pts_xy))
            
            for cnt in contour_0:    
                cnt = cnt/ grid_scale_factor *  (max_pt - min_pt)  + min_pt  # - pad
                
                plt.plot(cnt[:,0],
                         cnt[:,1], color='k', zorder=100, 
                         alpha=1, lw=3)#,
                
        plt.axis('off')
        plt.grid('off')
        
        if saveplotsfolder is not None:
            plt.savefig(os.path.join(saveplotsfolder,
                                     str(unique_conditions[cond_ii])+'_heatmap_density_embedding.svg'), bbox_inches='tight')
        plt.show()
        
    all_pts_density_select = np.vstack(all_pts_density_select)
    
    return all_pts_density, all_pts_density_select
    

def compute_phenotypic_trajectory(lower_dimension_pts_2D, 
                                    objects_time, 
                                    time_intervals,
                                    all_conditions=None, 
                                    unique_conditions=None,
                                    cmap='coolwarm',
                                    grid_scale_factor=500, 
                                    sigma_factor=0.25, 
                                    thresh_density_sigma = 3,
                                    debugviz=False):

    r""" Given the 2D phenomic landscape, construct the phenomic trajectory for each given condition and the given temporal sampling regime. 
    
    Parameters
    ----------
    lower_dimension_pts_2D : (N_objects, n_dim) array 
        dimensional reduced coordinates of SPOT SAM features. Can be of any dimension but the 1st two will be taken as the 2D phenomic landscape coordinates 
    objects_time : (N_objects,) array 
        the timepoint each object instance was obtained from 
    time_intervals : lenght (n_time_intervals+1,) list or array 
        the global time interval sampling, the density of objects mapping to the phenomic landscape within each interval will be used to construct the 'majority' phenomic coordinate to represent the phenotypic diversity in that interval.
    all_conditions : (N_objects,) array 
        the condition of each object instance. If not specified, all object instances are assumed to come from a single condition - 'na' class
    unique_conditions : list or array
        the unique conditions in the experiment. If not specified, the unique conditions will be obtained by np.unique(all_conditions) 
    cmap : str
        the color to show the point density if debugviz=True
    grid_scale_factor : int 
        the image size to compute the point density via image means. The larger the size, the finer the higher the resolution but slower the computation as filtering will take longer.  
    sigma_factor : float
        the scale correction factor for estimating the bandwidth of the smoothing gaussian filter. The sigma = sigma_factor * mean(pairwise_distances of points per condition per time interval)
    thresh_density_sigma : float  
        the density heatmap of points will be thresholded by mean(density) + thresh_density_sigma * std(density). The higher the threshold, the more the 'majority' coordinate reflects the average of the summation of phenotypes across the densest parts of the phenomic landscape       
    debugviz : bool
        if True, plot the image of density heatmap and computed 'majority' phenomic coordinate per time interval. 
        
    Returns
    -------
    all_trajectories : (n_conditions, n_time_intervals, 2) array
        The per condition 2D phenotype trajectory in phenomic landscape coordinates. 
    all_density_contours : list
        if debugviz=True, for each condition this will record the contour of the heatmap at thresholds of mean(density) + 1 * std(density), mean(density) + 2 * std(density), and mean(density) + 3 * std(density) for each time interval bin
    
    """
    import numpy as np 
    import pylab as plt 
    from sklearn.metrics import pairwise_distances
    import skimage.filters as skfilters
    from skimage.measure import find_contours
    
    # convert the points to images to estimate the point densities. 
    # pad = 1
    min_pt = np.min(lower_dimension_pts_2D) - 1
    max_pt = np.max(lower_dimension_pts_2D) + 1
    
    uu_tform = (lower_dimension_pts_2D - min_pt) / (max_pt - min_pt) * grid_scale_factor
    
    grid_pts_image = np.zeros((grid_scale_factor+1, grid_scale_factor+1), dtype=np.float32)
    
    
    # all_organoid_days = (all_organoid_TP-1) * all_organoid_Ts / 24.
    # day_bins = np.linspace(0, np.max(all_organoid_days), int(2*np.max(all_organoid_days))+1)
    
    
    if all_conditions is None:
        all_conditions = ['na']
        
    if unique_conditions is None:
        unique_conditions = np.unique(all_conditions)
    
    
    all_trajectories = []
    all_density_contours = []
    
    
    # iterate over unique conditions. 
    for cond_ii, cond in enumerate(unique_conditions):
        
        traj_cond_time = []
        contour_cond_time = []
        
        select_cond = np.hstack(all_conditions) == cond
        
        # iterate over each time interval 
        for day_bin_ii in np.arange(len(time_intervals)-1)[:]:
    
            select_day = np.logical_and(objects_time>=time_intervals[day_bin_ii], 
                                        objects_time<=time_intervals[day_bin_ii+1])
            select = np.logical_and(select_cond, select_day) # joint data selector of condition and time interval 
            
            """
            some of these were not filmed long enough!.
            """
            if np.sum(select) == 0: 
                traj_cond_time.append(np.hstack([np.nan, np.nan]))
                contour_cond_time.append([])
            else:
                xy = uu_tform[select].copy()
                sigma = sigma_factor*np.nanmean(pairwise_distances(uu_tform[select])) # how to get a better estimate... 
                
                grid_pts_xy = grid_pts_image.copy().astype(np.float32)
                grid_pts_xy[xy[:,0].astype(np.int32), xy[:,1].astype(np.int32)] += 1. # have to keep adding.
                grid_pts_xy = skfilters.gaussian(grid_pts_xy, sigma=sigma, mode='reflect', preserve_range=True) # depends on the sigma. 
                
                
                level1mask = grid_pts_xy>=np.mean(grid_pts_xy)+thresh_density_sigma*np.std(grid_pts_xy) # 3x or 2x? 
                grid_pt_mean = np.mean(np.array(np.where(level1mask>0)).T, axis=0)
                traj_cond_time.append(grid_pt_mean)
                
                # contour_0 = find_contours(im, np.mean(im)+level*np.std(im))
                contour_cond_time_levels = []
                    
                if debugviz: 
                    
                    plt.figure()
                    plt.imshow(grid_pts_xy, cmap='coolwarm')
                    plt.plot(grid_pt_mean[1], grid_pt_mean[0], 'ro')
                    # for level in [2.8, 3, 3.2]:
                    for level in [1, 2, 3]:
                        contour_0 = find_contours(grid_pts_xy, np.mean(grid_pts_xy)+level*np.std(grid_pts_xy))
                        contour_cond_time_levels.append(contour_0)
                        for cnt in contour_0:
                            # cnt = cnt/float(grid_size[0]) * (max_lim[0][1] - max_lim[0][0]) + max_lim[0][0]
                            # ax.plot(cnt[:, 0],
                            #         cnt[:,1], color=color, zorder=zorder, 
                            #         alpha=1, lw=3)#,
                            plt.plot(cnt[:,1],
                                     cnt[:,0], color='k', zorder=100, 
                                     alpha=1, lw=3)#,
                    plt.show()
                    
                contour_cond_time.append(contour_cond_time_levels)
            
        all_density_contours.append(contour_cond_time)
        all_trajectories.append(traj_cond_time)
        
    all_trajectories = np.array(all_trajectories) # this is regular. 
    
    # rescale back into the actual coordinate space
    all_trajectories = all_trajectories / grid_scale_factor *  (max_pt - min_pt)  + min_pt 
    
    return all_trajectories, all_density_contours


def construct_affinity_matrix_phenotypic_trajectory(trajectories_condition,
                                                    affinity='dtw',
                                                    use_std_denom=True):
    r""" Construct the affinity matrix between phenotypic trajectories to compare differences
    
    Parameters
    ----------
    trajectories_condition : 
        
    affinity : 'dtw' or 'euclidean'
        
    use_std_denom : bool
        if True, use the affinity = np.exp(-pair_dist_matrix/pair_dist_matrix.std()) as the affinity matrix. if False, use the affinity = np.exp(-pair_dist_matrix/pair_dist_matrix.mean()) as the affinity matrix
    
    Returns
    -------
    affinity_matrix : 
    
    """
    import numpy as np 
    # we can use a euclidean distance matrix. 
    from sklearn.metrics.pairwise import pairwise_distances
    from dtaidistance import dtw_ndim
    
    # find the valid times 
    valid_times = np.sum(np.isnan(trajectories_condition[...,0]), axis=0)
    valid_times_select = valid_times == 0 

    N_conditions = len(trajectories_condition)
    
    if affinity == 'dtw':
        pair_dist_matrix = np.zeros((N_conditions, N_conditions))
        
        for ii in np.arange(N_conditions):
            for jj in np.arange(N_conditions):
                d = dtw_ndim.distance(trajectories_condition[ii][valid_times_select], trajectories_condition[jj][valid_times_select])    
                pair_dist_matrix[ii,jj] = d
                
    elif affinity == 'euclidean':
        pair_dist_matrix = pairwise_distances(trajectories_condition[:,valid_times_select].reshape(len(trajectories_condition), -1))
        
    else:
        print('not valid affinity matrix type ')
    
    # what kernel 
    if use_std_denom: 
        affinity_matrix = np.exp(-pair_dist_matrix/pair_dist_matrix.std())
    else:
        affinity_matrix = np.exp(-pair_dist_matrix/np.mean(pair_dist_matrix))
            
    return affinity_matrix
    

def hcluster_phenotypic_trajectory_affinity_matrix(affinity_matrix,
                                                   uniq_conditions,
                                                   linkage='average', 
                                                   metric='euclidean',
                                                   cmap='vlag', 
                                                   figsize=(10,10),
                                                   save_dpi=300,
                                                   savefolder=None):
    
    r""" Uses pandas clustermap to visualize the affinity matrix as a heatmap and hierarchically cluster the matrix. 
    
    Parameters
    ----------
    affinity_matrix : (N_unique_conditions, N_unique_conditions) array 
        Pairwise trajectory affinity matrix between all unique conditions 
    uniq_conditions : (N_unique_conditions,) array
        Unique conditions 
    linkage : str
        linkage used in hierarchical clustering c.f. seaborn.clustermap
    metric : str    
        metric used in the hierarchical clustering c.f. seaborn.clustermap
    cmap : str
        colormap to visualize the affinity matrix 
    figsize : 2-tuple
        size of figure canvas
    save_dpi : int
        dpi of the saved figure if savefolder is not None 
    savefolder : str
        folder path to save the clustered heatmap and dendrogram figure
        
    Returns
    -------
    g : seaborn.clustermap object
        result of clustering as a seaborn.clustermap object 
    
    """
    import seaborn as sns
    import pandas as pd 
    import os 
    
    df = pd.DataFrame(affinity_matrix, 
                      index=uniq_conditions, 
                      columns=None)
    
    # fig, ax = plt.subplots(figsize=(.5*100,.5*16))
    g = sns.clustermap(df, 
                        method=linkage, # use ward? ---- we should use ward here to abide by visual  ---- Use this .... 
                        metric=metric,
                        cmap=cmap,
                        figsize=figsize)#,
#                        dendrogram_ratio=(0.05,0.1))
#    g.ax_cbar.remove()
    g.ax_row_dendrogram.remove()
    g.ax_heatmap.xaxis.set_ticks(np.arange(len(g.dendrogram_col.reordered_ind))+.5)
    g.ax_heatmap.xaxis.set_ticklabels(uniq_conditions[g.dendrogram_col.reordered_ind])

    if savefolder is not None:
        g.savefig(os.path.join(savefolder, 
                               'trajectory_clustering_%s-linkage_%s-metric.svg' %(str(linkage), str(metric))), 
                  bbox_inches='tight', 
                  pad_inches=0, 
                  dpi=save_dpi)
    
    return g 


def find_representative_images_per_cluster_PCA(all_feats,
                                                all_embedding_coords,
                                                all_object_patches, 
                                               all_object_patches_sizes,
                                               all_cluster_labels, 
                                               unique_cluster_labels, 
                                               rev_sign=False,
                                               mosaic_size=0.7,
                                               percentile_cutoff = 0.9, 
                                               n_rows_mosaic = 2, 
                                               n_cols_mosaic = 2,
                                               pca_random_state=0,
                                               debugviz=False):
    
    r""" Sample the most representative patches according to PCA per cluster, returning them as a mosaic in standardized size and real size. 
    
    Parameters
    ----------
    all_feats : (N_objects, n_features) array
        the SPOT SAM features without dimensional reduction
    all_embedding_coords : (N_objects, n_dim) array
        the low-dimensional reduction of SPOT SAM features
    all_object_patches : (N_objects, m, n) or (N_objects, m, n, 3) array
        the equal sized cropped object images for all object instances
    all_object_patches_sizes : (N_objects, M, N) array
        the real-size image dimensions for each object instance
    all_cluster_labels : (N_objects,)
        cluster ids of all object instances
    unique_cluster_labels : (N_unique_objects,) array
        the unique cluster ids
    rev_sign : bool
        if True, flip the sign of the PC1 component. We always rank descending so that largest positive to largest negative value
    n_rows_mosaic : int
        number of image patches to be used as rows in the mosaic
    n_cols_mosaic : int 
        number of image patches to be used as rows in the mosaic
    pca_random_state : int
        the random seed to be used in PCA.
    debugviz : bool
        if True, shows the mosaic panel as they are created. 
    
    Returns
    -------
    representative_image_dict : dict
        representative images found and their associated metadata as a dictionary with the following entries
        
        * representative_images_std_size : (n_unique_clusters, n_rows_mosaic, n_cols_mosaic) image patches
            representative image patches per cluster in standarized size arranged as a (n_rows_mosaic, n_cols_mosaic) image mosaic
        * representative_images_real_size : (n_unique_clusters, n_rows_mosaic, n_cols_mosaic) image patches
            representative image patches per cluster in real size, with appropriate extra space padding arranged as a (n_rows_mosaic, n_cols_mosaic) image mosaic
        * representative_images_raw_shape_dimensions : (n_unique_clusters, n_rows_mosaic*n_cols_mosaic, 2) array
            image dimensions of the raw image crops of the representative images per cluster
        * representative_images_embedding_coords : (n_unique_clusters, n_rows_mosaic, n_cols_mosaic, n_dim) array 
            the low-dimensional reduction of SPOT SAM features corresponding to the found representative image patches
        * representative_images_indices : (n_unique_clusters, n_rows_mosaic*n_cols_mosaic) array
            the index of the image patches that were sampled as the representative image for the cluster
        
    """
    from sklearn.decomposition import PCA
    import skimage.transform as sktform 
    import numpy as np 
    # sample_images.append(rand_index)
    import skimage.exposure as skexposure
    import pylab as plt 
    
    sample_images = []
    sample_images_patch_sizes = []
    sample_images_realsize = []
    sample_images_embedding_coords = []
    sample_images_indices = []
    
    N_total = n_rows_mosaic*n_cols_mosaic

    # define the final size. 
    patch_m, patch_n = np.int32(np.max(all_object_patches_sizes, axis=0) * mosaic_size)
    
    for lab_ii, lab in enumerate(unique_cluster_labels): 
        
        select = all_cluster_labels == lab
        select_index = np.arange(len(all_cluster_labels))[select]
        
        pca_cluster = PCA(n_components=1, 
                          random_state=pca_random_state, 
                          whiten=True)
        pca_score_clusters = pca_cluster.fit_transform(all_feats[select])
        
        # np.random.shuffle(rand_index)
        pca_score_clusters = pca_score_clusters.ravel()
        
        if rev_sign:
            pca_score_clusters = pca_score_clusters * -1
        
        # valid = pca_score_clusters<=np.percentile(pca_score_clusters,95)
        valid = pca_score_clusters <= np.percentile(pca_score_clusters, percentile_cutoff*100)
        
        # regate
        select_index = select_index[valid]
        pca_score_clusters = pca_score_clusters[valid]
        
        select_index_ = select_index[np.argsort(pca_score_clusters)[::-1]] # was abs? we shud check! 
        # rand_index_ = (rand_index[valid])[np.argsort(pca_score_clusters[valid])[::-1]]
        select_index = select_index_[:N_total]
        
        sample_images_indices.append(select_index)
        sample_images_embedding_coords.append(all_embedding_coords[select_index])
        
        """
        a. standard size. 
        """
        img_panel = all_object_patches[select_index].copy()
        # print(img_panel.shape)
        
        if len(img_panel.shape) == 3: 
            _, M, N = img_panel.shape[:]
            # print('test', img_panel.shape)
            # img_panel = img_panel.reshape(n_rows_mosaic*M, n_cols_mosaic*N, order='C')
            
        elif len(img_panel.shape) == 4: 
            _, M, N, ch = img_panel.shape[:]
            # # img_panel = img_panel.reshape(n_rows_mosaic*M, n_cols_mosaic*N, ch, order='C')
            # img_mosaic = np.zeros((n_rows_mosaic*M, 
            #                        n_cols_mosaic*N, 
            #                        ch))
        else:
            print('error')
            
        img_mosaic = np.zeros((n_rows_mosaic*M, 
                                   n_cols_mosaic*N)) 
            
        for ii in np.arange(n_rows_mosaic):
            for jj in np.arange(n_cols_mosaic):
                kk = ii*n_cols_mosaic + jj
                patch = img_panel[kk].copy()
                
                img_mosaic[ii*M:ii*M+M, 
                          jj*N:jj*N+N] = patch.copy()
                
        img_panel = img_mosaic.copy()
        # img_panel = np.vstack([np.hstack([img_panel[0], img_panel[1]]), np.hstack([img_panel[2], img_panel[3]])])
        # ok we have the panel sizes! information therefore we can save this information out. 
        
        if debugviz:
            plt.figure()
            plt.title('Cluster_'+str(lab)+'_std-size')
            plt.imshow(img_panel)
            plt.show()
            
        
        sample_images.append(img_panel)
        sample_images_patch_sizes.append(all_object_patches_sizes[select_index])
        
        
        """
        make it the correct size. 
        """
        sample_images_realsize

        all_patches_clust = [] 
        all_patches_sizes = []
        mean_I = []

        for r_ind in select_index: # iterate over each. 
            img_panel_r_ind = all_object_patches[r_ind]
            # also get the image size!
            img_panel_r_ind_size = all_object_patches_sizes[r_ind]
            # resize 
            img_panel_r_ind = sktform.resize(img_panel_r_ind, output_shape=img_panel_r_ind_size, order=1, preserve_range=True)
            
            img_panel_r_ind = skexposure.rescale_intensity(img_panel_r_ind) # rescale the intensity so we can see? 
            # mean_I.append(np.max(img_panel_r_ind))
            mean_I.append(np.min(img_panel_r_ind)) # save the background
            
            all_patches_clust.append(img_panel_r_ind)
            all_patches_sizes.append(img_panel_r_ind.shape)
        
        # img_panel = np.percentile(mean_I,75)*np.ones((n_rows*patch_m, 
        #                       n_cols*patch_n)) #* np.mean(mean_I)
        
        if len(img_panel_r_ind.shape) == 2:
            img_panel = np.min(mean_I)*np.ones((n_rows_mosaic*patch_m, 
                                                n_cols_mosaic*patch_n)) #* np.mean(mean_I)
        else:
            img_panel = np.min(mean_I)*np.ones((n_rows_mosaic*patch_m, 
                                                n_cols_mosaic*patch_n,
                                                img_panel_r_ind.shape[-1])) #* np.mean(mean_I)
        
        # montage
        for ii in np.arange(n_rows_mosaic):
            for jj in np.arange(n_cols_mosaic):
                kk = ii*n_cols_mosaic + jj
                patch = all_patches_clust[kk].copy()
                # center this.
                mm, nn = patch.shape[:2]
                offset_mm = (patch_m - mm)//2
                offset_nn = (patch_n - nn)//2
                
                img_panel[ii*patch_m+offset_mm:ii*patch_m+patch.shape[0]+offset_mm, 
                          jj*patch_n+offset_nn:jj*patch_n+patch.shape[1]+offset_nn] = patch.copy()
        
        if debugviz:
            plt.figure()
            plt.title('Cluster_'+str(lab)+'_real-size')
            plt.imshow(img_panel)
            plt.show()
        
        sample_images_realsize.append(img_panel)
        
    representative_images_std_size = np.array(sample_images)
    representative_images_raw_shape_dimensions = np.array(sample_images_patch_sizes)
    representative_images_real_size = np.array(sample_images_realsize)
    representative_images_embedding_coords = np.array(sample_images_embedding_coords)
    representative_images_indices = np.array(sample_images_indices)
    
    representative_image_dict = {'representative_images_std_size': representative_images_std_size,
                                 'representative_images_real_size':representative_images_real_size,
                                 'representative_images_raw_shape_dimensions':representative_images_raw_shape_dimensions,
                                 'representative_images_embedding_coords':representative_images_embedding_coords,
                                 'representative_images_indices': representative_images_indices}
    
    return representative_image_dict


def compute_temporal_cluster_proportions(cluster_labels, 
                                        objects_time, 
                                        time_intervals,
                                        all_conditions=None, 
                                        unique_conditions=None):
    r""" Compute the proportions of objects per phenotype cluster in the given time intervals
    
    Parameters
    ----------
    cluster_labels : (N_objects,) array 
        the cluster ids each object belongs to
    objects_time : (N_objects,) array 
        the timepoint each object instance was obtained from 
    time_intervals : lenght (n_time_intervals+1,) list or array 
        the global time interval sampling, the density of objects mapping to the phenomic landscape within each interval will be used to construct the 'majority' phenomic coordinate to represent the phenotypic diversity in that interval.
    all_conditions : (N_objects,) array 
        the condition of each object instance. If not specified, all object instances are assumed to come from a single condition - 'na' class
    unique_conditions : list or array
        the unique conditions in the experiment. If not specified, the unique conditions will be obtained by np.unique(all_conditions) 

    Returns
    -------    
    uniq_clust_labels : (n_unique_labels,) array
        unique cluster label ids 
    proportion_time_condition : (n_unique_condition, n_time_intervals, n_unique_labels) array
        the fractional proportion of objects in each time interval in each phenotype cluster
    
    """
    import numpy as np 
    
    uniq_clust_labels = np.unique(cluster_labels)
    
    # day_bins_composition = np.linspace(0, np.max(all_organoid_days), int(8*np.max(all_organoid_days))+1)
    if all_conditions is None:
        all_conditions = np.hstack(['na']*len(cluster_labels))
        
    if unique_conditions is None:
        unique_conditions = np.unique(all_conditions)
    
    proportion_time_condition = []
    
    for cond_ii in np.arange(len(unique_conditions))[:]:
        # print(cond_ii)

        proportion_time = []
        select_cond = np.hstack(all_conditions)==unique_conditions[cond_ii]

        for day_bin_ii in np.arange(len(time_intervals)-1)[:]:
    
            select_day = np.logical_and(objects_time>=time_intervals[day_bin_ii], 
                                        objects_time<=time_intervals[day_bin_ii+1])
            select = np.logical_and(select_cond, select_day)
            # print(day_bin_ii, np.sum(select))
            
            if np.sum(select) == 0:
                proportion_time.append(np.hstack([np.nan]*len(uniq_clust_labels)))
            else:
                # histogram
                lab_select = cluster_labels[select].copy()
                hist_lab = np.hstack([np.sum(lab_select==lab) for lab in uniq_clust_labels]) 
                hist_lab = hist_lab/float(np.sum(hist_lab))
                
                proportion_time.append(hist_lab)

        proportion_time = np.array(proportion_time)
        proportion_time_condition.append(proportion_time) # this is where the order becomes imposed!
    proportion_time_condition = np.array(proportion_time_condition)

    # proportion_time_condition = proportion_time_condition.transpose() # 
    
    return uniq_clust_labels, proportion_time_condition


# HMM modeling 
def get_labels_for_trajectory( all_object_trajectories,
                               all_cluster_labels,
                               all_conditions=None, 
                               all_unique_conditions = None):
    
    r""" Use the object instances that form individual object trajectories to transform the phenotype clusters of individual object instances to label object instances in each object trajectory
    
    Parameters
    ----------
    all_object_trajectories : list of array
        list of all trajectories where each trajectory is provided as the index of the object instance in the trajectory in chronological order and can be used to index the corresponding cluster label and conditions
    all_cluster_labels : (N_objects,) array
        the phenotype cluster label of each object instance 
    all_conditions : (N_objects,) array
        the experimental condition label of each object instance 
    all_unique_conditions : (N_unique_conditions,) array 
        the unique experimental conditions
    
    Returns
    -------
    all_object_label_trajectories : list of array
        list of all trajectories where each trajectory is the associated phenotype label for each cell instance within
        
    """
    import numpy as np 
    

    if all_conditions is None:
        all_conditions = np.hstack(['na']*len(all_cluster_labels))
    if all_unique_conditions is None:
        all_unique_conditions = np.unique(all_conditions)
    
    # find the per trajectory condition. 
    all_object_trajectories_classes = np.hstack([all_conditions[tra_ind][0] for tra_ind in all_object_trajectories])
    
    all_object_label_trajectories = []
    
    for condition in all_unique_conditions:

        all_trajectories_select = [all_object_trajectories[ii] for ii in np.arange(len(all_object_trajectories)) if all_object_trajectories_classes[ii] == condition]

        # produce the label sequences. 
        all_label_seqs = []
        
        for select_id in np.arange(len(all_trajectories_select))[:]:
            
            test_traj_clust_labels = all_cluster_labels[all_trajectories_select[select_id]].copy()
            all_label_seqs.append(test_traj_clust_labels)
            
        all_object_label_trajectories.append(all_label_seqs)
    
    return all_object_label_trajectories


def compute_phenotype_cluster_transitions_from_label_trajectories(all_object_label_trajectories,
                                                                  all_cluster_labels,
                                                                  pseudocounts=1,
                                                                  time_interval = 1,
                                                                  diffs_only=True):
    r""" computer Pr(cluster_j at time t+1 | cluster_i at time t) based on frequency statistics, equivalent to beta-dirichlet stats
   
    Parameters
    ----------
    all_object_label_trajectories : list
        list of all trajectories, where the object instances in each trajectory are given by their index in the table
    all_cluster_labels : (N_objects,) array
        the phenotype cluster label of each object instance
    pseudocounts : float
        normalization count
       
    Returns
    -------
    all_models : list of [transition_matrix, transition_counts_matrix] per condition.
   
    """
    import numpy as np
   
    all_uniq_cluster_labels = np.unique(all_cluster_labels) # find the number of unique clusters used to setup the transition matrix.
    n_labels = len(all_uniq_cluster_labels)
   
    all_models = []
   
    for condition_ii in np.arange(len(all_object_label_trajectories)):
       
        all_label_seqs = list(all_object_label_trajectories[condition_ii]) # array of integer label
       
        transition_counts = np.ones((n_labels, n_labels)) * pseudocounts # add this count uniformly
       
        # iterate over the sequences
        for lab_seq_ii in np.arange(len(all_label_seqs)):
            label_seq = all_label_seqs[lab_seq_ii]
           
            if len(label_seq)>=1+time_interval:
                # label_seq_t = np.hstack(label_seq[1:]).copy()
                # label_seq_t_plus_1 = np.hstack(label_seq[:-1]).copy()
               
                label_seq_tt = np.vstack([label_seq[time_interval-ppp:len(label_seq)-ppp][None,...] for ppp in np.arange(time_interval+1)])
                # should be an array the same number.
               
                # we only tabulate the entries for which the second is different
                # for tttt in np.arange(len(label_seq_t)):
                for tttt in np.arange(label_seq_tt.shape[1]):
                    if diffs_only==True:
                        # if label_seq_t[tttt]!= label_seq_t_plus_1[tttt]:
                        if np.max(label_seq_tt[:,tttt]) - np.min(label_seq_tt[:,tttt]) > 0:
                            # transition_counts[label_seq_t[tttt],
                            #                   label_seq_t_plus_1[tttt]] += 1
                            for labbb in np.setdiff1d(label_seq_tt[:,tttt], label_seq_tt[0,tttt]):
                                transition_counts[label_seq_tt[0,tttt],
                                                  labbb] += 1
                    else:
                        # transition_counts[label_seq_t[tttt],
                                          # label_seq_t_plus_1[tttt]] += 1 # add counts to the table.
                   
                        for xxx in np.arange(time_interval):
                            transition_counts[label_seq_tt[0,tttt],
                                              label_seq_tt[xxx+1,tttt]] += 1 # add counts to the table.
                           

        transition_probs = transition_counts/(transition_counts.sum(axis=1)[:,None]) # divide by the sum of the columns to have row normalization
        all_models.append([transition_probs, transition_counts])
       
    return all_models
    

def fit_categoricalHMM_model_to_phenotype_cluster_label_trajectories( all_object_label_trajectories,
                                                                     all_cluster_labels,
                                                                     hmm_algorithm = 'map', 
                                                                     hmm_random_state = 0,
                                                                     hmm_implementation='scaling'):
    
    r""" We use the hmmlearn package to fit categorical HMM model for each object trajectory. 
    
    Parameters
    ----------
    all_object_label_trajectories : list
        list of all trajectories, where the object instances in each trajectory are given by their index in the table 
    all_cluster_labels : (N_objects,) array
        the phenotype cluster label of each object instance 
    hmm_algorithm : str
        the algorithm used to fit the hmmlearn hidden markov model parameters. Default is 'map' for maximum a priori method
    hmm_random_state : int 
        the random seed use to fit the hmmlearn hidden markov model 
    hmm_implementation : str
        Determines if the forward-backward algorithm in the hmmlearn model fitting is implemented with logarithms (“log”), or using scaling (“scaling”).
        
    Returns
    -------
    all_models : list of [transition_matrix, hmmlearn model object] per condition. 
    all_reorder : list of reordering used internally to bring HMM states into alignment with phenotype cluster labels
    
    """
    from hmmlearn.hmm import CategoricalHMM 
    from scipy.optimize import linear_sum_assignment
    import numpy as np 
    
    all_uniq_cluster_labels = np.unique(all_cluster_labels)
    
    all_models = []
    all_reorder = []
    
    for condition_ii in np.arange(len(all_object_label_trajectories)):
        
        all_label_seqs = list(all_object_label_trajectories[condition_ii])
        
        # hmmlearn uses 'pooled' data
        X = np.concatenate(all_label_seqs)
        lengths = [len(xx) for xx in all_label_seqs]
        
        hmm_model = CategoricalHMM(n_components=len(all_uniq_cluster_labels),
                                   algorithm=hmm_algorithm,
                                   random_state=hmm_random_state, 
                                   implementation=hmm_implementation).fit(X.reshape(-1,1),lengths)
    
        # now map hmm_model to the input cluster space.
        out = hmm_model.decode(X.reshape(-1,1))
        out_decode = out[1].copy()
        
        """
        Do bipartite matching to associate the latent hidden state to the phenotype cluster label. 
        """
        unique_X = all_uniq_cluster_labels.copy()
        
        cost = np.zeros((len(unique_X), len(unique_X)))
        
        for ii in np.arange(len(unique_X)):
            for jj in np.arange(len(unique_X)):
                
                select_ii = X==unique_X[ii]
                select_jj = out_decode == unique_X[jj] # was ii 
        
                total = np.sum(np.logical_and(select_ii, select_jj)) # this is the agreement.
                cost[ii,jj] = total
                
        cost = cost.max()-cost
        
        in_inds, out_inds = linear_sum_assignment(cost)
    
        """
        Get the fitted latent transition matrix 
        """        
        Q = hmm_model.transmat_.copy()
        
        """
        reorder. 
        """
        #### save out as csv. 
        # reorder = out_inds[np.argsort(in_inds)]
        reorder = out_inds.copy()
        Q_table = Q[reorder,:].copy()
        Q_table = Q_table[:,reorder].copy()
        # Q_table = Q_table
        Q_array = Q_table.copy()
        
        all_models.append([Q_array, hmm_model])
        all_reorder.append(reorder)
        
        
    return all_models, all_reorder


def construct_obj_traj_from_uniq_obj_ids(all_obj_uniq_row_ids, 
                                         separator='_', 
                                         wanted_entries=[0,1,2,3],
                                         d_Frame = 1):
    
    r""" based on the unique object identifiers in the merged SPOT table which encodes the unique id of object instance, video filename object is from, the object_id within the video, and the frame_number the instance occurs
    
    Parameters
    ----------
    all_obj_uniq_row_ids : list or array
        list of unique object ids which needs to link into object trajectories, the last entry after splitting must be the frame number
    separator : str
        this is the character separator used to separate the meta information in the unique object id.
    wanted_entries : str
        this is the index that corresponds to the entities uniq_filename, filename, organoid_no after splitting a unique object id with the separator. 
    d_Frame : int 
        this is the expected number of frames between timepoints in the trajectory if timepoints are 'consecutive'. Thus if the data was not subsampled in time, the d_Frame=1 frame
    
    Returns
    -------
    object_trajectories_dict : dict
        dictionary containing the following metadata for analysis 
        
        * 'all_obj_trajectories': list
                This is a list of all the reconstructed trajectories. Each trajectory is given as a list of the input indices that make it up. 
        * 'all_obj_trajectories_times': list
                This is a list of all the reconstructed trajectories. Each trajectory is a list of the frame number that the object instances occurred at 
        * 'all_obj_trajectories_uniq_ids': list
                This is a list of unique ids assigned to each trajectory.
        * 'all_obj_uniq_row_ids_next': array
                This is an array the same size of the number of input object ids, specifying the index of the object corresponding to the immediate next timepoint 
        * 'all_obj_uniq_ids_uniq': array
                This is the array of unique objects. Note: this is different to the input unique object instance ids such that unique objects are all the instances belonging to the same object across all timepoints.
            
    """
    import numpy as np 
    from tqdm import tqdm 
    # build given the organoid_ids, the temporal trajectory from the flattened data of each individual organoid. 
    # id separation given by '_' last one = TP!
    
    # parse the uniq ids to get the uniq_filename, filename, organoid_no, frame_no
    all_obj_uniq_ids = np.hstack(['_'.join(np.hstack(idd.split(separator))[wanted_entries]) for idd in all_obj_uniq_row_ids])
    
    # these are consistent tracklets .... 
    all_obj_trajectories = [] 
    all_obj_trajectories_times = [] 
    all_obj_trajectories_uniq_ids = []
    all_obj_uniq_row_ids_next = np.zeros(len(all_obj_uniq_row_ids), dtype=np.int32) # this stores what we want. 
    all_obj_uniq_ids_uniq = np.unique(all_obj_uniq_ids)
    
    for idd in tqdm(all_obj_uniq_ids_uniq[:]):
        
        select_index = np.arange(len(all_obj_uniq_ids))[all_obj_uniq_ids==idd]
        select_row_ids = all_obj_uniq_row_ids[select_index]
        select_frames = np.hstack([int(ff.split('_')[-1]) for ff in select_row_ids])
        select_frames_sort_order = np.argsort(select_frames)
        
        if len(select_frames) == 1:
            all_obj_uniq_row_ids_next[select_index[select_frames_sort_order]] = -1 # no ongoing link
            all_obj_trajectories.append(select_index[select_frames_sort_order])
            all_obj_trajectories_uniq_ids.append(idd+'_Track-%s' %(str(1).zfill(3)))
            all_obj_trajectories_times.append(select_frames)
        else:
            """
            if the selected number of frames is greater than 1 then we have to start testing if this organoid is multiple consecutive segments all not. 
            """
            select_frames = select_frames[select_frames_sort_order]
            select_index = select_index[select_frames_sort_order]
            
            # assess for continuity. 
            select_frame_diff = np.diff(select_frames)
            segments = [[select_index[0]]] #initiate with the first.
            segments_times = [[select_frames[0]]]
            for iii in np.arange(len(select_frame_diff)):
                if select_frame_diff[iii] == d_Frame:
                    segments[-1].append(select_index[iii+1])
                    segments_times[-1].append(select_frames[iii+1])
                else:
                    segments.append([select_index[iii+1]])
                    segments_times.append([select_frames[iii+1]]) # start anew. 
            
            # iterate over the build segment.
            for ss_ii, ss in enumerate(segments):
                # based on this build the next index. 
                if len(ss) > 1:
                    select_index_next = np.hstack([ss[1:], -1]) # whoops!. 
                    all_obj_trajectories.append(ss)
                    all_obj_uniq_row_ids_next[ss] = select_index_next
                    all_obj_trajectories_uniq_ids.append(idd+'_Track-%s' %(str(ss_ii+1).zfill(3)))
                    all_obj_trajectories_times.append(segments_times[ss_ii])
                else:
                    all_obj_uniq_row_ids_next[ss] = -1 # no ongoing link
                    all_obj_trajectories.append(ss)
                    all_obj_trajectories_uniq_ids.append(idd+'_Track-%s' %(str(ss_ii+1).zfill(3)))
                    all_obj_trajectories_times.append(segments_times[ss_ii])
    
    
    object_trajectories_dict = {'all_obj_trajectories' : all_obj_trajectories, 
                                'all_obj_trajectories_times' : all_obj_trajectories_times, 
                                'all_obj_trajectories_uniq_ids' : all_obj_trajectories_uniq_ids,
                                'all_obj_uniq_row_ids_next' : all_obj_uniq_row_ids_next,
                                'all_obj_uniq_ids' : all_obj_uniq_ids}
    
    return object_trajectories_dict


def entropy_transition_matrix(trans_mat, 
                              normalize_rows=True):
    r""" Optionally row normalize an input transition matrix so that the sum of rows = 1. Then for each row, compute the shannon entropy. 

    Parameters
    ----------
    trans_mat : 2D numpy array 
        Transition table giving the probability of transition from row i to column j. Can be un-normalized.  
    normalize_rows : bool
        If set, this automatically normalizes the row transition probabilities prior to computing entropy 

    Returns
    -------
    trans_mat_entropy : (n_states,) array
        entropy of transition for each state of row i

    """
    import scipy.stats as spstats
    import numpy as np 
    
    trans_mat_copy = trans_mat.copy()
    if normalize_rows:
        trans_mat_copy = trans_mat_copy/(np.nansum(trans_mat, axis=1)[...,None]+1e-20)
    trans_mat_entropy = np.hstack([spstats.entropy(tt, nan_policy='omit') for tt in trans_mat_copy])
    
    return trans_mat_entropy


### automatic drawing of the hmm graph with edges colored by probability. 
def draw_HMM_transition_graph(trans_table, ax, node_colors=None, 
                              edgescale=10, edgelabelpos=1., 
                              pow_scale=1.,
                              figsize=(10,10), 
                              savefile=None):
    r""" Draw a given Markov transition table as a graph diagram. Nodes are organized as a circle of fixed radius

    Parameters
    ----------
    trans_table : pandas.DataFrame or 2D numpy array 
        Markov transition table where each row sums to 1, giving the probability of transition from row i to column j. 
    ax : Matplotlib axes object
        this is the character separator used to separate the meta information in the unique object id.
    node_colors : list or array
        the desired color to color each graph node. If None, the default is the Spectral colormap
    edgescale : int
        controls the width of the arrows
    pow_scale : float
        raise the transition matrix entries by this exponent, to enhance the strongest transitions.
    edgelabelpos : float
        controls the distance away from the center that graph nodes are drawn at. 
    figsize : 2-tuple
        size of the matplotlib plot
    savefile : str
        if specified, the displayed image will be saved to this path. 

    Returns
    -------
    None

    """

    from hmmviz import TransGraph
    import numpy as np 
    import pylab as plt 
    import pandas as pd 
    import seaborn as sns 
   
    # if array first cast to table. 
    if isinstance(trans_table, pd.DataFrame):
        transition_table = trans_table.copy()
    else:
        transition_table = pd.DataFrame(np.array(trans_table),
                                        index=np.arange(len(trans_table))+1, 
                                        columns=np.arange(len(trans_table))+1)
        
    node_list = np.arange(len(trans_table)) + 1
    
    graph = TransGraph(transition_table)
    trans_table_prob = transition_table.values.copy()
    
    
    fig, ax = plt.subplots(figsize=figsize)

    nodelabels = {ii:ii for ii in node_list}
    
    if node_colors is None:
        node_colors = sns.color_palette('Spectral', len(node_list)) # create a default. 
    
    colors = {ii: node_colors[ii-1] for ii in node_list}
   
    # print(colors)
    # print(nodelabels)
    # nodelabels = {ii:ii for ii in uniq_clust_labels+1}
    # colors = {ii+1:clust_labels_colors[ii] for ii in uniq_clust_labels}
   
    # edgecolors = {('sunny','rainy'): 'orange',
    #            ('sunny','sunny'): 'red',
    #            ('rainy','sunny'): (1,1,0,0.5),
    #            ('rainy','rainy'): (1,1,0,0.1)} # this works?
    edgecolors = {} # this works?  
    for ii in np.arange(len(node_list)):
        for jj in np.arange(len(node_list)):
            edge = (ii+1,jj+1)
            color = np.hstack([colors[ii+1], trans_table_prob[ii,jj]**pow_scale])
            # print(edge, color)
            edgecolors[edge] = color
    # print(edgecolors)
   
    graph.draw(ax=ax,
               # r=2,
        nodelabels=nodelabels,
        nodecolors=colors,
        edgecolors=edgecolors,
        edgelabels=False,
        edgewidths = 20,
        edgescale=edgescale,
        edgelabelpos=edgelabelpos,
        nodefontsize=16)
   
    # plt.show()
    if savefile is not None:
        fig.savefig(savefile,
                    dpi=300, bbox_inches='tight')
       
    return [] 

