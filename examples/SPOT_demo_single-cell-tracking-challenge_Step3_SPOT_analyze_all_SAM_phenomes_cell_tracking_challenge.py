# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:00:02 2020

@author: felix
"""

if __name__=="__main__":
    
    import numpy as np 
    import pylab as plt 
    import scipy.io as spio
    import os 
    import skimage.io as skio 
    # import MOSES.Utility_Functions.file_io as fio
    import glob
    import skimage.morphology as skmorph
    from scipy.ndimage.morphology import binary_fill_holes
#    from detect_organoid import clean_and_seg_organoids_Xiao_VC
    from skimage.restoration import denoise_nl_means
    from skimage.filters.rank import median, entropy
    import seaborn as sns
    from skimage.filters import gaussian
    from skimage.exposure import rescale_intensity
    from skimage.measure import find_contours
    from tqdm import tqdm
    
    from scipy.ndimage.measurements import center_of_mass
    import skimage.measure as skmeasure
    from skimage.util import img_as_ubyte
    from skimage.filters import threshold_otsu
    import skimage.transform as sktform 
    
    import seaborn as sns 
    import glob 
    
    from hmmviz import TransGraph

    import numpy as np 
    import glob 
    import os 
    import pylab as plt 
    from tqdm import tqdm 
    import skimage.exposure as skexposure 
    import scipy.io as spio 
    import pandas as pd 
    
    import skimage.filters as skfilters
    import skimage.transform as sktform
        
    import umap 
    import seaborn as sns
    
    """
    Define the imports here. 
    """
    import SPOT.Utility_Functions.file_io as fio 
    import SPOT.Analysis.preprocessing as preprocess
    import SPOT.Analysis.sam_analysis as SAM
    import SPOT.Utility_Functions.plotting as plotting
    
    
    rootfolder = r'./data/cell_tracking'
    saverootfolder = rootfolder
    
    saveplotsfolder = r'../test_cell-tracking-challenge_results'
    # saveplotsfolder = r'C:\Users\s205272\Documents\Work\Research\Lu_Lab\SAM_paper_2023\Other_Manuscripts\Cell_Tracking_Challenge_Results_PCA-init'
    fio.mkdir(saveplotsfolder)
    
    
    # cellfolders = [r'PhC-C2DH-U373\01', #, # done  ---- done 
    #                 r'PhC-C2DH-U373\02'] #done 

    # saveplotsfolder_cells = os.path.join(saveplotsfolder, 
    #                                       '2023-10-04_U373_modules_github_test')
    # mkdir(saveplotsfolder_cells)
    
    cellfolders = [r'PhC-C2DH-U373/01', 
                   r'PhC-C2DH-U373/02']
    # cellfolders = [r'Fluo-N2DH-SIM+\01', #, # done  ---- done 
                    # r'Fluo-N2DH-SIM+\02'] #done 

    saveplotsfolder_cells = os.path.join(saveplotsfolder, 
                                           'PhC-C2DH-U373_analysis')
                                           # '2023-05-23_N2DH-SIM+_modules_github_test')
    fio.mkdir(saveplotsfolder_cells)
    
    
# =============================================================================
#     Do the rest of the analysis now. 
# =============================================================================
    
    analysisfolders = []
    expts = []
    
    for cellfolder in tqdm(cellfolders[:]):
        infolder = os.path.join(rootfolder,cellfolder)
        rootname, basename = os.path.split(cellfolder)
        saveresultsfolder = os.path.join(saverootfolder, rootname, basename); # mkdir(saveresultsfolder)
        
        expt = rootname+'_'+basename
        
        expts.append(expt)
        analysisfolders.append(saveresultsfolder)
        
    expts = ['','']
    """
    load
    """
    all_feats, metadict = fio.load_SPOT_features_files(analysisfolders, 
                                                        expts,
                                                        boundaryfile_suffix='final_boundaries.mat',
                                                        boundarykey='boundary',
                                                        patchfile_suffix='final_img_patches_noresize_filter.mat',
                                                        shapefeatsfile_suffix='final_shape_metrics_filter.csv',
                                                        appearfeatsfile_suffix='final_appearance_metrics_filter.csv', 
                                                        motionfeatsfile_suffix='final_motion_metrics_filter.csv',
                                                        read_chunksize=2000)
                                   

    """
    Do the full combine across the 2 datasets.
    """
    all_uniq_org_ids = metadict['all_object_uniq_row_ids']
    all_dataset_boundaries = metadict['all_object_boundaries']
    
    all_patches = list(metadict['all_object_patches'])    
    all_patches = np.array(all_patches, dtype=object)[None,...] # to make the same. 
    std_size=(64,64)
    all_patches_std_size = np.array([sktform.resize(pp, output_shape=std_size, order=1, preserve_range=True) for pp in all_patches[0]])
    all_patches_sizes = metadict['all_object_patches_sizes']
    
    # all_expts = metadict['']

# =============================================================================
# =============================================================================
# #     Remove features that shouldn't have been computed 
# =============================================================================
# =============================================================================
    all_feats = all_feats.astype(np.float64)
    feature_names = metadict['feature_names']                  
    
    """
    parse out the division 
    """
    all_division = np.squeeze(all_feats[:,feature_names=='div_bool']).copy() # this needs to  be taken out of all_feats. 

    
    # number 2 is identification of velocity vector columns. 
    mean_global_velocity_xy_cols = []
    for feat_ii, feature_name in enumerate(feature_names):
        if 'mean_disp_global_' in feature_name and 'flow' not in feature_name:
            mean_global_velocity_xy_cols.append(feat_ii)
    mean_global_velocity_xy_cols = np.hstack(mean_global_velocity_xy_cols)
            
    mean_global_flow_velocity_xy_cols = []
    for feat_ii, feature_name in enumerate(feature_names):
        if 'mean_disp_global_flow_' in feature_name:
            mean_global_flow_velocity_xy_cols.append(feat_ii)
    mean_global_flow_velocity_xy_cols = np.hstack(mean_global_flow_velocity_xy_cols)

    div_bool_cols = []
    for feat_ii, feature_name in enumerate(feature_names):
        if 'div_bool' in feature_name:
            div_bool_cols.append(feat_ii)
    div_bool_cols = np.hstack(div_bool_cols)

    select_feats_index = np.setdiff1d(np.arange(len(feature_names)), 
                                  np.hstack([mean_global_velocity_xy_cols,
                                              mean_global_flow_velocity_xy_cols,
                                              div_bool_cols]))
    
    
    
    
    # remove the vectorial information as well as cell division only information.  
    all_feats = all_feats[:,select_feats_index].copy()
    feature_names = feature_names[select_feats_index].copy()


    # =============================================================================
    #   remove threshold adjacency stats.   
    #===========================================================================        
    # area_feat_ind = np.arange(len(feature_names))[np.hstack(['area' in feat_name for feat_name in feature_names])][0]
    # drop the threshold adjacency -> this seems to be the most culprit for a bipartition? 
    thresh_adj_ind = np.arange(len(feature_names))[np.hstack(['thresh_adj' in feat_name for feat_name in feature_names])]
    feature_inds = np.setdiff1d(np.arange(len(feature_names)), thresh_adj_ind) # this step seems a must must. 
    # feature_inds = np.arange(len(feature_names))
    feature_names = feature_names[feature_inds]
    all_feats = all_feats[:,feature_inds]

# =============================================================================
# =============================================================================
# #     Now we can preprocess all the features. 
# =============================================================================
# =============================================================================

    # scale normalize curvature features. 
    all_feats, feature_names = preprocess.scale_normalize_curvature_features(all_feats, 
                                                                              feature_names)
    
    # remove all zero features
    all_feats, feature_names = preprocess.remove_zero_features(all_feats, feature_names)    

    
    # remove high var features
    all_feats, feature_names = preprocess.remove_high_variance_features(all_feats, 
                                                                        feature_names, 
                                                                        variance_threshold_sigma=2)

    # transform ECC features (this is different)
    all_feats, feature_names, ECT_rbf_feature_tformer = preprocess.kernel_dim_reduction_ECC_features(all_feats.astype(np.float64), 
                                                                                                      feature_names, 
                                                                                                      n_dim=100, 
                                                                                                      gamma=None, # 1. was closer?  
                                                                                                      random_state=1) # not the same? 
    # all_feats = np.int32(all_feats*10e6) / 10e6
    # all_feats = spio.loadmat('saved_out_ECCproc_feats (1)')['all_feats']
    
# =============================================================================
#       Normalize.
# =============================================================================
    from sklearn.preprocessing import StandardScaler, power_transform
        
    sd_tform = StandardScaler()
    all_feats = power_transform(sd_tform.fit_transform(all_feats))

    # temporal variation selection 
    all_feats, feature_names, coeffs = preprocess.select_time_varying_features(all_feats.astype(np.float64), 
                                                                                feature_names, 
                                                                                metadict['all_object_TP'], 
                                                                                ridge_alpha=1.)

    print(all_feats.shape)
    
    
    # =============================================================================
    # =============================================================================
    # =============================================================================
    # # #     SAM module analysis ( double check the quantities all check out. )
    # =============================================================================
    # =============================================================================
    # =============================================================================

    # measure the contribution 
    (feature_type, feature_scope), (SAM_expr, Scope_expr), (SAM_contribution, scope_contribution) = SAM.compute_SAM_features_and_scope_contribution(all_feats, 
                                                                                                                                                    feature_names,
                                                                                                                                                    random_state=0)
    print(SAM_contribution)
    print(scope_contribution)
    

    # =============================================================================
    #   Use hcluster to discover modules.          
    # =============================================================================
        
    clustermap, SAM_modules, SAM_modules_ordered_colorbar, SAM_modules_featurenames, SAM_modules_feature_indices = SAM.hierarchical_cluster_features_into_SAM_modules(all_feats, 
                                                                                                                       feature_names, 
                                                                                                                       feature_type=feature_type,
                                                                                                                       feature_scope=feature_scope,
                                                                                                                       hcluster_heatmap_color='vlag',
                                                                                                                       hcluster_method='average', 
                                                                                                                       hcluster_metric='euclidean',
                                                                                                                       lam_smooth=1e1, 
                                                                                                                       p_smooth=0.5, 
                                                                                                                       niter_smooth=10,
                                                                                                                       min_peak_distance=5,
                                                                                                                       debugviz=False,
                                                                                                                       savefile=None) # if savefile is set, and debugviz = True, this will save the intermediate plots to the specified folder path.
    
    # =============================================================================
    #     Obtain the characteristic signature for each module 
    # =============================================================================
    
    feature_module_expr, feature_module_expr_contrib = SAM.compute_individual_feature_contributions_in_SAM_modules(all_feats, 
                                                                                                                    SAM_modules_feature_indices,
                                                                                                                    random_state=0)
        
    # =============================================================================
    #     grab the top patches that are enriched in each modules. 
    # =============================================================================
    # create a regular. 
    # all_object_patches = metadict['']
    
    
    object_SAM_module_purity_scores, sample_images_modules, sample_object_index_modules = SAM.compute_most_representative_image_patches( feature_module_expr,
                                                                                                                                        all_patches[0], 
                                                                                                                                        n_rows = 3, 
                                                                                                                                        n_cols = 3,
                                                                                                                                        rescale_intensity=False)
    
    # =============================================================================
    #      Now we do the dynamic phenotyping analysis
    # ============================================================================= 
    umap_fit = umap.UMAP(n_neighbors=100, 
                          random_state=0,
                          metric='euclidean') # this seems to look better?
    uu = umap_fit.fit_transform(all_feats.astype(np.float64)) 
    
    
    """
    plot the cropped patches onto the umap coordinates.
    """
    fig, ax = plt.subplots(figsize=(15,15))
    
    plotting.plot_image_patches(positions=uu,
                                patches=metadict['all_object_patches'],
                                ax=ax,
                                subsample=1,
                                zoom=0.35)
    plt.show()
        
    
    
    """
    plot the density map 
    """
    
    all_pts_density, all_pts_density_select = SAM.compute_heatmap_density_image_Embedding(uu, 
                                                                                            all_conditions=None, 
                                                                                            unique_conditions=None,
                                                                                            cmap='coolwarm',
                                                                                            grid_scale_factor=500, 
                                                                                            sigma_factor=0.25, 
                                                                                            contour_sigma_levels=[1,2,3,3.5,4, 4.5, 5],
                                                                                            saveplotsfolder=None)
                  
    """
    select certain statistics and plot -- as they may have disappeared, these need to be based on the prefiltered but still normalized
    """
    area_ind = np.arange(len(feature_names))[np.hstack(['equivalent_diameter' in name for name in feature_names])]
    ecc_ind = np.arange(len(feature_names))[np.hstack(['major_minor_axis_ratio' in name for name in feature_names])]
    flow_ind = np.arange(len(feature_names))[np.hstack(['mean_speed_global' in name for name in feature_names])]
    intensity_ind = np.arange(len(feature_names))[np.hstack(['mean_intensity' in name for name in feature_names])]

    if len(area_ind) > 0 :
        fig, ax = plt.subplots(figsize=(15,15))
        plt.title('Area '+feature_names[area_ind[0]])
        ax.scatter(uu[:,0], 
                    uu[:,1], c=all_feats[:,area_ind[0]], cmap='coolwarm', vmin=-0.5, vmax=0.5)
    #    ax.set_aspect(1)
    #    ax.set_xlim([-10,10])
        plt.axis('off')
        plt.grid('off')
        plt.savefig(os.path.join(saveplotsfolder_cells, 
                                  'umap_area.svg'), dpi=300, bbox_inches='tight')
        plt.show()
    
    if len(ecc_ind)>0:
        fig, ax = plt.subplots(figsize=(15,15))
        plt.title('Eccentricity '+feature_names[ecc_ind[0]])
        ax.scatter(uu[:,0], 
                    uu[:,1], c=all_feats[:,ecc_ind[0]], cmap='coolwarm', vmin=-0.5, vmax=0.5)
    #    ax.set_aspect(1)
    #    ax.set_xlim([-10,10])
        plt.axis('off')
        plt.grid('off')
        plt.savefig(os.path.join(saveplotsfolder_cells, 
                              'umap_eccentricity.svg'), dpi=300, bbox_inches='tight')
        plt.show()
    
    if len(flow_ind)>0:
        fig, ax = plt.subplots(figsize=(15,15))
        plt.title('Speed '+feature_names[flow_ind[0]])
        ax.scatter(uu[:,0], 
                    uu[:,1], c=all_feats[:,flow_ind[0]], cmap='coolwarm', vmin=-0.5, vmax=0.5)
    #    ax.set_aspect(1)
    #    ax.set_xlim([-10,10])
        plt.axis('off')
        plt.grid('off')
        plt.savefig(os.path.join(saveplotsfolder_cells, 
                          'umap_mean_speed_flow.svg'), dpi=300, bbox_inches='tight')
        plt.show()
    
    if len(intensity_ind)>0:
        fig, ax = plt.subplots(figsize=(15,15))
        plt.title('Intensity '+feature_names[intensity_ind[0]])
        ax.scatter(uu[:,0], 
                    uu[:,1], c=all_feats[:,intensity_ind[0]], cmap='coolwarm', vmin=-0.5, vmax=0.5)
    #    ax.set_aspect(1)
    #    ax.set_xlim([-10,10])
        plt.axis('off')
        plt.grid('off')
        plt.savefig(os.path.join(saveplotsfolder_cells, 
                          'umap_mean_intensity.svg'), dpi=300, bbox_inches='tight')    
        plt.show()
            
# =============================================================================
#     Plot the sam module expr onto the umap. 
# =============================================================================

    for mod_ii in np.arange(feature_module_expr.shape[-1]):    
        fig, ax = plt.subplots(figsize=(15,15))
        plt.title('Module '+str(mod_ii))
        ax.scatter(uu[:,0], 
                    uu[:,1], c=feature_module_expr[:,mod_ii], cmap='coolwarm', vmin=-6, vmax=6)
    #    ax.set_aspect(1)
    #    ax.set_xlim([-10,10])
        plt.axis('off')
        plt.grid('off')
        plt.savefig(os.path.join(saveplotsfolder_cells, 
                          'umap_module-%s.svg' %(str(mod_ii).zfill(3))), dpi=300, bbox_inches='tight')    
        plt.show()
    
    
# =============================================================================
#     Plot the division!
# =============================================================================

    # all_division
    fig, ax = plt.subplots(figsize=(15,15))
    plt.title('Division times')
    ax.scatter(uu[:,0], 
                uu[:,1], c=all_division, cmap='coolwarm', vmin=-1, vmax=1)
    ax.scatter(uu[all_division==1,0], 
                uu[all_division==1,1], c='r', s=100,zorder=1000)
#    ax.set_aspect(1)
#    ax.set_xlim([-10,10])
    plt.axis('off')
    plt.grid('off')
    plt.savefig(os.path.join(saveplotsfolder_cells, 
                      'umap_divisions.svg' ), dpi=300, bbox_inches='tight')    
    plt.show()
    
    
    
# =============================================================================
#     Do phenotype clustering prototype analysis. 
# =============================================================================
    
    clust_labels, clusterer, best_k, scan_results = SAM.KMeans_cluster_low_dimensional_embedding_with_scan(uu,
                                                                                                             k_range=(2,20),
                                                                                                             ploton=True)
                                                    
    
  
    # clust_labels = clusterer.predict(uu); # does this make more sense with the actual features? 
    clust_labels = clusterer.predict(uu);
    uniq_clust_labels = np.unique(clust_labels)
    clust_labels_colors = sns.color_palette('Set1', len(uniq_clust_labels))

    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(uu[:,0], 
                uu[:,1], c='lightgrey')
    for lab_ii, lab in enumerate(uniq_clust_labels):
        ax.scatter(uu[clust_labels==lab,0], 
                    uu[clust_labels==lab,1], c=clust_labels_colors[lab_ii], alpha=1)
        
        # get the mean coordinate for each coordinate and label .
        mean_clust_uu = np.mean(uu[clust_labels==lab],axis=0)
        ax.text(mean_clust_uu[0],
                mean_clust_uu[1], 
                str(lab_ii+1),
                va='center',
                ha='center',
                fontsize=24,
                fontname='Arial', zorder=1000)
        
    plt.axis('off')
    plt.grid('off')
    plt.savefig(os.path.join(saveplotsfolder_cells, 
                              'SAM_umap_kmeans_clusters.pdf'), dpi=300, bbox_inches='tight')
    
    # plt.savefig('Comb_umap_2021-07-06_Brittany_batch_correct_powt_kernelECT_8clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    """
    PC-based representative image patches 
    """
    representative_image_dict = SAM.find_representative_images_per_cluster_PCA(all_feats,
                                                                                uu,
                                                                                all_patches_std_size, 
                                                                               all_patches_sizes,
                                                                               clust_labels, 
                                                                               unique_cluster_labels=np.unique(clust_labels), 
                                                                               rev_sign=False,
                                                                               mosaic_size=0.95,
                                                                               percentile_cutoff = 1., # i.e. 100% 
                                                                               n_rows_mosaic = 2, 
                                                                               n_cols_mosaic = 2,
                                                                               pca_random_state=0,
                                                                               debugviz=False)
        
    representative_images_std_size = representative_image_dict['representative_images_std_size']
    representative_images_real_size = representative_image_dict['representative_images_real_size']
    
    
    # vis
    
    for cluster_ii in np.arange(len(representative_images_real_size)):
        
        plt.figure(figsize=(10,10))
        plt.title(cluster_ii)
        plt.imshow(representative_images_real_size[cluster_ii], cmap='gray')
        plt.show()
    
    
    
    """
    SAM module expression per cluster
    """
    unique_clust_labels, cluster_mean_scores, cluster_std_scores = SAM.compute_mean_score_features_cluster(clust_labels, 
                                                                                                           all_feats=feature_module_expr, 
                                                                                                           avg_func=np.nanmean)
    
    
    # plot the expression per cluster. Since the number of phenotype clusters is fairly low, we can use the joint canvas option.
    plotting.barplot_cluster_statistics(cluster_mean_scores,
                                        featnames=['Module %s' %(str(jj).zfill(3)) for jj in np.arange(cluster_mean_scores.shape[1])],
                                        colormap=plt.cm.gray,
                                        style='hbar',
                                        shared_canvas = True,
                                        # figsize=(1,4),
                                        figsize=(10,4),
                                        vmin=-6, 
                                        vmax=6,
                                        nticks=5,
                                        save_dpi=300, 
                                        saveplotsfolder=saveplotsfolder_cells)
    
    
    """
    Temporal stacked barplots of cluster 
    """
    uniq_cluster_conditions, proportion_time_condition = SAM.compute_temporal_cluster_proportions(clust_labels, 
                                                                                                    objects_time=metadict['all_object_TP'], 
                                                                                                    time_intervals=np.linspace(0, np.max(metadict['all_object_TP']), 16+1),
                                                                                                    all_conditions=None, 
                                                                                                    unique_conditions=None)

    # plot the stacked barplots as in the paper. 
    clust_labels_colors = sns.color_palette('Set1', len(np.unique(clust_labels)))
    plotting.stacked_barplot_temporal_cluster_proportions(proportion_time_condition, 
                                                         unique_conditions=['U373'], 
                                                         time_intervals = np.linspace(0, np.max(metadict['all_object_TP']), 16+1), 
                                                         clust_labels_colors=clust_labels_colors,
                                                         figsize=(7,5),
                                                         saveplotsfolder=saveplotsfolder_cells)
    
    
    """
    Phenotype trajectory
    """
    all_phenotype_trajectories, all_density_contours = SAM.compute_phenotypic_trajectory(uu, 
                                                                                            objects_time=metadict['all_object_TP'], 
                                                                                            time_intervals=np.linspace(0, np.max(metadict['all_object_TP']), 16+1),
                                                                                            all_conditions=None, 
                                                                                            unique_conditions=None,
                                                                                            cmap='coolwarm',
                                                                                            grid_scale_factor=500, 
                                                                                            sigma_factor=0.25, 
                                                                                            thresh_density_sigma = 3,
                                                                                            debugviz=False)
    
    # plot the trajectory atop the points. 
    plt.figure(figsize=(10,10))
    plt.scatter(uu[:,0], 
                uu[:,1], s=20, c='lightgrey')
    plt.plot(all_phenotype_trajectories[0][:,0], 
             all_phenotype_trajectories[0][:,1], 
             'ko-', lw=3, ms=5)
    plt.plot(all_phenotype_trajectories[0][0,0], 
             all_phenotype_trajectories[0][0,1], 
             'ro', ms=15, mec='k', mew=3)
    plt.grid('off')
    plt.axis('off')
    plt.savefig(os.path.join(saveplotsfolder_cells, 
                      'umap_phenotype_trajectory.svg' ), dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
    # =============================================================================
    # =============================================================================
    # #     Test we can do the HMM analysis and single cell trajectory transition analysis
    # =============================================================================
    # =============================================================================
      
    # build the object trajectories.
    object_trajectories_dict = SAM.construct_obj_traj_from_uniq_obj_ids(metadict['all_object_uniq_row_ids'], 
                                                                          separator='_', 
                                                                          wanted_entries=[1,2,3,4], # we need uniq_filename, filename, organoid_no, frame_no
                                                                          d_Frame = 1)
    
    all_obj_trajectories = object_trajectories_dict['all_obj_trajectories']
    
    # check this is correct. 
    # use the phenotype clusters to label the trajectory
    all_object_label_trajectories = SAM.get_labels_for_trajectory( all_obj_trajectories,
                                                                   clust_labels,
                                                                   all_conditions=None, 
                                                                   all_unique_conditions = None)
                  
    """
    HMM learning on the labeled trajectories,
    
    permute is the reordering of the hidden states to correspond with phenotype clusters. If using the methods associated with the fitted HMM model, then permute needs to be reapplied. 
    """
    # only one condition. 
    all_HMM_models, all_HMM_models_transmat_permute = SAM.fit_categoricalHMM_model_to_phenotype_cluster_label_trajectories( all_object_label_trajectories,
                                                                                          clust_labels,
                                                                                          hmm_algorithm = 'map', 
                                                                                          hmm_random_state = 0,
                                                                                          hmm_implementation='scaling')
       
    transition_matrix = all_HMM_models[0][0]
    hmm_model = all_HMM_models[0][1]
    
    
    """
    visualize the transition matrix
    """
    SAM.draw_HMM_transition_graph(transition_matrix, 
                                  ax=ax, 
                                  node_colors=clust_labels_colors, 
                                  edgescale=10, 
                                  edgelabelpos=.5, 
                                  figsize=(15,15),
                                  savefile=os.path.join(saveplotsfolder_cells, 'HMM_transition_graph.svg')) # savefile should be specified in .svg format to get the same looking arrows. For some reason .pdf doesn't give this. 
    plt.show()
    

