# -*- coding: utf-8 -*-
""" representative image mosaics (SPOT.Analysis.sam_analysis.find_representative_images_per_cluster_PCA): real-size
patches bigger than the mosaic cell (max patch size * mosaic_size) enlarge the cell instead of crashing, and the cell
is unchanged when every selected patch fits """
import numpy as np

from SPOT.Analysis import sam_analysis as sam


def run(sizes_per_cluster, mosaic_size=0.7):
    rng = np.random.default_rng(0)
    sizes = np.vstack(sizes_per_cluster)
    labels = np.repeat(np.arange(len(sizes_per_cluster)), [len(s) for s in sizes_per_cluster])
    patches = rng.integers(0, 255, size=(len(sizes), 32, 32)).astype(np.uint8)
    feats = rng.standard_normal((len(sizes), 5))
    coords = rng.standard_normal((len(sizes), 2))
    # 4 objects per cluster and percentile_cutoff=1 -> every object is placed in the 2x2 mosaic
    return sam.find_representative_images_per_cluster_PCA(feats, coords, patches, sizes, labels, np.unique(labels),
                                                          mosaic_size=mosaic_size, percentile_cutoff=1.0,
                                                          n_rows_mosaic=2, n_cols_mosaic=2)


def test_patch_larger_than_cell_enlarges_cell():
    # max sizes (80, 100) * 0.7 -> cell (56, 70); the (80, 30), (30, 100) and (60, 70) patches do not fit it
    rep = run([np.array([[80, 30], [30, 100], [60, 70], [20, 20]]),
               np.array([[20, 20]] * 4)])
    assert rep['representative_images_real_size'].shape == (2, 2 * 80, 2 * 100)
    assert rep['representative_images_std_size'].shape == (2, 2 * 32, 2 * 32)


def test_cell_unchanged_when_patches_fit():
    rep = run([np.array([[80, 30], [30, 100], [50, 60], [20, 20]]),
               np.array([[20, 20]] * 4)], mosaic_size=1.0)
    assert rep['representative_images_real_size'].shape == (2, 2 * 80, 2 * 100)
    rep = run([np.array([[50, 60], [20, 20], [40, 40], [20, 20]]),
               np.array([[20, 20]] * 4)], mosaic_size=1.2)
    assert rep['representative_images_real_size'].shape == (2, 2 * 60, 2 * 72)


def test_small_clusters_leave_cells_empty():
    # clusters of 1 and 3 objects in a 2x2 mosaic: no PCA on a single object, the missing cells stay empty
    rep = run([np.array([[30, 30]]), np.array([[20, 20], [25, 30], [30, 25]])])
    assert rep['representative_images_real_size'].shape[0] == 2
    assert list(rep['representative_images_indices'][0]) == [0, -1, -1, -1]
    assert np.sum(rep['representative_images_indices'][1] >= 0) == 3
    assert np.isnan(rep['representative_images_embedding_coords'][0, 1:]).all()


def test_colour_patches():
    rng = np.random.default_rng(1)
    sizes = np.array([[30, 40], [20, 20], [40, 30], [25, 25]])
    patches = rng.integers(0, 255, size=(4, 32, 32, 3)).astype(np.uint8)
    rep = sam.find_representative_images_per_cluster_PCA(rng.standard_normal((4, 5)), rng.standard_normal((4, 2)),
                                                         patches, sizes, np.zeros(4, dtype=int), np.array([0]),
                                                         percentile_cutoff=1.0)
    assert rep['representative_images_std_size'].shape == (1, 64, 64, 3)
    assert rep['representative_images_real_size'].shape[-1] == 3


def module_montage(patches, n_modules=3):
    rng = np.random.default_rng(2)
    expr = rng.standard_normal((len(patches), n_modules))
    return sam.compute_most_representative_image_patches(expr, patches, n_rows=3, n_cols=3)


def test_module_montage_fewer_objects_than_cells():
    rng = np.random.default_rng(3)
    patches = [rng.integers(0, 255, size=(rng.integers(10, 40), rng.integers(10, 40))).astype(np.uint8) for _ in range(5)]
    scores, panels, index = module_montage(patches)
    assert scores.shape == (5, 3)
    assert all(len(ii) == 5 for ii in index)
    m = max(p.shape[0] for p in patches)
    n = max(p.shape[1] for p in patches)
    assert all(p.shape == (3 * m, 3 * n) for p in panels)


def test_module_montage_colour_and_single_module():
    rng = np.random.default_rng(4)
    patches = [rng.integers(0, 255, size=(20 + ii, 25, 3)).astype(np.uint8) for ii in range(12)]
    scores, panels, index = module_montage(patches, n_modules=1)
    assert scores.shape == (12, 1)
    assert panels[0].shape == (3 * 31, 3 * 25, 3)
