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
