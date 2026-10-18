# -*- coding: utf-8 -*-
"""SPOT.Analysis.preprocessing.sam_feature_groups / weight_sam_features"""

import itertools

import numpy as np
import pytest

import fixtures
import SPOT.Features.features as sf
import SPOT.Analysis.preprocessing as pp
from conftest import run_quiet

EXPECTED_GROUPS = {
    'shape': ['curvature', 'contour geometry', 'binary morphology', 'Fourier', 'elliptic Fourier', 'Euler characteristic transform', 'shape context'],
    'appearance': ['global intensity', 'regional intensity', 'Haralick texture', 'SIFT'],
    'motion': ['global motion', 'regional speed', 'regional speed distribution', 'speed SIFT', 'curl SIFT', 'divergence SIFT', 'shear SIFT'],
}


@pytest.fixture(scope='module')
def sam_names():
    d = fixtures.load('synthetic')
    names = {}
    _, names['shape'], _ = run_quiet(sf.compute_boundary_morphology_features_fast, boundaries=d['boundaries'], imshape=d['imshape'])
    _, names['appearance'], _ = run_quiet(sf.compute_boundary_texture_features_fast, vid=d['vid'], boundaries=d['boundaries'])
    _, names['motion'], _ = run_quiet(sf.compute_boundary_motion_features_fast, vid_flow=d['flow'], boundaries=d['boundaries'])
    return {k: np.asarray(v).astype(str) for k, v in names.items()}


def test_every_sam_feature_is_grouped(sam_names):
    for s, names in sam_names.items():
        sets, groups = pp.sam_feature_groups(names)
        assert set(sets) == {s}, (s, set(sets))
        assert sorted(set(groups)) == sorted(EXPECTED_GROUPS[s]), (s, sorted(set(groups)))
    # the kernel reduced ECT features of kernel_dim_reduction_ECC_features
    assert list(pp.sam_feature_groups(['ECT_rbf_1', 'ECT_rbf_100'])[1]) == ['Euler characteristic transform'] * 2
    assert list(pp.sam_feature_groups(['not_a_sam_feature'])[0]) == ['other']


@pytest.mark.parametrize('group_weighting,set_weighting', list(itertools.product([True, False], repeat=2)))
def test_weight_sam_features(sam_names, group_weighting, set_weighting):
    names = np.hstack([sam_names['shape'], sam_names['appearance'], sam_names['motion'], ['not_a_sam_feature']])
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, len(names))) * rng.uniform(0.5, 3, len(names)) + rng.uniform(-5, 5, len(names))
    W, (mean, weight) = pp.weight_sam_features(X, names, group_weighting=group_weighting, set_weighting=set_weighting,
                                               return_weights=True)
    np.testing.assert_allclose(W, (X - mean) * weight)
    np.testing.assert_allclose(W.mean(0), 0, atol=1e-12)
    sets, groups = pp.sam_feature_groups(names)
    var, var_in = W.var(0), X.var(0)
    for s in dict.fromkeys(sets):
        in_set = sets == s
        total = 1. if set_weighting else var_in[in_set].sum()
        np.testing.assert_allclose(var[in_set].sum(), total)
        if group_weighting:
            g_totals = [var[in_set & (groups == g)].sum() for g in dict.fromkeys(groups[in_set])]
            np.testing.assert_allclose(g_totals, total / len(g_totals))
        else:
            # the set is scaled as a whole: relative feature variances are kept
            ratio = var[in_set] / var_in[in_set]
            np.testing.assert_allclose(ratio, ratio[0])


def test_sets_agree_with_get_SAM_feature_type(sam_names):
    from SPOT.Analysis.sam_analysis import get_SAM_feature_type
    names = np.hstack([sam_names['shape'], sam_names['appearance'], sam_names['motion']])
    sets, _ = pp.sam_feature_groups(names)
    sam_type, _ = get_SAM_feature_type(names)
    assert list(sets) == list(np.array(['shape', 'appearance', 'motion'])[sam_type])



def test_shear_is_motion():
    """ the flow shear features of pip SAM-SPOT 0.1.1 are motion (they were classed as shape) """
    from SPOT.Analysis.sam_analysis import get_SAM_feature_type
    names = np.array(['shear_sift_0', 'shear_sift_127', 'mean_shear_global_flow', 'stddev_shear_global_flow'])
    sam_type, _ = get_SAM_feature_type(names)
    assert list(sam_type) == [2, 2, 2, 2]
    sets, groups = pp.sam_feature_groups(names)
    assert list(sets) == ['motion'] * 4
    assert list(groups[:2]) == ['shear SIFT'] * 2

def test_legacy_export_names_with_feature_sets():
    """ older exports (data/cell_tracking) name appearance and motion SIFT 'sift_<n>' and have legacy features """
    import os
    import pandas as pd
    meta = {'Filename', 'Img_Channel_No', 'Org_ID', 'Frame_No', 'pixel_resolution[um]', 'Frame_Duration[h]',
            'Total_Video_Frame_No', 'div_bool'}
    base = os.path.join(fixtures.REPO, 'data', 'cell_tracking', 'PhC-C2DH-U373', '01')
    names, given = [], []
    for s in ('shape', 'appearance', 'motion'):
        cols = [c for c in pd.read_csv(os.path.join(base, 'final_%s_metrics_filter.csv' % s), nrows=1).columns if c not in meta]
        names += cols; given += [s] * len(cols)
    sets, groups = pp.sam_feature_groups(names, feature_sets=given)
    assert list(sets) == given
    groups_of = lambda s: set(groups[sets == s])
    assert groups_of('shape') == set(EXPECTED_GROUPS['shape']) - {'elliptic Fourier'}  # exported before the EFD features
    assert groups_of('appearance') == set(EXPECTED_GROUPS['appearance']) | {'thresh_adj_stat'}  # legacy feature group
    assert groups_of('motion') == set(EXPECTED_GROUPS['motion']) - {'shear SIFT'}  # exported before the shear features
    W = pp.weight_sam_features(np.random.default_rng(0).standard_normal((50, len(names))), names, feature_sets=given)
    for s in ('shape', 'appearance', 'motion'):
        np.testing.assert_allclose(W[:, sets == s].var(0).sum(), 1.)


def test_merged_table_suffixes():
    """ file_io merges the appearance / motion tables with suffixes for duplicate (legacy 'sift_<n>') names """
    sets, groups = pp.sam_feature_groups(['sift_1_appear', 'sift_1_motion', 'sift_motion_1', 'thresh_adj_stat_1_appear'])
    assert list(sets) == ['appearance', 'motion', 'motion', 'appearance']
    assert list(groups) == ['SIFT', 'speed SIFT', 'speed SIFT', 'thresh_adj_stat']
