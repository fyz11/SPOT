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
    'shape': ['curvature', 'contour geometry', 'binary morphology', 'Fourier', 'Euler characteristic transform', 'shape context'],
    'appearance': ['global intensity', 'regional intensity', 'Haralick texture', 'SIFT'],
    'motion': ['global motion', 'regional speed', 'regional speed distribution', 'speed SIFT', 'curl SIFT', 'divergence SIFT'],
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
