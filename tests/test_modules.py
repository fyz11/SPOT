""" SAM module selection (SPOT.Analysis.sam_analysis.hierarchical_cluster_features_into_SAM_modules):
the multi-scale selection finds nested module scales, never crashes, and the legacy rule is unchanged """
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from SPOT.Analysis import sam_analysis as sam

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden', 'modules_legacy.npz')


def nested(seed=0, n=400, n_super=3, n_sub=3, per=8, n_noise=20):
    """ n_super groups of n_sub modules of per correlated features each, plus n_noise independent features """
    rng = np.random.default_rng(seed)
    cols = []
    for _ in range(n_super):
        zs = rng.standard_normal(n)
        for _ in range(n_sub):
            zt = rng.standard_normal(n)
            cols += [zs + zt + 0.6 * rng.standard_normal(n) for _ in range(per)]
    cols += [rng.standard_normal(n) for _ in range(n_noise)]
    return np.column_stack(cols)


def run(X, **kw):
    names = np.array(['f%d' % i for i in range(X.shape[1])])
    z = np.zeros(len(names), dtype=int)
    out = sam.hierarchical_cluster_features_into_SAM_modules(X, names, feature_type=z, feature_scope=z, **kw)
    plt.close('all')
    return out


def n_big(labels, min_size=3):
    return int((np.bincount(labels)[1:] >= min_size).sum())


def test_legacy_unchanged():
    d = np.load(GOLDEN)
    out = run(d['X'], module_selection='legacy')
    np.testing.assert_array_equal(out[1], d['labels'])


@pytest.mark.parametrize('n_noise', [0, 20])
def test_multiscale_finds_nested_scales(n_noise):
    X = nested(n_noise=n_noise)
    out = run(X, return_scales=True)
    labels, scales = out[1], out[5]
    coarse = 3 + (n_noise >= 3)  # the independent features end up in one group at the coarse scale
    assert {coarse, 9} <= set(scales.n_modules)
    assert scales.selected.sum() == 1
    assert int(scales.n_modules[scales.selected].iloc[0]) in (coarse, 9)
    assert n_big(labels) == int(scales.n_modules[scales.selected].iloc[0])
    assert ((scales.stability >= -1) & (scales.stability <= 1)).all()
    # the 9 planted modules are recovered exactly at the fine scale
    fine = run(X, scale=9)[1]
    planted = np.repeat(np.arange(9), 8)
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(planted, fine[:72]) == 1.0
    for col in ['n_modules', 'n_small_modules', 'coverage', 'lifetime_gap', 'silhouette_gap', 'stability',
                'consensus_rank', 'selected']:
        assert col in scales


def test_scale_options():
    X = nested()
    coarse, fine = n_big(run(X, scale='coarse')[1]), n_big(run(X, scale='fine')[1])
    assert coarse <= fine
    assert n_big(run(X, scale=9)[1]) == 9


def test_no_structure_does_not_crash():
    X = np.random.default_rng(1).standard_normal((300, 40))
    out = run(X, return_scales=True)
    assert len(out[1]) == 40 and out[5].selected.sum() == 1
    out = run(X, module_selection='legacy')
    assert len(out[1]) == 40


def test_module_count_intervals():
    from scipy.cluster.hierarchy import fcluster
    X = nested(n_noise=0)
    Z = sam._feature_linkage(sam._corr(X))
    lo, hi, k = sam._module_count_intervals(Z, min_module_size=1)
    # with min_module_size=1 every cut counts all clusters: n - 1 - i after merge i
    assert list(k) == list(len(Z) - np.arange(len(Z)))
    for kk in (3, 9):
        assert n_big(sam._cut_modules(Z, kk), 3) == kk
        assert len(np.unique(fcluster(Z, sam._cut_height(Z, kk, 1), 'distance'))) == kk
