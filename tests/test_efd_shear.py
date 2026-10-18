"""The elliptic Fourier descriptor shape features and the shear-rate motion features: same names, order and values as
pip SAM-SPOT 0.1.1 (golden/pip011_efd_shear_u373.npz), their invariances / analytic values, and agreement of the
original, fast and torch implementations."""
import os

import numpy as np
import pytest

import fixtures
import SPOT.Features.features as sf
from conftest import run_quiet

PIP = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden', 'pip011_efd_shear_u373.npz'))


@pytest.fixture(scope='module')
def u373():
    return fixtures.load('u373')


def test_labels_and_values_match_pip_011(u373):
    feats, labels, _ = run_quiet(sf.compute_boundary_morphology_features, boundaries=u373['boundaries'], imshape=u373['imshape'])
    labels = np.asarray(labels).astype(str)
    assert list(labels) == list(PIP['shape_labels'])
    np.testing.assert_allclose(feats[..., np.char.startswith(labels, 'EFD_')], PIP['efd'], atol=1e-12, equal_nan=True)

    feats, labels, _ = run_quiet(sf.compute_boundary_motion_features, vid_flow=u373['flow'], boundaries=u373['boundaries'], use_mask=False,
                                 directional_motion_sift=False)  # the motion SIFT of pip 0.1.1
    labels = np.asarray(labels).astype(str)
    assert list(labels) == list(PIP['motion_labels'])
    sh = np.isin(labels, ['mean_shear_global_flow', 'stddev_shear_global_flow'])
    np.testing.assert_allclose(feats[..., sh], PIP['shear_global'], atol=1e-12, equal_nan=True)


def test_feature_order():
    labels = list(PIP['shape_labels'])
    last_fourier = max(i for i, n in enumerate(labels) if n.startswith('Fourier_'))
    assert labels[last_fourier + 1] == 'EFD_d_1' and labels[last_fourier + 37] == 'EFD_d_10'
    assert labels[last_fourier + 38].startswith('ECT_')
    labels = list(PIP['motion_labels'])
    assert labels.index('mean_shear_global_flow') == labels.index('stddev_divergence_global_flow') + 1
    assert labels.index('shear_sift_1') == labels.index('div_sift_128') + 1


def _contour(n_pts=300, rot=0., scale=1., shift=(0., 0.), start=0):
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    r = 20 * (1 + 0.3 * np.cos(3 * t) + 0.15 * np.sin(2 * t))  # an asymmetric shape
    xy = np.c_[r * np.cos(t), r * np.sin(t)]
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    xy = scale * xy @ R.T + np.asarray(shift)
    xy = np.roll(xy, start, axis=0)
    return np.vstack([xy, xy[:1]])


def test_efd_invariances():
    ref = sf.compute_elliptic_fourier_feats([_contour()])
    assert ref.shape == (1, 37)
    for kw in (dict(rot=0.7), dict(scale=2.5), dict(shift=(13., -4.)), dict(rot=2.1, scale=0.4, shift=(-3., 9.))):
        np.testing.assert_allclose(sf.compute_elliptic_fourier_feats([_contour(**kw)]), ref, atol=1e-8, err_msg=str(kw))
    # start point: up to the 180 degree ambiguity of the 1st harmonic ellipse (Kuhl & Giardina normalisation, as pip
    # SAM-SPOT 0.1.1), which can flip the sign of the even harmonics
    for kw in (dict(start=77), dict(rot=2.1, scale=0.4, shift=(-3., 9.), start=150)):
        np.testing.assert_allclose(np.abs(sf.compute_elliptic_fourier_feats([_contour(**kw)])), np.abs(ref), atol=1e-8,
                                   err_msg=str(kw))
    # unlike the Fourier magnitude features, the descriptor sees the mirror image (chirality)
    mirrored = _contour() * np.array([1., -1.])
    assert np.abs(sf.compute_elliptic_fourier_feats([mirrored]) - ref).max() > 1e-2


def test_shear_rate_analytic():
    yy, xx = np.mgrid[0:40, 0:50].astype(float)
    k = 0.3
    simple_shear = np.dstack([k * yy, np.zeros_like(xx)])  # u = (k y, 0)
    rotation = np.dstack([-k * yy, k * xx])  # rigid rotation
    dilation = np.dstack([k * xx, k * yy])  # isotropic expansion
    np.testing.assert_allclose(sf._shear_vector_flow(simple_shear), k, atol=1e-12)
    np.testing.assert_allclose(sf._shear_vector_flow(rotation), 0, atol=1e-12)
    np.testing.assert_allclose(sf._shear_vector_flow(dilation), 0, atol=1e-12)


@pytest.mark.parametrize('use_mask', [True, False])
def test_fast_matches_original(u373, use_mask):
    kw = dict(use_mask=use_mask, masks=u373['masks'] if use_mask else None)
    o = run_quiet(sf.compute_boundary_motion_features, vid_flow=u373['flow'], boundaries=u373['boundaries'], fast=False, **kw)
    f = run_quiet(sf.compute_boundary_motion_features, vid_flow=u373['flow'], boundaries=u373['boundaries'], **kw)
    assert list(o[1]) == list(f[1]) and sum('shear' in n for n in o[1]) == 130
    np.testing.assert_allclose(f[0], o[0], atol=1e-10, equal_nan=True)
    o = run_quiet(sf.compute_boundary_morphology_features, boundaries=u373['boundaries'], imshape=u373['imshape'], fast=False)
    f = run_quiet(sf.compute_boundary_morphology_features, boundaries=u373['boundaries'], imshape=u373['imshape'])
    assert list(o[1]) == list(f[1])
    np.testing.assert_allclose(f[0], o[0], atol=1e-10, equal_nan=True)


@pytest.mark.parametrize('use_mask', [True, False])
def test_torch_matches_original(u373, use_mask):
    pytest.importorskip('torch')
    from SPOT.Features import features_torch as ft
    kw = dict(use_mask=use_mask, masks=u373['masks'] if use_mask else None)
    o = run_quiet(sf.compute_boundary_motion_features, vid_flow=u373['flow'], boundaries=u373['boundaries'], **kw)
    t = run_quiet(ft.compute_boundary_motion_features_torch, vid_flow=u373['flow'], boundaries=u373['boundaries'], **kw)
    assert list(o[1]) == list(t[1])
    np.testing.assert_allclose(t[0], o[0], atol=1e-8, equal_nan=True)
    o = run_quiet(sf.compute_boundary_morphology_features, boundaries=u373['boundaries'], imshape=u373['imshape'])
    t = run_quiet(ft.compute_boundary_morphology_features_torch, boundaries=u373['boundaries'], imshape=u373['imshape'])
    assert list(o[1]) == list(t[1])
    np.testing.assert_allclose(t[0], o[0], atol=1e-10, equal_nan=True)


def test_flags_off_and_groups(u373):
    import SPOT.Analysis.preprocessing as pp
    from SPOT.Analysis.sam_analysis import get_SAM_feature_type
    _, labels, _ = run_quiet(sf.compute_boundary_morphology_features, boundaries=u373['boundaries'], imshape=u373['imshape'],
                             elliptic_fourier_features=False)
    assert not any(n.startswith('EFD_') for n in labels)
    _, labels, _ = run_quiet(sf.compute_boundary_motion_features, vid_flow=u373['flow'], boundaries=u373['boundaries'], shear_features=False)
    assert not any('shear' in n for n in labels)
    names = np.array(['EFD_d_1', 'EFD_a_2', 'mean_shear_global_flow', 'shear_sift_3'])
    sets, groups = pp.sam_feature_groups(names)
    assert list(sets) == ['shape', 'shape', 'motion', 'motion']
    assert list(groups) == ['elliptic Fourier', 'elliptic Fourier', 'global motion', 'shear SIFT']
    sam_type, scope = get_SAM_feature_type(names)
    assert list(sam_type) == [0, 0, 2, 2] and list(scope[:2]) == [1, 1]
