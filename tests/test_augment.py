# -*- coding: utf-8 -*-
"""SPOT.Features.augment: rotation of the inputs and rotation-averaged SAM features."""

import numpy as np
import pytest

import fixtures
import SPOT.Features.features as sf
import SPOT.Features.augment as ag
from conftest import run_quiet


@pytest.fixture(scope='module')
def data():
    return fixtures.load('synthetic')


@pytest.mark.parametrize('angle', [90, 180, 270])
def test_quarter_turns_are_exact_and_invertible(data, angle):
    vid, b = data['vid'], data['boundaries']
    rot, T = ag.rotate_video(vid, angle)
    assert np.array_equal(rot, np.rot90(vid, angle // 90, axes=(1, 2)))
    back, T_back = ag.rotate_video(rot, -angle)
    assert np.array_equal(back, vid)
    b_back = ag.rotate_boundaries(ag.rotate_boundaries(b, T), T_back)
    np.testing.assert_allclose(b_back, b, atol=1e-12)
    assert np.array_equal(np.isnan(b_back), np.isnan(b))
    # every contour point lands on the same pixel value (integer test points)
    pts = np.array([[3., 7.], [100., 20.], [0., 159.]])
    q = T(pts).astype(int)
    for (r, c), (r0, c0) in zip(q, pts.astype(int)):
        assert rot[0, r, c] == vid[0, r0, c0]


def test_interpolated_rotation_matches_rotated_contour(data):
    """ at 45 degrees the rotated mask and the mask of the rotated contour agree """
    cnt = data['boundaries'][0, 0]
    mask = np.zeros(data['imshape'], dtype=bool)
    from skimage.draw import polygon
    rr, cc = polygon(cnt[:, 0], cnt[:, 1], data['imshape'])
    mask[rr, cc] = True
    rot_mask, T = ag.rotate_video(mask[None], 45, order=0)
    cnt_rot = T(cnt)
    ref = np.zeros(rot_mask.shape[1:], dtype=bool)
    rr, cc = polygon(cnt_rot[:, 0], cnt_rot[:, 1], ref.shape)
    ref[rr, cc] = True
    iou = (ref & rot_mask[0]).sum() / (ref | rot_mask[0]).sum()
    assert iou > 0.95, iou


def test_invariant_features_identical_at_quarter_turns(data):
    feats, labels, norm, (per_angle, angles) = run_quiet(
        ag.compute_rotation_averaged_features, feature_fn=sf.compute_boundary_texture_features_fast,
        boundaries=data['boundaries'], vid=data['vid'], n_rotations=4, return_per_angle=True, verbose=False)
    assert list(angles) == [0., 90., 180., 270.]
    assert per_angle.shape[:1] == (4,) and feats.shape == per_angle.shape[1:]
    labels = list(labels)
    for name in ('mean_intensity', 'stddev_intensity'):
        col = per_angle[..., labels.index(name)]
        np.testing.assert_allclose(col, np.broadcast_to(col[0], col.shape), rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(feats, np.nanmean(per_angle, axis=0), equal_nan=True)
    sens = ag.rotation_sensitivity(per_angle, labels).set_index('feature')
    assert sens.loc['mean_intensity', 'sensitivity'] < 1e-10
    assert sens[sens.group == 'SIFT'].sensitivity.median() > 0.05  # SIFT is orientation dependent


def test_shape_area_identical_at_quarter_turns(data):
    feats, labels, norm, (per_angle, _) = run_quiet(
        ag.compute_rotation_averaged_features, feature_fn=sf.compute_boundary_morphology_features_fast,
        boundaries=data['boundaries'], imshape=data['imshape'], n_rotations=4, return_per_angle=True, verbose=False)
    labels = list(labels)
    for name in ('area', 'perimeter'):
        col = per_angle[..., labels.index(name)]
        np.testing.assert_allclose(col, np.broadcast_to(col[0], col.shape), rtol=1e-12, equal_nan=True)


def test_motion_recomputes_flow_per_angle(data, monkeypatch):
    import SPOT.Tracking.optical_flow as of
    calls = []
    real = of.extract_vid_optflow

    def counting(vid, flow_params, rescale_intensity=False):
        calls.append(vid.shape)
        return real(vid, flow_params, rescale_intensity=rescale_intensity)

    monkeypatch.setattr(of, 'extract_vid_optflow', counting)
    feats, labels, norm, (per_angle, angles) = run_quiet(
        ag.compute_rotation_averaged_features, feature_fn=sf.compute_boundary_motion_features_fast,
        boundaries=data['boundaries'], vid=data['vid'], angles=[0, 90, 45], return_per_angle=True, verbose=False)
    H, W = data['imshape']
    assert calls[0][1:] == (H, W) and calls[1][1:] == (W, H) and calls[2][1] > H  # 45 deg: enlarged canvas
    assert feats.shape == (data['boundaries'].shape[0], data['boundaries'].shape[1] - 1, len(labels))
    # the centroid speed comes from the contours only: identical at the exact quarter turn
    i = list(labels).index('mean_speed_global')
    np.testing.assert_allclose(per_angle[1, ..., i], per_angle[0, ..., i], rtol=0, atol=1e-10, equal_nan=True)


def test_fourier_remove_closing_point_is_rotation_invariant(data):
    """ without the repeated closing point the Fourier magnitudes do not depend on the rotation / contour start point.
    Real (U373) contours: the ideal synthetic shapes have Fourier coefficients that are exactly 0, whose round-off is
    amplified by the |.|**0.2 power normalisation of the features. """
    real = fixtures.load('u373')
    kw = dict(feature_fn=sf.compute_boundary_morphology_features_fast, boundaries=real['boundaries'][:, :4],
              imshape=real['imshape'], angles=[0, 30, 90], return_per_angle=True, verbose=False)
    sens = {}
    for remove in (False, True):
        _, labels, _, (per_angle, _) = run_quiet(ag.compute_rotation_averaged_features,
                                                 fourier_remove_closing_point=remove, **kw)
        s = ag.rotation_sensitivity(per_angle, labels)
        sens[remove] = s[s.group == 'Fourier'].sensitivity.max()
    assert sens[True] < 1e-8 < 0.01 < sens[False]
    # the same option in the original and torch variants
    f0, l0, _ = run_quiet(sf.compute_boundary_morphology_features, boundaries=data['boundaries'], imshape=data['imshape'],
                          fourier_remove_closing_point=True)
    f1, l1, _ = run_quiet(sf.compute_boundary_morphology_features_fast, boundaries=data['boundaries'],
                          imshape=data['imshape'], fourier_remove_closing_point=True)
    assert list(l0) == list(l1) and sum(str(n).startswith('Fourier') for n in l0) == 98
    np.testing.assert_array_equal(f0, f1)


def _probes(n, k, masked, n_probe=12, seed=0):
    from scipy import ndimage
    rng = np.random.default_rng(seed)
    ext = sf.SingleSiftExtractor(64)
    A, B = [], []
    for _ in range(n_probe):
        im = ndimage.gaussian_filter(rng.standard_normal((n, n)), 1.5) * 50 + 100
        if masked:
            yy, xx = np.mgrid[:n, :n]
            m = (yy - n // 2) ** 2 + (xx - n // 2) ** 2 <= (n // 2 - 2) ** 2
            A.append(sf._masked_sift_descriptor(im, m, resize='native'))
            B.append(sf._masked_sift_descriptor(np.rot90(im, k), np.rot90(m, k), resize='native'))
        else:
            d = [ext.process_image(x).ravel() for x in (im, np.rot90(im, k))]
            A.append(d[0] / np.linalg.norm(d[0])); B.append(d[1] / np.linalg.norm(d[1]))
    return np.array(A), np.array(B)


@pytest.mark.parametrize('masked', [False, True])
def test_sift_quarter_turn_permutation(masked):
    """ desc(rot90(im))[j] == desc(im)[P[j]]: exactly for the square grid, up to pixel discretisation of the sectors
    for the masked descriptor; P^k for k quarter turns """
    P = ag.sift_quarter_turn_permutation(use_mask=masked)
    assert sorted(P) == list(range(128))
    Pk = np.arange(128)
    for k in (1, 2, 3):
        Pk = Pk[P]
        A, B = _probes(63 if masked else 64, k, masked)
        err = np.abs(B - A[:, Pk]).max()
        others = np.abs(B - A[:, np.roll(Pk, 1)]).max()
        assert err < (1e-12 if not masked else 0.25 * others), (k, err, others)
    orbit = ag.sift_rotation_orbits(use_mask=masked)
    assert np.bincount(orbit).tolist() == [4] * 32


def test_collapse_sift_is_opt_in_and_invariant(data):
    kw = dict(feature_fn=sf.compute_boundary_texture_features_fast, boundaries=data['boundaries'], vid=data['vid'],
              n_rotations=4, return_per_angle=True, verbose=False)
    feats, labels, norm, (per_angle, _) = run_quiet(ag.compute_rotation_averaged_features, **kw)
    assert sum(l.startswith('sift_appear_') for l in labels) == 128  # default: all bins kept
    feats_c, labels_c, norm_c, (per_angle_c, _) = run_quiet(ag.compute_rotation_averaged_features, collapse_sift=True, **kw)
    sift = [l for l in labels_c if l.startswith('sift_appear_')]
    assert len(sift) == 32 and all('orbit' in l for l in sift)
    assert len(labels_c) == len(labels) - 96 == feats_c.shape[-1] == per_angle_c.shape[-1] == len(norm_c)
    # the orbit features are the means of the averaged bins, and invariant to quarter turns themselves
    orbit = ag.sift_rotation_orbits(use_mask=True)
    lab = list(labels)
    j = [lab.index('sift_appear_%d' % (i + 1)) for i in np.flatnonzero(orbit == 0)]
    np.testing.assert_allclose(feats_c[..., list(labels_c).index('sift_appear_orbit1')], np.nanmean(feats[..., j], -1),
                               equal_nan=True)
    sens = ag.rotation_sensitivity(per_angle_c, labels_c).set_index('feature')
    sens_all = ag.rotation_sensitivity(per_angle, labels).set_index('feature')
    assert sens.loc[sift, 'sensitivity'].median() < 0.5 * sens_all.loc[[l for l in lab if l.startswith('sift_appear_')], 'sensitivity'].median()
    # not collapsed if the angles are not closed under quarter turns
    out = run_quiet(ag.compute_rotation_averaged_features, feature_fn=sf.compute_boundary_texture_features_fast,
                    boundaries=data['boundaries'], vid=data['vid'], angles=[0, 90], collapse_sift=True, verbose=False)
    assert sum(l.startswith('sift_appear_') for l in out[1]) == 128
