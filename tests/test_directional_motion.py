"""directional_motion_sift=True: the flow passed to SIFT as the gradient field, sign-aware curl / divergence descriptors
and the principal shear axis SIFT; their quarter-turn permutations (rotation orbits) and the agreement of the
original, fast and torch implementations."""
import numpy as np
import pytest
from scipy import ndimage

import fixtures
import SPOT.Features.augment as ag
import SPOT.Features.features as sf
from conftest import run_quiet

FULL = np.ones((64, 64), dtype=bool)


def _uniform(angle, shape=(64, 64)):
    return np.dstack([np.full(shape, np.cos(angle)), np.full(shape, np.sin(angle))])


def _bins(desc, n_ch=8):
    return desc.reshape(n_ch, -1).sum(axis=1)


def test_flow_vector_sift_sees_direction():
    for k in range(8):
        d = sf._vector_sift_descriptor(_uniform(k * np.pi / 4), FULL, use_mask=False, sigma=0)
        assert np.argmax(_bins(d)) == k
        d = sf._vector_sift_descriptor(_uniform(k * np.pi / 4), FULL, use_mask=True, sigma=0)
        assert np.argmax(_bins(d)) == k
    # opposite flows: different descriptors (the magnitude SIFT cannot tell them apart)
    a = sf._vector_sift_descriptor(_uniform(0.3), FULL, use_mask=False)
    b = sf._vector_sift_descriptor(_uniform(0.3 + np.pi), FULL, use_mask=False)
    assert np.abs(a - b).max() > 0.1


def test_signed_descriptor():
    field = np.ones((64, 64))
    field[:, 32:] = -1  # positive left half, negative right half
    d = sf._signed_scalar_descriptor(field, FULL, use_mask=False, sigma=0).reshape(2, 4, 4)  # (channel, row, col)
    pos, neg = d
    assert pos[:, :2].sum() > 3 * pos[:, 2:].sum() and neg[:, 2:].sum() > 3 * neg[:, :2].sum()
    np.testing.assert_allclose(sf._signed_scalar_descriptor(-field, FULL, use_mask=False, sigma=0).reshape(2, 16),
                               d.reshape(2, 16)[::-1], atol=1e-12)
    dm = sf._signed_scalar_descriptor(-field, FULL, use_mask=True, sigma=0).reshape(2, 16)
    np.testing.assert_allclose(sf._signed_scalar_descriptor(field, FULL, use_mask=True, sigma=0).reshape(2, 16),
                               dm[::-1], atol=1e-12)


def test_shear_axis():
    yy, xx = np.mgrid[0:64, 0:64].astype(float)
    k = 0.2
    cases = {0: np.dstack([k * xx, 0 * yy]),  # stretching along x: axis 0 degrees
             4: np.dstack([0 * xx, k * yy]),  # stretching along y: axis 90 degrees (doubled: bin 4)
             2: np.dstack([k * yy, 0 * xx])}  # simple shear: principal axis 45 degrees (doubled: bin 2)
    for b, flow in cases.items():
        axis = sf._shear_axis_field(flow)
        np.testing.assert_allclose(np.linalg.norm(axis, axis=-1), sf._shear_vector_flow(flow), atol=1e-12)
        assert np.argmax(_bins(sf._vector_sift_descriptor(axis, FULL, use_mask=False, sigma=0))) == b


def _rot_vector(F):
    """ np.rot90 of a (rows, cols, 2) (x, y) vector field, rotating the vectors with the image """
    return np.dstack([np.rot90(F[..., 1]), -np.rot90(F[..., 0])])


def _random_flow(n, seed):
    rng = np.random.default_rng(seed)
    return np.dstack([ndimage.gaussian_filter(rng.standard_normal((n, n)), 3) for _ in range(2)])


@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('kind', ['flow_vector_sift_', 'curl_signed_', 'shear_axis_sift_'])
def test_quarter_turn_permutations(kind, masked):
    n = 63 if masked else 64
    yy, xx = np.mgrid[:n, :n]
    mask = ((yy - n // 2) ** 2 + (xx - n // 2) ** 2 <= (n // 2 - 2) ** 2) if masked else np.ones((n, n), bool)
    n_ch, shift = ag.DESCRIPTOR_SPECS[kind]
    P = ag.sift_quarter_turn_permutation(masked, n_ch, (4, 4), shift)
    kw = dict(use_mask=masked, resize='native', sigma=1.)
    A, B = [], []
    for seed in range(8):
        F = _random_flow(n, seed)
        G = _rot_vector(F)
        if kind == 'flow_vector_sift_':
            a, b = sf._vector_sift_descriptor(F, mask, **kw), sf._vector_sift_descriptor(G, np.rot90(mask), **kw)
        elif kind == 'shear_axis_sift_':
            a = sf._vector_sift_descriptor(sf._shear_axis_field(F), mask, **kw)
            b = sf._vector_sift_descriptor(sf._shear_axis_field(G), np.rot90(mask), **kw)
        else:
            curl = sf._curl_vector_flow(F)[..., -1]
            a = sf._signed_scalar_descriptor(curl, mask, **kw)
            b = sf._signed_scalar_descriptor(np.rot90(curl), np.rot90(mask), **kw)
        A.append(a); B.append(b)
    A, B = np.array(A), np.array(B)
    err = np.abs(B - A[:, P]).max()
    wrong = np.abs(B - A[:, np.roll(P, 1)]).max()
    assert err < (1e-6 if not masked else 0.3 * wrong), (err, wrong)


@pytest.fixture(scope='module')
def u373():
    return fixtures.load('u373')


@pytest.mark.parametrize('use_mask', [True, False])
def test_implementations_agree_and_layout(u373, use_mask):
    kw = dict(vid_flow=u373['flow'], boundaries=u373['boundaries'], use_mask=use_mask,
              masks=u373['masks'] if use_mask else None)
    o = run_quiet(sf.compute_boundary_motion_features, fast=False, directional_motion_sift=True, **kw)
    f = run_quiet(sf.compute_boundary_motion_features, directional_motion_sift=True, **kw)
    assert list(o[1]) == list(f[1])
    np.testing.assert_allclose(f[0], o[0], atol=1e-10, equal_nan=True)
    pytest.importorskip('torch')
    from SPOT.Features import features_torch as ft
    t = run_quiet(ft.compute_boundary_motion_features_torch, directional_motion_sift=True, **kw)
    assert list(o[1]) == list(t[1])
    np.testing.assert_allclose(t[0], o[0], atol=1e-8, equal_nan=True)
    # the directional blocks replace the gradient SIFT blocks at the same position
    g = run_quiet(sf.compute_boundary_motion_features, **kw)
    lab_d, lab_g = list(o[1]), list(g[1])
    first = lab_g.index('sift_motion_1')
    assert lab_d[:first] == lab_g[:first]
    sizes = {p: sum(n.startswith(p) for n in lab_d) for p in ('flow_vector_sift_', 'curl_signed_', 'div_signed_', 'shear_axis_sift_')}
    assert sizes == {'flow_vector_sift_': 128, 'curl_signed_': 32, 'div_signed_': 32, 'shear_axis_sift_': 128}
    assert not any(n.startswith(('sift_motion_', 'curl_sift_', 'div_sift_', 'shear_sift_')) for n in lab_d)


def test_rotation_averaged_orbits_and_groups():
    import SPOT.Analysis.preprocessing as pp
    from SPOT.Analysis.sam_analysis import get_SAM_feature_type
    d = fixtures.load('synthetic')
    feats, labels, norm = run_quiet(ag.compute_rotation_averaged_features, feature_fn=sf.compute_boundary_motion_features,
                                    boundaries=d['boundaries'], vid=d['vid'], n_rotations=4, verbose=False,
                                    directional_motion_sift=True)
    labels = np.asarray(labels).astype(str)
    counts = {p: int(np.char.startswith(labels, p + 'orbit').sum())
              for p in ('flow_vector_sift_', 'curl_signed_', 'div_signed_', 'shear_axis_sift_')}
    assert counts == {'flow_vector_sift_': 32, 'curl_signed_': 8, 'div_signed_': 8, 'shear_axis_sift_': 32}
    names = np.array(['flow_vector_sift_1', 'curl_signed_3', 'div_signed_31', 'shear_axis_sift_9'])
    sets, groups = pp.sam_feature_groups(names)
    assert list(sets) == ['motion'] * 4
    assert list(groups) == ['flow vector SIFT', 'curl signed', 'divergence signed', 'shear axis SIFT']
    sam_type, scope = get_SAM_feature_type(names)
    assert list(sam_type) == [2] * 4 and list(scope) == [2] * 4
