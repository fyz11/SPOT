# -*- coding: utf-8 -*-
"""compute_boundary_*_features_fast give exactly the outputs of the original functions (n_jobs=1), and up to
floating point round-off in parallel."""

import numpy as np
import pytest

import fixtures
import SPOT.Features.features as sf
from conftest import run_quiet

CALLS = fixtures.calls()
_ORIGINAL = {}


def original(case, d, key, fn_name, kw):
    if (case, key) not in _ORIGINAL:
        _ORIGINAL[(case, key)] = run_quiet(getattr(sf, fn_name), fast=False, **kw(d))
    return _ORIGINAL[(case, key)]


@pytest.mark.parametrize('n_jobs', [1, 2])
@pytest.mark.parametrize('key,fn_name,kw', CALLS, ids=[c[0] for c in CALLS])
def test_fast_equals_original(case_data, key, fn_name, kw, n_jobs):
    case, d = case_data
    feats, labels, norm = original(case, d, key, fn_name, kw)
    f_feats, f_labels, f_norm = run_quiet(getattr(sf, fn_name + '_fast'), n_jobs=n_jobs, **kw(d))
    assert list(np.asarray(f_labels).astype(str)) == list(np.asarray(labels).astype(str))
    assert np.array_equal(np.asarray(f_norm).astype(bool), np.asarray(norm).astype(bool))
    f_feats, feats = np.asarray(f_feats, np.float64), np.asarray(feats, np.float64)
    if n_jobs == 1:
        assert np.array_equal(f_feats, feats, equal_nan=True)
    else:
        # worker processes limit the BLAS threads (round-off of the BLAS-based Haralick statistics) and hold differently
        # aligned arrays (last-bit differences of SIMD log / arctan2 can move a shape context point across a bin edge)
        scm = np.char.startswith(np.asarray(labels).astype(str), 'Shape_Context')
        np.testing.assert_allclose(f_feats[..., ~scm], feats[..., ~scm], rtol=1e-10, atol=1e-12, equal_nan=True)
        if scm.any():
            bad = ~np.isclose(f_feats[..., scm], feats[..., scm], rtol=1e-10, atol=1e-12, equal_nan=True)
            assert bad.mean() < 0.1 and np.nanmax(np.abs(f_feats[..., scm] - feats[..., scm])) < 0.05


def test_ect_fast_equals_cubical_complex():
    rng = np.random.default_rng(3)
    dirs = sf._regular_directions(36, dims=2)
    for _ in range(20):
        im = rng.random((rng.integers(3, 30), rng.integers(3, 30))) > rng.uniform(0.2, 0.7)
        if im.sum() < 2:
            continue
        ref = sf.CubicalComplex(im).complexify().ECT(dirs, T=32)
        assert np.array_equal(sf._ect_fast(im, dirs, T=32), ref)


def test_tile_crop_equals_full_frame(case_data):
    case, d = case_data
    b = d['boundaries']
    for i in range(b.shape[0]):
        for t in range(b.shape[1]):
            cnt = b[i, t]
            if np.isnan(cnt[0, 0]):
                continue
            for n_r, n_theta, a0 in [(3, 1, None), (2, 4, 30.)]:
                full = sf._tile_uniform_radial_windows_boundary(d['imshape'], n_r, n_theta, cnt, zero_angle=a0)
                crop, (r0, c0) = sf._tile_uniform_radial_windows_boundary_crop(d['imshape'], n_r, n_theta, cnt,
                                                                                zero_angle=a0)
                expect = np.zeros_like(full)
                expect[r0:r0 + crop.shape[0], c0:c0 + crop.shape[1]] = crop
                assert np.array_equal(full, expect)


@pytest.mark.parametrize('fn_name', ['compute_boundary_morphology_features', 'compute_boundary_texture_features',
                                     'compute_boundary_motion_features'])
def test_fast_parallel_over_frames_single_object(fn_name):
    """ one object over many frames: the frames are split over the workers, same output as serial """
    d = fixtures.load('synthetic')
    kw = [c for c in CALLS if c[1] == fn_name][0][2](d)
    kw['boundaries'] = kw['boundaries'][:1]
    f1, l1, _ = run_quiet(getattr(sf, fn_name + '_fast'), n_jobs=1, **kw)
    f3, l3, _ = run_quiet(getattr(sf, fn_name + '_fast'), n_jobs=3, **kw)
    assert list(l1) == list(l3)
    np.testing.assert_allclose(np.asarray(f3, np.float64), np.asarray(f1, np.float64), rtol=1e-10, atol=1e-12, equal_nan=True)


def test_defaults_are_fast_and_masked(case_data):
    """ the public functions run the fast implementation and the masked appearance / motion features by default """
    import inspect
    case, d = case_data
    for fn in (sf.compute_boundary_texture_features, sf.compute_boundary_motion_features,
               sf.compute_boundary_texture_features_fast, sf.compute_boundary_motion_features_fast):
        p = inspect.signature(fn).parameters
        assert p['use_mask'].default is True and p['masked_sift_resize'].default == 'cap'
        assert p['siftshape'].default == (64, 64)
    for fn in (sf.compute_boundary_morphology_features, sf.compute_boundary_texture_features,
               sf.compute_boundary_motion_features):
        assert inspect.signature(fn).parameters['fast'].default is True
    a = run_quiet(sf.compute_boundary_texture_features, vid=d['vid'], boundaries=d['boundaries'])[0]
    b = run_quiet(sf.compute_boundary_texture_features, vid=d['vid'], boundaries=d['boundaries'], fast=False, use_mask=True)[0]
    assert np.array_equal(np.asarray(a, np.float64), np.asarray(b, np.float64), equal_nan=True)
