# -*- coding: utf-8 -*-
"""compute_boundary_*_features_torch match the original functions (identical rasterisation / ECT / curl, float64
round-off in the batched Haralick and SIFT reductions). Skipped without torch."""

import numpy as np
import pytest

import fixtures
import SPOT.Features.features as sf
from conftest import run_quiet

torch = pytest.importorskip('torch')
import SPOT.Features.features_torch as ft  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CALLS = fixtures.calls()
_ORIGINAL = {}


def original(case, d, key, fn_name, kw):
    if (case, key) not in _ORIGINAL:
        _ORIGINAL[(case, key)] = run_quiet(getattr(sf, fn_name), **kw(d))
    return _ORIGINAL[(case, key)]


@pytest.mark.parametrize('key,fn_name,kw', CALLS, ids=[c[0] for c in CALLS])
def test_torch_matches_original(case_data, key, fn_name, kw):
    case, d = case_data
    feats, labels, norm = original(case, d, key, fn_name, kw)
    t_feats, t_labels, t_norm = run_quiet(getattr(ft, fn_name + '_torch'), device=DEVICE, **kw(d))
    assert list(np.asarray(t_labels).astype(str)) == list(np.asarray(labels).astype(str))
    assert np.array_equal(np.asarray(t_norm).astype(bool), np.asarray(norm).astype(bool))
    np.testing.assert_allclose(np.asarray(t_feats, np.float64), np.asarray(feats, np.float64),
                               rtol=1e-8, atol=1e-9, equal_nan=True)


def test_polygon_torch_equals_skimage():
    from skimage.draw import polygon
    rng = np.random.default_rng(0)
    polys, shapes = [], []
    for k in range(300):
        n = rng.integers(3, 60)
        t = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = rng.uniform(2, 40) * (1 + 0.5 * rng.standard_normal(n).clip(-1.5, 1.5))
        cy, cx = rng.uniform(-10, 110, 2)
        r, c = cy + rad * np.sin(t), cx + rad * np.cos(t)
        if k % 3 == 0:
            r, c = np.round(r), np.round(c)  # integer vertices, on which pixels lie exactly
        polys.append((r, c))
        shapes.append(None if k % 4 == 0 else (100, 90))
    got = ft.polygon_torch(polys, shapes, device=DEVICE)
    for (r, c), s, (rr, cc) in zip(polys, shapes, got):
        ref = polygon(r, c, s)
        assert np.array_equal(ref[0], rr) and np.array_equal(ref[1], cc)


def test_ect_torch_equals_cpu():
    rng = np.random.default_rng(1)
    dirs = sf._regular_directions(36, dims=2)
    ims = [rng.random((rng.integers(2, 40), rng.integers(2, 40))) > rng.uniform(0.1, 0.8) for _ in range(50)]
    ims += [np.ones((1, 1), bool), np.ones((1, 5), bool)]
    ref = np.stack([sf.CubicalComplex(im).complexify().ECT(dirs, T=32) for im in ims])
    assert np.array_equal(ft.ect_torch(ims, dirs, T=32, device=DEVICE), ref)
