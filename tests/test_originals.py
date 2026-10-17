# -*- coding: utf-8 -*-
"""The original SAM feature functions reproduce the golden outputs (guards the compat fixes and newer environments)."""

import numpy as np
import pytest

import fixtures
import SPOT.Features.features as sf
from conftest import load_golden, run_quiet

CALLS = fixtures.calls()


def assert_matches_golden(feats, labels, g_feats, g_labels):
    """bit-exact in the environment the goldens were made in; across library versions float noise is allowed, and the
    shape context histograms may move a few contour points across a log-distance / angle bin edge."""
    feats = np.asarray(feats, np.float64)
    assert list(np.asarray(labels).astype(str)) == list(g_labels)
    scm = np.char.startswith(g_labels.astype(str), 'Shape_Context')
    np.testing.assert_allclose(feats[..., ~scm], g_feats[..., ~scm], rtol=1e-7, atol=1e-9, equal_nan=True)
    if scm.any():
        a, b = feats[..., scm], g_feats[..., scm]
        assert np.array_equal(np.isnan(a), np.isnan(b))
        bad = ~np.isclose(a, b, rtol=1e-7, atol=1e-9, equal_nan=True)
        assert np.nanmax(np.abs(a - b)) < 0.05 and bad.mean() < 0.1, (np.nanmax(np.abs(a - b)), bad.mean())


@pytest.mark.parametrize('key,fn_name,kw', CALLS, ids=[c[0] for c in CALLS])
def test_original_matches_golden(case_data, key, fn_name, kw):
    case, d = case_data
    feats, labels, norm = run_quiet(getattr(sf, fn_name), fast=False, **kw(d))  # the original implementation
    g_feats, g_labels, g_norm = load_golden(case, key)
    assert np.array_equal(np.asarray(norm).astype(bool), g_norm)
    assert_matches_golden(feats, labels, g_feats, g_labels)
