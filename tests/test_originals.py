# -*- coding: utf-8 -*-
"""The original SAM feature functions reproduce the golden outputs (guards the compat fixes and newer environments)."""

import numpy as np
import pytest

import fixtures
import SPOT.Features.features as sf
from conftest import load_golden, run_quiet

CALLS = fixtures.calls()


@pytest.mark.parametrize('key,fn_name,kw', CALLS, ids=[c[0] for c in CALLS])
def test_original_matches_golden(case_data, key, fn_name, kw):
    case, d = case_data
    feats, labels, norm = run_quiet(getattr(sf, fn_name), **kw(d))
    g_feats, g_labels, g_norm = load_golden(case, key)
    assert list(np.asarray(labels).astype(str)) == list(g_labels)
    assert np.array_equal(np.asarray(norm).astype(bool), g_norm)
    # bit-exact in the environment the goldens were made in; float noise allowed across library versions
    np.testing.assert_allclose(np.asarray(feats, np.float64), g_feats, rtol=1e-7, atol=1e-9, equal_nan=True)
