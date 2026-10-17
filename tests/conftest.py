import os
import sys
import warnings

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import fixtures  # noqa: E402

warnings.filterwarnings('ignore', category=RuntimeWarning)


@pytest.fixture(scope='session', params=fixtures.CASES)
def case_data(request):
    return request.param, fixtures.load(request.param)


def load_golden(case, key):
    g = np.load(os.path.join(HERE, 'golden', '%s_%s.npz' % (case, key)))
    return g['features'], g['labels'], g['norm_bool']


def run_quiet(fn, **kwargs):
    import contextlib
    import io
    with contextlib.redirect_stderr(io.StringIO()):
        return fn(**kwargs)
