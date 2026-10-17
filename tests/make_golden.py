# -*- coding: utf-8 -*-
"""
Save the outputs of the original SAM feature functions on the test fixtures as the reference ('golden') outputs that
every variant (*_fast, *_torch) and every newer Python / NumPy environment is checked against.

    python tests/make_golden.py

Writes tests/golden/<case>_<call>.npz (features, labels, norm_bool) and tests/golden/versions.txt.
"""

import contextlib
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import fixtures  # noqa: E402
import SPOT.Features.features as sf  # noqa: E402


def run(fn_name, kwargs):
    with contextlib.redirect_stderr(io.StringIO()):
        return getattr(sf, fn_name)(**kwargs)


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    out = os.path.join(HERE, 'golden')
    os.makedirs(out, exist_ok=True)
    for case in fixtures.CASES:
        d = fixtures.load(case)
        for key, fn_name, kw in fixtures.calls():
            feats, labels, norm = run(fn_name, kw(d))
            np.savez_compressed(os.path.join(out, '%s_%s.npz' % (case, key)), features=np.asarray(feats, np.float64),
                                labels=np.asarray(labels).astype(str), norm_bool=np.asarray(norm).astype(bool))
            print(case, key, np.shape(feats), 'NaN rows', int(np.isnan(np.asarray(feats)).all(-1).sum()), flush=True)
    import scipy, skimage, mahotas
    with open(os.path.join(out, 'versions.txt'), 'w') as f:
        f.write('python %s\nnumpy %s\nscipy %s\nskimage %s\nmahotas %s\n'
                % (sys.version.split()[0], np.__version__, scipy.__version__, skimage.__version__, mahotas.__version__))
