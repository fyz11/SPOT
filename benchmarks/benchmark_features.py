# -*- coding: utf-8 -*-
"""
Benchmark the SAM feature functions: original vs *_fast (serial and parallel) vs *_torch.

    python benchmarks/benchmark_features.py [--n-obj 8] [--n-frames 40] [--n-jobs 8 16] [--repeats 1]

Uses the real PhC-C2DH-U373 cell contours of data/cell_tracking (520 x 696 frames) on a synthetic textured video and
smooth flow (tests/fixtures.py). Writes benchmarks/benchmark_results.csv and benchmarks/benchmark_results.md.
"""

import argparse
import contextlib
import io
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))

import fixtures  # noqa: E402
import SPOT.Features.features as sf  # noqa: E402


def timed(fn, **kw):
    t0 = time.perf_counter()
    with contextlib.redirect_stderr(io.StringIO()):
        out = fn(**kw)
    return time.perf_counter() - t0, out


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-obj', type=int, default=8)
    ap.add_argument('--n-frames', type=int, default=40)
    ap.add_argument('--n-jobs', type=int, nargs='+', default=[8, 16])
    ap.add_argument('--skip-original', action='store_true')
    args = ap.parse_args()

    d = fixtures.u373(n_obj=args.n_obj, T=args.n_frames)
    n_samples = int((~np.isnan(d['boundaries'][:, :, 0, 0])).sum())
    print('U373: %d cells x %d frames (%d contours), frame %s' % (args.n_obj, args.n_frames, n_samples, d['imshape']))

    try:
        import torch
        import SPOT.Features.features_torch as ft
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gpu = torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'
        # warm up (CUDA context)
        ft.compute_boundary_texture_features_torch(d['vid'][:2], d['boundaries'][:1, :2], device=device)
    except ImportError:
        ft, device, gpu = None, None, None

    jobs = [('shape', 'compute_boundary_morphology_features', dict(boundaries=d['boundaries'], imshape=d['imshape'])),
            ('appearance', 'compute_boundary_texture_features', dict(vid=d['vid'], boundaries=d['boundaries'])),
            ('appearance (masked)', 'compute_boundary_texture_features',
             dict(vid=d['vid'], boundaries=d['boundaries'], use_mask=True, masks=d['masks'])),
            ('motion', 'compute_boundary_motion_features', dict(vid_flow=d['flow'], boundaries=d['boundaries']))]
    rows = []
    for family, fn_name, kw in jobs:
        ref = None
        variants = [] if args.skip_original else [('original', getattr(sf, fn_name), {})]
        variants += [('fast, n_jobs=1', getattr(sf, fn_name + '_fast'), dict(n_jobs=1))]
        variants += [('fast, n_jobs=%d' % j, getattr(sf, fn_name + '_fast'), dict(n_jobs=j)) for j in args.n_jobs]
        if ft is not None:
            variants += [('torch (%s)' % device, getattr(ft, fn_name + '_torch'), dict(device=device))]
        for name, fn, extra in variants:
            sec, out = timed(fn, **kw, **extra)
            feats = np.asarray(out[0], np.float64)
            if ref is None:
                ref = feats
            err = np.nanmax(np.abs(feats - ref) / (np.abs(ref) + 1e-9)) if feats.size else 0.
            rows.append(dict(features=family, variant=name, seconds=sec, ms_per_contour=1000 * sec / n_samples,
                             max_rel_diff_vs_first=err))
            print('%-20s %-18s %8.2fs  %7.1f ms/contour  max rel diff %.1e' % (family, name, sec, 1000 * sec / n_samples, err),
                  flush=True)
    df = pd.DataFrame(rows)
    for family in df.features.unique():
        base = df[(df.features == family)].seconds.iloc[0]
        df.loc[df.features == family, 'speedup'] = base / df.loc[df.features == family, 'seconds']
    df.to_csv(os.path.join(HERE, 'benchmark_results.csv'), index=False)
    import platform
    with open(os.path.join(HERE, 'benchmark_results.md'), 'w') as f:
        f.write('# SAM feature extraction benchmark\n\n')
        f.write('%d PhC-C2DH-U373 cell contours (%d cells x %d frames, %s frames); CPU: %s (%d logical cores); GPU: %s\n\n'
                % (n_samples, args.n_obj, args.n_frames, 'x'.join(map(str, d['imshape'])), platform.processor(),
                   os.cpu_count(), gpu))
        f.write('| features | variant | seconds | ms / contour | speed-up |\n|---|---|---:|---:|---:|\n')
        for r in df.itertuples():
            f.write('| %s | %s | %.2f | %.1f | %.1fx |\n' % (r.features, r.variant, r.seconds, r.ms_per_contour, r.speedup))
    print(df.round(3).to_string(index=False))
