# -*- coding: utf-8 -*-
"""SPOT.Analysis.preprocessing.ConfoundRegressor / regress_out_confounders / confound_association"""

import numpy as np
import pytest

import SPOT.Analysis.preprocessing as pp

N_FEATS = 12
BATCH_SHIFT = np.array([0., 1.5, -1.])
BATCH_SCALE = np.array([1., 1.8, 0.6])


def _corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def make_data(n=1500, size_shift=0., seed=0):
    """ features = batch scale * (true signal + noise) + nonlinear f(size) + intensity effect + batch shift
    + condition effect; size of the treated objects shifted by size_shift standard deviations """
    rng = np.random.default_rng(seed)
    cond = rng.integers(0, 2, n)
    batch = rng.integers(0, 3, n)
    z = rng.normal(size=n) + size_shift * cond
    size = np.exp(0.3 * z + 3.)  # skewed, like an equivalent diameter
    intensity = rng.normal(100., 20., n)
    latent = rng.normal(size=(n, 2))
    W = rng.normal(size=(2, N_FEATS))
    W /= np.linalg.norm(W, axis=0, keepdims=True)  # unit signal variance per feature
    signal = latent @ W
    f_size = np.tanh(1.5 * z)[:, None] * rng.uniform(1., 2., N_FEATS)[None, :]
    f_int = ((intensity - 100.) / 20.)[:, None] * rng.uniform(0.5, 1., N_FEATS)[None, :]
    cond_effect = np.zeros(N_FEATS)
    cond_effect[:4] = 1.
    X = (BATCH_SCALE[batch][:, None] * (signal + 0.2 * rng.normal(size=(n, N_FEATS))) + f_size + f_int
         + BATCH_SHIFT[batch][:, None] + cond[:, None] * cond_effect[None, :])
    return dict(X=X, size=size, intensity=intensity, batch=batch, cond=cond, signal=signal,
                cond_effect=cond_effect, names=np.array(['f%d' % i for i in range(N_FEATS)]))


@pytest.fixture(scope='module')
def data():
    return make_data()


def _assoc(X, d, **kw):
    df = pp.confound_association(X, d['names'], continuous={'size': d['size'], 'intensity': d['intensity']},
                                 categorical={'batch': d['batch']}, **kw)
    return df.groupby('covariate')['association'].max()


def test_removes_confounds_and_keeps_signal(data):
    d = data
    before = _assoc(d['X'], d)
    assert before['size'] > 0.3 and before['intensity'] > 0.3 and before['batch'] > 0.1

    reg = pp.ConfoundRegressor(batch_scale=True)
    Xc = reg.fit_transform(d['X'], continuous={'size': d['size'], 'intensity': d['intensity']},
                           categorical={'batch': d['batch']})
    after = _assoc(Xc, d)
    assert after['size'] < 0.05 and after['intensity'] < 0.05 and after['batch'] < 0.005
    # re-standardised
    np.testing.assert_allclose(Xc.mean(0), 0, atol=1e-8)
    np.testing.assert_allclose(Xc.std(0), 1, atol=1e-8)
    # the true signal (plus the condition effect, independent of the covariates here) survives
    truth = d['signal'] + d['cond'][:, None] * d['cond_effect'][None, :]
    assert min(_corr(Xc[:, j], truth[:, j]) for j in range(N_FEATS)) > 0.9
    assert max(_corr(d['X'][:, j], truth[:, j]) for j in range(N_FEATS)) < 0.9  # it was not before


def test_batch_scale_equalises_variance(data):
    d = data
    kw = dict(continuous={'size': d['size'], 'intensity': d['intensity']}, categorical={'batch': d['batch']})
    sds = lambda X: np.array([X[d['batch'] == b].std(0) for b in range(3)])  # (3, n_feats)
    loc = pp.ConfoundRegressor(batch_scale=False).fit_transform(d['X'], **kw)
    ratio_loc = sds(loc).max(0) / sds(loc).min(0)
    assert ratio_loc.min() > 1.5  # BATCH_SCALE 1.8 / 0.6 of the signal + noise part
    both = pp.ConfoundRegressor(batch_scale=True).fit_transform(d['X'], **kw)
    ratio = sds(both).max(0) / sds(both).min(0)
    assert ratio.max() < 1.1


def test_linear_and_raw_options(data):
    d = data
    kw = dict(continuous={'size': d['size'], 'intensity': d['intensity']}, categorical={'batch': d['batch']})
    for spline, rank in [(False, True), (True, False), (False, False)]:
        Xc = pp.ConfoundRegressor(spline=spline, rank=rank).fit_transform(d['X'], **kw)
        assert np.isfinite(Xc).all()
        assert _assoc(Xc, d)['batch'] < 0.005
    # a linear model of the rank cannot fully remove the saturating tanh(size) effect, the spline can
    lin = pp.ConfoundRegressor(spline=False).fit_transform(d['X'], **kw)
    spl = pp.ConfoundRegressor().fit_transform(d['X'], **kw)
    r = lambda X: max(abs(_corr(np.tanh(1.5 * (np.log(d['size']) - 3.) / 0.3), X[:, j])) for j in range(N_FEATS))
    assert r(spl) < r(lin)


def test_reference_preserves_condition_difference():
    d = make_data(n=3000, size_shift=1.5, seed=1)
    ctrl = d['cond'] == 0
    kw = dict(continuous={'size': d['size'], 'intensity': d['intensity']}, categorical={'batch': d['batch']})
    diff = lambda X: X[~ctrl, :4].mean(0) - X[ctrl, :4].mean(0)
    # the raw difference mixes the condition effect (1) and the larger size of the treated objects
    assert diff(d['X']).min() > 1.5
    with_ref = pp.ConfoundRegressor(restandardize=False).fit_transform(d['X'], reference=ctrl, **kw)
    all_rows = pp.ConfoundRegressor(restandardize=False).fit_transform(d['X'], **kw)
    np.testing.assert_allclose(diff(with_ref), 1., atol=0.2)
    assert diff(all_rows).max() < 0.75  # partly regressed away as it is correlated with size
    # features without a condition effect show no difference either way
    assert np.abs(with_ref[~ctrl, 4:].mean(0) - with_ref[ctrl, 4:].mean(0)).max() < 0.2


def test_transform_matches_fit_transform_and_new_data(data):
    d = data
    n_fit = 1000
    cont = {'size': d['size'], 'intensity': d['intensity']}
    sub = lambda s: ({k: v[s] for k, v in cont.items()}, {'batch': d['batch'][s]})
    fit = slice(0, n_fit)
    new = slice(n_fit, None)
    reg = pp.ConfoundRegressor(batch_scale=True)
    ft = reg.fit_transform(d['X'][fit], *sub(fit))
    np.testing.assert_allclose(reg.transform(d['X'][fit], *sub(fit)), ft, atol=1e-10)
    # new rows: same result as transforming them together with the fit data
    Xn = reg.transform(d['X'][new], *sub(new))
    both = reg.transform(d['X'], *sub(slice(None)))
    np.testing.assert_allclose(Xn, both[new], atol=1e-10)
    assert _assoc(Xn, {k: (v[new] if k != 'names' else v) for k, v in d.items()})['size'] < 0.1
    # array covariates give the same result as dicts
    arr = pp.ConfoundRegressor(batch_scale=True).fit_transform(
        d['X'][fit], np.c_[d['size'][fit], d['intensity'][fit]], d['batch'][fit])
    np.testing.assert_allclose(arr, ft, atol=1e-10)
    # unseen batch, wrong covariates, non-finite input
    c, _ = sub(new)
    with pytest.raises(ValueError, match='not seen'):
        reg.transform(d['X'][new], c, {'batch': d['batch'][new] + 10})
    with pytest.raises(ValueError, match='do not match'):
        reg.transform(d['X'][new], {'size': d['size'][new]}, {'batch': d['batch'][new]})
    Xbad = d['X'][fit].copy()
    Xbad[0, 0] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        pp.ConfoundRegressor().fit(Xbad, *sub(fit))


def test_level_missing_from_reference_raises(data):
    d = data
    ref = d['batch'] != 2
    with pytest.raises(ValueError, match='reference'):
        pp.ConfoundRegressor().fit_transform(d['X'], categorical={'batch': d['batch']}, reference=ref)


def test_regress_out_confounders_covariate_features(data):
    d = data
    X = np.c_[d['X'], d['size'], d['intensity']]
    names = np.r_[d['names'], ['equivalent_diameter', 'mean_intensity']]
    cov = ['equivalent_diameter', 'mean_intensity']
    out, model = pp.regress_out_confounders(X, categorical={'batch': d['batch']}, feature_names=names,
                                            covariate_features=cov, return_model=True)
    assert out.shape == X.shape
    np.testing.assert_array_equal(out[:, -2:], X[:, -2:])
    ref = pp.ConfoundRegressor().fit_transform(d['X'], continuous={'equivalent_diameter': d['size'],
                                                                   'mean_intensity': d['intensity']},
                                               categorical={'batch': d['batch']})
    np.testing.assert_allclose(out[:, :-2], ref, atol=1e-10)
    assert isinstance(model, pp.ConfoundRegressor)

    dropped, names_out = pp.regress_out_confounders(X, categorical={'batch': d['batch']}, feature_names=names,
                                                    covariate_features=cov, keep_covariate_features=False)
    np.testing.assert_array_equal(names_out, d['names'])
    np.testing.assert_allclose(dropped, ref, atol=1e-10)
    with pytest.raises(ValueError):
        pp.regress_out_confounders(X, covariate_features=cov)
    with pytest.raises(ValueError):
        pp.regress_out_confounders(X, feature_names=names, covariate_features=['not_a_feature'])


def test_confound_association(data):
    d = data
    df = pp.confound_association(d['X'], d['names'], continuous={'size': d['size']},
                                 categorical={'batch': d['batch']})
    assert list(df.columns) == ['feature', 'covariate', 'kind', 'measure', 'association']
    assert len(df) == 2 * N_FEATS
    assert set(df['measure']) == {'abs_spearman', 'eta2'}
    assert ((df['association'] >= 0) & (df['association'] <= 1)).all()
    from scipy.stats import spearmanr
    rho = abs(spearmanr(d['X'][:, 0], d['size'])[0])
    got = df.query("feature == 'f0' and covariate == 'size'")['association'].item()
    assert got == pytest.approx(rho, abs=1e-10)
    # eta^2 of a pure level effect is 1, of independent noise ~0
    rng = np.random.default_rng(0)
    lv = rng.integers(0, 3, 600)
    Y = np.c_[lv * 1., rng.normal(size=600)]
    eta = pp.confound_association(Y, ['a', 'b'], categorical=lv)['association'].values
    assert eta[0] == pytest.approx(1.) and eta[1] < 0.02
