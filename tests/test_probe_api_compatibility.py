import warnings

import numpy as np

from mini_bang.tasks.flow_lenia_common.probe import PROBE_PARAMETERS, make_probe


def test_probe_preserves_predictions_without_deprecated_keyword(monkeypatch):
    from sklearn import linear_model
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    original = linear_model.LogisticRegression
    rng = np.random.default_rng(31)
    x, test = rng.normal(size=(24, 5)), rng.normal(size=(11, 5))
    y = (x[:, 0] + .5 * x[:, 1] > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        legacy = make_pipeline(StandardScaler(), original(**PROBE_PARAMETERS)).fit(x, y)
    def future_constructor(**kwargs):
        if 'penalty' in kwargs:
            raise TypeError('penalty has been removed')
        return original(**kwargs)
    monkeypatch.setattr(linear_model, 'LogisticRegression', future_constructor)
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        compatible = make_probe().fit(x, y)
    assert np.allclose(compatible.predict_proba(test), legacy.predict_proba(test), rtol=0, atol=1e-12)
    assert PROBE_PARAMETERS['penalty'] == 'l2'
