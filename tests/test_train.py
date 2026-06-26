import sys
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, brier_score_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.train import calibrar_modelo


def _dataset_gbm_overconfident(n=400, seed=42):
    """
    Dataset binario con GBM ya entrenado en todo (sobreconfiado: probas extremas).
    Devuelve (modelo_base, X_held, y_held, dates_train).
    """
    rng = np.random.RandomState(seed)
    X, y = make_classification(n_samples=n, n_features=6, random_state=seed)
    # split temporal simple: 75% train, 25% held-out
    split = int(n * 0.75)
    X_tr, y_tr = X[:split], y[:split]
    X_h,  y_h  = X[split:], y[split:]

    base = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=seed)
    base.fit(X_tr, y_tr)

    # fechas ficticias cronológicas para el fold temporal
    base_date = pd.Timestamp("2024-01-01")
    dates_train = (
        pd.date_range(base_date, periods=split, freq="D")
        .strftime("%Y%m%d").astype(int).to_numpy()
    )
    return base, X_tr, y_tr, X_h, y_h, dates_train


def test_calibrar_modelo_devuelve_objeto_con_predict_y_predict_proba():
    base, X_tr, y_tr, X_h, y_h, dates = _dataset_gbm_overconfident()
    cal = calibrar_modelo(base, X_tr, y_tr)
    assert hasattr(cal, 'predict')
    assert hasattr(cal, 'predict_proba')


def test_predict_proba_en_rango_y_suman_uno():
    base, X_tr, y_tr, X_h, y_h, dates = _dataset_gbm_overconfident()
    cal = calibrar_modelo(base, X_tr, y_tr)
    proba = cal.predict_proba(X_h)
    assert proba.shape == (len(X_h), 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


def test_calibracion_no_empeora_log_loss():
    """
    GBM sobreentrenado → probas extremas → mal calibrado.
    CalibratedClassifierCV(isotonic) debe igualar o mejorar log-loss en held-out.
    Margen eps=0.05 para ruido estadístico con n pequeño.
    """
    base, X_tr, y_tr, X_h, y_h, dates = _dataset_gbm_overconfident()
    cal = calibrar_modelo(base, X_tr, y_tr)

    ll_base = log_loss(y_h, base.predict_proba(X_h)[:, 1])
    ll_cal  = log_loss(y_h, cal.predict_proba(X_h)[:, 1])
    assert ll_cal <= ll_base + 0.05, (
        f"Calibrado ({ll_cal:.4f}) peor que base ({ll_base:.4f}) por más de eps"
    )


def test_calibracion_no_empeora_brier():
    base, X_tr, y_tr, X_h, y_h, dates = _dataset_gbm_overconfident()
    cal = calibrar_modelo(base, X_tr, y_tr)

    br_base = brier_score_loss(y_h, base.predict_proba(X_h)[:, 1])
    br_cal  = brier_score_loss(y_h, cal.predict_proba(X_h)[:, 1])
    assert br_cal <= br_base + 0.05, (
        f"Calibrado ({br_cal:.4f}) peor que base ({br_base:.4f}) por más de eps"
    )


def test_calibrar_modelo_acepta_dates_sin_error():
    base, X_tr, y_tr, X_h, y_h, dates = _dataset_gbm_overconfident()
    # No debe lanzar excepción al pasar dates (usa purged_time_series_splits)
    cal = calibrar_modelo(base, X_tr, y_tr, dates=dates)
    assert hasattr(cal, 'predict_proba')


def test_calibrar_modelo_method_sigmoid():
    base, X_tr, y_tr, X_h, y_h, dates = _dataset_gbm_overconfident()
    cal = calibrar_modelo(base, X_tr, y_tr, method="sigmoid")
    proba = cal.predict_proba(X_h)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


from src.train import comparar_calibracion

def test_comparar_calibracion_devuelve_mejor():
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 3))
    y = (X[:, 0] + rng.standard_normal(200) * 0.5 > 0).astype(int)
    modelo = GradientBoostingClassifier(n_estimators=10, random_state=42).fit(X, y)
    res = comparar_calibracion(modelo, X, y)
    assert set(res.keys()) == {'sigmoid_log_loss', 'isotonic_log_loss', 'mejor'}
    assert res['mejor'] in ('sigmoid', 'isotonic')
    assert res['sigmoid_log_loss'] > 0 and res['isotonic_log_loss'] > 0

def test_comparar_calibracion_mejor_es_menor_log_loss():
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    rng = np.random.default_rng(99)
    X = rng.standard_normal((200, 3))
    y = (X[:, 0] > 0).astype(int)
    modelo = GradientBoostingClassifier(n_estimators=10, random_state=42).fit(X, y)
    res = comparar_calibracion(modelo, X, y)
    if res['mejor'] == 'sigmoid':
        assert res['sigmoid_log_loss'] <= res['isotonic_log_loss']
    else:
        assert res['isotonic_log_loss'] <= res['sigmoid_log_loss']


# --- Tests E5: SoftVotingEnsemble + crear_ensemble ---

from src.train import SoftVotingEnsemble, crear_ensemble
from sklearn.linear_model import LogisticRegression


def _cuatro_modelos_simples():
    """Cuatro modelos ya entrenados en datos linealmente separables."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    y = (X[:, 0] > 0).astype(int)
    return {
        'a': LogisticRegression(C=1).fit(X, y),
        'b': LogisticRegression(C=0.1).fit(X, y),
        'c': LogisticRegression(C=10).fit(X, y),
        'd': LogisticRegression(C=0.5).fit(X, y),
    }, X, y


def test_soft_voting_predict_proba_shape():
    modelos, X, y = _cuatro_modelos_simples()
    ens = SoftVotingEnsemble(modelos)
    proba = ens.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_soft_voting_predict_proba_en_rango():
    modelos, X, y = _cuatro_modelos_simples()
    ens = SoftVotingEnsemble(modelos)
    proba = ens.predict_proba(X)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_soft_voting_predict_proba_suma_uno():
    modelos, X, y = _cuatro_modelos_simples()
    ens = SoftVotingEnsemble(modelos)
    proba = ens.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


def test_soft_voting_es_promedio_exacto():
    modelos, X, y = _cuatro_modelos_simples()
    ens = SoftVotingEnsemble(modelos)
    esperado = np.mean([m.predict_proba(X) for m in modelos.values()], axis=0)
    np.testing.assert_allclose(ens.predict_proba(X), esperado, atol=1e-12)


def test_soft_voting_predict_binario():
    modelos, X, y = _cuatro_modelos_simples()
    ens = SoftVotingEnsemble(modelos)
    preds = ens.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_crear_ensemble_tiene_predict_y_predict_proba():
    modelos, X, y = _cuatro_modelos_simples()
    ens = crear_ensemble(modelos)
    assert hasattr(ens, 'predict') and hasattr(ens, 'predict_proba')
