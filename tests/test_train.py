import sys
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, brier_score_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.train import calibrar_modelo, entrenar_modelo, crear_pipeline, coeficientes_modelo


def _dataset(n=400, seed=42):
    """Dataset binario + fechas cronológicas ficticias para el fold temporal."""
    X, y = make_classification(n_samples=n, n_features=5, random_state=seed)
    split = int(n * 0.75)
    X_tr, y_tr = X[:split], y[:split]
    X_h,  y_h  = X[split:], y[split:]
    base = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=seed)
    base.fit(X_tr, y_tr)
    dates_train = (
        pd.date_range("2024-01-01", periods=split, freq="D")
        .strftime("%Y%m%d").astype(int).to_numpy()
    )
    return base, X_tr, y_tr, X_h, y_h, dates_train


# --- calibrar_modelo ---

def test_calibrar_modelo_devuelve_objeto_con_predict_y_predict_proba():
    base, X_tr, y_tr, X_h, y_h, dates = _dataset()
    cal = calibrar_modelo(base, X_tr, y_tr)
    assert hasattr(cal, 'predict')
    assert hasattr(cal, 'predict_proba')


def test_predict_proba_en_rango_y_suman_uno():
    base, X_tr, y_tr, X_h, y_h, dates = _dataset()
    cal = calibrar_modelo(base, X_tr, y_tr)
    proba = cal.predict_proba(X_h)
    assert proba.shape == (len(X_h), 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


def test_calibracion_no_empeora_log_loss():
    base, X_tr, y_tr, X_h, y_h, dates = _dataset()
    cal = calibrar_modelo(base, X_tr, y_tr)
    ll_base = log_loss(y_h, base.predict_proba(X_h)[:, 1])
    ll_cal  = log_loss(y_h, cal.predict_proba(X_h)[:, 1])
    assert ll_cal <= ll_base + 0.05


def test_calibracion_no_empeora_brier():
    base, X_tr, y_tr, X_h, y_h, dates = _dataset()
    cal = calibrar_modelo(base, X_tr, y_tr)
    br_base = brier_score_loss(y_h, base.predict_proba(X_h)[:, 1])
    br_cal  = brier_score_loss(y_h, cal.predict_proba(X_h)[:, 1])
    assert br_cal <= br_base + 0.05


def test_calibrar_modelo_acepta_dates_sin_error():
    base, X_tr, y_tr, X_h, y_h, dates = _dataset()
    cal = calibrar_modelo(base, X_tr, y_tr, dates=dates)
    assert hasattr(cal, 'predict_proba')


def test_calibrar_modelo_default_sigmoid():
    base, X_tr, y_tr, X_h, y_h, dates = _dataset()
    cal = calibrar_modelo(base, X_tr, y_tr)
    proba = cal.predict_proba(X_h)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


# --- entrenar_modelo (LogReg + GridSearchCV) ---

def test_entrenar_modelo_devuelve_pipeline_y_score():
    base, X_tr, y_tr, X_h, y_h, dates = _dataset()
    modelo, cv_ll, params = entrenar_modelo(X_tr, y_tr, dates=dates)
    assert hasattr(modelo, 'predict_proba')
    assert cv_ll > 0
    assert 'logisticregression__C' in params


def test_entrenar_modelo_es_lineal():
    """El modelo entrenado debe ser un pipeline con LogisticRegression (no árbol)."""
    base, X_tr, y_tr, X_h, y_h, dates = _dataset()
    modelo, _, _ = entrenar_modelo(X_tr, y_tr, dates=dates)
    assert 'logisticregression' in modelo.named_steps
    assert not hasattr(modelo, 'feature_importances_')


def test_entrenar_modelo_sin_dates_usa_fallback():
    base, X_tr, y_tr, X_h, y_h, dates = _dataset()
    modelo, cv_ll, params = entrenar_modelo(X_tr, y_tr, dates=None)
    assert hasattr(modelo, 'predict_proba')


# --- coeficientes_modelo (explicabilidad) ---

def test_coeficientes_modelo_keys_y_estructura():
    base, X_tr, y_tr, X_h, y_h, dates = _dataset()
    feats = [f'f{i}' for i in range(5)]
    pipe = crear_pipeline().fit(X_tr, y_tr)
    coefs = coeficientes_modelo(pipe, feats)
    assert set(coefs.keys()) == set(feats)
    for v in coefs.values():
        assert 'coef' in v and 'odds_ratio' in v
        assert v['odds_ratio'] == pytest.approx(np.exp(v['coef']))


def test_coeficientes_modelo_ordenado_por_magnitud():
    base, X_tr, y_tr, X_h, y_h, dates = _dataset()
    feats = [f'f{i}' for i in range(5)]
    pipe = crear_pipeline().fit(X_tr, y_tr)
    coefs = coeficientes_modelo(pipe, feats)
    magnitudes = [abs(v['coef']) for v in coefs.values()]
    assert magnitudes == sorted(magnitudes, reverse=True)
