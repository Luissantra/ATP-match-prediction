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
