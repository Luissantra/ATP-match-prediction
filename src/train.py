"""
Entrenamiento del modelo único: Regresión Logística calibrada.
==============================================================

Tras el estudio de feature importance + ablación (test ciego 2025, n=2861), un
modelo lineal (LogReg) iguala a GBM/RF/XGBoost en AUC y log-loss: toda la señal es
lineal en las diferencias de ELO/rank. Se elige LogReg por ser el óptimo de
rigor + explicabilidad (coeficientes = odds-ratio interpretables) + minimalismo.
El multi-modelo y el ensemble se retiran: no aportaban señal medible (IC95% ±0.009).
"""

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.cv import purged_time_series_splits


def _build_cv(dates, embargo_days, fallback):
    """CV temporal con embargo si hay fechas; si no, fallback (TimeSeriesSplit o int)."""
    if dates is not None:
        return list(purged_time_series_splits(dates, n_splits=5, embargo_days=embargo_days))
    return fallback


def crear_pipeline():
    """LogReg estandarizada. El StandardScaler deja los coeficientes en unidades de
    desviación estándar → comparables entre features de escalas distintas (ELO vs rank)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )


def entrenar_modelo(X, y, dates=None, embargo_days=7, param_grid=None):
    """
    Entrena LogReg con GridSearchCV(neg_log_loss) + CV temporal purgado.

    Returns
    -------
    (best_pipeline, cv_log_loss, best_params)
    """
    if param_grid is None:
        param_grid = {'logisticregression__C': [0.01, 0.1, 1.0, 10.0]}
    cv = _build_cv(dates, embargo_days, TimeSeriesSplit(n_splits=5))

    # El producto es la probabilidad de victoria → optimizamos log-loss, no accuracy.
    gs = GridSearchCV(
        estimator=crear_pipeline(),
        param_grid=param_grid,
        cv=cv,
        scoring='neg_log_loss',
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X, y)
    return gs.best_estimator_, -gs.best_score_, gs.best_params_


def calibrar_modelo(modelo_base, X, y, dates=None, embargo_days=7, method="sigmoid"):
    """
    Calibra con CV temporal purgado (mismo CV que entrenamiento, sin leakage).
    Para n grande + modelo lineal, sigmoid (Platt) es estable; isotonic sobreajusta
    escalones en folds pequeños y mete varianza.
    """
    cv = _build_cv(dates, embargo_days, 5)
    calibrado = CalibratedClassifierCV(estimator=clone(modelo_base), method=method, cv=cv)
    calibrado.fit(X, y)
    return calibrado


def coeficientes_modelo(pipeline, features):
    """
    Explicabilidad del modelo: coeficiente y odds-ratio por feature.

    El pipeline estandariza, así que cada coeficiente es el efecto de mover esa feature
    +1 desviación estándar sobre el log-odds de victoria de A. odds_ratio = exp(coef):
    >1 favorece a A, <1 favorece a B. Devuelto ordenado por |coef| descendente.

    Returns
    -------
    dict {feature: {'coef': float, 'odds_ratio': float}}
    """
    lr = pipeline.named_steps['logisticregression']
    coefs = lr.coef_[0]
    out = {
        feat: {'coef': float(c), 'odds_ratio': float(np.exp(c))}
        for feat, c in zip(features, coefs)
    }
    return dict(sorted(out.items(), key=lambda kv: abs(kv[1]['coef']), reverse=True))
