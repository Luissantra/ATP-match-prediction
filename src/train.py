from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from src.cv import purged_time_series_splits


def entrenar_modelo(X_train, y_train, param_grid=None, dates=None, embargo_days=7):
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 150],
        }

    # CV temporal con embargo: rompe la fuga blanda en la frontera train/val
    # (partidos contiguos comparten estado ELO casi idéntico). Sin fechas → fallback.
    if dates is not None:
        cv = list(purged_time_series_splits(dates, n_splits=5, embargo_days=embargo_days))
    else:
        cv = TimeSeriesSplit(n_splits=5)

    # El producto es la probabilidad de victoria → optimizamos log-loss, no accuracy.
    # accuracy elige hiperparámetros que aciertan el binario pero calibran mal la proba.
    grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring='neg_log_loss',
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor CV log-loss: {-grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def calibrar_modelo(modelo_base, X, y, dates=None, embargo_days=7, method="isotonic"):
    """
    Envuelve modelo_base en CalibratedClassifierCV usando fold temporal para evitar
    leakage. Si dates viene, usa purged_time_series_splits (mismo CV que entrenamiento).
    """
    if dates is not None:
        cv = list(purged_time_series_splits(dates, n_splits=5, embargo_days=embargo_days))
    else:
        cv = 5

    calibrado = CalibratedClassifierCV(estimator=clone(modelo_base), method=method, cv=cv)
    calibrado.fit(X, y)
    return calibrado


def comparar_calibracion(modelo_base, X, y, dates=None, embargo_days=7):
    """
    Compara sigmoid (Platt) vs isotonic por log-loss en CV temporal purgado.
    Isotonic sobreajusta en folds pequeños (n<500); sigmoid más estable.
    Devuelve {'sigmoid_log_loss', 'isotonic_log_loss', 'mejor'}.
    """
    from sklearn.metrics import log_loss as _log_loss
    resultados = {}
    for method in ('sigmoid', 'isotonic'):
        cal = calibrar_modelo(modelo_base, X, y,
                              dates=dates, embargo_days=embargo_days,
                              method=method)
        proba = cal.predict_proba(X)[:, 1]
        resultados[f'{method}_log_loss'] = _log_loss(y, proba, labels=[0, 1])
    mejor = 'sigmoid' if resultados['sigmoid_log_loss'] <= resultados['isotonic_log_loss'] else 'isotonic'
    resultados['mejor'] = mejor
    return resultados


_DEFINICIONES_MODELOS = [
    (
        "logreg",
        LogisticRegression(max_iter=1000, random_state=42),
        {'C': [0.01, 0.1, 1.0, 10.0]},
    ),
    (
        "randomforest",
        RandomForestClassifier(random_state=42),
        {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]},
    ),
    (
        "gbm",
        GradientBoostingClassifier(random_state=42),
        {'max_depth': [3, 4, 5], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [100, 150]},
    ),
    (
        "xgboost",
        XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
        {'n_estimators': [100, 150], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
    ),
]


def entrenar_todos_los_modelos(X, y, dates=None, embargo_days=7):
    """
    Entrena LogReg baseline, RandomForest, GBM y XGBoost con GridSearchCV (neg_log_loss)
    + CV temporal purgado. Cada modelo se calibra con CalibratedClassifierCV(isotonic).

    Returns
    -------
    (modelos_calibrados, base_estimators) — dicts {nombre: modelo}.
    base_estimators permite graficar importancia sin recalcular el grid search.
    """
    if dates is not None:
        cv = list(purged_time_series_splits(dates, n_splits=5, embargo_days=embargo_days))
    else:
        cv = TimeSeriesSplit(n_splits=5)

    modelos_calibrados = {}
    base_estimators = {}
    cv_scores = {}
    for nombre, estimador, param_grid in _DEFINICIONES_MODELOS:
        print(f"\n  [{nombre}] GridSearchCV...")
        gs = GridSearchCV(
            estimator=estimador,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_log_loss',
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X, y)
        cv_log_loss = -gs.best_score_
        print(f"    best_params={gs.best_params_}  cv_log_loss={cv_log_loss:.4f}")
        base_estimators[nombre] = gs.best_estimator_
        cv_scores[nombre] = cv_log_loss
        modelos_calibrados[nombre] = calibrar_modelo(gs.best_estimator_, X, y,
                                                     dates=dates, embargo_days=embargo_days)

    return modelos_calibrados, base_estimators, cv_scores
