from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

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
