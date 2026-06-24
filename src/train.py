from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def entrenar_modelo(X_train, y_train, param_grid=None):
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 150],
        }

    # El producto es la probabilidad de victoria → optimizamos log-loss, no accuracy.
    # accuracy elige hiperparámetros que aciertan el binario pero calibran mal la proba.
    grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=5),
        scoring='neg_log_loss',
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor CV log-loss: {-grid_search.best_score_:.4f}")

    return grid_search.best_estimator_
