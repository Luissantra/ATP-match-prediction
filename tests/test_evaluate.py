import sys
import os

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.evaluate import evaluar


def _modelo_perfecto():
    """Modelo entrenado sobre datos linealmente separables → predice perfecto."""
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    modelo = LogisticRegression().fit(X, y)
    return modelo, X, y


def test_devuelve_las_cuatro_metricas():
    modelo, X, y = _modelo_perfecto()
    m = evaluar(modelo, X, y)
    assert set(m.keys()) == {'accuracy', 'log_loss', 'brier', 'auc'}


def test_accuracy_perfecta_en_datos_separables():
    modelo, X, y = _modelo_perfecto()
    assert evaluar(modelo, X, y)['accuracy'] == pytest.approx(1.0)


def test_auc_perfecta_en_datos_separables():
    modelo, X, y = _modelo_perfecto()
    assert evaluar(modelo, X, y)['auc'] == pytest.approx(1.0)


def test_brier_en_rango_valido():
    modelo, X, y = _modelo_perfecto()
    brier = evaluar(modelo, X, y)['brier']
    assert 0.0 <= brier <= 1.0


def test_log_loss_no_negativo():
    modelo, X, y = _modelo_perfecto()
    assert evaluar(modelo, X, y)['log_loss'] >= 0.0
