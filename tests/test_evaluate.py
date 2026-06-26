import sys
import os
import tempfile

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.evaluate import evaluar, graficar_reliability_diagram, graficar_histograma_probas


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


def test_reliability_diagram_crea_archivo():
    modelo, X, y = _modelo_perfecto()
    orig_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            graficar_reliability_diagram(modelo, X, y)
            assert os.path.exists("plots/reliability_diagram.png")
        finally:
            os.chdir(orig_dir)


def test_reliability_diagram_acepta_n_bins():
    modelo, X, y = _modelo_perfecto()
    orig_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            graficar_reliability_diagram(modelo, X, y, n_bins=5)
            assert os.path.exists("plots/reliability_diagram.png")
        finally:
            os.chdir(orig_dir)


def test_histograma_probas_crea_archivo():
    modelo, X, y = _modelo_perfecto()
    orig_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            graficar_histograma_probas(modelo, X, y)
            assert os.path.exists("plots/histograma_probas.png")
        finally:
            os.chdir(orig_dir)
