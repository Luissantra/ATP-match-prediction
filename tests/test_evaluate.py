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


# --- Tests Q1: bootstrap_ic95 + evaluar_con_ic ---

from src.evaluate import bootstrap_ic95, evaluar_con_ic


def _datos_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    proba = np.clip(y * 0.6 + rng.uniform(-0.2, 0.2, size=200), 0.01, 0.99)
    return y, proba


def test_bootstrap_ic95_keys():
    y, proba = _datos_bootstrap()
    ic = bootstrap_ic95(y, proba, metric='auc', n_iter=200, seed=42)
    assert set(ic.keys()) == {'mean', 'lower', 'upper'}


def test_bootstrap_ic95_bounds():
    y, proba = _datos_bootstrap()
    ic = bootstrap_ic95(y, proba, metric='auc', n_iter=200, seed=42)
    assert 0.5 <= ic['lower'] <= ic['mean'] <= ic['upper'] <= 1.0


def test_bootstrap_ic95_log_loss():
    y, proba = _datos_bootstrap()
    ic = bootstrap_ic95(y, proba, metric='log_loss', n_iter=200, seed=42)
    assert ic['lower'] > 0 and ic['lower'] <= ic['mean'] <= ic['upper']


def test_bootstrap_ic95_brier():
    y, proba = _datos_bootstrap()
    ic = bootstrap_ic95(y, proba, metric='brier', n_iter=200, seed=42)
    assert 0 <= ic['lower'] <= ic['mean'] <= ic['upper'] <= 1.0


def test_bootstrap_ic95_deterministic():
    y, proba = _datos_bootstrap()
    r1 = bootstrap_ic95(y, proba, n_iter=100, seed=7)
    r2 = bootstrap_ic95(y, proba, n_iter=100, seed=7)
    assert r1 == r2


def test_evaluar_con_ic_has_ic_keys():
    X = np.array([[-2.], [-1.], [1.], [2.]])
    y = np.array([0, 0, 1, 1])
    modelo = LogisticRegression().fit(X, y)
    m = evaluar_con_ic(modelo, X, y, n_iter=50, seed=42)
    assert 'auc_ic' in m and 'log_loss_ic' in m and 'brier_ic' in m
    assert set(m['auc_ic'].keys()) == {'mean', 'lower', 'upper'}


# --- Tests Q1: evaluar_baseline_elo ---

import pandas as pd
from src.evaluate import evaluar_baseline_elo


def test_baseline_elo_devuelve_metricas():
    rng = np.random.default_rng(1)
    n = 100
    diff_elo = rng.uniform(-400, 400, size=n)
    # probabilidad ELO cruda
    proba_elo = 1 / (1 + 10 ** (-diff_elo / 400))
    # simular resultados sesgados por el ELO
    y = (rng.uniform(size=n) < proba_elo).astype(int)
    df = pd.DataFrame({'diff_elo_general': diff_elo})
    met = evaluar_baseline_elo(df, y, n_iter=50)
    assert set(met.keys()) >= {'accuracy', 'log_loss', 'brier', 'auc', 'auc_ic', 'log_loss_ic'}
    assert 0.0 < met['log_loss'] < 1.0
    assert 0.5 <= met['auc'] <= 1.0


def test_baseline_elo_requiere_columna():
    import pytest
    df = pd.DataFrame({'other_col': [1, 2, 3]})
    with pytest.raises(KeyError):
        evaluar_baseline_elo(df, [0, 1, 0])


# --- Tests Q3: diagnosticar_gap_cv_test ---

from src.evaluate import diagnosticar_gap_cv_test


def test_diagnosticar_gap_devuelve_string():
    msg = diagnosticar_gap_cv_test(cv_best_score=0.620, test_log_loss=0.683, n_test=137)
    assert isinstance(msg, str)
    assert len(msg) > 50


def test_diagnosticar_gap_no_dice_confirmado():
    msg = diagnosticar_gap_cv_test(cv_best_score=0.620, test_log_loss=0.683, n_test=137)
    assert 'confirmado' not in msg.lower()
