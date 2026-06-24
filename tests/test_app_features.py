import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import app
from src.features import FEATURES, DEFAULT_LEVEL_NUM


@pytest.fixture(autouse=True)
def _stub_state(monkeypatch):
    """Inyecta estado en memoria sin necesidad de los .pkl reales."""
    monkeypatch.setattr(app, 'elo_general', {'A': 1600.0, 'B': 1500.0})
    monkeypatch.setattr(app, 'elo_superficie', {'Hard': {'A': 1700.0, 'B': 1500.0}})
    monkeypatch.setattr(app, 'stats_jugadores', {
        'A': {'rank': 5.0, 'age': 24.0},
        'B': {'rank': 20.0, 'age': 30.0},
    })
    monkeypatch.setattr(app, 'h2h', {('A', 'B'): {'A': 3, 'B': 1}})
    monkeypatch.setattr(app, 'form_final', {'A': 0.8, 'B': 0.4})


def test_devuelve_todas_las_features():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    assert set(feat.keys()) == set(FEATURES)


def test_diff_elo_usa_elo_hibrido():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    # A: 0.5*1600 + 0.5*1700 = 1650 ; B: 0.5*1500 + 0.5*1500 = 1500
    assert feat['diff_elo'] == pytest.approx(150.0)


def test_diff_h2h_usa_historial_real_no_cero():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    # A ganó 3 de 4 → 0.75 ; B 0.25 → diff = 0.5 (antes estaba hardcoded a 0.0)
    assert feat['diff_h2h'] == pytest.approx(0.5)


def test_diff_form_usa_forma_real_no_cero():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    assert feat['diff_form'] == pytest.approx(0.8 - 0.4)


def test_tourney_level_se_mapea_desde_parametro():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    assert feat['tourney_level_num'] == 5  # Grand Slam


def test_tourney_level_desconocido_usa_default_no_3():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level=None)
    assert feat['tourney_level_num'] == DEFAULT_LEVEL_NUM  # 1, no el viejo 3


def test_jugadores_desconocidos_son_neutros():
    feat = app.construir_features('X', 'Z', 'Hard', tourney_level='G')  # ninguno existe
    assert feat['diff_elo'] == pytest.approx(0.0)
    assert feat['diff_h2h'] == pytest.approx(0.0)
    assert feat['diff_form'] == pytest.approx(0.0)
