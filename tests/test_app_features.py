import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import app
from src.features import FEATURES, RANK_CAP


@pytest.fixture(autouse=True)
def _stub_state(monkeypatch):
    monkeypatch.setattr(app, 'elo_general', {'A': 1600.0, 'B': 1500.0})
    monkeypatch.setattr(app, 'elo_superficie', {'Hard': {'A': 1700.0, 'B': 1500.0}})
    monkeypatch.setattr(app, 'stats_jugadores', {
        'A': {'rank': 5.0, 'age': 24.0},
        'B': {'rank': 20.0, 'age': 30.0},
    })


def test_devuelve_todas_las_features():
    feat = app.construir_features('A', 'B', 'Hard')
    assert set(feat.keys()) == set(FEATURES)


def test_diff_elo_general():
    feat = app.construir_features('A', 'B', 'Hard')
    # A: elo_general=1600 ; B: elo_general=1500
    assert feat['diff_elo_general'] == pytest.approx(100.0)


def test_diff_elo_sup():
    feat = app.construir_features('A', 'B', 'Hard')
    # A: elo_sup_Hard=1700 ; B: elo_sup_Hard=1500
    assert feat['diff_elo_sup'] == pytest.approx(200.0)


def test_diff_rank_usa_rank_real():
    feat = app.construir_features('A', 'B', 'Hard')
    # A rank=5, B rank=20 → diff = 5-20 = -15
    assert feat['diff_rank'] == pytest.approx(-15.0)


def test_diff_age():
    feat = app.construir_features('A', 'B', 'Hard')
    # A age=24, B age=30 → diff = -6
    assert feat['diff_age'] == pytest.approx(-6.0)


def test_is_unranked_cero_cuando_ambos_rankeados():
    feat = app.construir_features('A', 'B', 'Hard')
    assert feat['is_unranked'] == 0


def test_is_unranked_detecta_jugador_desconocido():
    # 'X' no está en stats_jugadores → unranked; 'A' sí → no unranked
    feat = app.construir_features('X', 'A', 'Hard')
    assert feat['is_unranked'] == 1


def test_jugadores_desconocidos_son_neutros():
    feat = app.construir_features('X', 'Z', 'Hard')
    assert feat['diff_elo_general'] == pytest.approx(0.0)
    assert feat['diff_elo_sup'] == pytest.approx(0.0)
    # ambos desconocidos → is_unranked se cancela
    assert feat['is_unranked'] == 0
