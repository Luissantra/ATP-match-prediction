import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import app
from src.features import FEATURES, DEFAULT_LEVEL_NUM, RANK_CAP


@pytest.fixture(autouse=True)
def _stub_state(monkeypatch):
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


def test_diff_elo_general():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    # A: elo_general=1600 ; B: elo_general=1500
    assert feat['diff_elo_general'] == pytest.approx(100.0)


def test_diff_elo_sup():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    # A: elo_sup_Hard=1700 ; B: elo_sup_Hard=1500
    assert feat['diff_elo_sup'] == pytest.approx(200.0)


def test_diff_rank_usa_rank_real():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    # A rank=5, B rank=20 → diff = 5-20 = -15
    assert feat['diff_rank'] == pytest.approx(-15.0)


def test_is_unranked_cero_cuando_ambos_rankeados():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    assert feat['is_unranked'] == 0


def test_diff_h2h_usa_historial_real_no_cero():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    assert feat['diff_h2h'] == pytest.approx(0.5)


def test_diff_form_usa_forma_real_no_cero():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    assert feat['diff_form'] == pytest.approx(0.8 - 0.4)


def test_tourney_level_se_mapea_desde_parametro():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    assert feat['tourney_level_num'] == 5


def test_tourney_level_desconocido_usa_default_no_3():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level=None)
    assert feat['tourney_level_num'] == DEFAULT_LEVEL_NUM


def test_jugadores_desconocidos_son_neutros():
    feat = app.construir_features('X', 'Z', 'Hard', tourney_level='G')
    assert feat['diff_elo_general'] == pytest.approx(0.0)
    assert feat['diff_elo_sup'] == pytest.approx(0.0)
    assert feat['diff_h2h'] == pytest.approx(0.0)
    assert feat['diff_form'] == pytest.approx(0.0)
