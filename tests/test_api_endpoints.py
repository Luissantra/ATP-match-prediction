"""
Tests para los endpoints multi-modelo de E3:
  GET /api/models
  GET /api/predict?model=<nombre>
  GET /api/predict_all
"""
import sys
import os
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import app


NOMBRES_MODELOS = {"logreg", "randomforest", "gbm", "xgboost"}


def _mock_modelo():
    """LogReg trivial que devuelve probabilidades válidas. 8 features = len(FEATURES)."""
    m = LogisticRegression()
    m.fit([[0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]], [0, 1])
    return m


def _mock_todos_modelos():
    return {nombre: _mock_modelo() for nombre in NOMBRES_MODELOS}


def _mock_metrics():
    return {
        nombre: {'accuracy': 0.57, 'log_loss': 0.68, 'brier': 0.24, 'auc': 0.61}
        for nombre in NOMBRES_MODELOS
    }


@pytest.fixture(autouse=True)
def _estado(monkeypatch):
    monkeypatch.setattr(app, 'elo_general', {'A': 1600.0, 'B': 1500.0})
    monkeypatch.setattr(app, 'elo_superficie', {'Hard': {'A': 1700.0, 'B': 1500.0}})
    monkeypatch.setattr(app, 'stats_jugadores', {
        'A': {'rank': 5.0,  'age': 24.0},
        'B': {'rank': 20.0, 'age': 30.0},
    })
    monkeypatch.setattr(app, 'h2h', {('A', 'B'): {'A': 3, 'B': 1}})
    monkeypatch.setattr(app, 'form_final', {'A': 0.8, 'B': 0.4})
    monkeypatch.setattr(app, 'modelo', _mock_modelo())
    monkeypatch.setattr(app, 'todos_modelos', _mock_todos_modelos())
    monkeypatch.setattr(app, 'metrics_todos', _mock_metrics())


@pytest.fixture
def client():
    app.app.config['TESTING'] = True
    with app.app.test_client() as c:
        yield c


# ── GET /api/models ──────────────────────────────────────────────────────────

def test_api_models_devuelve_200(client):
    r = client.get('/api/models')
    assert r.status_code == 200


def test_api_models_devuelve_los_cuatro_modelos(client):
    data = client.get('/api/models').get_json()
    nombres = {m['nombre'] for m in data}
    assert nombres == NOMBRES_MODELOS


def test_api_models_contiene_metricas_requeridas(client):
    data = client.get('/api/models').get_json()
    for m in data:
        assert 'log_loss' in m
        assert 'brier' in m
        assert 'auc' in m
        assert 'accuracy' in m


def test_api_models_ordenado_por_log_loss(client):
    data = client.get('/api/models').get_json()
    losses = [m['log_loss'] for m in data]
    assert losses == sorted(losses)


# ── GET /api/predict?model= ──────────────────────────────────────────────────

def test_predict_sin_model_param_usa_gbm(client):
    r = client.get('/api/predict?player_a=A&player_b=B&surface=Hard')
    assert r.status_code == 200
    data = r.get_json()
    assert 'player_a' in data and 'player_b' in data


def test_predict_con_model_logreg(client):
    r = client.get('/api/predict?player_a=A&player_b=B&surface=Hard&model=logreg')
    assert r.status_code == 200
    data = r.get_json()
    assert data['model_used'] == 'logreg'


def test_predict_con_model_invalido_devuelve_400(client):
    r = client.get('/api/predict?player_a=A&player_b=B&surface=Hard&model=inexistente')
    assert r.status_code == 400


def test_predict_probas_suman_100_con_model(client):
    for nombre in NOMBRES_MODELOS:
        r = client.get(f'/api/predict?player_a=A&player_b=B&surface=Hard&model={nombre}')
        assert r.status_code == 200
        data = r.get_json()
        total = data['player_a']['prob_victory'] + data['player_b']['prob_victory']
        assert abs(total - 100.0) < 0.2, f"{nombre}: probas no suman 100 ({total})"


# ── GET /api/predict_all ─────────────────────────────────────────────────────

def test_predict_all_devuelve_200(client):
    r = client.get('/api/predict_all?player_a=A&player_b=B&surface=Hard')
    assert r.status_code == 200


def test_predict_all_contiene_todos_los_modelos(client):
    data = client.get('/api/predict_all?player_a=A&player_b=B&surface=Hard').get_json()
    assert set(data['predictions'].keys()) == NOMBRES_MODELOS


def test_predict_all_cada_modelo_tiene_prob_victory(client):
    data = client.get('/api/predict_all?player_a=A&player_b=B&surface=Hard').get_json()
    for nombre, pred in data['predictions'].items():
        assert 'prob_a' in pred, f"{nombre} sin prob_a"
        assert 'prob_b' in pred, f"{nombre} sin prob_b"
        assert 0.0 <= pred['prob_a'] <= 100.0
        assert 0.0 <= pred['prob_b'] <= 100.0


def test_predict_all_requiere_player_a_y_b(client):
    r = client.get('/api/predict_all?player_a=A&surface=Hard')
    assert r.status_code == 400


def test_predict_all_rechaza_superficie_invalida(client):
    r = client.get('/api/predict_all?player_a=A&player_b=B&surface=Tierra')
    assert r.status_code == 400


# ── GET /api/predict — validaciones básicas (G3) ────────────────────────────

def test_predict_superficie_invalida_devuelve_400(client):
    r = client.get('/api/predict?player_a=A&player_b=B&surface=Tierra')
    assert r.status_code == 400


def test_predict_sin_player_a_devuelve_400(client):
    r = client.get('/api/predict?player_b=B&surface=Hard')
    assert r.status_code == 400


def test_predict_sin_player_b_devuelve_400(client):
    r = client.get('/api/predict?player_a=A&surface=Hard')
    assert r.status_code == 400


def test_predict_mismo_jugador_devuelve_400(client):
    r = client.get('/api/predict?player_a=A&player_b=A&surface=Hard')
    assert r.status_code == 400


def test_predict_jugador_desconocido_devuelve_200_con_defaults(client):
    r = client.get('/api/predict?player_a=A&player_b=Desconocido&surface=Hard')
    assert r.status_code == 200
    data = r.get_json()
    assert data['player_b']['name'] == 'Desconocido'
    assert data['player_b']['rank'] == 'Sin Ranking'


def test_predict_jugador_desconocido_tiene_unknown_true(client):
    r = client.get('/api/predict?player_a=A&player_b=Desconocido&surface=Hard')
    data = r.get_json()
    assert data['player_b']['unknown'] is True
    assert data['player_a']['unknown'] is False


def test_predict_ambos_conocidos_no_tienen_unknown(client):
    r = client.get('/api/predict?player_a=A&player_b=B&surface=Hard')
    data = r.get_json()
    assert data['player_a']['unknown'] is False
    assert data['player_b']['unknown'] is False
