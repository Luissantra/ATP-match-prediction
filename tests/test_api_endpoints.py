"""
Tests para los endpoints del modelo único:
  GET /api/model     — métricas + coeficientes (explicabilidad)
  GET /api/predict   — predicción + validaciones
  GET /api/players
"""
import sys
import os
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import app


def _mock_modelo():
    """LogReg trivial que devuelve probabilidades válidas. 5 features = len(FEATURES)."""
    m = LogisticRegression()
    m.fit([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], [0, 1])
    return m


def _mock_metrics():
    return {'accuracy': 0.65, 'log_loss': 0.62, 'brier': 0.22, 'auc': 0.71}


def _mock_coefs():
    return {'diff_elo_general': {'coef': 0.5, 'odds_ratio': 1.65}}


@pytest.fixture(autouse=True)
def _estado(monkeypatch):
    monkeypatch.setattr(app, 'elo_general', {'A': 1600.0, 'B': 1500.0})
    monkeypatch.setattr(app, 'elo_superficie', {'Hard': {'A': 1700.0, 'B': 1500.0}})
    monkeypatch.setattr(app, 'stats_jugadores', {
        'A': {'rank': 5.0,  'age': 24.0},
        'B': {'rank': 20.0, 'age': 30.0},
    })
    monkeypatch.setattr(app, 'modelo', _mock_modelo())
    monkeypatch.setattr(app, 'metrics', _mock_metrics())
    monkeypatch.setattr(app, 'coeficientes', _mock_coefs())


@pytest.fixture
def client():
    app.app.config['TESTING'] = True
    with app.app.test_client() as c:
        yield c


# ── GET /api/model ───────────────────────────────────────────────────────────

def test_api_model_devuelve_200(client):
    assert client.get('/api/model').status_code == 200


def test_api_model_contiene_metricas_y_coeficientes(client):
    data = client.get('/api/model').get_json()
    assert data['nombre'] == 'logreg'
    assert set(data['metrics'].keys()) == {'accuracy', 'log_loss', 'brier', 'auc'}
    assert 'diff_elo_general' in data['coeficientes']


# ── GET /api/predict ─────────────────────────────────────────────────────────

def test_predict_devuelve_200(client):
    r = client.get('/api/predict?player_a=A&player_b=B&surface=Hard')
    assert r.status_code == 200
    data = r.get_json()
    assert 'player_a' in data and 'player_b' in data
    assert data['model_used'] == 'logreg'


def test_predict_probas_suman_100(client):
    data = client.get('/api/predict?player_a=A&player_b=B&surface=Hard').get_json()
    total = data['player_a']['prob_victory'] + data['player_b']['prob_victory']
    assert abs(total - 100.0) < 0.2


def test_predict_features_debug_tiene_5_features(client):
    data = client.get('/api/predict?player_a=A&player_b=B&surface=Hard').get_json()
    assert set(data['features_debug'].keys()) == {
        'diff_elo_general', 'diff_elo_sup', 'diff_rank', 'is_unranked', 'diff_age',
    }


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
    data = client.get('/api/predict?player_a=A&player_b=Desconocido&surface=Hard').get_json()
    assert data['player_b']['unknown'] is True
    assert data['player_a']['unknown'] is False


def test_predict_ambos_conocidos_no_tienen_unknown(client):
    data = client.get('/api/predict?player_a=A&player_b=B&surface=Hard').get_json()
    assert data['player_a']['unknown'] is False
    assert data['player_b']['unknown'] is False
