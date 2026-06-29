import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import draw


def _df_multi():
    """DataFrame con torneos de distintos niveles."""
    return pd.DataFrame([
        {'tourney_id': '2026-580', 'tourney_name': 'Wimbledon',   'surface': 'Grass', 'tourney_level': 'G',   'winner_name': 'A', 'loser_name': 'B'},
        {'tourney_id': '2026-580', 'tourney_name': 'Wimbledon',   'surface': 'Grass', 'tourney_level': 'G',   'winner_name': 'C', 'loser_name': 'D'},
        {'tourney_id': '2026-422', 'tourney_name': 'Halle',       'surface': 'Grass', 'tourney_level': '250', 'winner_name': 'E', 'loser_name': 'F'},
        {'tourney_id': '2026-500', 'tourney_name': 'Queen Club',  'surface': 'Grass', 'tourney_level': '500', 'winner_name': 'G', 'loser_name': 'H'},
        {'tourney_id': '2026-999', 'tourney_name': 'Davis Cup',   'surface': 'Hard',  'tourney_level': 'D',   'winner_name': 'I', 'loser_name': 'J'},
    ])


def test_listar_torneos_orden_prioridad():
    """GS > 500 > 250; Davis Cup excluido."""
    result = draw.listar_torneos(_df_multi())
    names = [t['name'] for t in result]
    assert names[0] == 'Wimbledon'
    assert names[1] == 'Queen Club'
    assert names[2] == 'Halle'
    assert 'Davis Cup' not in names


def test_listar_torneos_campos():
    """Cada torneo tiene los campos requeridos."""
    result = draw.listar_torneos(_df_multi())
    for t in result:
        assert set(t.keys()) >= {'name', 'surface', 'level', 'draw_size', 'tourney_id'}


def test_listar_torneos_draw_size():
    """draw_size es el número de jugadores únicos del torneo."""
    result = draw.listar_torneos(_df_multi())
    wimbledon = next(t for t in result if t['name'] == 'Wimbledon')
    assert wimbledon['draw_size'] == 4  # A, B, C, D


def test_listar_torneos_vacio():
    """DataFrame vacío devuelve lista vacía."""
    assert draw.listar_torneos(pd.DataFrame()) == []


def test_listar_torneos_solo_davis_cup():
    """Si solo hay Davis Cup, resultado es lista vacía."""
    df = pd.DataFrame([
        {'tourney_id': '2026-999', 'tourney_name': 'Davis Cup', 'surface': 'Hard',
         'tourney_level': 'D', 'winner_name': 'A', 'loser_name': 'B'},
    ])
    assert draw.listar_torneos(df) == []


def test_descargar_ongoing_timeout(monkeypatch):
    """Si requests lanza Timeout, descargar_ongoing lanza RuntimeError."""
    import requests as req_lib

    def mock_get(*args, **kwargs):
        raise req_lib.exceptions.Timeout("timeout simulado")

    monkeypatch.setattr(req_lib, 'get', mock_get)
    with pytest.raises(RuntimeError, match="No se pudo descargar"):
        draw.descargar_ongoing()


def test_descargar_ongoing_http_error(monkeypatch):
    """Si requests devuelve 500, descargar_ongoing lanza RuntimeError."""
    import requests as req_lib

    class MockResponse:
        status_code = 500
        def raise_for_status(self):
            raise req_lib.exceptions.HTTPError("500 Server Error")

    monkeypatch.setattr(req_lib, 'get', lambda *a, **kw: MockResponse())
    with pytest.raises(RuntimeError, match="No se pudo descargar"):
        draw.descargar_ongoing()
