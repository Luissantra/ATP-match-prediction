import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.elo import calcular_elos_historicos


def _make_df():
    """Tres partidos: A vence a B dos veces, B vence a C una vez."""
    return pd.DataFrame([
        {'tourney_date': 20240101, 'match_num': 1, 'winner_name': 'A', 'loser_name': 'B',
         'surface': 'Hard', 'winner_rank': 10, 'loser_rank': 20,
         'winner_age': 25.0, 'loser_age': 27.0, 'tourney_level': 'G'},
        {'tourney_date': 20240102, 'match_num': 1, 'winner_name': 'A', 'loser_name': 'B',
         'surface': 'Clay', 'winner_rank': 10, 'loser_rank': 20,
         'winner_age': 25.0, 'loser_age': 27.0, 'tourney_level': 'M'},
        {'tourney_date': 20240103, 'match_num': 1, 'winner_name': 'B', 'loser_name': 'C',
         'surface': 'Grass', 'winner_rank': 20, 'loser_rank': 50,
         'winner_age': 27.0, 'loser_age': 30.0, 'tourney_level': '250'},
    ])


def test_new_columns_exist(tmp_path):
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, *_ = calcular_elos_historicos(str(tmp_path), [2024])
    assert 'h2h_winner_ratio' in result.columns
    assert 'h2h_loser_ratio' in result.columns
    assert 'form_winner' in result.columns
    assert 'form_loser' in result.columns


def test_h2h_default_when_no_history(tmp_path):
    """Primer partido entre A y B: H2H debe ser 0.5 para ambos."""
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, *_ = calcular_elos_historicos(str(tmp_path), [2024])
    assert result.iloc[0]['h2h_winner_ratio'] == 0.5
    assert result.iloc[0]['h2h_loser_ratio'] == 0.5


def test_h2h_updates_after_first_match(tmp_path):
    """Segundo partido A vs B: A ganó el primero, ratio debe ser 1.0 para A."""
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, *_ = calcular_elos_historicos(str(tmp_path), [2024])
    assert result.iloc[1]['h2h_winner_ratio'] == 1.0
    assert result.iloc[1]['h2h_loser_ratio'] == 0.0


def test_form_default_when_no_history(tmp_path):
    """Primer partido de A: forma debe ser 0.5."""
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, *_ = calcular_elos_historicos(str(tmp_path), [2024])
    assert result.iloc[0]['form_winner'] == 0.5
    assert result.iloc[0]['form_loser'] == 0.5


def test_form_updates_after_matches(tmp_path):
    """Tercer partido (B vs C): B perdió 2 de 2 previos → form=0.0. C debuta → 0.5."""
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, *_ = calcular_elos_historicos(str(tmp_path), [2024])
    assert result.iloc[2]['form_winner'] == 0.0
    assert result.iloc[2]['form_loser'] == 0.5


def test_ratios_in_valid_range(tmp_path):
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, *_ = calcular_elos_historicos(str(tmp_path), [2024])
    for col in ['h2h_winner_ratio', 'h2h_loser_ratio', 'form_winner', 'form_loser']:
        assert result[col].between(0.0, 1.0).all(), f"{col} out of range"


def test_devuelve_h2h_y_form_finales(tmp_path):
    """Para inferencia: el motor debe exportar el estado final de H2H y forma."""
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    _, _, _, h2h, form_final = calcular_elos_historicos(str(tmp_path), [2024])

    # H2H final: A ganó 2 a B, B ganó 1 a C
    assert h2h[('A', 'B')] == {'A': 2, 'B': 0}
    assert h2h[('B', 'C')] == {'B': 1, 'C': 0}

    # Forma final (media de últimos resultados): A=[1,1]=1.0, B=[0,0,1]≈0.333, C=[0]=0.0
    assert form_final['A'] == pytest.approx(1.0)
    assert form_final['B'] == pytest.approx(1 / 3)
    assert form_final['C'] == pytest.approx(0.0)
