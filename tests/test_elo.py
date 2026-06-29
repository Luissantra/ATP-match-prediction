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


def test_devuelve_tres_valores(tmp_path):
    """El motor devuelve (df, elo_general, elo_superficie)."""
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    resultado = calcular_elos_historicos(str(tmp_path), [2024])
    assert len(resultado) == 3
    _, elo_general, elo_superficie = resultado
    assert isinstance(elo_general, dict)
    assert set(elo_superficie.keys()) == {'Clay', 'Grass', 'Hard'}


def test_no_emite_columnas_h2h_form(tmp_path):
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, *_ = calcular_elos_historicos(str(tmp_path), [2024])
    for col in ['h2h_winner_ratio', 'h2h_loser_ratio', 'form_winner', 'form_loser']:
        assert col not in result.columns


def test_columnas_elo_separado_existen(tmp_path):
    """Tras I3, el df debe exponer ELO general y de superficie por separado."""
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, *_ = calcular_elos_historicos(str(tmp_path), [2024])
    for col in ['elo_winner_general', 'elo_loser_general',
                'elo_winner_sup', 'elo_loser_sup']:
        assert col in result.columns, f"Falta columna {col}"


def test_elo_general_y_sup_son_distintos(tmp_path):
    """Los ratings general y de superficie divergen a medida que se juegan partidos."""
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, *_ = calcular_elos_historicos(str(tmp_path), [2024])
    # En el segundo partido (Clay), A tiene historial en Hard pero no en Clay:
    # general habrá cambiado, superficie Clay empieza en 1500
    row1 = result.iloc[1]
    assert row1['elo_winner_general'] != row1['elo_winner_sup']


def test_csv_con_columna_faltante_lanza_error_claro(tmp_path):
    """CSV sin 'winner_name' debe lanzar ValueError descriptivo, no KeyError crudo."""
    df = _make_df().drop(columns=['winner_name'])
    df.to_csv(tmp_path / "2024.csv", index=False)
    with pytest.raises(ValueError, match="winner_name"):
        calcular_elos_historicos(str(tmp_path), [2024])


def test_csv_con_varias_columnas_faltantes_lista_todas(tmp_path):
    """El error debe mencionar todas las columnas ausentes."""
    df = _make_df().drop(columns=['winner_name', 'loser_name'])
    df.to_csv(tmp_path / "2024.csv", index=False)
    with pytest.raises(ValueError, match="winner_name") as exc_info:
        calcular_elos_historicos(str(tmp_path), [2024])
    assert "loser_name" in str(exc_info.value)


def test_mov_sube_elo_mas_en_straight_sets(tmp_path):
    """2-0 debe mover el ELO más que 2-1."""
    df_straight = pd.DataFrame([{
        'tourney_date': 20240101, 'match_num': 1,
        'winner_name': 'A', 'loser_name': 'B', 'surface': 'Hard',
        'winner_rank': 10, 'loser_rank': 20, 'winner_age': 25.0, 'loser_age': 27.0,
        'tourney_level': '250', 'score': '6-3 6-2',
    }])
    df_deciding = df_straight.copy()
    df_deciding.loc[0, 'score'] = '4-6 7-5 6-3'

    df_straight.to_csv(tmp_path / "2024.csv", index=False)
    _, elo_s, *_ = calcular_elos_historicos(str(tmp_path), [2024], use_mov=True)
    (tmp_path / "2024.csv").write_text(df_deciding.to_csv(index=False))
    _, elo_d, *_ = calcular_elos_historicos(str(tmp_path), [2024], use_mov=True)
    # Ganador debe tener ELO mayor en straight sets
    assert elo_s['A'] > elo_d['A']


def test_k_schedule_debutante_mayor_cambio(tmp_path):
    """Debutante con K=48 debe cambiar más el ELO que un veterano con K=32."""
    df = pd.DataFrame([{
        'tourney_date': 20240101, 'match_num': 1,
        'winner_name': 'NUEVO', 'loser_name': 'NUEVO2', 'surface': 'Hard',
        'winner_rank': 100, 'loser_rank': 101, 'winner_age': 22.0, 'loser_age': 22.0,
        'tourney_level': '250', 'score': '6-4 6-4',
    }])
    df.to_csv(tmp_path / "2024.csv", index=False)
    _, elo_k, *_ = calcular_elos_historicos(str(tmp_path), [2024], use_k_schedule=True)
    _, elo_nok, *_ = calcular_elos_historicos(str(tmp_path), [2024], use_k_schedule=False)
    # Con K=48, el ganador debutante debe ganar más ELO que con K=32
    assert elo_k['NUEVO'] > elo_nok['NUEVO']


def test_sin_mov_sin_k_igual_que_antes(tmp_path):
    """use_mov=False + use_k_schedule=False debe reproducir el comportamiento anterior."""
    df = pd.DataFrame([{
        'tourney_date': 20240101, 'match_num': 1,
        'winner_name': 'A', 'loser_name': 'B', 'surface': 'Hard',
        'winner_rank': 10, 'loser_rank': 20, 'winner_age': 25.0, 'loser_age': 27.0,
        'tourney_level': 'G', 'score': '6-3 6-2',
    }])
    df.to_csv(tmp_path / "2024.csv", index=False)
    from src.elo import actualizar_ratings, calcular_expectativa
    _, elo_legacy, *_ = calcular_elos_historicos(str(tmp_path), [2024], use_mov=False, use_k_schedule=False)
    e_A = calcular_expectativa(1500.0, 1500.0)
    nuevo_A, _ = actualizar_ratings(1500.0, 1500.0, resultado_A=1, K=32)
    assert abs(elo_legacy['A'] - nuevo_A) < 1e-6
