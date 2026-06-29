import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_processing import preparar_datos_entrenamiento as _real_preparar

def preparar_datos_entrenamiento(df, *args, **kwargs):
    df = df.copy()
    for col in ['winner_matches_played', 'loser_matches_played']:
        if col not in df.columns:
            df[col] = 0.0
    for col in ['winner_tb_ratio', 'loser_tb_ratio']:
        if col not in df.columns:
            df[col] = 0.5
    return _real_preparar(df, *args, **kwargs)


def _make_df_with_elo():
    return pd.DataFrame([
        {
            'tourney_date': 20240101, 'surface': 'Hard', 'tourney_level': 'G',
            'winner_rank': 10.0, 'loser_rank': 20.0,
            'winner_age': 25.0, 'loser_age': 27.0,
            'elo_winner': 1650.0, 'elo_loser': 1500.0,
            'elo_winner_general': 1600.0, 'elo_loser_general': 1500.0,
            'elo_winner_sup': 1700.0, 'elo_loser_sup': 1500.0,
            'h2h_winner_ratio': 0.75, 'h2h_loser_ratio': 0.25,
            'form_winner': 0.8, 'form_loser': 0.4,
        },
        {
            'tourney_date': 20240102, 'surface': 'Clay', 'tourney_level': 'M',
            'winner_rank': 5.0, 'loser_rank': 30.0,
            'winner_age': 22.0, 'loser_age': 32.0,
            'elo_winner': 1575.0, 'elo_loser': 1475.0,
            'elo_winner_general': 1700.0, 'elo_loser_general': 1450.0,
            'elo_winner_sup': 1450.0, 'elo_loser_sup': 1500.0,
            'h2h_winner_ratio': 0.5, 'h2h_loser_ratio': 0.5,
            'form_winner': 0.6, 'form_loser': 0.3,
        },
    ])


def test_columnas_nuevas_existen():
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    assert 'diff_elo_general' in result.columns
    assert 'diff_elo_sup' in result.columns
    assert 'is_unranked' in result.columns


def test_columna_diff_elo_vieja_no_existe():
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    assert 'diff_elo' not in result.columns


def test_total_feature_columns():
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    expected = {
        'year', 'tourney_date', 'surface',
        'diff_elo_general', 'diff_elo_sup',
        'diff_rank', 'is_unranked',
        'diff_age', 'diff_matches_played', 'diff_tb_ratio', 'label',
    }
    assert set(result.columns) == expected


def test_rank_cap_limita_outlier():
    """rank=999 (wildcard) debe capar a RANK_CAP=250 en diff_rank."""
    df = pd.DataFrame([{
        'tourney_date': 20240101, 'surface': 'Hard', 'tourney_level': 'G',
        'winner_rank': 1.0, 'loser_rank': 999.0,
        'winner_age': 25.0, 'loser_age': 27.0,
        'elo_winner': 1650.0, 'elo_loser': 1500.0,
        'elo_winner_general': 1700.0, 'elo_loser_general': 1500.0,
        'elo_winner_sup': 1600.0, 'elo_loser_sup': 1500.0,
        'h2h_winner_ratio': 0.5, 'h2h_loser_ratio': 0.5,
        'form_winner': 0.5, 'form_loser': 0.5,
    }])
    result = preparar_datos_entrenamiento(df)
    # cap(1) - cap(999) = 1-250=-249 o cap(999)-cap(1) = 250-1=249
    assert abs(result.iloc[0]['diff_rank']) <= 249


def test_is_unranked_detecta_wildcard():
    """Cuando un jugador NO tiene ranking real (NaN → wildcard/qualifier), is_unranked != 0.
    Un rank numérico alto (p.ej. 999/2000) NO debe marcarse como sin-ranking."""
    df = pd.DataFrame([{
        'tourney_date': 20240101, 'surface': 'Hard', 'tourney_level': 'G',
        'winner_rank': 1.0, 'loser_rank': np.nan,
        'winner_age': 25.0, 'loser_age': 27.0,
        'elo_winner': 1650.0, 'elo_loser': 1500.0,
        'elo_winner_general': 1700.0, 'elo_loser_general': 1500.0,
        'elo_winner_sup': 1600.0, 'elo_loser_sup': 1500.0,
        'h2h_winner_ratio': 0.5, 'h2h_loser_ratio': 0.5,
        'form_winner': 0.5, 'form_loser': 0.5,
    }])
    result = preparar_datos_entrenamiento(df)
    assert result.iloc[0]['is_unranked'] != 0


def test_is_unranked_cero_cuando_ambos_rankeados():
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    assert result.iloc[0]['is_unranked'] == 0
    assert result.iloc[1]['is_unranked'] == 0


def test_rank_numerico_alto_no_es_unranked():
    """Fix del centinela: un rank real alto (2000) NO es 'sin ranking' (solo NaN lo es)."""
    df = pd.DataFrame([{
        'tourney_date': 20240101, 'surface': 'Hard', 'tourney_level': 'G',
        'winner_rank': 1.0, 'loser_rank': 2000.0,
        'winner_age': 25.0, 'loser_age': 27.0,
        'elo_winner': 1650.0, 'elo_loser': 1500.0,
        'elo_winner_general': 1700.0, 'elo_loser_general': 1500.0,
        'elo_winner_sup': 1600.0, 'elo_loser_sup': 1500.0,
    }])
    result = preparar_datos_entrenamiento(df)
    assert result.iloc[0]['is_unranked'] == 0


def test_label_balanced():
    rows = [
        {
            'tourney_date': 20240101 + i, 'surface': 'Hard', 'tourney_level': 'G',
            'winner_rank': 10.0, 'loser_rank': 20.0,
            'winner_age': 25.0, 'loser_age': 27.0,
            'elo_winner': 1600.0, 'elo_loser': 1500.0,
            'elo_winner_general': 1600.0, 'elo_loser_general': 1500.0,
            'elo_winner_sup': 1600.0, 'elo_loser_sup': 1500.0,
            'h2h_winner_ratio': 0.5, 'h2h_loser_ratio': 0.5,
            'form_winner': 0.5, 'form_loser': 0.5,
        }
        for i in range(100)
    ]
    df = pd.DataFrame(rows)
    result = preparar_datos_entrenamiento(df)
    ratio = result['label'].mean()
    # seed=42 fijo en preparar_datos_entrenamiento → resultado determinista: 47/100
    assert ratio == 0.47


def _make_df_con_elo(n=20):
    return pd.DataFrame({
        'tourney_date': [20240101] * n,
        'winner_name': [f'A{i}' for i in range(n)],
        'loser_name':  [f'B{i}' for i in range(n)],
        'elo_winner_general': np.full(n, 1500.0),
        'elo_loser_general':  np.full(n, 1500.0),
        'elo_winner_sup':     np.full(n, 1500.0),
        'elo_loser_sup':      np.full(n, 1500.0),
        'winner_rank': np.full(n, 10.0),
        'loser_rank':  np.full(n, 20.0),
        'winner_age':  np.full(n, 25.0),
        'loser_age':   np.full(n, 27.0),
        'h2h_winner_ratio': np.full(n, 0.5),
        'h2h_loser_ratio':  np.full(n, 0.5),
        'form_winner': np.full(n, 0.5),
        'form_loser':  np.full(n, 0.5),
        'tourney_level': ['250'] * n,
        'surface': ['Hard'] * n,
    })


def test_preparar_datos_seed_reproducible():
    df = _make_df_con_elo(n=50)
    r1 = preparar_datos_entrenamiento(df.copy(), seed=42)
    r2 = preparar_datos_entrenamiento(df.copy(), seed=42)
    assert (r1['label'].values == r2['label'].values).all()


def test_preparar_datos_seed_diferente_da_diferente_shuffle():
    df = _make_df_con_elo(n=20)
    r1 = preparar_datos_entrenamiento(df.copy(), seed=42)
    r2 = preparar_datos_entrenamiento(df.copy(), seed=99)
    # Different seeds → different label distributions (may occasionally be equal but very unlikely with n=20)
    assert not (r1['label'].values == r2['label'].values).all()


def test_simetrizacion_coherencia_label_diff_rank():
    """label=1 (A=ganador) → diff_rank negativo (ganador mejor rankeado que perdedor)."""
    n = 200
    df = _make_df_con_elo(n=n)
    df['winner_rank'] = 10.0
    df['loser_rank'] = 50.0
    result = preparar_datos_entrenamiento(df.copy(), seed=42)

    label_1 = result[result['label'] == 1]
    label_0 = result[result['label'] == 0]

    # label=1: A=winner(10) - B=loser(50) = -40
    assert (label_1['diff_rank'] == 10 - 50).all()
    # label=0: A=loser(50) - B=winner(10) = +40
    assert (label_0['diff_rank'] == 50 - 10).all()


def test_simetrizacion_coherencia_label_diff_elo():
    """label=1 → diff_elo_general positivo (ganador mejor ELO); label=0 → negativo."""
    n = 200
    df = _make_df_con_elo(n=n)
    df['elo_winner_general'] = 1600.0
    df['elo_loser_general'] = 1400.0
    result = preparar_datos_entrenamiento(df.copy(), seed=42)

    label_1 = result[result['label'] == 1]
    label_0 = result[result['label'] == 0]

    assert (label_1['diff_elo_general'] == 200.0).all()
    assert (label_0['diff_elo_general'] == -200.0).all()
