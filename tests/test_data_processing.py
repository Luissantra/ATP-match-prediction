import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_processing import preparar_datos_entrenamiento, LEVEL_MAP


def _make_df_with_elo():
    return pd.DataFrame([
        {
            'tourney_date': 20240101, 'surface': 'Hard', 'tourney_level': 'G',
            'winner_rank': 10.0, 'loser_rank': 20.0,
            'winner_age': 25.0, 'loser_age': 27.0,
            'elo_winner': 1600.0, 'elo_loser': 1500.0,
            'h2h_winner_ratio': 0.75, 'h2h_loser_ratio': 0.25,
            'form_winner': 0.8, 'form_loser': 0.4,
        },
        {
            'tourney_date': 20240102, 'surface': 'Clay', 'tourney_level': 'M',
            'winner_rank': 5.0, 'loser_rank': 30.0,
            'winner_age': 22.0, 'loser_age': 32.0,
            'elo_winner': 1700.0, 'elo_loser': 1450.0,
            'h2h_winner_ratio': 0.5, 'h2h_loser_ratio': 0.5,
            'form_winner': 0.6, 'form_loser': 0.3,
        },
    ])


def test_new_feature_columns_exist():
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    assert 'diff_h2h' in result.columns
    assert 'diff_form' in result.columns
    assert 'tourney_level_num' in result.columns


def test_total_feature_columns():
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    expected = {'year', 'tourney_date', 'surface', 'diff_elo', 'diff_rank', 'diff_age',
                'diff_h2h', 'diff_form', 'tourney_level_num', 'label'}
    assert set(result.columns) == expected


def test_level_map_grand_slam():
    assert LEVEL_MAP['G'] == 5


def test_level_map_masters():
    assert LEVEL_MAP['M'] == 4


def test_level_map_500():
    assert LEVEL_MAP['500'] == 2
    assert LEVEL_MAP['A'] == 2


def test_level_map_250():
    assert LEVEL_MAP['250'] == 1
    assert LEVEL_MAP['D'] == 1


def test_tourney_level_encoded_correctly():
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    assert result.iloc[0]['tourney_level_num'] == 5  # G
    assert result.iloc[1]['tourney_level_num'] == 4  # M


def test_diff_h2h_absolute_value():
    """Valor absoluto de diff_h2h primera fila: |0.75 - 0.25| = 0.5"""
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    assert abs(result.iloc[0]['diff_h2h']) == pytest.approx(0.5, abs=1e-9)


def test_diff_form_absolute_value():
    """Valor absoluto de diff_form primera fila: |0.8 - 0.4| = 0.4"""
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    assert abs(result.iloc[0]['diff_form']) == pytest.approx(0.4, abs=1e-9)


def test_label_balanced():
    rows = [
        {
            'tourney_date': 20240101 + i, 'surface': 'Hard', 'tourney_level': 'G',
            'winner_rank': 10.0, 'loser_rank': 20.0,
            'winner_age': 25.0, 'loser_age': 27.0,
            'elo_winner': 1600.0, 'elo_loser': 1500.0,
            'h2h_winner_ratio': 0.5, 'h2h_loser_ratio': 0.5,
            'form_winner': 0.5, 'form_loser': 0.5,
        }
        for i in range(100)
    ]
    df = pd.DataFrame(rows)
    result = preparar_datos_entrenamiento(df)
    ratio = result['label'].mean()
    assert 0.4 < ratio < 0.6
