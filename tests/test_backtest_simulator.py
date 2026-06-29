import os
import sys
import pandas as pd
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.elo import calcular_elos_historicos
from src.backtest import preparar_features_test_estatico, ejecutar_backtest_checkpoint
from src.simulator import simular_torneo_montecarlo, predecir_probabilidad_matchup


def _make_mock_history_df():
    """Genera un historial de partidos simulados para pruebas."""
    return pd.DataFrame([
        {'tourney_date': 20240101, 'match_num': 1, 'winner_name': 'Djokovic', 'loser_name': 'Ruud',
         'surface': 'Hard', 'winner_rank': 1.0, 'loser_rank': 8.0,
         'winner_age': 36.0, 'loser_age': 25.0, 'score': '6-4 6-4'},
        {'tourney_date': 20240102, 'match_num': 1, 'winner_name': 'Sinner', 'loser_name': 'Alcaraz',
         'surface': 'Hard', 'winner_rank': 4.0, 'loser_rank': 2.0,
         'winner_age': 22.0, 'loser_age': 20.0, 'score': '7-6(5) 6-4'},
        {'tourney_date': 20240115, 'match_num': 1, 'winner_name': 'Djokovic', 'loser_name': 'Sinner',
         'surface': 'Hard', 'winner_rank': 1.0, 'loser_rank': 4.0,
         'winner_age': 36.0, 'loser_age': 22.0, 'score': '6-3 7-5'},
        # Partido de torneo en fecha 20240120
        {'tourney_date': 20240120, 'match_num': 1, 'winner_name': 'Alcaraz', 'loser_name': 'Ruud',
         'surface': 'Hard', 'winner_rank': 2.0, 'loser_rank': 8.0,
         'winner_age': 20.0, 'loser_age': 25.0, 'score': '6-2 6-3'},
    ])


def test_calcular_elos_hasta_fecha(tmp_path):
    """Verifica que calcular_elos_historicos respeta la fecha de corte hasta_fecha."""
    df = _make_mock_history_df()
    df.to_csv(tmp_path / "2024.csv", index=False)

    # Si cortamos en 20240115, debe incluir solo los dos primeros partidos
    df_res, elo_gen, _, *_ = calcular_elos_historicos(str(tmp_path), [2024], hasta_fecha=20240115)
    assert len(df_res) == 2
    # El partido de Djokovic vs Sinner en fecha 15 y Alcaraz vs Ruud en fecha 20 no deben procesarse
    assert 'Djokovic' in elo_gen
    assert 'Sinner' in elo_gen
    assert 'Ruud' in elo_gen
    # Djokovic jugó y ganó en fecha 01, Sinner jugó y ganó en fecha 02.
    # El rating de Djokovic debe haber subido de 1500, pero su partido del 15 no se procesó.
    # Así que sus ratings reflejan solo el primer partido.


def test_preparar_features_test_estatico():
    """Verifica que las features estáticas de test se construyan correctamente sin leakages."""
    df_test = pd.DataFrame([
        {'winner_name': 'Djokovic', 'loser_name': 'Sinner', 'surface': 'Hard',
         'winner_rank': 1.0, 'loser_rank': 4.0, 'winner_age': 36.0, 'loser_age': 22.0}
    ])
    elo_gen = {'Djokovic': 1700.0, 'Sinner': 1600.0}
    elo_sup = {'Hard': {'Djokovic': 1750.0, 'Sinner': 1620.0}}
    stats_jugadores = {
        'Djokovic': {'rank': 1.0, 'age': 36.0, 'matches_played': 1000, 'tb_wins': 200, 'tb_played': 300},
        'Sinner': {'rank': 4.0, 'age': 22.0, 'matches_played': 200, 'tb_wins': 40, 'tb_played': 60}
    }

    df_feats = preparar_features_test_estatico(df_test, elo_gen, elo_sup, stats_jugadores, seed=42)
    assert len(df_feats) == 1
    assert 'diff_elo_general' in df_feats.columns
    assert 'diff_elo_sup' in df_feats.columns
    assert 'diff_rank' in df_feats.columns
    assert 'is_unranked' in df_feats.columns
    assert 'diff_age' in df_feats.columns
    assert 'diff_matches_played' in df_feats.columns
    assert 'diff_tb_ratio' in df_feats.columns

    # Calcular ratios esperados
    tb_ratio_djok = 202.0 / 304.0
    tb_ratio_sinn = 42.0 / 64.0

    # Verificar que las diferencias sean correctas (respetando la simetrización aleatoria)
    label = df_feats.iloc[0]['label']
    if label == 1:
        # A es Djokovic (Winner), B es Sinner (Loser)
        assert df_feats.iloc[0]['diff_elo_general'] == 1700.0 - 1600.0
        assert df_feats.iloc[0]['diff_elo_sup'] == 1750.0 - 1620.0
        assert df_feats.iloc[0]['diff_rank'] == 1.0 - 4.0
        assert df_feats.iloc[0]['diff_age'] == 36.0 - 22.0
        assert df_feats.iloc[0]['diff_matches_played'] == 1000.0 - 200.0
        assert abs(df_feats.iloc[0]['diff_tb_ratio'] - (tb_ratio_djok - tb_ratio_sinn)) < 1e-6
    else:
        # A es Sinner (Loser), B es Djokovic (Winner)
        assert df_feats.iloc[0]['diff_elo_general'] == 1600.0 - 1700.0
        assert df_feats.iloc[0]['diff_elo_sup'] == 1620.0 - 1750.0
        assert df_feats.iloc[0]['diff_rank'] == 4.0 - 1.0
        assert df_feats.iloc[0]['diff_age'] == 22.0 - 36.0
        assert df_feats.iloc[0]['diff_matches_played'] == 200.0 - 1000.0
        assert abs(df_feats.iloc[0]['diff_tb_ratio'] - (tb_ratio_sinn - tb_ratio_djok)) < 1e-6


def test_simular_torneo_potencia_de_dos():
    """Verifica que el simulador valide que el número de jugadores sea potencia de 2."""
    modelo_dummy = CalibratedClassifierCV(estimator=make_pipeline(StandardScaler(), LogisticRegression()))
    
    # 3 jugadores (no es potencia de 2) debe lanzar ValueError
    with pytest.raises(ValueError, match="potencia de 2"):
        simular_torneo_montecarlo(
            ['A', 'B', 'C'], 'Hard', modelo_dummy, {}, {}, {}
        )


def test_simular_torneo_comportamiento_esperado():
    """Prueba el comportamiento lógico del simulador de Monte Carlo con un modelo dummy ajustado."""
    # Entrenar un modelo real extremadamente simple sobre features dummy para tener predict_proba funcional
    X = np.array([[100.0, 100.0, -10.0, 0, 2.0, 0.0, 0.0],
                  [100.0, 100.0, -10.0, 0, 2.0, 0.0, 0.0],
                  [-100.0, -100.0, 10.0, 0, -2.0, 0.0, 0.0],
                  [-100.0, -100.0, 10.0, 0, -2.0, 0.0, 0.0]])
    y = np.array([1, 1, 0, 0])
    
    modelo_base = make_pipeline(StandardScaler(), LogisticRegression())
    modelo_base.fit(X, y)
    
    modelo = CalibratedClassifierCV(estimator=modelo_base, cv=2)
    modelo.fit(X, y)

    # 4 jugadores
    draw = ['Favorito', 'Perdedor1', 'Perdedor2', 'Perdedor3']
    
    # ELOs y stats
    elo_gen = {'Favorito': 1800.0, 'Perdedor1': 1400.0, 'Perdedor2': 1400.0, 'Perdedor3': 1400.0}
    elo_sup = {'Hard': {'Favorito': 1800.0, 'Perdedor1': 1400.0, 'Perdedor2': 1400.0, 'Perdedor3': 1400.0}}
    stats = {
        'Favorito': {'rank': 1, 'age': 25, 'matches_played': 100, 'tb_wins': 10, 'tb_played': 15},
        'Perdedor1': {'rank': 100, 'age': 25, 'matches_played': 10, 'tb_wins': 1, 'tb_played': 2},
        'Perdedor2': {'rank': 100, 'age': 25, 'matches_played': 10, 'tb_wins': 1, 'tb_played': 2},
        'Perdedor3': {'rank': 100, 'age': 25, 'matches_played': 10, 'tb_wins': 1, 'tb_played': 2}
    }

    df_prob = simular_torneo_montecarlo(
        draw, 'Hard', modelo, elo_gen, elo_sup, stats, n_simulaciones=100, seed=42
    )

    # El Favorito debe tener una probabilidad de ganar muy superior a los demás
    assert df_prob.index[0] == 'Favorito'
    assert df_prob.loc['Favorito', 'Winner'] > 50.0
    # La suma de probabilidades de ganar debe ser 100%
    assert np.isclose(df_prob['Winner'].sum(), 100.0)
    # Las columnas correctas deben existir ('SF', 'F', 'Winner' para draw de 4)
    assert set(df_prob.columns) == {'SF', 'F', 'Winner'}
