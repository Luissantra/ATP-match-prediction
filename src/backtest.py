"""
Engine de Walk-Forward Validation (Backtesting)
==============================================

Valida el rendimiento del modelo simulando predicciones sobre torneos históricos
completos. Para cada torneo (checkpoint), calcula los ratings ELO y estadísticas
de los jugadores acumulados hasta el día de inicio del torneo, entrena un modelo
fresco con todos los datos previos y predice las eliminatorias del torneo de forma
completamente out-of-sample y libre de data leakage.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, accuracy_score

from src.elo import calcular_elos_historicos
from src.data_processing import preparar_datos_entrenamiento
from src.features import FEATURES, RANK_CAP, elo_hibrido, vector_from_features
from src.train import entrenar_modelo, calibrar_modelo


def preparar_features_test_estatico(df_test_raw, elo_general, elo_superficie, stats_jugadores, seed=42):
    """
    Construye las features de test de manera estática usando las estadísticas y ratings
    de los jugadores al inicio del torneo (antes de jugar cualquier ronda).
    """
    rows = []
    rng = np.random.default_rng(seed)
    shuffle = rng.random(len(df_test_raw)) > 0.5

    for idx, (_, row) in enumerate(df_test_raw.iterrows()):
        ganador = row['winner_name']
        perdedor = row['loser_name']
        surface = row['surface']
        if surface not in elo_superficie:
            surface = 'Hard'

        # Ratings al inicio del torneo
        g_gen = elo_general.get(ganador, 1500.0)
        p_gen = elo_general.get(perdedor, 1500.0)
        g_sup = elo_superficie[surface].get(ganador, 1500.0)
        p_sup = elo_superficie[surface].get(perdedor, 1500.0)

        # Ranks y Edades
        # Si no están en stats_jugadores, usamos los valores de la fila del partido
        rank_w = stats_jugadores.get(ganador, {}).get('rank', row['winner_rank'])
        rank_l = stats_jugadores.get(perdedor, {}).get('rank', row['loser_rank'])
        age_w = stats_jugadores.get(ganador, {}).get('age', row['winner_age'])
        age_l = stats_jugadores.get(perdedor, {}).get('age', row['loser_age'])

        # Imputaciones fallback
        rank_w = 999.0 if pd.isna(rank_w) else float(rank_w)
        rank_l = 999.0 if pd.isna(rank_l) else float(rank_l)
        age_w = 26.0 if pd.isna(age_w) else float(age_w)
        age_l = 26.0 if pd.isna(age_l) else float(age_l)

        unranked_w = int(rank_w >= 999)
        unranked_l = int(rank_l >= 999)

        # Simetrizar
        is_shuffled = shuffle[idx]
        if is_shuffled:
            # Jugador A es Ganador, Jugador B es Perdedor
            diff_elo_general = g_gen - p_gen
            diff_elo_sup = g_sup - p_sup
            diff_rank = min(rank_w, RANK_CAP) - min(rank_l, RANK_CAP)
            is_unranked = unranked_w - unranked_l
            diff_age = age_w - age_l
            label = 1
        else:
            # Jugador A es Perdedor, Jugador B es Ganador
            diff_elo_general = p_gen - g_gen
            diff_elo_sup = p_sup - g_sup
            diff_rank = min(rank_l, RANK_CAP) - min(rank_w, RANK_CAP)
            is_unranked = unranked_l - unranked_w
            diff_age = age_l - age_w
            label = 0

        rows.append({
            'diff_elo_general': diff_elo_general,
            'diff_elo_sup': diff_elo_sup,
            'diff_rank': diff_rank,
            'is_unranked': is_unranked,
            'diff_age': diff_age,
            'label': label
        })

    return pd.DataFrame(rows)


def ejecutar_backtest_checkpoint(data_dir, años, checkpoint_date, tourney_name, surface):
    """
    Ejecuta el entrenamiento y prueba para un único checkpoint de torneo.
    """
    # 1. Calcular el ELO histórico de los partidos previos al inicio del torneo
    df_pre_tourney, elo_gen, elo_sup = calcular_elos_historicos(
        data_dir, años, hasta_fecha=checkpoint_date
    )

    # 2. Construir stats_jugadores acumulados hasta esa fecha
    stats_jugadores = {}
    for _, row in df_pre_tourney.iterrows():
        for role in [('winner_name', 'winner_rank', 'winner_age'),
                     ('loser_name',  'loser_rank',  'loser_age')]:
            name = row[role[0]]
            stats_jugadores[name] = {
                'rank': float(row[role[1]]) if not pd.isna(row[role[1]]) else 999.0,
                'age':  float(row[role[2]]) if not pd.isna(row[role[2]]) else 26.0,
            }

    # 3. Preparar dataset de entrenamiento (solo partidos previos)
    df_features_train = preparar_datos_entrenamiento(df_pre_tourney)
    X_train = df_features_train[FEATURES].values
    y_train = df_features_train['label'].values
    dates_train = df_features_train['tourney_date'].values

    # 4. Entrenar y calibrar modelo fresh
    modelo_base, _, _ = entrenar_modelo(X_train, y_train, dates=dates_train)
    modelo = calibrar_modelo(modelo_base, X_train, y_train, dates=dates_train)

    # 5. Cargar los partidos reales jugados en el torneo (el año del checkpoint)
    año_tourney = int(str(checkpoint_date)[:4])
    filepath = os.path.join(data_dir, f"{año_tourney}.csv")
    if not os.path.exists(filepath):
        # Si no está en el año, probar en el siguiente o buscar en el archivo
        raise FileNotFoundError(f"No se encontró el archivo del año {año_tourney} en {data_dir}")

    df_año = pd.read_csv(filepath)
    # Filtrar partidos específicos del torneo
    # Nota: algunos datasets pueden tener pequeñas variaciones en el nombre
    df_test_raw = df_año[
        (df_año['tourney_name'].str.contains(tourney_name, case=False, na=False)) &
        (df_año['tourney_date'] == int(checkpoint_date))
    ].copy()

    if len(df_test_raw) == 0:
        # Fallback: buscar solo por fecha
        df_test_raw = df_año[df_año['tourney_date'] == int(checkpoint_date)].copy()

    if len(df_test_raw) == 0:
        raise ValueError(f"No se encontraron partidos para el torneo '{tourney_name}' en la fecha {checkpoint_date}")

    # 6. Preparar features de test estático
    df_features_test = preparar_features_test_estatico(
        df_test_raw, elo_gen, elo_sup, stats_jugadores
    )
    X_test = df_features_test[FEATURES].values
    y_test = df_features_test['label'].values

    # 7. Predecir
    y_prob = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    # Calcular métricas
    metrics = {
        'n': len(y_test),
        'accuracy': accuracy_score(y_test, y_pred),
        'log_loss': log_loss(y_test, y_prob),
        'brier': brier_score_loss(y_test, y_prob),
        'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan
    }

    return metrics, modelo, elo_gen, elo_sup, stats_jugadores


def ejecutar_backtest_completo(data_dir="data", años=None):
    """
    Ejecuta el backtest walk-forward sobre los 4 Grand Slams de 2025.
    """
    if años is None:
        años = [2020, 2021, 2022, 2023, 2024, 2025]

    checkpoints = [
        {'name': 'Australian Open 2025', 'date': 20250113, 'surface': 'Hard'},
        {'name': 'Roland Garros 2025',   'date': 20250526, 'surface': 'Clay'},
        {'name': 'Wimbledon 2025',       'date': 20250630, 'surface': 'Grass'},
        {'name': 'US Open 2025',         'date': 20250824, 'surface': 'Hard'},
    ]

    resultados = []
    print("\n=== Iniciando Backtest Walk-Forward (Grand Slams 2025) ===")

    for cp in checkpoints:
        print(f"\nEvaluando Checkpoint: {cp['name']} ({cp['date']})...")
        try:
            met, _, _, _, _ = ejecutar_backtest_checkpoint(
                data_dir, años, cp['date'], cp['name'].split(' 2025')[0], cp['surface']
            )
            print(f"  Resultados: N={met['n']} | Acc={met['accuracy']:.3%}"
                  f" | LogLoss={met['log_loss']:.4f} | AUC={met['auc']:.4f} | Brier={met['brier']:.4f}")
            resultados.append((cp['name'], met))
        except Exception as e:
            print(f"  Error al evaluar {cp['name']}: {e}")

    # Calcular consolidado ponderado por N
    if not resultados:
        print("No se completó ningún backtest de checkpoint.")
        return None

    total_n = sum(met['n'] for _, met in resultados)
    avg_acc = sum(met['accuracy'] * met['n'] for _, met in resultados) / total_n
    avg_loss = sum(met['log_loss'] * met['n'] for _, met in resultados) / total_n
    avg_brier = sum(met['brier'] * met['n'] for _, met in resultados) / total_n
    # Para AUC hacemos promedio simple sobre torneos válidos
    valid_aucs = [met['auc'] for _, met in resultados if not np.isnan(met['auc'])]
    avg_auc = np.mean(valid_aucs) if valid_aucs else np.nan

    print("\n=======================================================")
    print("=== CONSOLIDADO GLOBAL WALK-FORWARD (GRAND SLAMS) ===")
    print(f"  Partidos Totales Evaluados: {total_n}")
    print(f"  Accuracy Promedio:          {avg_acc:.3%}")
    print(f"  Log-Loss Promedio:          {avg_loss:.4f}")
    print(f"  Brier Score Promedio:       {avg_brier:.4f}")
    print(f"  AUC Promedio:               {avg_auc:.4f}")
    print("=======================================================")

    return {
        'checkpoints': resultados,
        'global': {
            'total_n': total_n,
            'accuracy': avg_acc,
            'log_loss': avg_loss,
            'brier': avg_brier,
            'auc': avg_auc
        }
    }


if __name__ == '__main__':
    ejecutar_backtest_completo()
