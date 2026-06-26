"""
Módulo de Preprocesamiento de Datos e Ingeniería de Características
===================================================================

Este módulo maneja la carga, limpieza de datos, imputación de valores faltantes y
la generación de un dataset balanceado simétrico neutral para el entrenamiento
de algoritmos de aprendizaje supervisado.

La Simetrización en Ciencia de Datos Deportivos:
----------------------------------------------
En una base de datos histórica de partidos, cada registro suele venir tabulado como:
    [winner_name, loser_name, winner_rank, loser_rank, ...]

Si entrenamos un modelo directamente con variables como:
    diferencia_rank = winner_rank - loser_rank

El objetivo (label) siempre sería '1' (el ganador ganó). Un modelo supervisado aprendería
una regla trivial inútil para predecir futuros partidos en los que no conocemos el resultado.

Para resolver esto, aplicamos un método cognitivo y estadístico de simetrización:
- Generamos un vector de máscara booleana aleatoria (shuffle_mask) con un 50% de probabilidad.
- Si es True: Asignamos Jugador A = Ganador, Jugador B = Perdedor.
  * La etiqueta (label) es 1 (gana A).
  * Las diferencias se calculan como (A - B).
- Si es False: Asignamos Jugador A = Perdedor, Jugador B = Ganador.
  * La etiqueta (label) es 0 (gana B, es decir, A pierde).
  * Las diferencias se calculan como (A - B).

Este balanceo genera un dataset neutral donde la variable objetivo está perfectamente distribuida
al 50% (evitando sesgos sistemáticos) y el modelo aprende la verdadera frontera de decisión.
"""

import pandas as pd
import numpy as np

from src.features import RANK_CAP  # fuente única; re-exportado para compatibilidad

def preparar_datos_entrenamiento(df_con_elo, seed=42):
    """
    Realiza la imputación estadística y la simetrización de características
    para preparar el dataset de entrenamiento y test del modelo.

    Imputaciones realizadas:
    - Rankings faltantes: Se rellenan con '999'. En el circuito profesional ATP,
      un ranking extremadamente bajo o nulo suele indicar un jugador que entró
      por invitación (wildcard) o fase clasificatoria (qualifier), teniendo una
      desventaja implícita.
    - Edades faltantes: Se imputan con la mediana de edad por robustez ante outliers.

    Parameters
    ----------
    df_con_elo : pd.DataFrame
        DataFrame de entrada con ratings ELO calculados.

    Returns
    -------
    pd.DataFrame
        Un nuevo DataFrame con las características calculadas de forma simétrica:
        ['year', 'diff_elo', 'diff_rank', 'diff_age', 'label'].
    """
    # 1. Copiar para evitar Side-Effects
    df = df_con_elo.copy()
    
    # 2. Marcar jugadores SIN ranking real (wildcard/qualifier) antes de imputar.
    #    Usamos la máscara NaN original, no el centinela 999: en los datos hay ranks
    #    reales hasta ~2100 que no deben confundirse con "sin ranking".
    winner_unranked = df['winner_rank'].isna()
    loser_unranked = df['loser_rank'].isna()

    # Imputar nulos de ranking y edad
    df['winner_rank'] = df['winner_rank'].fillna(999)
    df['loser_rank'] = df['loser_rank'].fillna(999)

    mediana_winner_age = df['winner_age'].median()
    mediana_loser_age = df['loser_age'].median()
    df['winner_age'] = df['winner_age'].fillna(mediana_winner_age)
    df['loser_age'] = df['loser_age'].fillna(mediana_loser_age)
    
    # 3. Crear máscara aleatoria de simetrización
    rng = np.random.default_rng(seed)
    shuffle = rng.random(len(df)) > 0.5

    # 4. Simetrización vectorizada: A = ganador si shuffle, A = perdedor si no

    # Ranks crudos (ya imputados) para el cap; máscara unranked simetrizada aparte
    rank_a_raw = np.where(shuffle, df['winner_rank'], df['loser_rank'])
    rank_b_raw = np.where(shuffle, df['loser_rank'],  df['winner_rank'])
    unranked_a = np.where(shuffle, winner_unranked, loser_unranked)
    unranked_b = np.where(shuffle, loser_unranked,  winner_unranked)

    # ELO general y superficie separados (el modelo aprende el peso)
    elo_gen_a = np.where(shuffle, df['elo_winner_general'], df['elo_loser_general'])
    elo_gen_b = np.where(shuffle, df['elo_loser_general'],  df['elo_winner_general'])
    elo_sup_a = np.where(shuffle, df['elo_winner_sup'],     df['elo_loser_sup'])
    elo_sup_b = np.where(shuffle, df['elo_loser_sup'],      df['elo_winner_sup'])

    age_a  = np.where(shuffle, df['winner_age'],       df['loser_age'])
    age_b  = np.where(shuffle, df['loser_age'],        df['winner_age'])

    return pd.DataFrame({
        'year':             df['tourney_date'].astype(str).str[:4].astype(int).values,
        'tourney_date':     df['tourney_date'].values,  # yyyymmdd, para el embargo temporal del CV
        'surface':          df['surface'].values if 'surface' in df.columns else 'Hard',
        'diff_elo_general': elo_gen_a - elo_gen_b,
        'diff_elo_sup':     elo_sup_a - elo_sup_b,
        'diff_rank':        np.minimum(rank_a_raw, RANK_CAP) - np.minimum(rank_b_raw, RANK_CAP),
        'is_unranked':      unranked_a.astype(int) - unranked_b.astype(int),
        'diff_age':         age_a - age_b,
        'label':            np.where(shuffle, 1, 0),
    })
