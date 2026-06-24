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

from src.features import LEVEL_MAP  # fuente única; re-exportado para compatibilidad

def preparar_datos_entrenamiento(df_con_elo):
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
    
    # 2. Imputar nulos de ranking y edad
    df['winner_rank'] = df['winner_rank'].fillna(999)
    df['loser_rank'] = df['loser_rank'].fillna(999)
    
    mediana_winner_age = df['winner_age'].median()
    mediana_loser_age = df['loser_age'].median()
    df['winner_age'] = df['winner_age'].fillna(mediana_winner_age)
    df['loser_age'] = df['loser_age'].fillna(mediana_loser_age)
    
    # 3. Crear máscara aleatoria de simetrización
    np.random.seed(42)  # Semilla fija para reproducibilidad científica
    shuffle = np.random.rand(len(df)) > 0.5

    # 4. Simetrización vectorizada: A = ganador si shuffle, A = perdedor si no
    elo_a  = np.where(shuffle, df['elo_winner'],       df['elo_loser'])
    elo_b  = np.where(shuffle, df['elo_loser'],        df['elo_winner'])
    rank_a = np.where(shuffle, df['winner_rank'],      df['loser_rank'])
    rank_b = np.where(shuffle, df['loser_rank'],       df['winner_rank'])
    age_a  = np.where(shuffle, df['winner_age'],       df['loser_age'])
    age_b  = np.where(shuffle, df['loser_age'],        df['winner_age'])
    h2h_a  = np.where(shuffle, df.get('h2h_winner_ratio', 0.5), df.get('h2h_loser_ratio', 0.5))
    h2h_b  = np.where(shuffle, df.get('h2h_loser_ratio', 0.5),  df.get('h2h_winner_ratio', 0.5))
    form_a = np.where(shuffle, df.get('form_winner', 0.5),       df.get('form_loser', 0.5))
    form_b = np.where(shuffle, df.get('form_loser', 0.5),        df.get('form_winner', 0.5))

    level_col = df['tourney_level'].astype(str) if 'tourney_level' in df.columns else pd.Series(['250'] * len(df))
    tourney_level_num = level_col.map(lambda x: LEVEL_MAP.get(x, 1))

    return pd.DataFrame({
        'year':             df['tourney_date'].astype(str).str[:4].astype(int).values,
        'tourney_date':     df['tourney_date'].values,  # yyyymmdd, para el embargo temporal del CV
        'surface':          df['surface'].values if 'surface' in df.columns else 'Hard',
        'diff_elo':         elo_a - elo_b,
        'diff_rank':        rank_a - rank_b,
        'diff_age':         age_a - age_b,
        'diff_h2h':         h2h_a - h2h_b,
        'diff_form':        form_a - form_b,
        'tourney_level_num': tourney_level_num.values,
        'label':            np.where(shuffle, 1, 0),
    })

def crear_dataset_visual(filepath):
    """
    Carga un archivo anual individual y genera variables simétricas enriquecidas,
    incluyendo la altura (height) de los jugadores para el análisis exploratorio visual (EDA).

    Parameters
    ----------
    filepath : str
        Ruta al archivo CSV anual (ej: 'data/2024.csv').

    Returns
    -------
    pd.DataFrame
        DataFrame preparado para visualización con variables de diferencias y etiquetas de texto.
    """
    df = pd.read_csv(filepath)
    
    # Limpieza e imputación de rankings y alturas
    df['winner_rank'] = df['winner_rank'].fillna(999)
    df['loser_rank'] = df['loser_rank'].fillna(999)
    
    mediana_ht = df['winner_ht'].median() if not df['winner_ht'].isnull().all() else 185.0
    df['winner_ht'] = df['winner_ht'].fillna(mediana_ht)
    df['loser_ht'] = df['loser_ht'].fillna(mediana_ht)
    
    df['winner_age'] = df['winner_age'].fillna(df['winner_age'].median() if not df['winner_age'].isnull().all() else 26.0)
    df['loser_age'] = df['loser_age'].fillna(df['loser_age'].median() if not df['loser_age'].isnull().all() else 26.0)
    
    np.random.seed(42)
    shuffle_mask = np.random.rand(len(df)) > 0.5
    
    features = []
    for i in range(len(df)):
        row = df.iloc[i]
        if shuffle_mask[i]:
            rank_A, rank_B = row['winner_rank'], row['loser_rank']
            age_A, age_B = row['winner_age'], row['loser_age']
            ht_A, ht_B = row['winner_ht'], row['loser_ht']
            label = 1
        else:
            rank_A, rank_B = row['loser_rank'], row['winner_rank']
            age_A, age_B = row['loser_age'], row['winner_age']
            ht_A, ht_B = row['loser_ht'], row['winner_ht']
            label = 0
            
        features.append({
            'diff_rank': rank_A - rank_B,
            'diff_age': age_A - age_B,
            'diff_ht': ht_A - ht_B,
            'label': label
        })
        
    return pd.DataFrame(features)
