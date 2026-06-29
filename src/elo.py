"""
Módulo de Ratings ELO para Predicción de Tenis (ATP)
===================================================

Este módulo proporciona las bases matemáticas y el motor de procesamiento para calcular
los ratings ELO históricos de los jugadores en el circuito profesional ATP.

El sistema ELO es un método estadístico dinámico diseñado originalmente para el ajedrez por Arpad Elo.
En este proyecto, se adapta al tenis agregando una ponderación híbrida que combina el rendimiento global
y la especialización del jugador en cada tipo de superficie (Clay, Grass, Hard).

Fundamentos Matemáticos:
------------------------
1. Probabilidad Esperada (Expectativa de Victoria):
   Representa la probabilidad teórica de que el Jugador A derrote al Jugador B, basada en la diferencia
   de sus ratings previos. Se calcula mediante una función logística:

       E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))

   Donde R_A es el rating del Jugador A y R_B es el del Jugador B. Si la diferencia es 0, ambos tienen
   un 50% de probabilidad. Si R_A - R_B = 400, el Jugador A tiene un ~91% de probabilidad de victoria.

2. Ecuación de Actualización:
   Tras jugarse el partido, los ratings de ambos tenistas se actualizan de acuerdo al resultado real
   y la expectativa inicial:

       R'_A = R_A + K * (S_A - E_A)
       R'_B = R_B + K * (S_B - E_B)

   Donde:
     - S_A es el resultado real para A (1 si gana, 0 si pierde).
     - E_A es la expectativa calculada previamente.
     - K es el Factor K (tasa de aprendizaje). Controla la magnitud del ajuste. Un valor típico es 32.
       Si un favorito gana (S_A = 1, E_A = 0.90), el cambio es mínimo (ajuste de +3.2 pts).
       Si el "underdog" da la sorpresa (S_A = 1, E_A = 0.10), el ajuste es drástico (ajuste de +28.8 pts).
"""

import re
import pandas as pd
import numpy as np
import os

from src.features import elo_hibrido


def _extraer_sets(score: str):
    """
    Extrae (winner_sets, loser_sets) del campo 'score' del CSV ATP.
    Formato típico: "6-4 6-2" / "7-6(5) 4-6 7-5" / "6-3 RET".
    Returns (0, 0) si el score no es parseable (RET, W/O, vacío).
    """
    if not score or not isinstance(score, str):
        return (0, 0)
    if re.search(r'\bRET\b|\bW/O\b', score, re.IGNORECASE):
        return (0, 0)
    sets = re.findall(r'(\d+)-(\d+)(?:\(\d+\))?', score)
    if not sets:
        return (0, 0)
    winner_sets = sum(1 for w, l in sets if int(w) > int(l))
    loser_sets  = sum(1 for w, l in sets if int(l) > int(w))
    return (winner_sets, loser_sets)


def _mov_factor(winner_sets: int, loser_sets: int) -> float:
    """
    Multiplicador del factor-K basado en margin of victory.
    Formula: 1.0 + 0.25 * (winner_sets - loser_sets - 1)
    Resultado: straight sets → >1.0, deciding set → 1.0.
    Si no hay datos (0,0) devuelve 1.0.
    """
    if winner_sets == 0 and loser_sets == 0:
        return 1.0
    return max(1.0, 1.0 + 0.25 * (winner_sets - loser_sets - 1))


def _k_for_player(n_partidos: int) -> float:
    """
    K-schedule: K alto para debutantes (cold-start), decrece a 32 para titulares.
    Reduce el rating-drift que se produce cuando un debutante con K=32 baja
    demasiado lento el ELO de los rivales que pierde.
    """
    if n_partidos < 10:
        return 48.0
    if n_partidos < 30:
        return 40.0
    return 32.0


def calcular_expectativa(rating_A, rating_B):
    """
    Calcula la probabilidad esperada (expectativa logística) de que gane el Jugador A.

    Parameters
    ----------
    rating_A : float
        Rating ELO actual del Jugador A.
    rating_B : float
        Rating ELO actual del Jugador B.

    Returns
    -------
    float
        Probabilidad esperada entre 0 y 1 de que el Jugador A gane.
    """
    expectativa_A = 1 / (1 + 10 ** ((rating_B - rating_A) / 400))
    return expectativa_A

def actualizar_ratings(rating_A, rating_B, resultado_A, K=32):
    """
    Actualiza de forma simétrica los ratings de dos jugadores en base al resultado de un partido.

    Parameters
    ----------
    rating_A : float
        Rating ELO previo del Jugador A.
    rating_B : float
        Rating ELO previo del Jugador B.
    resultado_A : int
        Resultado real del partido respecto al Jugador A (1 si ganó, 0 si perdió).
    K : int, opcional
        Factor de ajuste (tasa de aprendizaje). Por defecto es 32.

    Returns
    -------
    tuple of float
        Nuevos ratings full-precision: (nuevo_rating_A, nuevo_rating_B).
        No se redondea aquí para evitar error acumulado; redondear solo al exportar.
    """
    # 1. Calcular expectativa de A
    e_A = calcular_expectativa(rating_A, rating_B)

    # 2. La expectativa de B es el complemento de A (ya que es un juego de suma cero)
    e_B = 1 - e_A

    # 3. Resultado real para B (es el opuesto al de A)
    resultado_B = 1 - resultado_A

    # 4. Calcular nuevos ratings aplicando la fórmula de Arpad Elo
    nuevo_rating_A = rating_A + K * (resultado_A - e_A)
    nuevo_rating_B = rating_B + K * (resultado_B - e_B)

    return nuevo_rating_A, nuevo_rating_B

def calcular_elos_historicos(data_dir, años, use_mov=True, use_k_schedule=True, hasta_fecha=None):
    """
    Procesa un conjunto de datasets anuales de partidos de tenis cronológicamente,
    manteniendo y actualizando los ratings ELO dinámicos a lo largo del tiempo.

    Para modelar el rendimiento en tenis de manera precisa, el rating final del jugador
    se compone como un promedio ponderado (50% ELO general y 50% ELO específico de superficie).
    Esto captura tanto la consistencia absoluta del tenista como su especialización táctica
    (ej. especialistas de arcilla o campeones en césped).

    Parameters
    ----------
    data_dir : str
        Ruta al directorio que contiene los archivos anuales (ej: '2024.csv').
    años : list of int
        Lista de años a procesar en orden cronológico (ej: [2020, 2021, 2022]).
    hasta_fecha : int or str, opcional
        Fecha de corte en formato YYYYMMDD. Si se proporciona, se ignoran los
        partidos jugados en esa fecha o después.

    Returns
    -------
    df_completo : pd.DataFrame
        DataFrame ordenado cronológicamente que contiene todos los partidos procesados
        con las columnas ELO previas al partido (general, superficie e híbrido).
    elo_general : dict
        Rating ELO general de cada jugador al final del periodo procesado.
    elo_superficie : dict
        Rating ELO por superficie ({'Clay'|'Grass'|'Hard': {jugador: rating}}).
    """
    # Diccionario para almacenar el ELO actual de cada jugador (inicializado a 1500)
    elo_general = {}

    # ELOs específicos de superficie (Clay, Grass, Hard)
    elo_superficie = {
        'Clay': {},
        'Grass': {},
        'Hard': {}
    }

    n_partidos = {}  # contador de partidos jugados por jugador (para K-schedule)

    lista_dfs = []
    
    # 1. Cargar y concatenar los años indicados en orden
    for año in años:
        filepath = os.path.join(data_dir, f"{año}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            lista_dfs.append(df)
            
    if not lista_dfs:
        raise FileNotFoundError(f"No se encontraron archivos CSV para los años {años} en el directorio '{data_dir}'.")

    df_completo = pd.concat(lista_dfs, ignore_index=True)

    COLUMNAS_REQUERIDAS = ['tourney_date', 'match_num', 'winner_name', 'loser_name', 'surface']
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df_completo.columns]
    if faltantes:
        raise ValueError(f"CSV incompleto — columnas requeridas ausentes: {', '.join(faltantes)}")
    
    # 2. Ordenar cronológicamente para evitar fuga de información hacia el pasado (data leakage)
    df_completo = df_completo.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)
    
    # Filtrar por fecha de corte si se especifica
    if hasta_fecha is not None:
        hasta_fecha_int = int(hasta_fecha)
        df_completo = df_completo[df_completo['tourney_date'] < hasta_fecha_int].reset_index(drop=True)
    
    elo_ganador_previo = []
    elo_perdedor_previo = []
    elo_ganador_gen_previo = []
    elo_perdedor_gen_previo = []
    elo_ganador_sup_previo = []
    elo_perdedor_sup_previo = []

    print(f"-> Procesando {len(df_completo)} partidos cronológicamente...")
    for idx, row in df_completo.iterrows():
        ganador = row['winner_name']
        perdedor = row['loser_name']
        superficie = row['surface']
        
        # Si la superficie no es reconocida, se asume 'Hard' por ser la más común
        if superficie not in elo_superficie:
            superficie = 'Hard'

        # Obtener ratings previos del ganador (inicializa a 1500 si es debutante)
        g_general = elo_general.get(ganador, 1500.0)
        p_general = elo_general.get(perdedor, 1500.0)
        
        g_superficie = elo_superficie[superficie].get(ganador, 1500.0)
        p_superficie = elo_superficie[superficie].get(perdedor, 1500.0)
        
        # Ponderación ELO Híbrido (50% General + 50% Superficie) — fuente única en src.features
        # Esto reduce el sesgo del ranking ATP oficial que a menudo subestima especialistas
        elo_final_g = elo_hibrido(g_general, g_superficie)
        elo_final_p = elo_hibrido(p_general, p_superficie)
        
        elo_ganador_previo.append(elo_final_g)
        elo_perdedor_previo.append(elo_final_p)
        elo_ganador_gen_previo.append(g_general)
        elo_perdedor_gen_previo.append(p_general)
        elo_ganador_sup_previo.append(g_superficie)
        elo_perdedor_sup_previo.append(p_superficie)

        # --- Actualizar ELO con MOV + K-schedule ---
        score_str = str(row.get('score', '')) if 'score' in row.index else ''
        mov = _mov_factor(*_extraer_sets(score_str)) if use_mov else 1.0
        K_g = _k_for_player(n_partidos.get(ganador, 0)) if use_k_schedule else 32
        K_p = _k_for_player(n_partidos.get(perdedor, 0)) if use_k_schedule else 32

        e_g = calcular_expectativa(g_general, p_general)
        nuevo_g_gen = g_general + K_g * mov * (1 - e_g)
        nuevo_p_gen = p_general + K_p * mov * (0 - (1 - e_g))
        e_g_sup = calcular_expectativa(g_superficie, p_superficie)
        nuevo_g_sup = g_superficie + K_g * mov * (1 - e_g_sup)
        nuevo_p_sup = p_superficie + K_p * mov * (0 - (1 - e_g_sup))

        elo_general[ganador] = nuevo_g_gen
        elo_general[perdedor] = nuevo_p_gen
        elo_superficie[superficie][ganador] = nuevo_g_sup
        elo_superficie[superficie][perdedor] = nuevo_p_sup

        # Actualizar contadores de partidos
        n_partidos[ganador] = n_partidos.get(ganador, 0) + 1
        n_partidos[perdedor] = n_partidos.get(perdedor, 0) + 1

    df_completo['elo_winner'] = elo_ganador_previo
    df_completo['elo_loser'] = elo_perdedor_previo
    df_completo['elo_winner_general'] = elo_ganador_gen_previo
    df_completo['elo_loser_general'] = elo_perdedor_gen_previo
    df_completo['elo_winner_sup'] = elo_ganador_sup_previo
    df_completo['elo_loser_sup'] = elo_perdedor_sup_previo

    return df_completo, elo_general, elo_superficie
