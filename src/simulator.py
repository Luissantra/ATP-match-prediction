"""
Motor de Simulación de Torneo Monte Carlo
========================================

Simula cuadros de eliminatorias de tenis (brackets de tamaño potencia de 2)
mediante simulación de Monte Carlo. Calcula la probabilidad de cada jugador
de avanzar a cada ronda y coronarse campeón, basándose en las predicciones
de probabilidad de victoria de nuestro modelo LogReg calibrado.
"""

import numpy as np
import pandas as pd

from src.features import RANK_CAP, elo_hibrido, vector_from_features


def predecir_probabilidad_matchup(player_a, player_b, surface, modelo, elo_general, elo_superficie, stats_jugadores):
    """
    Calcula la probabilidad de que Player A derrote a Player B.
    Misma lógica de inferencia que app.py para evitar sesgos.
    """
    if player_a == player_b:
        return 0.5

    # Obtener ratings
    gen_a = elo_general.get(player_a, 1500.0)
    gen_b = elo_general.get(player_b, 1500.0)
    sup_a = elo_superficie.get(surface, {}).get(player_a, 1500.0)
    sup_b = elo_superficie.get(surface, {}).get(player_b, 1500.0)

    rank_a = stats_jugadores.get(player_a, {}).get('rank', 999.0)
    rank_b = stats_jugadores.get(player_b, {}).get('rank', 999.0)
    age_a = stats_jugadores.get(player_a, {}).get('age', 26.0)
    age_b = stats_jugadores.get(player_b, {}).get('age', 26.0)

    # is_unranked desde el flag exportado (máscara NaN real), sin recalcular rank>=999.
    info_a = stats_jugadores.get(player_a)
    info_b = stats_jugadores.get(player_b)
    unranked_a = 1 if info_a is None else int(info_a.get('is_unranked', int(rank_a >= 999)))
    unranked_b = 1 if info_b is None else int(info_b.get('is_unranked', int(rank_b >= 999)))

    feat = {
        'diff_elo_general': gen_a - gen_b,
        'diff_elo_sup':     sup_a - sup_b,
        'diff_rank':        min(rank_a, RANK_CAP) - min(rank_b, RANK_CAP),
        'is_unranked':      unranked_a - unranked_b,
        'diff_age':         age_a - age_b,
    }

    features = np.array([vector_from_features(feat)])
    # predict_proba devuelve [prob_perder, prob_ganar]
    probs = modelo.predict_proba(features)[0]
    return float(probs[1])


def simular_torneo_montecarlo(initial_draw, surface, modelo, elo_general, elo_superficie, stats_jugadores, n_simulaciones=10000, seed=42):
    """
    Simula el torneo completo N veces y calcula la probabilidad de cada jugador
    de alcanzar cada ronda.

    Parameters
    ----------
    initial_draw : list of str
        Lista de nombres de jugadores en el orden inicial del bracket (tamaño potencia de 2, ej: 8, 16, 32, 64, 128).
    surface : str
        Superficie del torneo ('Hard', 'Clay', 'Grass').
    modelo : object
        Modelo LogReg calibrado cargado de modelos_atp.pkl.
    elo_general : dict
    elo_superficie : dict
    stats_jugadores : dict
    n_simulaciones : int
    seed : int

    Returns
    -------
    pd.DataFrame
        DataFrame indexado por jugador, con columnas que representan el % de probabilidad de alcanzar cada ronda.
    """
    n_players = len(initial_draw)
    # Validar potencia de 2
    if n_players < 2 or (n_players & (n_players - 1)) != 0:
        raise ValueError(f"El tamaño del cuadro inicial ({n_players}) debe ser una potencia de 2 mayor o igual a 2.")

    rng = np.random.default_rng(seed)

    # Identificar las rondas según el tamaño del cuadro
    # Ej: para 8 jugadores, rondas = ['QF', 'SF', 'F', 'Winner']
    # Para 128 jugadores, rondas = ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F', 'Winner']
    rondas_posibles = {
        128: ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F', 'Winner'],
        64:  ['R64', 'R32', 'R16', 'QF', 'SF', 'F', 'Winner'],
        32:  ['R32', 'R16', 'QF', 'SF', 'F', 'Winner'],
        16:  ['R16', 'QF', 'SF', 'F', 'Winner'],
        8:   ['QF', 'SF', 'F', 'Winner'],
        4:   ['SF', 'F', 'Winner'],
        2:   ['F', 'Winner']
    }
    rondas = rondas_posibles.get(n_players, [f"Ronda_{i}" for i in range(int(np.log2(n_players)) + 1)])

    # Diccionario para acumular conteos de rondas alcanzadas por jugador
    # Estructura: {jugador: {ronda: count}}
    conteos = {p: {r: 0 for r in rondas} for p in initial_draw if p is not None}

    # Pre-calcular una matriz de probabilidades de matchups de todos contra todos para optimizar
    # O bien calcular dinámicamente con una caché
    cache_matchups = {}

    def get_prob_matchup(p1, p2):
        if p1 is None or p2 is None:
            return 0.5
        pair = tuple(sorted((p1, p2)))
        if pair not in cache_matchups:
            prob = predecir_probabilidad_matchup(p1, p2, surface, modelo, elo_general, elo_superficie, stats_jugadores)
            cache_matchups[pair] = prob
        
        # Devolver prob de p1 ganando a p2
        prob_sorted = cache_matchups[pair]
        return prob_sorted if p1 == pair[0] else (1.0 - prob_sorted)

    print(f"Simulando torneo de {n_players} jugadores ({n_simulaciones} iteraciones)...")

    for sim in range(n_simulaciones):
        current_players = list(initial_draw)
        
        # Simular ronda por ronda
        for r_idx, r_name in enumerate(rondas[:-1]):
            # Marcar que todos los jugadores en current_players alcanzaron esta ronda
            for p in current_players:
                if p is not None:
                    conteos[p][r_name] += 1
            
            next_players = []
            for i in range(0, len(current_players), 2):
                p1 = current_players[i]
                p2 = current_players[i+1]
                
                if p1 is None and p2 is None:
                    next_players.append(None)
                elif p1 is None:
                    next_players.append(p2)
                elif p2 is None:
                    next_players.append(p1)
                else:
                    # Simular partido entre p1 y p2
                    p1_win_prob = get_prob_matchup(p1, p2)
                    if rng.random() < p1_win_prob:
                        next_players.append(p1)
                    else:
                        next_players.append(p2)
            
            current_players = next_players

        # Registrar al campeón final (el último sobreviviente)
        campeon = current_players[0]
        if campeon is not None:
            conteos[campeon][rondas[-1]] += 1

    # Construir DataFrame final con los porcentajes de probabilidad
    df_prob = pd.DataFrame.from_dict(conteos, orient='index')
    df_prob = (df_prob / n_simulaciones) * 100.0

    # Ordenar por probabilidad de ser campeón descendente
    df_prob = df_prob.sort_values(by=rondas[-1], ascending=False)
    
    return df_prob
