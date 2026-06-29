#!/usr/bin/env python
"""
Script CLI: Simulación del torneo ATP en curso de mayor categoría.
Descarga el draw desde TML-Database y simula con el modelo pkl existente.
"""

import os
import sys
import pickle

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.draw import descargar_ongoing, listar_torneos
from src.simulator import simular_torneo_montecarlo


def main():
    print("=== SIMULADOR DE TORNEOS MONTE CARLO ===\n")

    # 1. Cargar modelos existentes
    print("[1/4] Cargando modelos desde models/...")
    with open("models/modelos_atp.pkl", "rb") as f:
        modelo = pickle.load(f)
    with open("models/stats_jugadores.pkl", "rb") as f:
        metadata = pickle.load(f)

    elo_general = metadata['elo_general']
    elo_superficie = metadata['elo_superficie']
    stats_jugadores = metadata['stats']
    print(f"  Modelo cargado. {len(elo_general)} jugadores con ELO.")

    # 2. Descargar torneos en curso
    print("\n[2/4] Descargando ongoing_tourneys.csv desde TML-Database...")
    df_ongoing = descargar_ongoing()
    print(f"  {len(df_ongoing)} partidos descargados.")

    torneos = listar_torneos(df_ongoing)
    if not torneos:
        print("  No hay torneos ATP en curso. Abortando.")
        sys.exit(0)

    torneo = torneos[0]
    print(f"  Torneo seleccionado: {torneo['name']} (nivel {torneo['level']}, {torneo['surface']})")

    # 3. Reconstruir draw inicial
    print("\n[3/4] Reconstruyendo cuadro inicial...")
    df_torney = df_ongoing[df_ongoing['tourney_name'] == torneo['name']].copy()

    for r in ['R128', 'R64', 'R32', 'R16', 'QF']:
        if r in df_torney['round'].values:
            first_round = r
            break
    else:
        first_round = df_torney['round'].iloc[0]

    df_first = df_torney[df_torney['round'] == first_round].sort_values('match_num')
    initial_draw = []
    for _, row in df_first.iterrows():
        initial_draw.append(row['winner_name'])
        initial_draw.append(row['loser_name'])

    # Rellenar hasta la siguiente potencia de 2 (byes)
    import math
    n = len(initial_draw)
    next_pow2 = 2 ** math.ceil(math.log2(n)) if n > 1 else 2
    if next_pow2 != n:
        initial_draw.extend([None] * (next_pow2 - n))

    print(f"  Ronda inicial: {first_round} — {len(initial_draw)} jugadores (incl. byes).")

    # 4. Simulación Monte Carlo
    n_sims = 10000
    print(f"\n[4/4] Ejecutando {n_sims} simulaciones...")
    surface = torneo['surface']
    df_prob = simular_torneo_montecarlo(
        initial_draw, surface, modelo, elo_general, elo_superficie, stats_jugadores,
        n_simulaciones=n_sims, seed=42
    )

    # Reporte
    rondas = list(df_prob.columns)
    header_cols = ['ELO Gen', 'Rank'] + rondas
    print(f"\n{'Jugador':<26} {'ELO Gen':>7} {'Rank':>5}  " + '  '.join(f"{r:>6}" for r in rondas))
    print("=" * (26 + 10 + 8 * len(rondas)))

    for player, row in df_prob.head(16).iterrows():
        elo = elo_general.get(player, 1500.0)
        rank = stats_jugadores.get(player, {}).get('rank', 999.0)
        rank_str = str(int(rank)) if rank < 999 else 'S/R'
        probs = '  '.join(f"{row[r]:>5.1f}%" for r in rondas)
        print(f"{player:<26} {elo:>7.1f} {rank_str:>5}  {probs}")

    print("\nSimulación completada.")


if __name__ == '__main__':
    main()
