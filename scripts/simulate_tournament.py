#!/usr/bin/env python
"""
Script de Simulación de Torneo: Australian Open 2026
===================================================

Este script automatiza el flujo completo para predecir un torneo entero:
1. Calcula los ratings ELO y estadísticas acumulados hasta el inicio del torneo (2026-01-18).
2. Entrena y calibra dinámicamente un modelo fresco con todos los datos históricos previos.
3. Reconstruye el cuadro de 128 jugadores de primera ronda (R128) desde data/ongoing_tourneys.csv.
4. Ejecuta 10,000 simulaciones de Monte Carlo.
5. Imprime el reporte de probabilidades del torneo en formato Markdown.
"""

import os
import sys
import pandas as pd
import numpy as np

# Alinear path para importar módulos de src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.elo import calcular_elos_historicos
from src.data_processing import preparar_datos_entrenamiento
from src.features import FEATURES
from src.train import entrenar_modelo, calibrar_modelo
from src.simulator import simular_torneo_montecarlo


def main():
    print("=== SIMULADOR DE TORNEOS MONTE CARLO ===")
    
    # Parámetros del Australian Open 2026
    data_dir = "data"
    años = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    ao_start_date = 20260118  # 18 de Enero de 2026
    surface = "Hard"
    
    # 1. Calcular ELOs y estadísticas antes de la fecha de inicio del torneo (sin leakage)
    print(f"\n[1/5] Calculando ELOs y estadísticas previos al {ao_start_date}...")
    df_pre, elo_gen, elo_sup, stats_acumuladas = calcular_elos_historicos(
        data_dir, años, hasta_fecha=ao_start_date
    )
    
    stats_jugadores = {}
    for _, row in df_pre.iterrows():
        for role in [('winner_name', 'winner_rank', 'winner_age'),
                     ('loser_name',  'loser_rank',  'loser_age')]:
            name = row[role[0]]
            sa = stats_acumuladas.get(name, {})
            stats_jugadores[name] = {
                'rank': float(row[role[1]]) if not pd.isna(row[role[1]]) else 999.0,
                'age':  float(row[role[2]]) if not pd.isna(row[role[2]]) else 26.0,
                'matches_played': sa.get('matches_played', 0),
                'tb_wins': sa.get('tb_wins', 0),
                'tb_played': sa.get('tb_played', 0),
            }
            
    print(f"  Ratings calculados para {len(elo_gen)} jugadores.")
    
    # 2. Entrenar y calibrar modelo fresh con datos históricos previos
    print("\n[2/5] Entrenando y calibrando modelo dinámico...")
    df_features_train = preparar_datos_entrenamiento(df_pre)
    X_train = df_features_train[FEATURES].values
    y_train = df_features_train['label'].values
    dates_train = df_features_train['tourney_date'].values
    
    modelo_base, _, _ = entrenar_modelo(X_train, y_train, dates=dates_train)
    modelo = calibrar_modelo(modelo_base, X_train, y_train, dates=dates_train)
    print("  Modelo LogReg calibrado y listo.")
    
    # 3. Cargar el draw inicial de 128 jugadores del torneo
    print("\n[3/5] Reconstruyendo el cuadro (draw) del Australian Open 2026...")
    ongoing_path = os.path.join(data_dir, "ongoing_tourneys.csv")
    if not os.path.exists(ongoing_path):
        raise FileNotFoundError(f"No se encontró el archivo de torneos en curso: {ongoing_path}")
        
    df_ongoing = pd.read_csv(ongoing_path)
    df_ao_r128 = df_ongoing[
        (df_ongoing['tourney_name'] == 'Australian Open') & 
        (df_ongoing['round'] == 'R128')
    ].copy()
    
    if len(df_ao_r128) == 0:
        raise ValueError("No se encontraron partidos de R128 para el Australian Open en ongoing_tourneys.csv")
        
    # Ordenar por match_num para reconstruir la estructura binaria del cuadro
    df_ao_r128 = df_ao_r128.sort_values(by='match_num')
    
    initial_draw = []
    for _, row in df_ao_r128.iterrows():
        # Colocar al ganador y al perdedor del partido juntos en la primera ronda del simulador
        initial_draw.append(row['winner_name'])
        initial_draw.append(row['loser_name'])
        
    print(f"  Cuadro inicial cargado con {len(initial_draw)} jugadores.")
    
    # 4. Ejecutar simulación Monte Carlo
    print("\n[4/5] Ejecutando simulación Monte Carlo (10,000 iteraciones)...")
    df_prob = simular_torneo_montecarlo(
        initial_draw, surface, modelo, elo_gen, elo_sup, stats_jugadores,
        n_simulaciones=10000, seed=42
    )
    
    # 5. Mostrar resultados
    print("\n[5/5] Resultados de la simulación (Top 15 Favoritos):")
    print("==========================================================================================")
    print(f"| {'Jugador':<24} | {'ELO Gen':<7} | {'Rank':<5} | {'R64 %':<7} | {'R32 %':<7} | {'R16 %':<7} | {'QF %':<6} | {'SF %':<6} | {'F %':<5} | {'W %':<5} |")
    print("==========================================================================================")
    
    top_15 = df_prob.head(15)
    for idx, (player_name, row) in enumerate(top_15.iterrows(), 1):
        p_elo = elo_gen.get(player_name, 1500.0)
        p_rank = stats_jugadores.get(player_name, {}).get('rank', 999.0)
        rank_str = str(int(p_rank)) if p_rank < 999 else "S/R"
        
        print(f"| {player_name:<24} | {p_elo:<7.1f} | {rank_str:<5} | {row['R64']:>6.1f}% | {row['R32']:>6.1f}% | {row['R16']:>6.1f}% | {row['QF']:>5.1f}% | {row['SF']:>5.1f}% | {row['F']:>4.1f}% | {row['Winner']:>4.1f}% |")
        
    print("==========================================================================================")
    print("\nSimulación completada con éxito. Nota: Las probabilidades estiman el desempeño general basado")
    print("en ratings ELO, edad y ranking agregados al inicio del torneo.")


if __name__ == '__main__':
    main()
