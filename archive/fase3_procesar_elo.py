import pandas as pd
import numpy as np
import os
from fase3_elo import calcular_expectativa, actualizar_ratings

def calcular_elos_historicos(data_dir, años):
    # Diccionarios para almacenar el ELO actual de cada jugador
    # Si un jugador no existe, empezará con 1500
    elo_general = {}
    
    # ELOs por superficie: Clay, Grass, Hard
    elo_superficie = {
        'Clay': {},
        'Grass': {},
        'Hard': {}
    }
    
    lista_dfs = []
    
    # 1. Cargar y ordenar cronológicamente los años indicados
    for año in años:
        filepath = os.path.join(data_dir, f"{año}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            lista_dfs.append(df)
            
    df_completo = pd.concat(lista_dfs, ignore_index=True)
    # Ordenar por fecha de torneo y número de partido para consistencia cronológica
    df_completo = df_completo.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)
    
    # Listas para almacenar los ELOs calculados antes de cada partido
    elo_ganador_previo = []
    elo_perdedor_previo = []
    
    print("Calculando ratings ELO paso a paso...")
    for idx, row in df_completo.iterrows():
        ganador = row['winner_name']
        perdedor = row['loser_name']
        superficie = row['surface']
        
        # Validar que la superficie sea conocida, si no, usar Hard por defecto
        if superficie not in elo_superficie:
            superficie = 'Hard'
            
        # 2. Recuperar ratings actuales (o 1500 si es su primer partido)
        g_general = elo_general.get(ganador, 1500.0)
        p_general = elo_general.get(perdedor, 1500.0)
        
        g_superficie = elo_superficie[superficie].get(ganador, 1500.0)
        p_superficie = elo_superficie[superficie].get(perdedor, 1500.0)
        
        # 3. Ponderar ELO (50% general, 50% superficie) para obtener el rating de entrada al partido
        elo_final_g = 0.5 * g_general + 0.5 * g_superficie
        elo_final_p = 0.5 * p_general + 0.5 * p_superficie
        
        # Guardar ratings previos al inicio del partido
        elo_ganador_previo.append(elo_final_g)
        elo_perdedor_previo.append(elo_final_p)
        
        # 4. Actualizar ratings (el ganador tiene resultado 1 en la fórmula de actualización)
        # Actualizamos tanto el ELO general como el ELO de superficie
        nuevo_g_gen, nuevo_p_gen = actualizar_ratings(g_general, p_general, resultado_A=1)
        nuevo_g_sup, nuevo_p_sup = actualizar_ratings(g_superficie, p_superficie, resultado_A=1)
        
        # Guardar los nuevos ratings en los diccionarios
        elo_general[ganador] = nuevo_g_gen
        elo_general[perdedor] = nuevo_p_gen
        elo_superficie[superficie][ganador] = nuevo_g_sup
        elo_superficie[superficie][perdedor] = nuevo_p_sup
        
    # Añadir las nuevas características al DataFrame
    df_completo['elo_winner'] = elo_ganador_previo
    df_completo['elo_loser'] = elo_perdedor_previo
    
    return df_completo, elo_general

if __name__ == "__main__":
    # Procesaremos los datos del 2020 al 2024 para tener suficiente historial
    años_historial = [2020, 2021, 2022, 2023, 2024]
    df_con_elo, ratings_finales = calcular_elos_historicos("data", años_historial)
    
    print("\nRatings ELO general final - Top 5 jugadores al acabar 2024:")
    top_jugadores = sorted(ratings_finales.items(), key=lambda x: x[1], reverse=True)[:5]
    for jugador, elo in top_jugadores:
        print(f"{jugador}: {elo}")
