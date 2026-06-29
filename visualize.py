import os
import pandas as pd
import numpy as np
import matplotlib
# Configurar Matplotlib en modo no interactivo
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.elo import calcular_elos_historicos
from src.data_processing import preparar_datos_entrenamiento
from src.features import FEATURES


def graficar_evolucion_elo(data_dir, años):
    """
    Calcula los ratings ELO históricos y grafica la evolución temporal de los
    ratings para los 5 mejores jugadores al finalizar el periodo.

    Direct labeling al final de cada línea, alta relación señal/ruido, colores curados.
    """
    print("\nGenerando Gráfico de Evolución Temporal de ELO...")

    # 1. Calcular el ELO histórico (devuelve df, elo_general, elo_superficie, stats_acumuladas)
    df_completo, ratings_finales, _, *_ = calcular_elos_historicos(data_dir, años)

    # 2. Identificar el Top 5 de jugadores al final del periodo
    top_5 = [j for j, _ in sorted(ratings_finales.items(), key=lambda x: x[1], reverse=True)[:5]]
    print(f"Top 5 jugadores: {top_5}")

    df_completo = df_completo.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)
    df_completo['fecha'] = pd.to_datetime(df_completo['tourney_date'], format='%Y%m%d', errors='coerce')

    plt.figure(figsize=(12, 6))
    colores = {
        top_5[0]: '#e74c3c',
        top_5[1]: '#2ecc71',
        top_5[2]: '#3498db',
        top_5[3]: '#f1c40f',
        top_5[4]: '#9b59b6',
    }

    for jugador in top_5:
        partidos = df_completo[(df_completo['winner_name'] == jugador) | (df_completo['loser_name'] == jugador)].copy()
        elos, fechas = [], []
        for _, row in partidos.iterrows():
            fechas.append(row['fecha'])
            elos.append(row['elo_winner'] if row['winner_name'] == jugador else row['elo_loser'])

        serie = pd.Series(elos, index=fechas).sort_index()
        serie_suave = serie.rolling(window=15, min_periods=1).mean()

        color = colores.get(jugador, '#95a5a6')
        plt.plot(serie_suave.index, serie_suave.values, label=jugador, color=color, linewidth=2.5, alpha=0.95)

        # Direct labeling al final de la línea
        if not serie_suave.empty:
            plt.text(serie_suave.index[-1] + pd.Timedelta(days=20), serie_suave.values[-1],
                     f" {jugador} ({int(serie_suave.values[-1])})",
                     va='center', ha='left', color=color, fontsize=9.5, fontweight='bold')

    ax = plt.gca()
    xlims = ax.get_xlim()
    ax.set_xlim(xlims[0], xlims[1] + 250)

    plt.title("Evolución Temporal del Rating ELO Híbrido (Top 5 ATP)\nTendencia y estado de forma",
              fontsize=14, pad=15, fontweight='bold', color="#2c3e50")
    plt.xlabel("Línea de Tiempo", fontsize=11, color="#34495e")
    plt.ylabel("Rating ELO Híbrido (Puntos)", fontsize=11, color="#34495e")
    plt.grid(axis='y', linestyle='--', alpha=0.4, color='#95a5a6')
    sns.despine(top=True, right=True)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    out = os.path.join("plots", "evolucion_elo_top.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Guardado: {out}")


def graficar_correlaciones(data_dir, años):
    """
    EDA coherente con el modelo: matriz de correlación de las features REALES (las que
    consume el modelo) frente al label. Sustituye el antiguo pairplot de altura, que
    exploraba una variable (diff_ht) que el modelo no usa. Revela colinealidad
    (p.ej. diff_rank ↔ diff_elo) y qué features correlacionan con el resultado.
    """
    print("\nGenerando matriz de correlación de las features del modelo...")
    df_completo, _, _, *_ = calcular_elos_historicos(data_dir, años)
    feats = preparar_datos_entrenamiento(df_completo)

    cols = FEATURES + ['label']
    corr = feats[cols].corr()

    labels = {
        'diff_elo_general': 'ELO General', 'diff_elo_sup': 'ELO Superficie',
        'diff_rank': 'Ranking', 'is_unranked': 'Sin Ranking',
        'diff_age': 'Edad', 
        'diff_matches_played': 'Experiencia', 'diff_tb_ratio': 'Tie-breaks',
        'label': 'Victoria A',
    }
    nombres = [labels.get(c, c) for c in cols]

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={'shrink': 0.8},
                xticklabels=nombres, yticklabels=nombres, annot_kws={'size': 9})
    plt.title("Correlación entre Features del Modelo y el Resultado\n¿Colinealidad? ¿Qué correlaciona con la victoria?",
              fontsize=12, pad=15, weight='bold')
    plt.xticks(rotation=40, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    out = os.path.join("plots", "correlacion_features.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Guardado: {out}")


if __name__ == "__main__":
    data_dir = "data"
    años = [2020, 2021, 2022, 2023, 2024, 2025]
    graficar_evolucion_elo(data_dir, años)
    graficar_correlaciones(data_dir, años)
