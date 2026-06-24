import os
import pandas as pd
import numpy as np
import matplotlib
# Configurar Matplotlib en modo no interactivo
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.elo import calcular_elos_historicos
from src.data_processing import crear_dataset_visual

def graficar_evolucion_elo(data_dir, años):
    """
    Calcula los ratings ELO históricos y grafica la evolución temporal de los
    ratings para los 5 mejores jugadores al finalizar el periodo.

    Aplica principios cognitivos de diseño de visualización:
    - Direct Labeling: Coloca etiquetas directas sobre el final de cada línea para
      evitar que el ojo se desplace constantemente entre el gráfico y la leyenda.
    - Alta relación señal/ruido: Mantiene el fondo limpio (sin bordes innecesarios, cuadrícula tenue).
    - Colores curados con contraste intencional.
    """
    print("\nGenerando Gráfico de Evolución Temporal de ELO (2020-2025)...")
    
    # 1. Calcular el ELO histórico de todos los jugadores
    df_completo, ratings_finales = calcular_elos_historicos(data_dir, años)
    
    # 2. Identificar el Top 5 de jugadores al final del periodo
    top_5 = [jugador for jugador, rating in sorted(ratings_finales.items(), key=lambda x: x[1], reverse=True)[:5]]
    print(f"Top 5 jugadores seleccionados para trazar su evolución: {top_5}")
    
    # Asegurar orden cronológico de partidos
    df_completo = df_completo.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)
    df_completo['fecha'] = pd.to_datetime(df_completo['tourney_date'], format='%Y%m%d', errors='coerce')
    
    plt.figure(figsize=(12, 6))
    
    # Colores personalizados seleccionados con intención cognitiva para los mejores jugadores
    colores = {
        top_5[0]: '#e74c3c',  # Rojo fuerte para el n°1 (ej. Sinner)
        top_5[1]: '#2ecc71',  # Verde brillante (ej. Alcaraz)
        top_5[2]: '#3498db',  # Azul (ej. Djokovic/Medvedev)
        top_5[3]: '#f1c40f',  # Amarillo/Oro
        top_5[4]: '#9b59b6'   # Púrpura
    }
    
    for jugador in top_5:
        # Filtrar partidos donde el jugador haya participado
        partidos_jugador = df_completo[(df_completo['winner_name'] == jugador) | (df_completo['loser_name'] == jugador)].copy()
        
        # Extraer el rating ELO previo al partido de la columna correspondiente
        elos = []
        fechas = []
        
        for _, row in partidos_jugador.iterrows():
            fechas.append(row['fecha'])
            if row['winner_name'] == jugador:
                elos.append(row['elo_winner'])
            else:
                elos.append(row['elo_loser'])
                
        # Crear serie temporal
        serie_elo = pd.Series(elos, index=fechas).sort_index()
        
        # Suavizado ligero mediante media móvil para resaltar la tendencia macro
        serie_suave = serie_elo.rolling(window=15, min_periods=1).mean()
        
        color_linea = colores.get(jugador, '#95a5a6')
        plt.plot(serie_suave.index, serie_suave.values, label=jugador, color=color_linea, linewidth=2.5, alpha=0.95)
        
        # PRINCIPIO COGNITIVO: Direct Labeling (Etiquetado Directo)
        # Coloca el nombre del jugador al final de su línea (última fecha registrada)
        if not serie_suave.empty:
            ultima_fecha = serie_suave.index[-1]
            ultimo_elo = serie_suave.values[-1]
            plt.text(
                ultima_fecha + pd.Timedelta(days=20), ultimo_elo, 
                f" {jugador} ({int(ultimo_elo)})", 
                va='center', ha='left', 
                color=color_linea, fontsize=9.5, fontweight='bold'
            )
            
    # Ajustar límites del eje X para dar espacio a las etiquetas directas de texto
    ax = plt.gca()
    xlims = ax.get_xlim()
    # Extender el límite derecho en ~250 días
    ax.set_xlim(xlims[0], xlims[1] + 250)
    
    plt.title("Evolución Temporal del Rating ELO Híbrido (Top 5 ATP 2020-2025)\nVisualización limpia de la tendencia y estado de forma", fontsize=14, pad=15, fontweight='bold', color="#2c3e50")
    plt.xlabel("Línea de Tiempo", fontsize=11, color="#34495e")
    plt.ylabel("Rating ELO Híbrido (Puntos)", fontsize=11, color="#34495e")
    
    # Cuadrícula horizontal sutil para facilitar la lectura del rating sin saturar el fondo
    plt.grid(axis='y', linestyle='--', alpha=0.4, color='#95a5a6')
    sns.despine(top=True, right=True) # Elimina los bordes superior y derecho
    
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    output_image = os.path.join("plots", "evolucion_elo_top.png")
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"📊 Gráfico de evolución temporal de ELO guardado con éxito como '{output_image}'!")

if __name__ == "__main__":
    data_dir = "data"
    años_completos = [2020, 2021, 2022, 2023, 2024, 2025]
    
    # 1. Graficar Evolución Temporal de ELO (Top 5)
    graficar_evolucion_elo(data_dir, años_completos)
    
    # 2. Generar el Pair Plot clásico de Seaborn (EDA de Diferencias Físicas y de Ranking)
    filepath_2024 = os.path.join(data_dir, "2024.csv")
    print(f"\nCargando datos de {filepath_2024} y preparando dataset para visualización de correlaciones...")
    df_visual = crear_dataset_visual(filepath_2024)
    
    print("Generando el Pair Plot con Seaborn...")
    sns.set_theme(style="ticks")
    
    # Mapeamos las etiquetas numéricas a texto legible para la leyenda del gráfico
    df_visual['Resultado_Jugador_A'] = df_visual['label'].map({1: 'Victoria', 0: 'Derrota'})
    
    # Graficar dispersiones cruzadas
    g = sns.pairplot(
        df_visual[['diff_rank', 'diff_age', 'diff_ht', 'Resultado_Jugador_A']],
        hue='Resultado_Jugador_A',
        palette={'Victoria': '#2ecc71', 'Derrota': '#e74c3c'},  # Verde para victoria, rojo para derrota
        diag_kind="kde",
        plot_kws={'alpha': 0.35, 's': 15, 'edgecolor': 'none'}
    )
    
    # Ajustes cognitivos de legibilidad del Pair Plot
    g.fig.suptitle("Análisis Exploratorio de Diferencias en Tenis (ATP 2024)\nCorrelaciones entre Edad, Ranking y Altura con el Resultado", fontsize=13, weight='bold', y=1.02)
    
    # Guardar la imagen localmente
    os.makedirs("plots", exist_ok=True)
    output_pairplot = os.path.join("plots", "pairplot_tenis.png")
    g.savefig(output_pairplot, dpi=300)
    plt.close()
    print(f"📊 Gráfico Pair Plot guardado con éxito como '{output_pairplot}'!")
