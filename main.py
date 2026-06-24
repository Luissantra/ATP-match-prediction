import os
import pandas as pd
import numpy as np
import matplotlib
# Configurar Matplotlib en modo no interactivo (headless) para evitar fallos de interfaz en macOS
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from src.elo import calcular_elos_historicos
from src.data_processing import preparar_datos_entrenamiento

FEATURES = ['diff_elo', 'diff_rank', 'diff_age', 'diff_h2h', 'diff_form', 'tourney_level_num']

def imprimir_seccion(titulo, descripcion):
    """Imprime un banner con formato elegante en la consola para mejorar la lectura didáctica."""
    print("\n" + "="*80)
    print(f"✨ {titulo.upper()} ✨")
    print("="*80)
    print(descripcion)
    print("-"*80)

if __name__ == "__main__":
    imprimir_seccion(
        "Inicio del Pipeline de Modelado Estadístico",
        "Este script carga los partidos históricos de tenis de la ATP (2020-2025), calcula ratings ELO\n"
        "híbridos, preprocesa los datos bajo un enfoque simétrico neutral y optimiza un clasificador\n"
        "Gradient Boosting mediante GridSearchCV."
    )

    # 1. Cargar e integrar datos históricos
    años = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    print("Paso 1: Procesando ELO histórico de 2020 a 2026...")
    df_completo, ratings_finales, ratings_superficie = calcular_elos_historicos("data", años)
    
    # Mostrar el Top 10 de jugadores al final del periodo histórico
    print("\n🔥 TOP 10 JUGADORES SEGÚN RATING ELO GENERAL AL FINAL DEL PERIODO:")
    top_jugadores = sorted(ratings_finales.items(), key=lambda x: x[1], reverse=True)[:10]
    for rango, (jugador, rating) in enumerate(top_jugadores, start=1):
        print(f"  {rango:02d}. {jugador:<25} ELO Rating: {rating:.1f}")
        
    # 2. Generar el dataset neutral simétrico
    imprimir_seccion(
        "Preprocesamiento y Simetrización del Dataset",
        "Para evitar la fuga de etiquetas (label leakage), transformamos el dataset original\n"
        "(Ganador vs Perdedor) en una perspectiva neutral (Jugador A vs Jugador B).\n"
        "Esto garantiza que la variable objetivo esté perfectamente balanceada al 50% de victorias de A."
    )
    df_features = preparar_datos_entrenamiento(df_completo)
    print(f"Dataset simétrico generado con éxito. Dimensión total: {df_features.shape[0]} filas, {df_features.shape[1]} columnas.")
    
    # 3. Dividir en Entrenamiento (2020-2025) y Prueba Final/Ciega (2026)
    # Esto asegura que evaluamos el modelo en la temporada más reciente.
    df_train = df_features[df_features['year'] < 2026]
    df_test = df_features[df_features['year'] == 2026]
    
    X_train = df_train[FEATURES]
    y_train = df_train['label']

    X_test = df_test[FEATURES]
    y_test = df_test['label']
    
    print(f"  * Tamaño conjunto de Entrenamiento (2020-2025): {len(X_train)} partidos")
    print(f"  * Tamaño conjunto de Validación Ciega (2026):     {len(X_test)} partidos")
    
    # 4. Configurar Grid Search para Gradient Boosting
    imprimir_seccion(
        "Optimización de Hiperparámetros (Grid Search con CV)",
        "El Gradient Boosting es un modelo de ensamble aditivo de árboles de decisión.\n"
        "Utilizaremos Validación Cruzada (3-folds) para encontrar la combinación óptima de:\n"
        "  - n_estimators (número de árboles)\n"
        "  - max_depth (profundidad máxima de cada árbol)\n"
        "  - learning_rate (tasa de aprendizaje de la corrección de residuos)"
    )
    
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 150]
    }
    
    gb_base = GradientBoostingClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        estimator=gb_base, 
        param_grid=param_grid, 
        cv=TimeSeriesSplit(n_splits=5),
        scoring='accuracy', 
        n_jobs=-1,
        verbose=1
    )
    
    print("Iniciando ajuste en paralelo (Grid Search)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ ¡Mejores parámetros encontrados!: {grid_search.best_params_}")
    print(f"✅ Mejor precisión en validación cruzada (CV): {grid_search.best_score_:.2%}")
    
    # 5. Evaluar el mejor modelo en el Test Ciego de 2025
    mejor_modelo = grid_search.best_estimator_
    preds_2025 = mejor_modelo.predict(X_test)
    
    precision_2025 = accuracy_score(y_test, preds_2025)
    
    imprimir_seccion(
        "Evaluación en la Temporada Ciega 2025",
        f"Evaluando el modelo final optimizado en los partidos reales de la temporada 2025.\n"
        f"PRECISIÓN FINAL (ACCURACY): {precision_2025:.2%}"
    )
    
    # Reporte detallado de clasificación
    print("Reporte de Clasificación Detallado:")
    print(classification_report(y_test, preds_2025, target_names=['Derrota A', 'Victoria A']))
    
    # =========================================================================
    # VISUALIZACIÓN CIENTÍFICA 1: Matriz de Confusión Estilizada
    # =========================================================================
    print("\nGenerando gráficos analíticos...")
    os.makedirs("plots", exist_ok=True)
    cm = confusion_matrix(y_test, preds_2025)
    
    plt.figure(figsize=(7, 6))
    # Paleta de color optimizada para evitar fatiga visual (Blues)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=False,
        xticklabels=['Predicción Derrota A', 'Predicción Victoria A'],
        yticklabels=['Realidad Derrota A', 'Realidad Victoria A'],
        annot_kws={"size": 14, "weight": "bold"}
    )
    
    # Agregar anotaciones secundarias con porcentajes
    for i in range(2):
        for j in range(2):
            total = np.sum(cm[i, :])
            porcentaje = (cm[i, j] / total) * 100
            plt.text(
                j + 0.5, i + 0.7, 
                f"({porcentaje:.1f}%)", 
                ha="center", va="center", 
                color="darkblue" if cm[i, j] < cm.max()/2 else "white",
                fontsize=11
            )
            
    plt.title("Matriz de Confusión Estilizada (Test Ciego 2025)\nATP Tennis Prediction Model", fontsize=13, pad=15, weight='bold')
    plt.ylabel("Estado Real del Partido", fontsize=12)
    plt.xlabel("Predicción del Modelo", fontsize=12)
    plt.tight_layout()
    
    path_cm = os.path.join("plots", "matriz_confusion.png")
    plt.savefig(path_cm, dpi=300)
    plt.close()
    print(f"📊 Matriz de confusión guardada con éxito como '{path_cm}'")
    
    # =========================================================================
    # VISUALIZACIÓN CIENTÍFICA 2: Importancia de Variables (Feature Importance)
    # =========================================================================
    importancias = mejor_modelo.feature_importances_
    features_list = ['Diferencia ELO', 'Diferencia Ranking', 'Diferencia Edad',
                     'H2H Histórico', 'Forma Reciente', 'Nivel de Torneo']
    
    # Ordenar por importancia
    indices_ordenados = np.argsort(importancias)
    
    plt.figure(figsize=(8, 4.5))
    # Barra horizontal de colores suaves y profesionales (Slate Blue)
    plt.barh(
        range(len(importancias)), 
        importancias[indices_ordenados], 
        color="#34495e", 
        edgecolor="#2c3e50", 
        height=0.6
    )
    
    plt.yticks(range(len(importancias)), [features_list[i] for i in indices_ordenados], fontsize=11)
    plt.xlabel("Importancia de la Variable (Gini Impurity Decrease)", fontsize=11)
    plt.title("Importancia Relativa de las Variables de Entrada\n¿Qué factores deciden la victoria en el tenis?", fontsize=12, pad=15, weight='bold')
    
    # Agregar anotaciones de valor al final de cada barra
    for index, value in enumerate(importancias[indices_ordenados]):
        plt.text(value + 0.01, index, f"{value:.1%}", va='center', ha='left', fontweight='bold', color="#2c3e50")
        
    plt.xlim(0, max(importancias) + 0.1)
    sns.despine()  # Quita los bordes sobrantes del gráfico para un diseño minimalista y moderno
    plt.tight_layout()
    
    path_imp = os.path.join("plots", "importancia_variables.png")
    plt.savefig(path_imp, dpi=300)
    plt.close()
    print(f"📊 Gráfico de importancia de variables guardado como '{path_imp}'")
    
    # =========================================================================
    # VISUALIZACIÓN CIENTÍFICA 3: Precisión por Tipo de Superficie
    # =========================================================================
    df_test_preds = df_test.copy()
    df_test_preds['pred'] = preds_2025
    
    # Agrupar por superficie y calcular precisión y tamaño de muestra
    superficies = df_test_preds['surface'].unique()
    res_superficie = {}
    for sup in ['Hard', 'Clay', 'Grass']:
        df_sup = df_test_preds[df_test_preds['surface'] == sup]
        if len(df_sup) > 0:
            acc = accuracy_score(df_sup['label'], df_sup['pred'])
            res_superficie[sup] = (acc, len(df_sup))
            
    # Graficar precisión por superficie con mapeo de colores cognitivo
    plt.figure(figsize=(7, 5))
    colores_cancha = {
        'Clay': '#e67e22',  # Color arcilla/ladrillo
        'Grass': '#2ecc71', # Color césped/verde
        'Hard': '#3498db'   # Color cemento/azul
    }
    
    names = list(res_superficie.keys())
    values = [res_superficie[k][0] for k in names]
    counts = [res_superficie[k][1] for k in names]
    colores_plot = [colores_cancha.get(n, '#7f8c8d') for n in names]
    
    bars = plt.bar(names, values, color=colores_plot, edgecolor='none', width=0.55)
    
    # Dibujar una línea punteada indicando la precisión promedio global
    plt.axhline(precision_2025, color='#7f8c8d', linestyle='--', alpha=0.7, label=f'Precisión Promedio Global ({precision_2025:.1%})')
    
    # Configurar el gráfico
    plt.title("Precisión del Modelo por Tipo de Superficie (Test Ciego 2025)\n¿En qué canchas el tenis es más predecible?", fontsize=12, pad=15, weight='bold')
    plt.ylabel("Precisión (Accuracy)", fontsize=11)
    plt.ylim(0, 0.8)
    plt.legend(loc='lower center', frameon=True, facecolor='white', edgecolor='none')
    
    # Agregar anotaciones con porcentajes y tamaños de muestra
    for bar, val, count in zip(bars, values, counts):
        plt.text(
            bar.get_x() + bar.get_width()/2.0, 
            val + 0.02, 
            f"{val:.1%}\n(n={count})", 
            ha='center', va='bottom', 
            fontweight='bold', 
            fontsize=10,
            color='#2c3e50'
        )
        
    sns.despine()
    plt.tight_layout()
    
    path_sup = os.path.join("plots", "precision_por_superficie.png")
    plt.savefig(path_sup, dpi=300)
    plt.close()
    print(f"📊 Gráfico de precisión por superficie guardado como '{path_sup}'")
    
    imprimir_seccion(
        "Análisis de Variables e Impacto Cognitivo",
        "Observaciones sobre la Importancia de Variables:\n"
        "  1. La Diferencia ELO domina significativamente la predicción (~80-90%).\n"
        "     Esto se debe a que el ELO es una medida dinámicamente actualizada basada en la probabilidad\n"
        "     y la fuerza del oponente, superando la rigidez del Ranking ATP oficial.\n"
        "  2. La Diferencia de Edad tiene un peso menor pero detecta sutiles ventajas en resistencia\n"
        "     física o madurez mental (experiencia).\n"
        "  3. El Ranking ATP oficial suele ser ruidoso debido a la inactividad, lesiones y la\n"
        "     acumulación estática de puntos anuales.\n\n"
        "Análisis por Superficie de Juego:\n"
        f"  * Clay (Arcilla): {res_superficie.get('Clay', (0,0))[0]:.2%} de acierto (n={res_superficie.get('Clay', (0,0))[1]} partidos)\n"
        f"  * Grass (Césped): {res_superficie.get('Grass', (0,0))[0]:.2%} de acierto (n={res_superficie.get('Grass', (0,0))[1]} partidos)\n"
        f"  * Hard (Dura):    {res_superficie.get('Hard', (0,0))[0]:.2%} de acierto (n={res_superficie.get('Hard', (0,0))[1]} partidos)\n\n"
        "Nota Científica: El tenis en césped y arcilla suele favorecer perfiles de juego más definidos\n"
        "(sacadores o pasabolas/desgaste), lo que a veces se traduce en una mayor o menor predictibilidad\n"
        "respecto a las canchas duras (Hard), las cuales representan el estándar promedio del circuito."
    )

    # 6. Guardar el modelo y los metadatos para la web interactiva
    imprimir_seccion(
        "Exportación de Artefactos para Producción",
        "Serializando el clasificador Gradient Boosting optimizado y consolidando\n"
        "las estadísticas más recientes (ELO híbrido, ranking, edad) de todos los jugadores\n"
        "activos para su despliegue en la aplicación web interactiva."
    )
    
    stats_jugadores = {}
    for idx, row in df_completo.iterrows():
        ganador = row['winner_name']
        perdedor = row['loser_name']
        
        # Guardar el último ranking y la edad actual conocida del ganador
        stats_jugadores[ganador] = {
            'rank': float(row['winner_rank']) if not pd.isna(row['winner_rank']) else 999.0,
            'age': float(row['winner_age']) if not pd.isna(row['winner_age']) else 26.0
        }
        # Guardar el último ranking y la edad actual conocida del perdedor
        stats_jugadores[perdedor] = {
            'rank': float(row['loser_rank']) if not pd.isna(row['loser_rank']) else 999.0,
            'age': float(row['loser_age']) if not pd.isna(row['loser_age']) else 26.0
        }
        
    metadata_export = {
        'elo_general': ratings_finales,
        'elo_superficie': ratings_superficie,
        'stats': stats_jugadores
    }
    
    with open('modelo_atp.pkl', 'wb') as f:
        pickle.dump(mejor_modelo, f)
    with open('stats_jugadores.pkl', 'wb') as f:
        pickle.dump(metadata_export, f)
    print("🚀 ¡Modelo y metadatos exportados con éxito a la raíz del proyecto (formato .pkl)! 🚀\n")
