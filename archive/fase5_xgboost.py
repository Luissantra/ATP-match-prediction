import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  # 🛠️ Reemplazamos XGBoost por Scikit-Learn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from fase3_procesar_elo import calcular_elos_historicos
from fase4_random_forest import preparar_datos_entrenamiento

if __name__ == "__main__":
    # 1. Cargar e integrar datos históricos ampliando hasta 2025
    años = [2020, 2021, 2022, 2023, 2024, 2025]
    print("Paso 1: Procesando ELO histórico de 2020 a 2025...")
    df_completo, _ = calcular_elos_historicos("data", años)
    
    # 2. Generar el dataset neutral simétrico
    df_features = preparar_datos_entrenamiento(df_completo)
    
    # 3. Dividir en Entrenamiento (2020-2024) y Prueba Final (2025)
    df_train = df_features[df_features['year'] < 2025]
    df_test = df_features[df_features['year'] == 2025]
    
    X_train = df_train[['diff_elo', 'diff_rank', 'diff_age']]
    y_train = df_train['label']
    
    X_test = df_test[['diff_elo', 'diff_rank', 'diff_age']]
    y_test = df_test['label']
    
    print(f"\nTamaño entrenamiento (2020-2024): {len(X_train)} partidos")
    print(f"Tamaño prueba ciega (2025): {len(X_test)} partidos")
    
    # 4. Configurar Grid Search para Gradient Boosting
    print("\nPaso 2: Iniciando Grid Search con Validación Cruzada...")
    
    # Parámetros a probar (los mismos conceptos)
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 150]
    }
    
    gb_base = GradientBoostingClassifier(random_state=42)
    
    # Realizamos Validación Cruzada de 3 pliegues (cv=3)
    grid_search = GridSearchCV(
        estimator=gb_base, 
        param_grid=param_grid, 
        cv=3, 
        scoring='accuracy', 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n¡Mejores parámetros encontrados!: {grid_search.best_params_}")
    print(f"Mejor precisión en entrenamiento (CV): {grid_search.best_score_:.2%}")
    
    # 5. Evaluar el mejor modelo en el Test Ciego de 2025
    mejor_modelo = grid_search.best_estimator_
    preds_2025 = mejor_modelo.predict(X_test)
    
    precision_2025 = accuracy_score(y_test, preds_2025)
    print(f"\n🚀 PRECISIÓN FINAL (Accuracy) EN LA TEMPORADA 2025: {precision_2025:.2%}")
    
    # Reporte detallado de clasificación
    print("\nReporte de Clasificación Detallado:")
    print(classification_report(y_test, preds_2025, target_names=['Derrota A', 'Victoria A']))
