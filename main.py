import pickle
import sklearn
import pandas as pd
from src.elo import calcular_elos_historicos
from src.data_processing import preparar_datos_entrenamiento
from src.features import FEATURES
from src.train import entrenar_modelo, calibrar_modelo, comparar_calibracion, entrenar_todos_los_modelos
from src.evaluate import (
    evaluar, evaluar_con_ic, evaluar_baseline_elo, evaluar_y_graficar,
    graficar_learning_curve, graficar_reliability_diagram, graficar_histograma_probas,
    diagnosticar_gap_cv_test,
)
from src.cv import purged_time_series_splits

AÑOS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
TEST_YEAR = 2026

if __name__ == "__main__":
    print("=== ATP Prediction Pipeline ===")

    # 1. ELO histórico
    print(f"\n[1/4] Calculando ELO histórico ({AÑOS[0]}-{AÑOS[-1]})...")
    df_completo, ratings_finales, ratings_superficie, h2h, form_final = calcular_elos_historicos("data", AÑOS)

    top10 = sorted(ratings_finales.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 ELO:")
    for i, (jugador, rating) in enumerate(top10, 1):
        print(f"  {i:02d}. {jugador:<25} {rating:.1f}")

    # 2. Dataset simétrico
    print("\n[2/4] Preparando dataset simétrico...")
    df_features = preparar_datos_entrenamiento(df_completo)
    print(f"  Dataset: {df_features.shape[0]} filas x {df_features.shape[1]} columnas")

    df_train = df_features[df_features['year'] < TEST_YEAR]
    df_test  = df_features[df_features['year'] == TEST_YEAR]
    X_train, y_train = df_train[FEATURES], df_train['label']
    X_test,  y_test  = df_test[FEATURES],  df_test['label']
    print(f"  Entrenamiento: {len(X_train)} partidos | Test ciego {TEST_YEAR}: {len(X_test)} partidos")

    # 3. Entrenar todos los modelos (LogReg, RF, GBM, XGBoost) + calibrar
    print("\n[3/5] Entrenando multi-modelo con GridSearchCV + CV temporal con embargo...")
    dates_train = df_train['tourney_date'].values
    X_train_arr = X_train.values
    y_train_arr = y_train.values

    todos_modelos = entrenar_todos_los_modelos(X_train_arr, y_train_arr, dates=dates_train)

    # GBM calibrado = modelo principal (compatibilidad con app.py)
    modelo = todos_modelos['gbm']
    # Modelo base sin calibrar para plots de importancia de features
    modelo_base_gbm = entrenar_modelo(X_train, y_train, dates=dates_train)

    # Q4: Elegir calibración óptima para GBM (sigmoid vs isotonic)
    print("\n[3b/5] Comparando calibración sigmoid vs isotonic (GBM)...")
    res_cal = comparar_calibracion(modelo_base_gbm, X_train_arr, y_train_arr, dates=dates_train)
    print(f"  sigmoid log-loss={res_cal['sigmoid_log_loss']:.4f}  "
          f"isotonic log-loss={res_cal['isotonic_log_loss']:.4f}  → mejor: {res_cal['mejor']}")
    if res_cal['mejor'] == 'sigmoid':
        todos_modelos['gbm'] = calibrar_modelo(modelo_base_gbm, X_train_arr, y_train_arr,
                                               dates=dates_train, method='sigmoid')
        modelo = todos_modelos['gbm']
        print("  GBM recalibrado con sigmoid.")

    # 4. Evaluar y graficar (GBM calibrado)
    print("\n[4/5] Evaluando y generando gráficos...")
    evaluar_y_graficar(modelo, X_test, y_test, df_test, FEATURES,
                       modelo_para_importancia=modelo_base_gbm)
    cv_splits = list(purged_time_series_splits(df_train['tourney_date'].values, n_splits=5))
    graficar_learning_curve(modelo_base_gbm, X_train, y_train, cv_splits)
    graficar_reliability_diagram(modelo, X_test, y_test)
    graficar_histograma_probas(modelo, X_test, y_test)

    print("\n[4b/5] Baseline ELO-crudo vs ML (¿aporta el stack?)...")
    met_baseline = evaluar_baseline_elo(df_test, y_test)
    print(f"  BASELINE ELO-crudo:")
    print(f"    log-loss={met_baseline['log_loss']:.4f} [{met_baseline['log_loss_ic']['lower']:.4f}–{met_baseline['log_loss_ic']['upper']:.4f}]")
    print(f"    AUC     ={met_baseline['auc']:.4f} [{met_baseline['auc_ic']['lower']:.4f}–{met_baseline['auc_ic']['upper']:.4f}]")

    print("\n  MODELOS ML (con IC95% bootstrap, n≈137 → ±0.08-0.09 en AUC):")
    metrics_all = {}
    for nombre, m in todos_modelos.items():
        met_ic = evaluar_con_ic(m, X_test, y_test)
        metrics_all[nombre] = {k: v for k, v in met_ic.items() if k in ('accuracy', 'log_loss', 'brier', 'auc')}
        print(f"    {nombre:<14} log-loss={met_ic['log_loss']:.4f} [{met_ic['log_loss_ic']['lower']:.4f}–{met_ic['log_loss_ic']['upper']:.4f}]  "
              f"AUC={met_ic['auc']:.4f} [{met_ic['auc_ic']['lower']:.4f}–{met_ic['auc_ic']['upper']:.4f}]")
    print("  Nota: diferencias < 0.08 en AUC son ruido estadístico con n≈137.")

    # Q3: Diagnóstico causas del gap CV/test
    print("\n[Q3] Diagnóstico gap CV/test:")
    print(diagnosticar_gap_cv_test(
        cv_best_score=0.620,
        test_log_loss=metrics_all['gbm']['log_loss'],
        n_test=len(y_test),
    ))

    # 5. Exportar artefactos
    print("\n[5/5] Exportando artefactos...")
    stats_jugadores = {}
    for _, row in df_completo.iterrows():
        for role in [('winner_name', 'winner_rank', 'winner_age'),
                     ('loser_name',  'loser_rank',  'loser_age')]:
            name = row[role[0]]
            stats_jugadores[name] = {
                'rank': float(row[role[1]]) if not pd.isna(row[role[1]]) else 999.0,
                'age':  float(row[role[2]]) if not pd.isna(row[role[2]]) else 26.0,
            }

    # modelo_atp.pkl: GBM calibrado (compatibilidad app.py existente)
    with open('modelo_atp.pkl', 'wb') as f:
        pickle.dump(modelo, f)

    # modelos_atp.pkl: todos los modelos calibrados para la épica multi-modelo
    with open('modelos_atp.pkl', 'wb') as f:
        pickle.dump(todos_modelos, f)

    # metrics_atp.pkl: métricas test ciego para /api/models
    with open('metrics_atp.pkl', 'wb') as f:
        pickle.dump(metrics_all, f)

    with open('stats_jugadores.pkl', 'wb') as f:
        pickle.dump({'elo_general': ratings_finales, 'elo_superficie': ratings_superficie,
                     'stats': stats_jugadores, 'h2h': h2h, 'form': form_final,
                     'sklearn_version': sklearn.__version__}, f)

    print("  modelo_atp.pkl       — GBM calibrado (API principal)")
    print("  modelos_atp.pkl      — {logreg, randomforest, gbm, xgboost} calibrados")
    print("  metrics_atp.pkl      — métricas test ciego por modelo")
    print("  stats_jugadores.pkl  — ELO/rank/age/H2H/forma")
