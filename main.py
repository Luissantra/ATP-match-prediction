import pickle
import sklearn
import pandas as pd
from src.elo import calcular_elos_historicos
from src.data_processing import preparar_datos_entrenamiento
from src.features import FEATURES
from src.train import calibrar_modelo, comparar_calibracion, entrenar_todos_los_modelos
from src.evaluate import (
    evaluar, evaluar_con_ic, evaluar_baseline_elo, evaluar_y_graficar,
    graficar_learning_curve, graficar_reliability_diagram, graficar_histograma_probas,
    diagnosticar_gap_cv_test,
)
from src.cv import purged_time_series_splits

AÑOS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
TRAIN_END_YEAR = 2025   # entrena con < TRAIN_END_YEAR (2020-2024)
TEST_YEAR      = 2025   # test principal: año completo, n suficiente para IC razonables
SECONDARY_YEAR = 2026   # evaluación secundaria: año parcial (referencial, n pequeño)

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

    df_train     = df_features[df_features['year'] < TRAIN_END_YEAR]
    df_test      = df_features[df_features['year'] == TEST_YEAR]
    df_secondary = df_features[df_features['year'] == SECONDARY_YEAR]
    X_train, y_train = df_train[FEATURES], df_train['label']
    X_test,  y_test  = df_test[FEATURES],  df_test['label']
    X_sec,   y_sec   = df_secondary[FEATURES], df_secondary['label']
    print(f"  Entrenamiento (2020-{TRAIN_END_YEAR-1}): {len(X_train)} partidos")
    print(f"  Test principal ({TEST_YEAR}):             {len(X_test)} partidos")
    print(f"  Eval. secundaria ({SECONDARY_YEAR}):          {len(X_sec)} partidos (referencial)")

    # 3. Entrenar todos los modelos (LogReg, RF, GBM, XGBoost) + calibrar
    print("\n[3/5] Entrenando multi-modelo con GridSearchCV + CV temporal con embargo...")
    dates_train = df_train['tourney_date'].values
    X_train_arr = X_train.values
    y_train_arr = y_train.values

    todos_modelos, base_estimators, cv_scores = entrenar_todos_los_modelos(
        X_train_arr, y_train_arr, dates=dates_train
    )

    # GBM calibrado = modelo principal; base sin calibrar para plots de importancia
    modelo_base_gbm = base_estimators['gbm']
    modelo = todos_modelos['gbm']

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

    # 4. Evaluar y graficar (GBM calibrado sobre test principal 2025)
    print(f"\n[4/5] Evaluando test principal ({TEST_YEAR}) y generando gráficos...")
    evaluar_y_graficar(modelo, X_test, y_test, df_test, FEATURES,
                       modelo_para_importancia=modelo_base_gbm)
    cv_splits = list(purged_time_series_splits(df_train['tourney_date'].values, n_splits=5))
    graficar_learning_curve(modelo_base_gbm, X_train, y_train, cv_splits)
    graficar_reliability_diagram(modelo, X_test, y_test)
    graficar_histograma_probas(modelo, X_test, y_test)

    print("\n[4b/5] Baseline ELO-crudo vs ML (¿aporta el stack?)...")
    met_baseline = evaluar_baseline_elo(df_test, y_test)
    n_test = len(y_test)
    ic_aprox = 1.0 / (2 * (n_test ** 0.5))
    print(f"  BASELINE ELO-crudo (n={n_test}):")
    print(f"    log-loss={met_baseline['log_loss']:.4f} [{met_baseline['log_loss_ic']['lower']:.4f}–{met_baseline['log_loss_ic']['upper']:.4f}]")
    print(f"    AUC     ={met_baseline['auc']:.4f} [{met_baseline['auc_ic']['lower']:.4f}–{met_baseline['auc_ic']['upper']:.4f}]")

    print(f"\n  MODELOS ML — test principal {TEST_YEAR} (n={n_test}, IC95% AUC ≈ ±{ic_aprox:.3f}):")
    metrics_all = {}
    for nombre, m in todos_modelos.items():
        met_ic = evaluar_con_ic(m, X_test, y_test)
        metrics_all[nombre] = {k: v for k, v in met_ic.items() if k in ('accuracy', 'log_loss', 'brier', 'auc')}
        print(f"    {nombre:<14} log-loss={met_ic['log_loss']:.4f} [{met_ic['log_loss_ic']['lower']:.4f}–{met_ic['log_loss_ic']['upper']:.4f}]  "
              f"AUC={met_ic['auc']:.4f} [{met_ic['auc_ic']['lower']:.4f}–{met_ic['auc_ic']['upper']:.4f}]")

    # Q3: Diagnóstico gap CV/test usando cv_score real del GBM
    print(f"\n[Q3] Diagnóstico gap CV→test ({TEST_YEAR}):")
    print(diagnosticar_gap_cv_test(
        cv_best_score=cv_scores['gbm'],
        test_log_loss=metrics_all['gbm']['log_loss'],
        n_test=n_test,
    ))

    # Evaluación secundaria 2026 (referencial — n pequeño, sin peso en decisiones)
    if len(y_sec) > 0:
        n_sec = len(y_sec)
        ic_sec = 1.0 / (2 * (n_sec ** 0.5))
        print(f"\n[4c/5] Evaluación secundaria {SECONDARY_YEAR} (n={n_sec}, IC95% AUC ≈ ±{ic_sec:.3f} — solo referencial):")
        print("  Advertencia: este conjunto NO debe usarse para tomar decisiones de diseño.")
        for nombre, m in todos_modelos.items():
            met_sec = evaluar_con_ic(m, X_sec, y_sec)
            print(f"    {nombre:<14} log-loss={met_sec['log_loss']:.4f} [{met_sec['log_loss_ic']['lower']:.4f}–{met_sec['log_loss_ic']['upper']:.4f}]  "
                  f"AUC={met_sec['auc']:.4f} [{met_sec['auc_ic']['lower']:.4f}–{met_sec['auc_ic']['upper']:.4f}]")

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

    # modelos_atp.pkl: todos los modelos calibrados (incluye gbm = modelo principal)
    with open('models/modelos_atp.pkl', 'wb') as f:
        pickle.dump(todos_modelos, f)

    # metrics_atp.pkl: métricas test principal (2025) para /api/models
    with open('models/metrics_atp.pkl', 'wb') as f:
        pickle.dump(metrics_all, f)

    with open('models/stats_jugadores.pkl', 'wb') as f:
        pickle.dump({'elo_general': ratings_finales, 'elo_superficie': ratings_superficie,
                     'stats': stats_jugadores, 'h2h': h2h, 'form': form_final,
                     'sklearn_version': sklearn.__version__}, f)

    print(f"  modelos_atp.pkl      — {{logreg, randomforest, gbm, xgboost}} calibrados (entrenados 2020-{TRAIN_END_YEAR-1})")
    print(f"  metrics_atp.pkl      — métricas test {TEST_YEAR} por modelo")
    print("  stats_jugadores.pkl  — ELO/rank/age/H2H/forma")
