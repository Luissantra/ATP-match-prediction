import pickle
import sklearn
import pandas as pd
from src.elo import calcular_elos_historicos
from src.data_processing import preparar_datos_entrenamiento
from src.features import FEATURES
from src.train import entrenar_modelo, calibrar_modelo, coeficientes_modelo
from src.evaluate import (
    evaluar_con_ic, evaluar_baseline_elo, evaluar_y_graficar,
    graficar_learning_curve, graficar_reliability_diagram, graficar_histograma_probas,
    graficar_permutation_importance, graficar_coeficientes, diagnosticar_gap_cv_test,
)
from src.cv import purged_time_series_splits

AÑOS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
# --- Fase de Validación ---
VAL_TRAIN_END_YEAR = 2025   # Validación: entrena con < VAL_TRAIN_END_YEAR (2020-2024)
TEST_YEAR          = 2025   # Test principal (out-of-sample para validación)
SECONDARY_YEAR     = 2026   # Evaluación secundaria (para el modelo de producción)

# --- Fase de Producción (Refit) ---
PROD_TRAIN_END_YEAR = 2026  # Producción: entrena con < PROD_TRAIN_END_YEAR (2020-2025)

if __name__ == "__main__":
    print("=== ATP Prediction Pipeline ===")

    # 1. ELO histórico
    print(f"\n[1/5] Calculando ELO histórico ({AÑOS[0]}-{AÑOS[-1]})...")
    df_completo, ratings_finales, ratings_superficie = calcular_elos_historicos("data", AÑOS)

    top10 = sorted(ratings_finales.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 ELO:")
    for i, (jugador, rating) in enumerate(top10, 1):
        print(f"  {i:02d}. {jugador:<25} {rating:.1f}")

    # 2. Dataset simétrico
    print("\n[2/5] Preparando dataset simétrico...")
    df_features = preparar_datos_entrenamiento(df_completo)
    print(f"  Dataset: {df_features.shape[0]} filas x {df_features.shape[1]} columnas")

    df_train     = df_features[df_features['year'] < VAL_TRAIN_END_YEAR]
    df_test      = df_features[df_features['year'] == TEST_YEAR]
    df_secondary = df_features[df_features['year'] == SECONDARY_YEAR]
    df_prod      = df_features[df_features['year'] < PROD_TRAIN_END_YEAR]

    X_train, y_train = df_train[FEATURES], df_train['label']
    X_test,  y_test  = df_test[FEATURES],  df_test['label']
    X_sec,   y_sec   = df_secondary[FEATURES], df_secondary['label']
    X_prod,  y_prod  = df_prod[FEATURES],  df_prod['label']

    print(f"  Features ({len(FEATURES)}): {FEATURES}")
    print(f"  Entrenamiento de Validación (2020-{VAL_TRAIN_END_YEAR-1}): {len(X_train)} partidos")
    print(f"  Test principal de Validación ({TEST_YEAR}):        {len(X_test)} partidos")
    print(f"  Entrenamiento de Producción (2020-{PROD_TRAIN_END_YEAR-1}): {len(X_prod)} partidos")
    print(f"  Eval. secundaria en producción ({SECONDARY_YEAR}): {len(X_sec)} partidos (referencial)")

    # 3. Entrenar modelos (Validación y Producción)
    # 3a. Entrenamiento del modelo de VALIDACIÓN para calcular métricas honestas
    print("\n[3a/5] Entrenando modelo de VALIDACIÓN con GridSearchCV + CV temporal con embargo...")
    dates_train = df_train['tourney_date'].values
    X_train_arr = X_train.values
    y_train_arr = y_train.values

    modelo_base_val, cv_log_loss_val, best_params_val = entrenar_modelo(X_train_arr, y_train_arr, dates=dates_train)
    print(f"  [Validación] best_params={best_params_val}  cv_log_loss={cv_log_loss_val:.4f}")
    modelo_val = calibrar_modelo(modelo_base_val, X_train_arr, y_train_arr, dates=dates_train)

    # 3b. Refit del modelo de PRODUCCIÓN utilizando todo el historial (2020-2025)
    print("\n[3b/5] Entrenando modelo final de PRODUCCIÓN (Refit completo 2020-2025)...")
    dates_prod = df_prod['tourney_date'].values
    X_prod_arr = X_prod.values
    y_prod_arr = y_prod.values

    modelo_base_prod, _, best_params_prod = entrenar_modelo(X_prod_arr, y_prod_arr, dates=dates_prod)
    print(f"  [Producción] best_params={best_params_prod}")
    modelo_prod = calibrar_modelo(modelo_base_prod, X_prod_arr, y_prod_arr, dates=dates_prod)

    # Explicabilidad: coeficientes / odds-ratio del modelo base de producción
    coefs_prod = coeficientes_modelo(modelo_base_prod, FEATURES)
    print("\n  Coeficientes del modelo de PRODUCCIÓN (log-odds por +1 std · odds-ratio):")
    for feat, v in coefs_prod.items():
        print(f"    {feat:18s} coef={v['coef']:+.3f}  OR={v['odds_ratio']:.3f}")

    # 4. Evaluar y graficar (modelo de validación calibrado sobre test principal 2025)
    print(f"\n[4/5] Evaluando modelo de validación en test principal ({TEST_YEAR}) y generando gráficos...")
    cm_data = evaluar_y_graficar(modelo_val, X_test, y_test, df_test, FEATURES)
    cv_splits = list(purged_time_series_splits(dates_train, n_splits=5))
    graficar_learning_curve(modelo_base_val, X_train, y_train, cv_splits)
    rel_data = graficar_reliability_diagram(modelo_val, X_test, y_test)
    hist_data = graficar_histograma_probas(modelo_val, X_test, y_test)
    graficar_permutation_importance(modelo_val, X_test.values, y_test.values, FEATURES)
    graficar_coeficientes(coefs_prod)

    print("\n[4b/5] Baseline ELO-híbrido vs ML (¿aporta el modelo de validación sobre la resta de ELO?)...")
    met_baseline = evaluar_baseline_elo(df_test, y_test)
    n_test = len(y_test)
    ic_aprox = 1.0 / (2 * (n_test ** 0.5))
    print(f"  BASELINE ELO-híbrido (n={n_test}):")
    print(f"    log-loss={met_baseline['log_loss']:.4f} [{met_baseline['log_loss_ic']['lower']:.4f}–{met_baseline['log_loss_ic']['upper']:.4f}]")
    print(f"    AUC     ={met_baseline['auc']:.4f} [{met_baseline['auc_ic']['lower']:.4f}–{met_baseline['auc_ic']['upper']:.4f}]")

    print(f"\n  MODELO LogReg (Validación) — test principal {TEST_YEAR} (n={n_test}, IC95% AUC ≈ ±{ic_aprox:.3f}):")
    met_ic = evaluar_con_ic(modelo_val, X_test, y_test)
    metrics = {k: v for k, v in met_ic.items() if k in ('accuracy', 'log_loss', 'brier', 'auc')}
    metrics['plots_data'] = {
        'confusion_matrix': cm_data,
        'reliability': rel_data,
        'histogram': hist_data
    }
    print(f"    log-loss={met_ic['log_loss']:.4f} [{met_ic['log_loss_ic']['lower']:.4f}–{met_ic['log_loss_ic']['upper']:.4f}]  "
          f"AUC={met_ic['auc']:.4f} [{met_ic['auc_ic']['lower']:.4f}–{met_ic['auc_ic']['upper']:.4f}]")

    # Diagnóstico gap CV/test
    print(f"\n[Q3] Diagnóstico gap CV→test ({TEST_YEAR}):")
    print(diagnosticar_gap_cv_test(
        cv_best_score=cv_log_loss_val,
        test_log_loss=metrics['log_loss'],
        n_test=n_test,
    ))

    # Evaluación secundaria 2026 (sobre el modelo de PRODUCCIÓN, ya que es out-of-sample)
    if len(y_sec) > 0:
        n_sec = len(y_sec)
        ic_sec = 1.0 / (2 * (n_sec ** 0.5))
        print(f"\n[4c/5] Evaluación secundaria {SECONDARY_YEAR} sobre el modelo de PRODUCCIÓN (n={n_sec}, IC95% AUC ≈ ±{ic_sec:.3f} — solo referencial):")
        print("  Advertencia: este conjunto NO debe usarse para tomar decisiones de diseño.")
        met_sec = evaluar_con_ic(modelo_prod, X_sec, y_sec)
        print(f"    log-loss={met_sec['log_loss']:.4f} [{met_sec['log_loss_ic']['lower']:.4f}–{met_sec['log_loss_ic']['upper']:.4f}]  "
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

    # modelos_atp.pkl: el modelo único calibrado (LogReg de producción)
    with open('models/modelos_atp.pkl', 'wb') as f:
        pickle.dump(modelo_prod, f)

    # metrics_atp.pkl: métricas del modelo de validación sobre el test principal (2025)
    with open('models/metrics_atp.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    with open('models/stats_jugadores.pkl', 'wb') as f:
        pickle.dump({'elo_general': ratings_finales, 'elo_superficie': ratings_superficie,
                     'stats': stats_jugadores, 'coeficientes': coefs_prod,
                     'sklearn_version': sklearn.__version__,
                     'trained_through': PROD_TRAIN_END_YEAR - 1,
                     'tested_on': TEST_YEAR}, f)

    print(f"  modelos_atp.pkl      — LogReg calibrado (PRODUCCIÓN, entrenado 2020-{PROD_TRAIN_END_YEAR-1})")
    print(f"  metrics_atp.pkl      — métricas de VALIDACIÓN (test {TEST_YEAR})")
    print(f"  stats_jugadores.pkl  — ELO/rank/age + coeficientes + metadatos (vigencia)")
