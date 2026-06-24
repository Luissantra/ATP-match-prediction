import pickle
import sklearn
import pandas as pd
from src.elo import calcular_elos_historicos
from src.data_processing import preparar_datos_entrenamiento
from src.features import FEATURES
from src.train import entrenar_modelo
from src.evaluate import evaluar_y_graficar

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

    # 3. Entrenar
    print("\n[3/4] Entrenando con GridSearchCV + TimeSeriesSplit...")
    modelo = entrenar_modelo(X_train, y_train)

    # 4. Evaluar y graficar
    print("\n[4/4] Evaluando y generando gráficos...")
    evaluar_y_graficar(modelo, X_test, y_test, df_test, FEATURES)

    # 5. Exportar artefactos
    stats_jugadores = {}
    for _, row in df_completo.iterrows():
        for role in [('winner_name', 'winner_rank', 'winner_age'),
                     ('loser_name',  'loser_rank',  'loser_age')]:
            name = row[role[0]]
            stats_jugadores[name] = {
                'rank': float(row[role[1]]) if not pd.isna(row[role[1]]) else 999.0,
                'age':  float(row[role[2]]) if not pd.isna(row[role[2]]) else 26.0,
            }

    with open('modelo_atp.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    with open('stats_jugadores.pkl', 'wb') as f:
        pickle.dump({'elo_general': ratings_finales, 'elo_superficie': ratings_superficie,
                     'stats': stats_jugadores, 'h2h': h2h, 'form': form_final,
                     'sklearn_version': sklearn.__version__}, f)

    print("\nModelo y metadatos exportados a modelo_atp.pkl y stats_jugadores.pkl")
