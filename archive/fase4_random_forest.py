import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fase3_procesar_elo import calcular_elos_historicos

def preparar_datos_entrenamiento(df_con_elo):
    # Imputar nulos de ranking y edad
    df_con_elo['winner_rank'] = df_con_elo['winner_rank'].fillna(999)
    df_con_elo['loser_rank'] = df_con_elo['loser_rank'].fillna(999)
    df_con_elo['winner_age'] = df_con_elo['winner_age'].fillna(df_con_elo['winner_age'].median())
    df_con_elo['loser_age'] = df_con_elo['loser_age'].fillna(df_con_elo['loser_age'].median())
    
    np.random.seed(42)
    shuffle_mask = np.random.rand(len(df_con_elo)) > 0.5
    
    features = []
    
    for i in range(len(df_con_elo)):
        row = df_con_elo.iloc[i]
        
        # Recuperar variables clave
        w_elo, l_elo = row['elo_winner'], row['elo_loser']
        w_rank, l_rank = row['winner_rank'], row['loser_rank']
        w_age, l_age = row['winner_age'], row['loser_age']
        
        # Simetrizar la perspectiva
        if shuffle_mask[i]:
            diff_elo = w_elo - l_elo
            diff_rank = w_rank - l_rank
            diff_age = w_age - l_age
            label = 1
        else:
            diff_elo = l_elo - w_elo
            diff_rank = l_rank - w_rank
            diff_age = l_age - w_age
            label = 0
            
        features.append({
            'year': int(str(row['tourney_date'])[:4]),
            'diff_elo': diff_elo,
            'diff_rank': diff_rank,
            'diff_age': diff_age,
            'label': label
        })
        
    return pd.DataFrame(features)

if __name__ == "__main__":
    # 1. Cargar e integrar datos históricos con ELO
    años = [2020, 2021, 2022, 2023, 2024]
    df_completo, _ = calcular_elos_historicos("data", años)
    
    # 2. Generar el dataset neutral simétrico
    df_features = preparar_datos_entrenamiento(df_completo)
    
    # 3. Dividir temporalmente:
    # Entrenamiento: 2020-2023
    # Prueba: 2024 (Blind Test)
    df_train = df_features[df_features['year'] < 2024]
    df_test = df_features[df_features['year'] == 2024]
    
    X_train = df_train[['diff_elo', 'diff_rank', 'diff_age']]
    y_train = df_train['label']
    
    X_test = df_test[['diff_elo', 'diff_rank', 'diff_age']]
    y_test = df_test['label']
    
    print(f"\nTamaño conjunto de entrenamiento: {len(X_train)} partidos")
    print(f"Tamaño conjunto de prueba (2024): {len(X_test)} partidos")
    
    # 4. Entrenar el Bosque Aleatorio
    # Usaremos 100 árboles y limitaremos la profundidad de cada uno a 5 niveles
    modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    modelo_rf.fit(X_train, y_train)
    
    # 5. Predecir y evaluar
    preds = modelo_rf.predict(X_test)
    precision = accuracy_score(y_test, preds)
    
    print(f"\n¡Precisión (Accuracy) del Bosque Aleatorio en 2024: {precision:.2%}!")
    
    # Importancia de las variables
    for col, imp in zip(X_train.columns, modelo_rf.feature_importances_):
        print(f"Importancia de la variable '{col}': {imp:.2%}")
