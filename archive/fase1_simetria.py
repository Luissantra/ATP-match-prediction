import pandas as pd
import numpy as np

def crear_dataset_simetrico(filepath):
    df = pd.read_csv(filepath)
    
    # Rellenar nulos de ranking
    df['winner_rank'] = df['winner_rank'].fillna(999)
    df['loser_rank'] = df['loser_rank'].fillna(999)
    
    # Semilla para reproducibilidad
    np.random.seed(42)
    # Genera un vector aleatorio de True/False del mismo tamaño que el DataFrame
    shuffle_mask = np.random.rand(len(df)) > 0.5
    
    # Inicializamos las listas de datos estructurados
    features = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Si la máscara es True, el Jugador A es el Ganador (y = 1)
        if shuffle_mask[i]:
            player_A = row['winner_name']
            player_B = row['loser_name']
            rank_A = row['winner_rank']
            rank_B = row['loser_rank']
            age_A = row['winner_age']
            age_B = row['loser_age']
            label = 1
        # Si la máscara es False, el Jugador A es el Perdedor (y = 0)
        else:
            player_A = row['loser_name']
            player_B = row['winner_name']
            rank_A = row['loser_rank']
            rank_B = row['winner_rank']
            age_A = row['loser_age']
            age_B = row['winner_age']
            label = 0
            
        # Calcular diferencias
        diff_rank = rank_A - rank_B
        diff_age = age_A - age_B
        
        features.append({
            'player_A': player_A,
            'player_B': player_B,
            'diff_rank': diff_rank,
            'diff_age': diff_age,
            'label': label
        })
        
    return pd.DataFrame(features)

if __name__ == "__main__":
    df_simetrico = crear_dataset_simetrico("data/2024.csv")
    print(f"Total de partidos procesados: {len(df_simetrico)}")
    print(f"Porcentaje de victorias de A (label 1): {df_simetrico['label'].mean():.2%}")
    print("\nPrimeras 5 filas del dataset simétrico:")
    print(df_simetrico.head())
