import pandas as pd
import numpy as np

def limpiar_y_preparar_datos(filepath):
    # 1. Cargar el dataset usando Pandas
    df = pd.read_csv(filepath)
    
    # Mostramos cuántos valores nulos hay originalmente en los rankings
    print(f"Nulos originales en winner_rank: {df['winner_rank'].isnull().sum()}")
    print(f"Nulos originales en loser_rank: {df['loser_rank'].isnull().sum()}")
    
    # 2. Rellena los valores nulos (NaN) en 'winner_rank' y 'loser_rank' con 999
    # (Esto representa a jugadores que no tienen un ranking ATP oficial)
    df['winner_rank'] = df['winner_rank'].fillna(999).astype(int)
    df['loser_rank'] = df['loser_rank'].fillna(999).astype(int)
    
    # 3. Para esta primera aproximación, supongamos que el Jugador A es siempre el ganador
    # y el Jugador B es siempre el perdedor.
    # Calcula la diferencia de edad: edad_A - edad_B (winner_age - loser_age)
    df['diff_age'] = df['winner_age'] - df['loser_age']
    
    # 4. Calcula la diferencia de ranking: ranking_A - ranking_B (winner_rank - loser_rank)
    df['diff_rank'] = df['winner_rank'] - df['loser_rank']
    
    return df

# Probar la función con el archivo de 2024
if __name__ == "__main__":
    df_limpio = limpiar_y_preparar_datos("data/2024.csv")
    print("\nPrimeras 5 filas con las nuevas características:")
    print(df_limpio[['winner_name', 'loser_name', 'diff_age', 'diff_rank']].head())
