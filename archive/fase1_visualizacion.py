import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def crear_dataset_visual(filepath):
    df = pd.read_csv(filepath)
    
    # Limpieza e imputación de rankings
    df['winner_rank'] = df['winner_rank'].fillna(999)
    df['loser_rank'] = df['loser_rank'].fillna(999)
    
    # Imputar alturas vacías con la mediana de altura
    mediana_ht = df['winner_ht'].median()
    df['winner_ht'] = df['winner_ht'].fillna(mediana_ht)
    df['loser_ht'] = df['loser_ht'].fillna(mediana_ht)
    
    np.random.seed(42)
    shuffle_mask = np.random.rand(len(df)) > 0.5
    
    features = []
    for i in range(len(df)):
        row = df.iloc[i]
        if shuffle_mask[i]:
            rank_A, rank_B = row['winner_rank'], row['loser_rank']
            age_A, age_B = row['winner_age'], row['loser_age']
            ht_A, ht_B = row['winner_ht'], row['loser_ht']
            label = 1
        else:
            rank_A, rank_B = row['loser_rank'], row['winner_rank']
            age_A, age_B = row['loser_age'], row['winner_age']
            ht_A, ht_B = row['loser_ht'], row['winner_ht']
            label = 0
            
        features.append({
            'diff_rank': rank_A - rank_B,
            'diff_age': age_A - age_B,
            'diff_ht': ht_A - ht_B,
            'label': label
        })
        
    return pd.DataFrame(features)

if __name__ == "__main__":
    df_visual = crear_dataset_visual("data/2024.csv")
    
    print("Generando el Pair Plot con Seaborn...")
    sns.set_theme(style="ticks")
    
    # Mapeamos las etiquetas numéricas a texto legible para la leyenda del gráfico
    df_visual['Resultado_Jugador_A'] = df_visual['label'].map({1: 'Victoria', 0: 'Derrota'})
    
    # Graficar
    g = sns.pairplot(
        df_visual[['diff_rank', 'diff_age', 'diff_ht', 'Resultado_Jugador_A']],
        hue='Resultado_Jugador_A',
        palette={'Victoria': '#2ecc71', 'Derrota': '#e74c3c'},  # Verde para victoria, rojo para derrota
        diag_kind="kde",
        plot_kws={'alpha': 0.4, 's': 20, 'edgecolor': 'none'}
    )
    
    # Guardar la imagen localmente
    output_image = "pairplot_tenis.png"
    g.savefig(output_image, dpi=300)
    print(f"¡Gráfico guardado con éxito como '{output_image}'!")
