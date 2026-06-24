import numpy as np

def calcular_expectativa(rating_A, rating_B):
    """
    Calcula la probabilidad esperada de que gane el Jugador A.
    """
    expectativa_A = 1 / (1 + 10 ** ((rating_B - rating_A) / 400))
    return expectativa_A

def actualizar_ratings(rating_A, rating_B, resultado_A, K=32):
    """
    Actualiza los ratings de ambos jugadores tras un partido.
    resultado_A es 1 si gana el Jugador A, o 0 si pierde.
    Retorna (nuevo_rating_A, nuevo_rating_B)
    """
    # 1. Calcular expectativa de A
    e_A = calcular_expectativa(rating_A, rating_B)
    
    # 2. La expectativa de B es simplemente 1 - expectativa de A
    e_B = 1 - e_A
    
    # 3. Resultado real para B (es el opuesto al de A)
    resultado_B = 1 - resultado_A
    
    # 4. Calcular nuevos ratings aplicando la fórmula de actualización
    nuevo_rating_A = rating_A + K * (resultado_A - e_A)
    nuevo_rating_B = rating_B + K * (resultado_B - e_B)
    
    return round(nuevo_rating_A, 1), round(nuevo_rating_B, 1)

if __name__ == "__main__":
    # Caso 1: Dos jugadores iguales
    r_A, r_B = 1500, 1500
    e_A = calcular_expectativa(r_A, r_B)
    print(f"Expectativa (1500 vs 1500): {e_A:.4f}")
    
    # Si gana el jugador A
    nuevo_A, nuevo_B = actualizar_ratings(r_A, r_B, resultado_A=1)
    print(f"Nuevos ratings tras victoria de A: Jugador A = {nuevo_A}, Jugador B = {nuevo_B}\n")
    
    # Caso 2: Un jugador muy favorito (1800) contra un "underdog" (1400)
    r_favorito, r_underdog = 1800, 1400
    e_fav = calcular_expectativa(r_favorito, r_underdog)
    print(f"Expectativa Favorito (1800 vs 1400): {e_fav:.4f}")
    
    # Si gana el favorito
    n_fav_gana, n_und_pierde = actualizar_ratings(r_favorito, r_underdog, resultado_A=1)
    print(f"Si gana el favorito: Fav = {n_fav_gana}, Und = {n_und_pierde}")
    
    # Si da la sorpresa el underdog (gana underdog, es decir, favorito resultado_A = 0)
    n_fav_pierde, n_und_gana = actualizar_ratings(r_favorito, r_underdog, resultado_A=0)
    print(f"Si da la sorpresa el underdog: Fav = {n_fav_pierde}, Und = {n_und_gana}")
