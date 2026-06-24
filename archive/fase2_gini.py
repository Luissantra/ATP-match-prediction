import numpy as np

def calcular_gini(y):
    """
    Calcula el índice de Gini para un vector y de etiquetas binarias (0 y 1).
    """
    # Si el grupo de datos está vacío, la impureza es 0
    if len(y) == 0:
        return 0.0
        
    # 1. Calcula las proporciones de victorias (1) y derrotas (0)
    # Pista: Puedes contar cuántos 1s hay dividiendo entre el total, etc.
    p_victoria = np.sum(y == 1) / len(y)
    p_derrota = 1 - p_victoria
    
    # 2. Aplica la fórmula matemática del Gini
    gini = 1 - (p_victoria**2 + p_derrota**2)
    
    return gini

if __name__ == "__main__":
    # Casos de prueba para verificar tu función:
    nodo_puro_v = np.array([1, 1, 1, 1, 1])
    nodo_puro_d = np.array([0, 0, 0, 0])
    nodo_mezclado_perfecto = np.array([1, 1, 0, 0])
    nodo_mezclado_tipico = np.array([1, 1, 1, 0]) # 3 victorias, 1 derrota
    
    print(f"Gini nodo puro victorias: {calcular_gini(nodo_puro_v):.4f}")
    print(f"Gini nodo puro derrotas: {calcular_gini(nodo_puro_d):.4f}")
    print(f"Gini mezclado 50/50: {calcular_gini(nodo_mezclado_perfecto):.4f}")
    print(f"Gini mezclado 75/25: {calcular_gini(nodo_mezclado_tipico):.4f}")
