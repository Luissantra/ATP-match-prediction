from scipy.optimize import moduleTNC
import numpy as np
from fase2_gini import calcular_gini  # Reutilizamos tu función Gini

def encontrar_mejor_division(X, y):
    """
    Encuentra el mejor umbral de división para una característica unidimensional X.
    Retorna el mejor umbral y la ganancia de información máxima.
    """
    gini_padre = calcular_gini(y)
    n_total = len(y)
    
    mejor_ganancia = -1.0
    mejor_umbral = None
    
    # 1. Obtener valores únicos ordenados de la característica X
    valores_ordenados = np.sort(np.unique(X))
    
    # Calculamos los puntos medios entre valores consecutivos como posibles umbrales
    umbrales = (valores_ordenados[:-1] + valores_ordenados[1:]) / 2
    
    for umbral in umbrales:
        # 2. Divide las etiquetas y en dos grupos (izq y der) usando el umbral
        mascara_izq = X <= umbral
        y_izq = y[mascara_izq]
        y_der = y[~mascara_izq]
        
        # Si alguno de los grupos queda vacío, no es una división válida
        if len(y_izq) == 0 or len(y_der) == 0:
            continue
            
        # 3. Calcula el Gini ponderado de los nodos hijos
        gini_hijos = (len(y_izq) / n_total) * calcular_gini(y_izq) + (len(y_der) / n_total) * calcular_gini(y_der)
        
        # 4. Calcula la Ganancia de Información 
        ganancia = (gini_padre - gini_hijos)
        
        # 5. Guarda el umbral si supera la mejor ganancia encontrada
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_umbral = umbral
            
    return mejor_umbral, mejor_ganancia

if __name__ == "__main__":
    # Caso de prueba simple:
    # Si la diferencia de ranking es muy negativa, gana el jugador A (1).
    # Si es muy positiva, gana el jugador B (por tanto A pierde, 0).
    X_test = np.array([-50, -30, -10, 10, 30, 50])
    y_test = np.array([1, 1, 1, 0, 0, 0])
    
    umbral, ganancia = encontrar_mejor_division(X_test, y_test)
    print(f"Mejor umbral de división: {umbral}")
    print(f"Ganancia de información obtenida: {ganancia:.4f}")

