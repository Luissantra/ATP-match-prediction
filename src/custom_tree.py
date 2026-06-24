import numpy as np

def calcular_gini(y):
    """
    Calcula el índice de Gini para un vector y de etiquetas binarias (0 y 1).
    """
    if len(y) == 0:
        return 0.0
        
    p_victoria = np.sum(y == 1) / len(y)
    p_derrota = 1 - p_victoria
    
    gini = 1 - (p_victoria**2 + p_derrota**2)
    return gini

def encontrar_mejor_division(X, y):
    """
    Encuentra el mejor umbral de división para una característica unidimensional X.
    Retorna el mejor umbral y la ganancia de información máxima.
    """
    gini_padre = calcular_gini(y)
    n_total = len(y)
    
    mejor_ganancia = -1.0
    mejor_umbral = None
    
    valores_ordenados = np.sort(np.unique(X))
    if len(valores_ordenados) < 2:
        return None, 0.0
        
    umbrales = (valores_ordenados[:-1] + valores_ordenados[1:]) / 2
    
    for umbral in umbrales:
        mascara_izq = X <= umbral
        y_izq = y[mascara_izq]
        y_der = y[~mascara_izq]
        
        if len(y_izq) == 0 or len(y_der) == 0:
            continue
            
        gini_hijos = (len(y_izq) / n_total) * calcular_gini(y_izq) + (len(y_der) / n_total) * calcular_gini(y_der)
        ganancia = gini_padre - gini_hijos
        
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_umbral = umbral
            
    return mejor_umbral, mejor_ganancia

class Nodo:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature       # Índice de la característica a evaluar
        self.threshold = threshold   # Umbral de corte
        self.left = left             # Rama izquierda
        self.right = right           # Rama derecha
        self.value = value           # Si es hoja, contiene la predicción (0 o 1)

    def es_hoja(self):
        return self.value is not None

class ArbolDecisionPropio:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.raiz = None

    def fit(self, X, y):
        self.raiz = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        
        # Condiciones de parada: si el nodo es puro, llegamos al max_depth, o no quedan muestras
        if depth >= self.max_depth or calcular_gini(y) == 0 or n_samples < 2:
            prediccion_hoja = 1 if np.sum(y == 1) >= np.sum(y == 0) else 0
            return Nodo(value=prediccion_hoja)

        # Buscar la mejor división entre todas las características
        mejor_feat, mejor_thr, mejor_gain = None, None, -1.0
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thr, gain = encontrar_mejor_division(X_column, y)
            if thr is not None and gain > mejor_gain:
                mejor_gain = gain
                mejor_thr = thr
                mejor_feat = feat_idx

        # Si no hay ganancia posible, creamos hoja
        if mejor_gain <= 0:
            prediccion_hoja = 1 if np.sum(y == 1) >= np.sum(y == 0) else 0
            return Nodo(value=prediccion_hoja)

        # Dividir y construir subárboles recursivamente
        left_mask = X[:, mejor_feat] <= mejor_thr
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return Nodo(feature=mejor_feat, threshold=mejor_thr, left=left_child, right=right_child)

    def predict(self, X):
        return np.array([self._predict_one(self.raiz, x) for x in X])

    def _predict_one(self, nodo, x):
        """
        Recorre el árbol de forma recursiva para una fila 'x'.
        """
        if nodo.es_hoja():
            return nodo.value

        if x[nodo.feature] <= nodo.threshold:
            return self._predict_one(nodo.left, x)
        else:
            return self._predict_one(nodo.right, x)
