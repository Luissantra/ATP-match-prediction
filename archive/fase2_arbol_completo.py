import numpy as np
from fase2_gini import calcular_gini
from fase2_division import encontrar_mejor_division

class Nodo:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature       # Índice de la característica a evaluar (0 o 1)
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
        # Si el nodo actual es una hoja, devuelve su valor
        if nodo.es_hoja():
            return nodo.value

        
        
        # Si el valor de la característica en 'x' (x[nodo.feature]) es menor o igual
        # al umbral del nodo (nodo.threshold), ve a la izquierda. Si no, ve a la derecha.
        # Recuerda que debes llamar a self._predict_one de forma recursiva.
            
        # El valor de la característica a evaluar es: x[nodo.feature]
        # El umbral de corte es: nodo.threshold
        # Si la condición se cumple, vamos a la izquierda (nodo.left)
        # Si no, vamos a la derecha (nodo.right)
        
        if x[nodo.feature] <= nodo.threshold:
            return self._predict_one(nodo.left, x)
        else:
            return self._predict_one(nodo.right, x)

if __name__ == "__main__":
    # Generamos datos sintéticos de tenis
    # X contiene: [diff_rank, diff_age]
    X_train = np.array([
        [-100, -2.0], # A es mucho mejor rankeado y más joven (Gana A = 1)
        [-50, 1.0],   # A es mejor rankeado pero mayor (Gana A = 1)
        [10, -5.0],   # A es peor rankeado pero más joven (Pierde A = 0)
        [80, 3.0],    # A es peor rankeado y mayor (Pierde A = 0)
    ])
    y_train = np.array([1, 1, 0, 0])

    arbol = ArbolDecisionPropio(max_depth=2)
    arbol.fit(X_train, y_train)

    # Predecir un partido nuevo:
    # Jugador A tiene ranking 20 y Jugador B tiene 70 (diff_rank = 20 - 70 = -50)
    # Jugador A tiene 25 años y Jugador B tiene 30 (diff_age = 25 - 30 = -5)
    partido_nuevo = np.array([[-50, -5.0]])
    prediccion = arbol.predict(partido_nuevo)
    
    print(f"Predicción para el nuevo partido (diff_rank=-50, diff_age=-5): {prediccion[0]}")
    print(f"¿Predijo que el Jugador A gana? {'SÍ' if prediccion[0] == 1 else 'NO'}")
