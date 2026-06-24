# ATP Tennis Prediction System 🎾🤖

Este proyecto implementa un sistema predictivo de aprendizaje automático optimizado para pronosticar resultados de partidos de tenis profesionales de la ATP. 

Utilizando datos históricos reales del circuito profesional (2020-2025), el modelo predice la probabilidad de victoria cruzando el estado de forma de los tenistas (mediante un sistema de rating ELO híbrido), su ranking ATP oficial y la diferencia de edad.

---

## 🚀 Características Clave

### 1. Sistema ELO Híbrido (Global + Superficie)
A diferencia de los modelos deportivos simples, este sistema calcula un **ELO híbrido** ponderado para cada partido:
*   **50% ELO General:** Mide el rendimiento absoluto del tenista frente a la fortaleza histórica de sus rivales.
*   **50% ELO por Superficie:** Captura la especialización en canchas de **Clay (Arcilla)**, **Grass (Césped)** y **Hard (Dura)**.

### 2. Preprocesamiento Neutro Simétrico
Para evitar la fuga de etiquetas (*label leakage*), se aplica un algoritmo de simetrización aleatoria. Esto crea una perspectiva neutral de juego (Jugador A frente a Jugador B) y garantiza que el dataset de entrenamiento esté perfectamente balanceado (50% de victorias de A), permitiendo que el clasificador aprenda la verdadera frontera de decisión estadística.

### 3. Modelado Aditivo Secuencial (Gradient Boosting)
El modelo final utiliza un clasificador **Gradient Boosting** optimizado mediante **GridSearchCV** con validación cruzada. El algoritmo corrige de forma secuencial y aditiva los errores de clasificación residuales de los árboles de decisión anteriores.

---

## 📈 Visualizaciones Analíticas Avanzadas
Las visualizaciones generadas por el sistema están diseñadas bajo **principios de ciencias cognitivas** (alta relación señal/ruido, contraste visual intencionado y etiquetado directo):

1.  **`plots/evolucion_elo_top.png`**: Serie temporal suavizada (media móvil de 15 partidos) de la evolución del ELO para el Top 5 de jugadores del ranking. Cuenta con etiquetado directo al final de las curvas para evitar la fatiga visual de consultar leyendas.
2.  **`plots/precision_por_superficie.png`**: Gráfico de barras de la precisión del modelo agrupada por tipo de cancha. Revela, por ejemplo, que en césped (**Grass**) el modelo alcanza una precisión del **67.34%**, debido a dinámicas de juego más predecibles en saques e intercambios rápidos.
3.  **`plots/matriz_confusion.png`**: Matriz de confusión del test ciego 2025 que detalla tasas de falsos positivos y negativos.
4.  **`plots/importancia_variables.png`**: Importancia Gini de las características, demostrando el dominio del ELO (~85-90%) frente al ranking oficial ATP.

---

## 📁 Estructura del Proyecto

```
├── src/
│   ├── __init__.py
│   ├── elo.py              # Ecuaciones de ELO y motor de procesamiento histórico
│   ├── data_processing.py  # Imputación de datos y balanceo simétrico neutral
│   └── custom_tree.py      # Árbol de decisión implementado desde cero (referencia)
├── plots/                  # Visualizaciones analíticas de alto contraste (.png)
├── archive/                # Historial de scripts de aprendizaje (Fases 1 a 5)
├── data/                   # Archivos anuales de partidos (de 2020 a 2026)
├── main.py                 # Script de entrenamiento y evaluación del pipeline
├── visualize.py            # Generador de gráficos EDA y evolución temporal
├── ideas_futuro.md         # Propuestas para ingeniería de variables avanzada
├── .gitignore              # Archivos y carpetas excluidas del control de versiones
└── README.md               # Este archivo de documentación
```

---

## 🛠️ Instalación y Ejecución

### 1. Clonar el repositorio y configurar el entorno virtual
```bash
git clone <URL_DEL_REPOSITORIO>
cd "ATP Prediction"

# Crear y activar el entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows usa: venv\Scripts\activate

# Instalar dependencias
pip install pandas numpy scikit-learn seaborn matplotlib
```

### 2. Entrenar y evaluar el modelo
Ejecuta el pipeline principal de extremo a extremo (entrenamiento 2020-2024 y prueba ciega en la temporada 2025):
```bash
python main.py
```

### 3. Generar visualizaciones y series temporales
Genera el Pair Plot y la evolución temporal del Top 5 de jugadores:
```bash
python visualize.py
```

---

## 📊 Resultados Científicos

*   **Dataset de Entrenamiento (2020-2024):** 13,273 partidos.
*   **Mejores Hiperparámetros (CV):** `{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 150}`
*   **Precisión CV en Entrenamiento:** **65.36%**
*   **Precisión en Test Ciego (Temporada 2025):** **64.52%**
*   **Predictibilidad por Superficie (2025):**
    *   **Grass (Césped):** 67.34%
    *   **Hard (Dura):** 64.38%
    *   **Clay (Arcilla):** 63.79%

---

## 📚 Fuentes de Datos y Agradecimientos

Este proyecto se nutre de datos reales y actualizados diariamente del circuito profesional de tenis:
*   **TML-Database (TennisMyLife):** La base de datos principal de partidos es proporcionada por [TennisMyLife](https://stats.tennismylife.org/), un recurso excelente con datos unificados e identificadores ATP consistentes.
*   **Jeff Sackmann / tennis_atp:** Inspirador original del esquema de base de datos bajo licencia [Creative Commons Non-Commercial Share Alike](https://github.com/JeffSackmann/tennis_atp).

Agradecemos a la comunidad de analistas deportivos que hacen accesible esta información para investigación académica y desarrollo tecnológico.

