# ATP Tennis Prediction System 🎾🤖

Este proyecto implementa un sistema predictivo de aprendizaje automático optimizado para pronosticar resultados de partidos de tenis profesionales de la ATP. 

Utilizando datos históricos reales del circuito profesional (2020-2026), el modelo predice la probabilidad de victoria cruzando el estado de forma de los tenistas (mediante un sistema de rating ELO híbrido), su ranking ATP oficial y la diferencia de edad.

---

## 🚀 Características Clave

### 1. Sistema ELO Híbrido (Global + Superficie)
A diferencia de los modelos deportivos simples, este sistema calcula un **ELO híbrido** ponderado para cada partido:
*   **50% ELO General:** Mide el rendimiento absoluto del tenista frente a la fortaleza histórica de sus rivales.
*   **50% ELO por Superficie:** Captura la especialización en canchas de **Clay (Arcilla)**, **Grass (Césped)** y **Hard (Dura)**.

### 2. Preprocesamiento Neutro Simétrico
Para evitar la fuga de etiquetas (*label leakage*), se aplica un algoritmo de simetrización aleatoria. Esto crea una perspectiva neutral de juego (Jugador A frente a Jugador B) y garantiza que el dataset de entrenamiento esté perfectamente balanceado (50% de victorias de A), permitiendo que el clasificador aprenda la verdadera frontera de decisión estadística.

### 3. Modelado Aditivo Secuencial (Gradient Boosting)
El modelo final utiliza un clasificador **Gradient Boosting** optimizado mediante **GridSearchCV**. La validación cruzada es **temporal con embargo** (`TimeSeriesSplit` purgado): descarta los partidos contiguos en la frontera train/val para evitar la fuga blanda por estado ELO compartido. El scoring se optimiza con **log-loss** (no accuracy), porque el producto es la *probabilidad* de victoria, no el acierto binario.

### 4. Features (6) y sin train/serve skew
`diff_elo`, `diff_rank`, `diff_age`, `diff_h2h`, `diff_form`, `tourney_level_num`. El vector se construye desde una **fuente única** (`src/features.py`) tanto en entrenamiento como en inferencia: la API reconstruye H2H y forma reales del historial persistido (no usa valores neutros hardcodeados).

---

## 📈 Visualizaciones Analíticas Avanzadas
Las visualizaciones generadas por el sistema están diseñadas bajo **principios de ciencias cognitivas** (alta relación señal/ruido, contraste visual intencionado y etiquetado directo):

1.  **`plots/evolucion_elo_top.png`**: Serie temporal suavizada (media móvil de 15 partidos) de la evolución del ELO para el Top 5 de jugadores del ranking. Cuenta con etiquetado directo al final de las curvas para evitar la fatiga visual de consultar leyendas.
2.  **`plots/precision_por_superficie.png`**: Gráfico de barras de la precisión del modelo agrupada por tipo de cancha. Revela, por ejemplo, que en césped (**Grass**) el modelo alcanza una precisión del **67.34%**, debido a dinámicas de juego más predecibles en saques e intercambios rápidos.
3.  **`plots/matriz_confusion.png`**: Matriz de confusión del test ciego 2026 que detalla tasas de falsos positivos y negativos.
4.  **`plots/importancia_variables.png`**: Importancia Gini de las características, demostrando el dominio del ELO frente al ranking oficial ATP.
5.  **`plots/learning_curve.png`**: Curva de aprendizaje (log-loss train vs validación) bajo CV temporal con embargo. Diagnostica sobreajuste vs falta de señal: el modelo está limitado por datos/señal (la validación sigue descendiendo), no roto por overfit.

---

## 📁 Estructura del Proyecto

```
├── src/
│   ├── __init__.py
│   ├── features.py         # Fuente única del vector de features (FEATURES, elo_hibrido, LEVEL_MAP)
│   ├── elo.py              # Ecuaciones ELO + H2H + forma; motor histórico cronológico
│   ├── data_processing.py  # Imputación y balanceo simétrico neutral (simetrización)
│   ├── cv.py               # Validación cruzada temporal con embargo (purged TimeSeriesSplit)
│   ├── train.py            # GradientBoostingClassifier + GridSearchCV (scoring neg_log_loss)
│   ├── evaluate.py         # Métricas (accuracy/log-loss/Brier/AUC) + plots + learning curve
│   └── custom_tree.py      # Árbol de decisión desde cero (referencia, no se usa en el pipeline)
├── templates/
│   └── index.html          # Interfaz web de la SPA de predicción
├── static/
│   ├── style.css           # Estilos CSS dinámicos por superficie
│   └── script.js           # Lógica interactiva y buscador predictivo de la web
├── tests/                  # Suite pytest (54 tests)
├── docs/
│   └── ROADMAP.md          # Backlog priorizado de la revisión técnica (P0/P1/P2 + multi-modelo)
├── plots/                  # Visualizaciones analíticas (.png)
├── archive/                # Historial de scripts de aprendizaje (Fases 1 a 5)
├── data/                   # Archivos anuales de partidos (2020 a 2026)
├── app.py                  # API web Flask (/api/players, /api/predict)
├── main.py                 # Orquestador del pipeline de entrenamiento y evaluación
├── visualize.py            # Generador de gráficos EDA y evolución temporal
├── requirements.txt        # Dependencias con versiones pineadas
├── .gitignore              # Archivos y carpetas excluidas del control de versiones
└── README.md               # Este archivo de documentación
```

---

## 🛠️ Instalación y Ejecución

### 1. Clonar el repositorio y configurar el entorno virtual
```bash
git clone https://github.com/Luissantra/ATP-match-prediction.git
cd "ATP-match-prediction"

# Crear y activar el entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows usa: venv\Scripts\activate

# Instalar dependencias (versiones pineadas)
pip install -r requirements.txt
```

### 2. Entrenar el modelo y exportar datos
Ejecuta el pipeline principal para procesar los datos históricos (2020-2026), ajustar los hiperparámetros del clasificador Gradient Boosting y exportar los modelos serializados (`.pkl`):
```bash
python main.py
```

### 3. Iniciar la aplicación web interactiva
Inicia el servidor de **desarrollo** Flask para realizar predicciones a la carta:
```bash
python app.py
```
Luego abre en tu navegador: **[http://localhost:8000](http://localhost:8000)**.

> ⚠️ `python app.py` usa el servidor de desarrollo de Werkzeug (un solo hilo), **no apto para producción**.
> Para un despliegue mínimo usa un servidor WSGI con varios workers:
> ```bash
> gunicorn -w 4 -b 0.0.0.0:8000 app:app
> ```
> Cada worker carga los `.pkl` al arrancar (estado global de solo lectura → sin problemas de concurrencia).

### 4. Generar visualizaciones estáticas analíticas
Genera el Pair Plot y la evolución temporal de los ELOs del Top 5 de jugadores:
```bash
python visualize.py
```

---

## 📊 Resultados Científicos

Métricas sobre el **test ciego 2026** (n≈137). El accuracy por sí solo engaña en un problema probabilístico, así que se reportan log-loss, Brier y AUC:

| Métrica | Valor | Referencia (azar) |
|---|---|---|
| AUC | **0.615** | 0.50 |
| Log-loss | **0.683** | 0.693 |
| Brier | **0.244** | 0.25 |
| Accuracy | **56.9%** | 50% |

*   **Mejores Hiperparámetros:** `{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100}`
*   **CV temporal con embargo (log-loss):** ~0.620
*   **Lectura honesta:** el modelo **discrimina** (AUC > 0.5) pero de forma débil. El gap CV(0.62)/test(0.68) **persiste tras aplicar embargo**, lo que indica *distribution shift* de la temporada 2026 y no fuga de datos. La learning curve muestra sobreajuste leve y validación aún descendente → el sistema está limitado por señal/datos, con techo en la predecibilidad intrínseca del tenis.

---

## 📚 Fuentes de Datos y Agradecimientos

Este proyecto se nutre de datos reales y actualizados diariamente del circuito profesional de tenis:
*   **TML-Database (TennisMyLife):** La base de datos principal de partidos es proporcionada por [TennisMyLife](https://stats.tennismylife.org/), un recurso excelente con datos unificados e identificadores ATP consistentes.
*   **Jeff Sackmann / tennis_atp:** Inspirador original del esquema de base de datos bajo licencia [Creative Commons Non-Commercial Share Alike](https://github.com/JeffSackmann/tennis_atp).

Agradecemos a la comunidad de analistas deportivos que hacen accesible esta información para investigación académica y desarrollo tecnológico.

