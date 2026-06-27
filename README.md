---
title: ATP Match Forecast
emoji: 🎾
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ATP Tennis Match Prediction

Sistema predictivo de aprendizaje automático para pronosticar resultados de partidos ATP, con énfasis en rigor estadístico: el producto es la *probabilidad* de victoria, no solo el acierto binario.

Datos históricos reales 2020–2026. Train 2020–2024, **test ciego 2025** (n=2861, IC95% AUC ≈ ±0.009).

---

## Características técnicas

### 1. ELO híbrido con MOV y K-schedule
Rating dinámico por jugador: general + por superficie (Clay / Grass / Hard). Cada partido actualiza ambos con:
- **Margin of Victory**: straight sets mueven más el ELO que sets igualados.
- **K-schedule**: K=48 para debutantes (<10 partidos), K=40 (<30), K=32 establecido — reduce el cold-start drift.
- **ELO híbrido**: el modelo aprende el peso relativo general/superficie (no 50/50 fijo).

### 2. Dataset simétrico sin label leakage
Simetrización vectorizada: Ganador/Perdedor → Jugador A/B aleatorio con balance 50/50. Todas las features se calculan *pre-partido*, sin información del resultado actual. `is_unranked` usa la máscara NaN real del ranking (no confunde un rank numérico alto con "sin ranking").

### 3. Modelo único, lineal y explicable
Regresión logística estandarizada y calibrada (`GridSearchCV(scoring='neg_log_loss')` sobre `C` + CV temporal purgado + `CalibratedClassifierCV` sigmoid). Un estudio de permutation importance + ablación sobre el test 2025 mostró que GBM/RandomForest/XGBoost **no superan** a la LogReg (diferencias dentro del IC95% ±0.009): la señal es lineal en las diferencias de ELO/ranking. Se elige LogReg por el óptimo rigor + explicabilidad (coeficientes = odds-ratio) + minimalismo.

### 4. 5 features, fuente única, sin train/serve skew
`src/features.py` es la fuente de verdad: entrenamiento e inferencia construyen el mismo vector en el mismo orden.

```
diff_elo_general  diff_elo_sup     # ELO general y de superficie por separado
diff_rank         is_unranked      # Ranking capeado a 250 + indicador wildcard/qualifier
diff_age                           # Diferencia de edad
```

Se podaron `diff_h2h`, `diff_form` y `tourney_level_num`: permutation importance ~0 y su ablación movía el AUC dentro del ruido. El ELO ya absorbe la forma; el H2H es débil tras controlar por ELO.

### 5. CV temporal con embargo
`purged_time_series_splits` descarta filas a <7 días de la frontera train/val: rompe la fuga blanda por estado ELO compartido entre partidos contiguos.

---

## Resultados (test ciego 2025, n=2861)

| Modelo | AUC | Log-loss | IC95% AUC |
|--------|-----|----------|-----------|
| **LogReg calibrada** | **0.709** | **0.6225** | [0.690–0.728] |
| Baseline ELO-híbrido | 0.694 | 0.6318 | [0.676–0.714] |

- Accuracy: **65.0%** · Brier: 0.217 (azar = 50% / 0.25)
- Gap CV→test: Δ=+0.007 — prácticamente nulo
- El modelo supera al baseline ELO-híbrido (mismo acceso a la superficie): AUC +0.015, log-loss −0.009, fuera del IC — el lift es real y viene de rank/edad/sin-ranking
- LogReg iguala a GBM/RandomForest/XGBoost (diferencias dentro del IC): la complejidad no añade señal demostrable
- Coeficientes (odds-ratio por +1 desviación estándar): ELO general 1.41, ELO superficie 1.38, ranking 0.73, sin-ranking 0.91, edad 0.92
- Evaluación secundaria 2026 (n=137, IC95% AUC ≈ ±0.043): solo referencial, no usar para decisiones

---

## Estructura del proyecto

```
src/
├── features.py         # Fuente única: FEATURES (5), RANK_CAP, elo_hibrido()
├── elo.py              # ELO histórico (MOV + K-schedule), sin leakage
├── data_processing.py  # Simetrización vectorizada, balance 50/50
├── cv.py               # purged_time_series_splits — embargo temporal 7 días
├── train.py            # LogReg: entrenar_modelo, calibrar_modelo, coeficientes_modelo
└── evaluate.py         # evaluar(), bootstrap_ic95(), evaluar_baseline_elo(), plots

templates/index.html    # SPA "court-side telemetry": identidad por superficie
static/
├── format.js           # Funciones puras de presentación (testeadas con node --test)
└── script.js           # Estado, fetch y render; llama /api/predict, /api/model

tests/                  # 124 tests (pytest) + 4 (node)
notebooks/
└── atp_resumen.ipynb   # Didáctico (legacy — pendiente de actualizar al modelo de 5 features)
docs/
└── ROADMAP.md          # Backlog priorizado + decisiones de poda
archive/                # Scripts de fases anteriores (referencia)
data/                   # CSVs anuales 2020–2026

app.py                  # Flask: /api/players, /api/predict, /api/model
main.py                 # Pipeline: ELO → dataset → entrenar LogReg → evaluar → exportar
visualize.py            # EDA (evolución ELO Top 5 + correlación de features)
requirements.txt        # Dependencias pineadas
Dockerfile              # Imagen para HuggingFace Spaces (gunicorn, puerto PORT)
```

---

## Instalación y ejecución

```bash
git clone https://github.com/Luissantra/ATP-match-prediction.git
cd "ATP-match-prediction"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
# Entrenar y exportar artefactos (.pkl) + plots
python main.py

# Servidor de desarrollo (http://localhost:8000)
python app.py

# Servidor de producción (WSGI, varios workers)
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# Visualizaciones EDA y evolución ELO Top 5
python visualize.py

# Tests
python -m pytest -q
```

> `python app.py` usa Werkzeug (un solo hilo, no apto para producción). Para producción usa gunicorn.

---

## Despliegue (Docker / HuggingFace Spaces)

El repositorio incluye `Dockerfile` y la cabecera YAML de HuggingFace Spaces (`sdk: docker`) en este README. La app lee el puerto de la variable `PORT` (default 8000 local, 7860 en HF).

```bash
# Build y run local
docker build -t atp-forecast .
docker run --rm -p 7860:7860 atp-forecast
# → http://localhost:7860
```

Para HuggingFace Spaces: crear un Space de tipo *Docker* y empujar el repo (los `.pkl` van versionados, ~430 KB en total, sin necesidad de Git LFS). El Space construye la imagen y sirve con gunicorn automáticamente.

---

## API

```
GET /api/players                                          → lista jugadores ordenados por ELO
GET /api/predict?player_a=X&player_b=Y&surface=Z          → predicción (LogReg calibrada)
GET /api/model                                            → métricas test 2025 + coeficientes
```

Parámetros: `surface` ∈ {Hard, Clay, Grass}.

La respuesta de `/api/predict` incluye, por jugador, `elo_surfaces` con el ELO en las tres superficies (`{"Hard": …, "Clay": …, "Grass": …}`) para la gráfica comparativa del frontend, además de `elo_general`, `elo_surface` (la elegida), `elo_hybrid`, `rank`, `age`, `prob_victory` y `unknown`.

---

## Fuentes de datos

- **TML-Database (TennisMyLife):** base de datos principal de partidos ATP 2020–2026.
- **Jeff Sackmann / tennis_atp:** esquema e inspiración original. Licencia [CC Non-Commercial Share Alike](https://github.com/JeffSackmann/tennis_atp).
