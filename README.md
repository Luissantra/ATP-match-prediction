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
Simetrización vectorizada: Ganador/Perdedor → Jugador A/B aleatorio con balance 50/50. H2H y forma reciente calculados *pre-partido*, sin información del resultado actual.

### 3. Multi-modelo calibrado
Cuatro clasificadores con `GridSearchCV(scoring='neg_log_loss')` + CV temporal purgado + `CalibratedClassifierCV`:

| Modelo | Rol |
|--------|-----|
| LogisticRegression | Baseline lineal |
| RandomForest | Ensemble bagging |
| GradientBoosting | Modelo principal (gbm) |
| XGBoost | Ensemble boosting alternativo |

Con n=2861 los IC95% son suficientemente estrechos (~±0.009 en AUC) para comparar modelos de verdad.

### 4. 8 features, fuente única, sin train/serve skew
`src/features.py` es la fuente de verdad: entrenamiento e inferencia construyen el mismo vector en el mismo orden.

```
diff_elo_general  diff_elo_sup     # ELO general y de superficie por separado
diff_rank         is_unranked      # Ranking capeado a 250 + indicador wildcard/qualifier
diff_age          diff_h2h         # Edad y head-to-head histórico
diff_form         tourney_level_num # Forma reciente y nivel de torneo
```

### 5. CV temporal con embargo
`purged_time_series_splits` descarta filas a <7 días de la frontera train/val: rompe la fuga blanda por estado ELO compartido entre partidos contiguos.

---

## Resultados (test ciego 2025, n=2861)

| Modelo | AUC | Log-loss | IC95% AUC |
|--------|-----|----------|-----------|
| GBM (principal) | **0.707** | **0.6259** | [0.689–0.726] |
| XGBoost | 0.709 | 0.6238 | [0.691–0.728] |
| LogReg | 0.709 | 0.6232 | [0.691–0.728] |
| RandomForest | 0.706 | 0.6264 | [0.687–0.724] |
| **Baseline ELO-crudo** | 0.693 | 0.6368 | [0.675–0.712] |

- Accuracy GBM: **64.7%** (azar = 50%)
- Gap CV→test: Δ=+0.009 — prácticamente nulo
- ML supera al ELO-crudo: diferencia AUC ~0.014, fuera del IC — real con este n
- Los 4 modelos son indistinguibles entre sí (ICs solapados)
- Evaluación secundaria 2026 (n=137, IC95% AUC ≈ ±0.043): solo referencial, no usar para decisiones

---

## Estructura del proyecto

```
src/
├── features.py         # Fuente única: FEATURES, LEVEL_MAP, RANK_CAP, elo_hibrido()
├── elo.py              # ELO histórico (MOV + K-schedule + H2H + forma), sin leakage
├── data_processing.py  # Simetrización vectorizada, balance 50/50
├── cv.py               # purged_time_series_splits — embargo temporal 7 días
├── train.py            # 4 modelos con GridSearchCV + calibración
└── evaluate.py         # evaluar(), bootstrap_ic95(), evaluar_baseline_elo(), plots

templates/index.html    # SPA "court-side telemetry": identidad por superficie
static/
├── format.js           # Funciones puras de presentación (testeadas con node --test)
└── script.js           # Estado, fetch y render; llama /api/predict, /api/predict_all

tests/                  # 132 tests (pytest)
notebooks/
└── atp_resumen.ipynb   # Didáctico: ELO, simetrización, 8 features, CV con embargo, IC95%
docs/
└── ROADMAP.md          # Backlog priorizado
archive/                # Scripts de fases anteriores (referencia)
data/                   # CSVs anuales 2020–2026

app.py                  # Flask: /api/players, /api/predict, /api/predict_all, /api/models
main.py                 # Pipeline: ELO → dataset → entrenar 4 modelos → evaluar → exportar
visualize.py            # EDA y evolución ELO Top 5
requirements.txt        # Dependencias pineadas
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

## API

```
GET /api/players                                          → lista jugadores ordenados por ELO
GET /api/predict?player_a=X&player_b=Y&surface=Z          → predicción con GBM (por defecto)
GET /api/predict?...&model=logreg                         → predicción con modelo específico
GET /api/predict_all?player_a=X&player_b=Y&surface=Z      → predicción de los 4 modelos
GET /api/models                                           → métricas test 2025 por modelo
```

Parámetros: `surface` ∈ {Hard, Clay, Grass}, `tourney_level` ∈ {G, M, 500, 250, ...}.

---

## Fuentes de datos

- **TML-Database (TennisMyLife):** base de datos principal de partidos ATP 2020–2026.
- **Jeff Sackmann / tennis_atp:** esquema e inspiración original. Licencia [CC Non-Commercial Share Alike](https://github.com/JeffSackmann/tennis_atp).
