# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow de integración

Proyecto individual. Al completar trabajo en una rama: merge a main localmente y push directo. No se crean Pull Requests.

## Comandos

```bash
# Activar entorno virtual
source venv/bin/activate

# Instalar dependencias (versiones pineadas)
pip install -r requirements.txt

# Entrenar modelo y exportar artefactos (.pkl) + plots
python main.py

# Servidor de DESARROLLO (http://localhost:8000)
python app.py

# Servidor de PRODUCCIÓN (WSGI, varios workers)
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# Generar visualizaciones EDA y evolución ELO Top 5
python visualize.py

# Tests
python -m pytest -q
```

Dependencias en `requirements.txt` (versiones pineadas: flask, flask-cors, gunicorn, numpy, pandas, scikit-learn, matplotlib, seaborn).

## Arquitectura

Pipeline en dos etapas separadas. La **fuente única de verdad del vector de features** es `src/features.py`: tanto el entrenamiento como la inferencia construyen las features desde ahí para evitar *train/serve skew*.

**Entrenamiento (`main.py` → `src/`)**
1. `src/elo.py` — ratings ELO históricos (general + por superficie) iterando partidos cronológicamente, más H2H y forma reciente pre-partido (sin leakage). MOV + K-schedule activos. Devuelve estado final de H2H y forma para inferencia.
2. `src/data_processing.py` — simetrización vectorizada (Ganador/Perdedor → Jugador A/B) para evitar label leakage; balance 50/50. Propaga `tourney_date` para el embargo del CV.
3. `src/features.py` — `FEATURES` (8), `LEVEL_MAP`, `RANK_CAP`, `elo_hibrido()`, `vector_from_features()`. **No reordenar `FEATURES` sin reentrenar.**
4. `src/cv.py` — `purged_time_series_splits()`: `TimeSeriesSplit` con embargo temporal (purga las filas a <N días del fold de validación; rompe la fuga blanda en la frontera).
5. `src/train.py` — `entrenar_todos_los_modelos()`: LogReg, RF, GBM, XGBoost con `GridSearchCV(neg_log_loss)` + CV temporal + `CalibratedClassifierCV`. Devuelve `(calibrados, bases, cv_scores)`.
6. `src/evaluate.py` — `evaluar()`, `evaluar_con_ic()`, `bootstrap_ic95()`, `evaluar_baseline_elo()`; plots (matriz confusión, importancia, reliability diagram, histograma probas, learning curve, precisión por superficie).
7. `main.py` — orquesta. Split: train 2020–2024, test principal 2025 (n≈2861), eval secundaria 2026 (n≈137, referencial). Exporta `modelos_atp.pkl` (los 4 calibrados), `metrics_atp.pkl`, `stats_jugadores.pkl`.

**Servidor web (`app.py`)**
- Flask en puerto 8000. Estado global de solo lectura cargado de los `.pkl` (apto para varios workers gunicorn).
- Endpoints: `GET /api/players` (lista por ELO) y `GET /api/predict?player_a=X&player_b=Y&surface=Z&tourney_level=L`.
- `construir_features()` reconstruye H2H/forma **reales** desde el historial persistido y mapea `tourney_level` (default 1 = ATP 250). Misma semántica que el entrenamiento.
- `verificar_version_sklearn()` avisa si el pkl se entrenó con otra versión.
- Valida: superficie ∈ {Hard, Clay, Grass}, `player_a != player_b`, params presentes.

**Frontend (`templates/index.html`, `static/`)**
- SPA sin framework (vanilla). Diseño "court-side telemetry": identidad por superficie (fondo color de pista + líneas de cancha como divisores), tipografía Bricolage Grotesque / Inter / JetBrains Mono.
- `static/format.js` — funciones puras de presentación (`normalizeFactor`, `formatDiff`, `formatRank`, `mergeModels`), patrón dual browser+node, testeadas con `node --test tests/format.test.mjs`. Expone funciones como globales **y** en `window.ATPFormat`; por eso `static/script.js` va envuelto en un IIFE (evita colisión de identificadores al hacer destructuring).
- `static/script.js` — estado, fetch y render; usa `format.js`. Llama a `/api/predict`, `/api/predict_all` y `/api/models`.
- Señales mostradas: barras divergentes de `features_debug` (a quién favorece cada factor; son diferencias de feature, no peso del modelo), barra de probabilidad como cancha vista desde arriba, panel colapsable comparar 4 modelos, badge de jugador desconocido (`unknown`).
- Selectores: superficie (Hard/Clay/Grass) y nivel de torneo (G1 hecho: ATP 250/500/Masters/Grand Slam → keys de `LEVEL_MAP`).
- Assets enlazados con `?v=N` (cache-busting); súbelo al cambiar CSS/JS.

## Métricas (test ciego 2025, n=2861)

GBM: AUC=0.707, log-loss=0.6259, Brier=0.218, accuracy=64.7%. IC95% AUC ≈ ±0.009 — suficiente para comparar modelos. Gap CV→test Δ=+0.009 (prácticamente nulo). ML supera baseline ELO-crudo (AUC 0.707 vs 0.693, diferencia fuera del IC). Los 4 modelos son indistinguibles entre sí. Eval secundaria 2026 (n=137, IC ≈ ±0.043): solo referencial.

## Roadmap

`docs/ROADMAP.md` — backlog priorizado (P0/P1/P2 + épica multi-modelo) de la revisión técnica 2026-06-24. Todos los P0 críticos resueltos. Trabajo con TDD estricto, commit por fase.

## Datos

`data/` contiene CSVs anuales. El pipeline de entrenamiento usa 2020–2024 (train), 2025 (test), 2026 (eval secundaria). `archive/` contiene scripts de fases anteriores (referencia, no se ejecutan).
