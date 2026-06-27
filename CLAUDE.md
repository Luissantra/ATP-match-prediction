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

# Imagen Docker (HuggingFace Spaces). Lee puerto de PORT (default 7860 en la imagen)
docker build -t atp-forecast . && docker run --rm -p 7860:7860 atp-forecast

# Generar visualizaciones EDA y evolución ELO Top 5
python visualize.py

# Tests
python -m pytest -q
```

Dependencias en `requirements.txt` (versiones pineadas: flask, flask-cors, gunicorn, numpy, pandas, scikit-learn, matplotlib, seaborn).

## Arquitectura

Pipeline en dos etapas separadas. La **fuente única de verdad del vector de features** es `src/features.py`: tanto el entrenamiento como la inferencia construyen las features desde ahí para evitar *train/serve skew*.

**Entrenamiento (`main.py` → `src/`)**
1. `src/elo.py` — ratings ELO históricos (general + por superficie) iterando partidos cronológicamente, pre-partido (sin leakage). MOV + K-schedule activos. Devuelve `(df, elo_general, elo_superficie)`.
2. `src/data_processing.py` — simetrización vectorizada (Ganador/Perdedor → Jugador A/B) para evitar label leakage; balance 50/50. Propaga `tourney_date` para el embargo del CV. `is_unranked` desde la máscara NaN real del rank (no el centinela 999).
3. `src/features.py` — `FEATURES` (5: `diff_elo_general`, `diff_elo_sup`, `diff_rank`, `is_unranked`, `diff_age`), `RANK_CAP`, `elo_hibrido()`, `vector_from_features()`. **No reordenar `FEATURES` sin reentrenar.** (h2h/forma/nivel-torneo se podaron: perm. importance ~0; ver `docs/ROADMAP.md`.)
4. `src/cv.py` — `purged_time_series_splits()`: `TimeSeriesSplit` con embargo temporal (purga las filas a <N días del fold de validación; rompe la fuga blanda en la frontera).
5. `src/train.py` — modelo único: `entrenar_modelo()` (LogReg estandarizada + `GridSearchCV(neg_log_loss)` sobre `C` + CV temporal purgado), `calibrar_modelo()` (sigmoid/Platt), `coeficientes_modelo()` (odds-ratio para explicabilidad). LogReg iguala a GBM/RF/XGBoost en AUC con coeficientes interpretables.
6. `src/evaluate.py` — `evaluar()`, `evaluar_con_ic()`, `bootstrap_ic95()`, `evaluar_baseline_elo()` (baseline ELO **híbrido**), `graficar_coeficientes()`; plots (matriz confusión, permutation importance, coeficientes, reliability diagram, histograma probas, learning curve, precisión por superficie).
7. `main.py` — orquesta. Split: train 2020–2024, test principal 2025 (n≈2861), eval secundaria 2026 (n≈137, referencial). Exporta `modelos_atp.pkl` (LogReg calibrado, objeto único), `metrics_atp.pkl`, `stats_jugadores.pkl` (incluye `coeficientes`).

**Servidor web (`app.py`)**
- Flask. Puerto configurable vía env `PORT` (default 8000 local; la imagen Docker fija 7860 para HF Spaces). Estado global de solo lectura cargado de los `.pkl` (apto para varios workers gunicorn).
- Endpoints: `GET /api/players` (lista por ELO), `GET /api/predict?player_a=X&player_b=Y&surface=Z`, `GET /api/model` (métricas + coeficientes).
- `/api/predict` devuelve por jugador `elo_surfaces` (ELO en Hard/Clay/Grass) para la gráfica multi-superficie del frontend, además de `elo_general`/`elo_surface`/`elo_hybrid`/`rank`/`age`/`prob_victory`/`unknown`.
- `construir_features()` reconstruye las 5 features con la misma semántica que el entrenamiento. La inferencia sirve numpy (evita warning de feature-names).
- `verificar_version_sklearn()` avisa si el pkl se entrenó con otra versión.
- Valida: superficie ∈ {Hard, Clay, Grass}, `player_a != player_b`, params presentes.

**Frontend (`templates/index.html`, `static/`)**
- SPA sin framework (vanilla). Diseño "court-side telemetry": identidad por superficie (fondo color de pista + líneas de cancha como divisores), tipografía Bricolage Grotesque / Inter / JetBrains Mono.
- `static/format.js` — funciones puras de presentación (`normalizeFactor`, `formatDiff`, `formatRank`), patrón dual browser+node, testeadas con `node --test tests/format.test.mjs`. Expone funciones como globales **y** en `window.ATPFormat`; por eso `static/script.js` va envuelto en un IIFE (evita colisión de identificadores al hacer destructuring).
- `static/script.js` — estado, fetch y render; usa `format.js`. Llama a `/api/predict` y `/api/model`.
- Señales mostradas: barra de probabilidad como cancha vista desde arriba, gráfica ELO multi-superficie (`renderEloChart`, barras agrupadas Hard/Clay/Grass × 2 jugadores desde `elo_surfaces`), barras divergentes de `features_debug` (a quién favorece cada factor; son diferencias de feature, no peso del modelo), panel colapsable "Detalle del modelo" (métricas + coeficientes/odds-ratio; barras OR clamped a 50% del track), badge de jugador desconocido (`unknown`).
- Fondo por superficie: `.court-bg` con textura distinta (`body.surface-hard/clay/grass`): rejilla ortogonal / trama diagonal / franjas de césped.
- Selector: superficie (Hard/Clay/Grass).
- Assets enlazados con `?v=N` (cache-busting; actual `v=8`); incrementa N **en los tres** (style.css, format.js, script.js) al cambiar CSS/JS.

## Métricas (test ciego 2025, n=2861)

LogReg calibrada: AUC=0.709, log-loss=0.6225, Brier=0.217, accuracy=65.0%. IC95% AUC ≈ ±0.009. Gap CV→test Δ=+0.007 (prácticamente nulo). ML supera baseline ELO-híbrido (AUC 0.709 vs 0.694, log-loss 0.6225 vs 0.6318, fuera del IC). El lift sobre el ELO viene de rank/edad/sin-ranking; LogReg iguala a GBM/RF/XGBoost (la complejidad no añade señal). Eval secundaria 2026 (n=137, IC ≈ ±0.043): solo referencial.

## Roadmap

`docs/ROADMAP.md` — backlog priorizado (P0/P1/P2 + épica multi-modelo) de la revisión técnica 2026-06-24. Todos los P0 críticos resueltos. Trabajo con TDD estricto, commit por fase.

## Datos

`data/` contiene CSVs anuales. El pipeline de entrenamiento usa 2020–2024 (train), 2025 (test), 2026 (eval secundaria). `archive/` contiene scripts de fases anteriores (referencia, no se ejecutan).
