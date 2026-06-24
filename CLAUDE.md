# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
1. `src/elo.py` — ratings ELO históricos (general + por superficie) iterando partidos cronológicamente, más H2H y forma reciente pre-partido (sin leakage). ELO híbrido vía `features.elo_hibrido()` (50% general + 50% superficie). Devuelve además el estado final de H2H y forma para inferencia.
2. `src/data_processing.py` — simetrización vectorizada (Ganador/Perdedor → Jugador A/B) para evitar label leakage; balance 50/50. Propaga `tourney_date` para el embargo del CV.
3. `src/features.py` — `FEATURES`, `LEVEL_MAP`, `elo_hibrido()`, `vector_from_features()`. **No reordenar `FEATURES` sin reentrenar.**
4. `src/cv.py` — `purged_time_series_splits()`: `TimeSeriesSplit` con embargo temporal (purga las filas a <N días del fold de validación; rompe la fuga blanda en la frontera).
5. `src/train.py` — `GradientBoostingClassifier` con `GridSearchCV`, **scoring `neg_log_loss`** (el producto es la probabilidad, no el acierto binario), CV temporal con embargo.
6. `src/evaluate.py` — `evaluar(modelo, X, y)` reutilizable → `{accuracy, log_loss, brier, auc}`; plots (matriz confusión, importancia, precisión por superficie) + `graficar_learning_curve()`.
7. `main.py` — orquesta. Features: `[diff_elo, diff_rank, diff_age, diff_h2h, diff_form, tourney_level_num]`. Split temporal: train 2020–2025, test ciego 2026. Exporta `modelo_atp.pkl` y `stats_jugadores.pkl` (dict con `elo_general`, `elo_superficie`, `stats`, `h2h`, `form`, `sklearn_version`).

**Servidor web (`app.py`)**
- Flask en puerto 8000. Estado global de solo lectura cargado de los `.pkl` (apto para varios workers gunicorn).
- Endpoints: `GET /api/players` (lista por ELO) y `GET /api/predict?player_a=X&player_b=Y&surface=Z&tourney_level=L`.
- `construir_features()` reconstruye H2H/forma **reales** desde el historial persistido y mapea `tourney_level` (default 1 = ATP 250). Misma semántica que el entrenamiento.
- `verificar_version_sklearn()` avisa si el pkl se entrenó con otra versión.
- Valida: superficie ∈ {Hard, Clay, Grass}, `player_a != player_b`, params presentes.

**Frontend (`templates/index.html`, `static/`)**
- SPA sin framework. `static/script.js` llama a los endpoints REST. `static/style.css` con estilos dinámicos por superficie.
- Pendiente (G1): el frontend aún no envía `tourney_level` (la API ya lo acepta).

## Métricas (test ciego 2026)

AUC ~0.615, log-loss ~0.683, Brier ~0.244, accuracy ~57% (n≈137). El modelo discrimina débilmente; el gap CV/test es *distribution shift* de 2026, no leakage (confirmado con embargo + learning curve). Techo ≈ predecibilidad intrínseca del tenis.

## Roadmap

`docs/ROADMAP.md` — backlog priorizado (P0/P1/P2 + épica multi-modelo) de la revisión técnica 2026-06-24. Todos los P0 críticos resueltos. Trabajo con TDD estricto, commit por fase.

## Datos

`data/` contiene CSVs anuales. El pipeline de entrenamiento usa 2020–2026. `archive/` contiene scripts de fases anteriores (referencia, no se ejecutan).
