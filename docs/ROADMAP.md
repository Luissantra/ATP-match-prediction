# ROADMAP — ATP Match Prediction

> Backlog derivado de la revisión técnica 2026-06-24. **Backlog cerrado** (2026-06-26).
> **Poda de minimalismo** (2026-06-26): ver sección abajo. 124 tests (pytest) + 4 (node).
> **Nueva épica abierta** (2026-06-27): despliegue HuggingFace + pulido visual. Plan en `docs/superpowers/plans/2026-06-26-hf-deploy-visual-polish.md`.

## Próximo — Épica deploy + visual (2026-06-27)

**Prioridad P0:**
- **D1** — `Dockerfile` + port configurable (`PORT` env var) para HuggingFace Spaces
- **D2** — `README.md` con header YAML de HF Spaces (sdk: docker)

**Prioridad P1 (visual):**
- **V1** — Texturas de fondo diferenciadas por superficie: rejilla ortogonal (hard), trama diagonal (clay), franjas de césped segado (grass)
- **V2** — Fix barras OR: clamp a 50% del track (`* 100` → `* 50`), `overflow: hidden` en `.ftrack`
- **V3** — Gráfica ELO multi-superficie (Hard/Clay/Grass × 2 jugadores) en panel de resultados; requiere exponer `elo_surfaces` en `/api/predict`

## Estado final (post-poda)

Métricas finales (test ciego 2025, n=2861):

- LogReg calibrada: AUC=0.709, log-loss=0.6225, Brier=0.217, accuracy=65.0%
- IC95% AUC ≈ ±0.009
- Gap CV→test Δ=+0.007 (prácticamente nulo)
- Supera baseline ELO-híbrido (AUC 0.709 vs 0.694, log-loss 0.6225 vs 0.6318, fuera del IC)
- Modelo único: LogReg iguala a GBM/RF/XGBoost (la complejidad no añade señal)

## Poda de minimalismo (2026-06-26)

Estudio de permutation importance + ablación sobre test 2025 (n=2861):

- **Features 8 → 5.** Podadas `diff_h2h`, `diff_form`, `tourney_level_num`: permutation importance < 0.001 y ablación dentro del IC95% (±0.009). El lift sobre el ELO viene de rank/edad/sin-ranking. El ELO ya absorbe la forma; el H2H es débil tras controlar por ELO.
- **Modelos 4 → 1.** LogReg calibrada como modelo único. GBM/RF/XGBoost/ensemble no superaban a la LogReg (señal lineal). Retirados `SoftVotingEnsemble`, `comparar_calibracion`, `/api/predict_all`, `/api/models`, `?model=`, panel comparador, dependencia `xgboost`.
- **Explicabilidad.** `coeficientes_modelo()` (odds-ratio por +1 std), `graficar_coeficientes()`, endpoint `/api/model`, panel "Detalle del modelo".
- **Fixes de calidad.** `is_unranked` desde máscara NaN real (no centinela 999); baseline ELO **híbrido** (honesto, mismo acceso a superficie que el modelo); calibración sigmoid (Platt) por defecto; `visualize.py` arreglado (unpack 5→3) y EDA = correlación de features reales (antes pairplot de altura, no usada); `tourney_level`/H2H/forma retirados de ELO/inferencia/frontend.

## Resuelto

### P0 — Críticos
- **C1** — Train/serve skew: `tourney_level_num=3` constante en inferencia → `src/features.py` fuente única; `construir_features` mapea nivel real.
- **C2** — Métrica equivocada `accuracy` en GridSearch → `neg_log_loss`.
- **C3** — Gap CV/test diagnosticado: distribution shift 2026, no overfit. CV con embargo temporal (`purged_time_series_splits`).
- **C4** — Servidor Werkzeug en producción → `gunicorn`.

### P1 — Importantes
- **I1** — Calibración automática sigmoid/isotonic por log-loss (`calibrar_modelo`, `comparar_calibracion`).
- **I2** — `RANK_CAP=250`, feature `is_unranked`. FEATURES = 8.
- **I3** — ELO separado general/superficie (GBM aprende el peso). AUC 0.693→0.707.
- **I4** — Versión sklearn guardada y validada al cargar (`verificar_version_sklearn`). `validar_metadata_pkl` detecta corrupción de pkl. skops evaluado y descartado (no cubre dicts Python puros ni `SoftVotingEnsemble`; riesgo bajo con artefactos propios).
- **I5** — Jugador desconocido → `"unknown": true` en respuesta API.
- **I6** — CSV con columnas faltantes → `ValueError` descriptivo.
- **I7** — `elo_hibrido()` centralizada en `src/features.py`.
- **I8** — `crear_dataset_visual` vectorizado con `np.where`.
- **I9** — Tests coherencia simetrización (`label=1 ↔ diff_rank negativo ↔ diff_elo positivo`), determinismo, endpoints.
- **I10** — `graficar_reliability_diagram`, `graficar_histograma_probas`, `graficar_learning_curve`.

### P2 — Menores
- **M1** — ELO full-precision (sin redondeo acumulado).
- **M2** — `/api/predict` valida `player_a == player_b` → 400.
- **M3** — `custom_tree.py` → `archive/`.
- **M4** — Permutation importance como alternativa robusta al Gini.
- **M5** — `test_label_balanced` exacto (seed=42, assert 0.47).

### Épica multi-modelo
- **E1-E5** — LogReg/RF/GBM/XGBoost calibrados + ensemble soft-voting. `/api/models`, `/api/predict_all`, `?model=`. Panel colapsable frontend.

### Calidad estadística
- **Q1** — IC95% bootstrap en métricas + baseline ELO-crudo.
- **Q2** — ELO con MOV + K-schedule.
- **Q3** — Diagnóstico gap CV/test (`diagnosticar_gap_cv_test`).
- **Q4** — Calibración automática sigmoid/isotonic.
- **Q5** — Notebook honesto: IC, baseline, reliability diagram.
- **Q6** — RNG local (`np.random.default_rng`) sin efecto colateral global.

### Otros
- **G1** — Selector `tourney_level` en frontend (ATP 250/500/Masters/Grand Slam).
- **G2** — `verificar_version_sklearn` al cargar pkl.
- **G3** — Tests endpoint `/api/predict` vía test_client.
- **N1** — `notebooks/atp_resumen.ipynb` didáctico.
- **N2** — Rediseño UI "court-side telemetry" (identidad por superficie, barras divergentes).
