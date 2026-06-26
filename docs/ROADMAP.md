# ROADMAP — ATP Match Prediction

> Backlog derivado de la revisión técnica 2026-06-24. **Backlog cerrado** (2026-06-26). 153 tests.

## Estado final

Todos los ítems resueltos. Métricas finales (test ciego 2025, n=2861):

- GBM: AUC=0.707, log-loss=0.6259, Brier=0.218, accuracy=64.7%
- IC95% AUC ≈ ±0.009 — suficiente para comparar modelos
- Gap CV→test Δ=+0.009 (prácticamente nulo)
- ML supera baseline ELO-crudo (AUC 0.707 vs 0.693, diferencia fuera del IC)
- 4 modelos indistinguibles entre sí; ensemble soft-voting incluido

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
