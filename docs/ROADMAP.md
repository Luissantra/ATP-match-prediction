# ROADMAP — ATP Match Prediction

> Backlog de mejoras derivado de la revisión técnica del 2026-06-24 (data scientist senior).
> Cada hallazgo: `[CRITICIDAD]` — problema — fix. Orden: impacto en modelo → producción → código → viz.
> Estado: `[ ]` pendiente · `[~]` en curso · `[x]` hecho.

## 🧭 Próximos pasos (orden recomendado)

**P0 críticos (C1-C4) + I1 + Épica E1-E3 + I10 + G3 + I2 + I3 resueltos** (sesiones 2026-06-24 / 2026-06-26).

1. **✅ I1 · Calibración** — hecho.
2. **✅ E1+E2 · Multi-modelo entrenamiento + artefactos** — hecho.
3. **✅ E3 · API multi-modelo** — hecho. 92 tests total.
4. **✅ I10 · Reliability diagram + histograma de probas** — hecho.
5. **✅ G3 · Tests /api/predict** — hecho.
6. **✅ I2 + I3 · rank cap + ELO separado** — hecho. FEATURES = 8. AUC 0.615→0.629, log-loss 0.683→0.674.
7. **✅ E4 · Frontend multi-modelo** — hecho. Panel colapsable "comparar modelos" (`/api/predict_all` + `/api/models`), badge jugador desconocido.
8. **⏸ E5 · Ensemble soft-voting** — diferido.
9. **✅ N2 · Rediseño UI** — hecho. Dirección "court-side telemetry" (identidad por superficie, barras divergentes de factores). Incluye G1 (`tourney_level`) y E4. Funciones puras en `static/format.js` (TDD `node --test`). Spec: `docs/superpowers/specs/2026-06-26-frontend-redesign-design.md`.
10. **✅ M1 · ELO sin redondeo acumulado** — hecho. `actualizar_ratings` full-precision.
11. **✅ M3 · `custom_tree.py` → `archive/`** — hecho.
12. **✅ M5 · `test_label_balanced` exacto** — seed=42 fijo, assert 0.47.
13. **✅ I5 · Jugador desconocido → `"unknown": true`** — hecho. 2 tests nuevos.
14. **✅ I6 · Validar columnas CSV** — hecho. `ValueError` descriptivo. 2 tests nuevos.
15. **✅ N1 · Notebook didáctico** — hecho. `notebooks/atp_resumen.ipynb` (ELO, híbrido, simetrización, 8 features, CV con embargo, métricas honestas).
16. **✅ Épica Q · Calidad estadística** — hecho (2026-06-26). Rigor estadístico 4/10 → 7/10: IC95% bootstrap en métricas, baseline ELO-crudo, MOV + K-schedule en ELO, calibración automática sigmoid/isotonic, notebook honesto. 132 tests.
17. **✅ M4 · Permutation importance** — hecho. `permutation_importancia` + `graficar_permutation_importance` en `src/evaluate.py`. 140 tests.
18. **✅ I8** — `crear_dataset_visual` vectorizado. 142 tests.
19. **▶️ E5** — Ensemble soft-voting (siguiente).

Convención de trabajo: **TDD estricto, un commit por ítem/fase**, actualizar este roadmap al cerrar.

---

## P0 — Críticos (atacar ya)

- [x] **C1 · Train/serve skew: `tourney_level_num=3` constante en inferencia.** ✅ Resuelto (Fase 1).
  `src/features.py` = fuente única; `app.construir_features` reconstruye H2H/forma reales del historial persistido y mapea `tourney_level` (default 1=ATP250, no 3). Verificado: la predicción ahora varía con el nivel (65.6% G vs 60.4% default).
  Entreno usa el valor real (`LEVEL_MAP`), pero `app.py` lo fija a `3` (= Finals/Olympics, raro; la mediana del circuito es `1`/250). Cada predicción en vivo simula un partido de nivel Finals → sesgo sistemático. `diff_h2h=0.0`/`diff_form=0.0` son tolerables (0.0 es el neutro simétrico), `3` no.
  *Fix:* pasar `tourney_level` desde el frontend y mapearlo, o (preferido a corto plazo) entrenar un **modelo servible reducido** sin las 3 features que no existen en inferencia. Ver Plan §C1.

- [x] **C2 · Métrica equivocada: `scoring='accuracy'`.** ✅ Resuelto (Fase 2).
  GridSearch ahora optimiza `neg_log_loss`. `src/evaluate.py` expone `evaluar(modelo, X, y) -> {accuracy, log_loss, brier, auc}` (reutilizable para la épica multi-modelo) y el reporte imprime las 4. Test ciego 2026: AUC 0.615, log-loss 0.683, Brier 0.244, accuracy 56.9% — el modelo discrimina (AUC>0.5) aunque débilmente; el accuracy solo lo subestimaba.

- [x] **C3 · Gap CV vs test = ¿sobreajuste o CV optimista?** ✅ Resuelto/diagnosticado (Fase 3).
  `src/cv.py::purged_time_series_splits` añade embargo temporal (7 días) a `TimeSeriesSplit`; `train.py` lo usa en GridSearch y `evaluate.graficar_learning_curve` dibuja la curva.
  **Hallazgo:** con embargo, CV log-loss 0.620 vs test ciego 0.683. El gap **persiste** → NO era fuga blanda sino **distribution shift de 2026**. La learning curve muestra sobreajuste leve (gap train/val ~0.05, estrechándose) y validación aún descendente → el modelo está **limitado por señal/datos**, no roto por overfit. Más datos ayudarían; el techo es la predecibilidad intrínseca del tenis (~AUC 0.62-0.65).

- [x] **C4 · `app.run(debug=False)` = servidor de desarrollo Werkzeug.** ✅ Resuelto (Fase 4).
  `gunicorn==23.0.0` en requirements; README documenta `gunicorn -w 4 -b 0.0.0.0:8000 app:app` con aviso de que `python app.py` es solo desarrollo. Verificado: gunicorn (2 workers) sirve `/api/players` y `/api/predict`.

---

## Gaps detectados durante Fase 1 (valorados)

- [x] **G1 · Frontend no envía `tourney_level`** → hecho. Selector pill ATP 250/500/Masters/Grand Slam (keys de `LEVEL_MAP`); se envía a `/api/predict` y `/api/predict_all`.
- [x] **G2 · `cargar_modelo` no valida `sklearn_version`** ✅ Resuelto (Fase 4). `app.verificar_version_sklearn()` avisa si la versión del pkl difiere de la instalada (3 tests). Cubre la mitad pendiente de **I4**.
- [x] **G3 · Falta test de endpoint `/api/predict` vía `test_client`** ✅ Resuelto. Tests en `tests/test_api_endpoints.py`: superficie inválida, player_a/b faltantes, mismo jugador, jugador desconocido (defaults). 81 tests total.

---

## P1 — Importantes

### Modelo
- [x] **I1 · GBM sin calibrar.** ✅ `calibrar_modelo(base, X, y, dates, method)` en `src/train.py`; usa `purged_time_series_splits` para el fold de calibración. `main.py` imprime comparación base→calibrado en test ciego y exporta el calibrado.
- [x] **I2 · `rank=999` genera outliers en `diff_rank`** ✅ Resuelto. `RANK_CAP=250` en `src/features.py`. Cap aplicado en `data_processing.py` y `app.py`. Feature `is_unranked` añadida (∈ {-1,0,1}). FEATURES pasa de 6 a 8.
- [x] **I3 · ELO híbrido 50/50 arbitrario.** ✅ Resuelto. `elo.py` emite `elo_winner/loser_general` y `elo_winner/loser_sup`. `diff_elo` reemplazada por `diff_elo_general` + `diff_elo_sup`. Reentrenado: AUC 0.615→0.629, log-loss 0.683→0.674 (GBM test ciego 2026). 92 tests.

### Producción
- [~] **I4 · Pickle inseguro + riesgo de versión sklearn.** Versión: ✅ se guarda (Fase 1) y se **valida al cargar** (Fase 4, G2). Pendiente solo lo de seguridad: evaluar `skops` para carga sin ejecución de código arbitrario (riesgo bajo con artefactos propios).
- [x] **I5 · Jugador desconocido cae a defaults silenciosos** ✅ Resuelto. `"unknown": true/false` en cada jugador de la respuesta API.
- [x] **I6 · CSV con columnas faltantes → `KeyError` crudo** ✅ Resuelto. `ValueError` descriptivo con lista de columnas ausentes.

### Código
- [x] **I7 · Lógica ELO híbrido duplicada 3×** ✅ Resuelto (Fase 1). `elo_hibrido()` única en `src/features.py`, usada por `elo.py` y `app.py`. `LEVEL_MAP` también centralizado allí.
- [x] **I8 · `crear_dataset_visual` reimplementa simetrización con `iterrows`** ✅ Vectorizado con `np.where` (mismo patrón que `preparar_datos_entrenamiento`). 6 tests nuevos. 142 tests total.
- [~] **I9 · Tests no cubren lo crítico.** Parcial (Fase 1): `tests/test_features.py` (orden/longitud vector) + `tests/test_app_features.py` (h2h/form reales, mismo jugador, `tourney_level` default, jugadores desconocidos). Falta: test del endpoint `/api/predict` vía test_client (superficie inválida, 400s) y determinismo de simetrización.

### Viz
- [x] **I10 · Faltan 3 plots clave:** ✅ Resuelto. `graficar_reliability_diagram` e `graficar_histograma_probas` añadidas a `src/evaluate.py`; `graficar_learning_curve` ya existía. Las 3 se invocan desde `main.py`. 8 tests en `tests/test_evaluate.py`.
- [x] **N1 · Notebook didáctico** ✅ Hecho. `notebooks/atp_resumen.ipynb`: matemática del ELO (logística + actualización), ELO híbrido, simetrización (anti-leakage), las **8** features (no 6 — actualizado tras I2/I3), CV temporal con embargo, lectura honesta de métricas (AUC/log-loss/Brier vs accuracy + learning curve). Narrativa + celdas ejecutables reusando `src/`. Pendiente menor: ver Q5 (no enseña IC/baseline ELO-crudo, métricas no coinciden con producción).
- [x] **N2 · Actualizar los visuales de la web** (`templates/index.html`, `static/style.css`, `static/script.js`): hecho. Rediseño "court-side telemetry"; barras divergentes pintan `features_debug` (incl. `diff_h2h`, `diff_form`); corregido bug del campo obsoleto `diff_elo`. Lógica pura testeada en `static/format.js`.

---

## P2 — Menores
- [x] **M1 · `actualizar_ratings` redondea a 1 decimal en cada update** ✅ Resuelto. Full-precision en updates; redondear solo al mostrar.
- [x] **M2 · `/api/predict` no valida `player_a == player_b`.** ✅ Resuelto (Fase 1): devuelve 400.
- [x] **M3 · `custom_tree.py` no se usa en el pipeline.** ✅ Movido a `archive/`.
- [x] **M4 · Feature importance Gini engaña con features correladas** (diff_elo/diff_rank). ✅ `permutation_importancia` + `graficar_permutation_importance` en `src/evaluate.py`; scoring=neg_log_loss, barras ±1std. 4 tests nuevos. Integrado en `main.py` tras los plots existentes. 140 tests total.
- [x] **M5 · `test_label_balanced` flojo** ✅ Resuelto. Assert exacto `ratio == 0.47` (seed=42, 100 filas).

---

## Épica — Comparación multi-modelo (interactividad)

Registry de modelos + endpoints de comparación. Detalle en sección dedicada del Plan.

- [x] **E1 ·** ✅ `entrenar_todos_los_modelos(X, y, dates)` en `src/train.py`: LogReg, RF, GBM, XGBoost — cada uno con GridSearchCV(neg_log_loss) + CV temporal purgado + `calibrar_modelo`. 4 tests. `xgboost==3.2.0` en requirements.txt.
- [x] **E2 ·** ✅ `main.py` exporta `modelos_atp.pkl` (`{nombre: modelo_calibrado}`) + `metrics_atp.pkl` (`{nombre: {accuracy, log_loss, brier, auc}}`). `modelo_atp.pkl` (GBM calibrado) se mantiene para compatibilidad con `app.py`.
- [x] **E3 ·** ✅ API multi-modelo: `GET /api/models` (lista ordenada por log-loss), `?model=` en `/api/predict` (valida nombre, 400 si inválido), `GET /api/predict_all` (probas de los 4 modelos para el mismo partido). Helper `_predecir_con()` evita duplicar lógica de features. 13 tests nuevos en `tests/test_api_endpoints.py`. 77 tests total.
- [x] **E4 ·** Frontend: hecho. Panel colapsable "comparar los 4 modelos" con probabilidades del partido (`/api/predict_all`) + tabla de métricas test 2026 (`/api/models`), fila `gbm` resaltada. Badge de jugador desconocido (`unknown`). Decisión: sin dropdown de modelo en el formulario (gbm fijo); la comparación vive en el panel.
- [ ] **E5 ·** Ensemble soft-voting como modelo extra.

---

## Épica — Calidad estadística (revisión 2026-06-26)

Detalle y severidades en `docs/REVISION-CALIDAD-2026-06-26.md`. El problema no es la ingeniería (8/10) sino el rigor estadístico (4/10): ELO sin exprimir y conclusiones que ignoran n≈137.

### Prioridad alta
- [x] **Q1 · 🔴 Baseline ELO-crudo + IC en métricas.** ✅ `bootstrap_ic95` + `evaluar_con_ic` + `evaluar_baseline_elo` en `src/evaluate.py`. `main.py` imprime baseline ELO-crudo e IC95% para cada modelo. Con n≈137, IC95% AUC ≈ ±0.08 → diferencias < 0.08 son ruido estadístico. Nota: "AUC 0.615→0.629 por I3" cae dentro del IC — no era mejora demostrable. (R1+R2)
- [x] **Q2 · 🟠 ELO con MOV + K-schedule.** ✅ `_extraer_sets`, `_mov_factor`, `_k_for_player` en `src/elo.py`. Integrado en `calcular_elos_historicos(use_mov=True, use_k_schedule=True)`: straight sets → K×1.25–1.5, K=48/<10 partidos, K=40/<30, K=32 establecido. Reentrenar para medir delta AUC real con Q1. (R3)

### Prioridad media
- [x] **Q3 · 🟠 Separar causas del gap CV/test.** ✅ `diagnosticar_gap_cv_test` en `src/evaluate.py`. Lenguaje corregido: "consistente con distribution shift + optimismo de GridSearch" (no "confirmado"). Las 3 causas listadas: selection bias, sesgo estacional 2026 parcial, shift real. (R4)
- [x] **Q4 · 🟡 Calibración: sigmoid vs isotonic.** ✅ `comparar_calibracion` en `src/train.py` elige automáticamente por log-loss. `main.py` recalibra GBM con el método ganador. (R5)

### Notebook + higiene
- [x] **Q5 · 🟠 Notebook — honestidad métrica real.** ✅ `notebooks/atp_resumen.ipynb` §7: aviso no-coincidencia con producción, celda IC95% bootstrap AUC + log-loss, celda baseline ELO-crudo vs GBM, `graficar_reliability_diagram`. Notebook ejecuta sin errores. (R7+R8+R9+R10)
- [x] **Q6 · 🟡 RNG global → local.** ✅ `np.random.default_rng(seed)` en `preparar_datos_entrenamiento(seed=42)` y `crear_dataset_visual(seed=42)`. Sin efecto colateral en RNG global. (R6)

---

## Plan de resolución — P0

> Estrategia: **C1+I7 juntos** (mismo fix de raíz: una única función de features), luego **C2**, luego **C3**, luego **C4** (independiente, rápido).

### Fase 1 — C1 + I7: feature pipeline única, sin skew

**Causa raíz:** el vector de features se construye en 2 sitios independientes (`data_processing.preparar_datos_entrenamiento` para entreno, `app.predict` a mano para inferencia). Divergen → skew.

1. Crear `src/features.py` con la **fuente de verdad única**:
   - `elo_hibrido(gen, sup, w=0.5)` (reemplaza las 3 copias → cierra I7).
   - `construir_vector(player_a_stats, player_b_stats, surface, tourney_level, h2h, form)` → devuelve dict/array en el **mismo orden** que `FEATURES`.
2. Decidir el set servible. Dos opciones:
   - **(A, recomendada corto plazo)** Entrenar un modelo reducido a `[diff_elo, diff_rank, diff_age]` para la API → elimina las 3 features fantasma. Mantener el modelo full solo para evaluación batch/EDA.
   - **(B, mejor a medio plazo)** Servir las 6 reales: añadir selector de `tourney_level` en el frontend y persistir H2H/form por jugador en el pkl para reconstruirlos en inferencia.
3. `app.predict` y `main.py` consumen `src/features.py` → imposible que diverjan.
4. **Tests (TDD):** test que afirma que el vector de inferencia tiene el orden/longitud de `FEATURES` (cierra parte de I9); test de que `elo_hibrido` da el mismo valor en los 3 contextos.

*Verificación:* `pytest`, y predicción manual del mismo enfrentamiento por la API vs cálculo directo → deben coincidir.

### Fase 2 — C2: métrica probabilística

1. `src/train.py`: `scoring='neg_log_loss'` en GridSearchCV.
2. `src/evaluate.py`: añadir `log_loss`, `brier_score_loss`, `roc_auc_score` al reporte de test ciego, junto a accuracy.
3. Función reutilizable `evaluar(modelo, X, y) -> dict` (la reutilizará la épica multi-modelo).

*Verificación:* re-entrenar; confirmar que `best_params_` puede cambiar y que el reporte imprime las 4 métricas.

### Fase 3 — C3: CV honesto

1. Implementar embargo temporal: tras cada split de `TimeSeriesSplit`, descartar de train las filas dentro de una ventana de N días del inicio de val (purging). Empezar con N = 7 días.
2. Añadir learning curve a `evaluate.py` (train vs val score sobre tamaños crecientes) → diagnostica si el gap es sobreajuste o falta de datos.
3. Reportar CV final con `neg_log_loss`, no accuracy.

*Verificación:* el `best_score_` de CV debería acercarse al test ciego (gap < ~5 pts si el embargo corrige la fuga blanda).

### Fase 4 — C4: WSGI de producción

1. Añadir `gunicorn` a `requirements.txt`.
2. Documentar arranque prod en `CLAUDE.md`/`README.md`: `gunicorn -w 4 -b 0.0.0.0:8000 app:app`.
3. Confirmar que `cargar_modelo()` corre por worker (lazy load ya presente lo cubre).

*Verificación:* arrancar con gunicorn, hit a `/api/players` y `/api/predict`.

---

### Orden sugerido de ejecución
`Fase 1 (C1+I7) → Fase 2 (C2) → Fase 3 (C3) → Fase 4 (C4) → P1 (I1, I9 primero) → Épica multi-modelo`.

C1+I7 es prerequisito de la épica multi-modelo (todos los modelos deben consumir el feature pipeline único).
