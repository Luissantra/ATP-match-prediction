# ROADMAP — ATP Match Prediction

> Backlog de mejoras derivado de la revisión técnica del 2026-06-24 (data scientist senior).
> Cada hallazgo: `[CRITICIDAD]` — problema — fix. Orden: impacto en modelo → producción → código → viz.
> Estado: `[ ]` pendiente · `[~]` en curso · `[x]` hecho.

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

- [ ] **C4 · `app.run(debug=False)` = servidor de desarrollo Werkzeug.**
  No apto para producción (single-thread).
  *Fix:* `gunicorn -w 4 app:app`. Estado global read-only → sin problema de concurrencia (solo RAM ×4).

---

## Gaps detectados durante Fase 1 (valorados)

- [ ] **G1 · Frontend no envía `tourney_level`** → la UI siempre usa el default (1). La API ya lo acepta. *Decisión: diferir* — es trabajo de UI (dropdown en `index.html` + `script.js`) y requiere verificación en navegador; no bloquea la calidad del modelo. Cerrar tras las fases P0.
- [ ] **G2 · `cargar_modelo` no valida `sklearn_version`** del pkl contra la versión instalada. *Decisión: diferir* — riesgo bajo (versión pineada en requirements); se agrupa con I4 en el hardening de producción (C4).
- [ ] **G3 · Falta test de endpoint `/api/predict` vía `test_client`** (superficie inválida, faltan params, jugador desconocido marcado). *Decisión: near-term* — barato; cerrar junto a I9 antes de la épica multi-modelo.

---

## P1 — Importantes

### Modelo
- [ ] **I1 · GBM sin calibrar.** `predict_proba` mal calibrado por defecto. Envolver en `CalibratedClassifierCV(method='isotonic')` sobre fold temporal.
- [ ] **I2 · `rank=999` genera outliers en `diff_rank`** (wildcard vs Top1 → ~998). Añadir indicador `is_unranked` + cap (≈250), o usar `log(rank)`.
- [ ] **I3 · ELO híbrido 50/50 arbitrario.** Pasar `elo_general` y `elo_superficie` como 2 features y dejar que el GBM aprenda el peso; o validar el 0.5 por grid.

### Producción
- [~] **I4 · Pickle inseguro + riesgo de versión sklearn.** Parcial: `sklearn_version` ya se guarda en el pkl (Fase 1). Falta **validar** al cargar (warn si difiere) y evaluar `skops`. Pin ya presente (`scikit-learn==1.9.0`).
- [ ] **I5 · Jugador desconocido cae a defaults silenciosos** (1500/999/26). Marcar `"unknown": true` en la respuesta y avisar en UI.
- [ ] **I6 · CSV con columnas faltantes → `KeyError` crudo** (`src/elo.py:163`). Validar columnas requeridas por archivo con error claro.

### Código
- [x] **I7 · Lógica ELO híbrido duplicada 3×** ✅ Resuelto (Fase 1). `elo_hibrido()` única en `src/features.py`, usada por `elo.py` y `app.py`. `LEVEL_MAP` también centralizado allí.
- [ ] **I8 · `crear_dataset_visual` reimplementa simetrización con `iterrows`** (lento, diverge de la versión vectorizada, seed global con efecto colateral). Unificar o borrar.
- [~] **I9 · Tests no cubren lo crítico.** Parcial (Fase 1): `tests/test_features.py` (orden/longitud vector) + `tests/test_app_features.py` (h2h/form reales, mismo jugador, `tourney_level` default, jugadores desconocidos). Falta: test del endpoint `/api/predict` vía test_client (superficie inválida, 400s) y determinismo de simetrización.

### Viz
- [ ] **I10 · Faltan 3 plots clave:** curva de calibración (reliability diagram), learning curve, histograma de probas predichas.

---

## P2 — Menores
- [ ] **M1 · `actualizar_ratings` redondea a 1 decimal en cada update** (`elo.py:95`) → error acumulado sobre miles de partidos. Redondear solo al exportar.
- [x] **M2 · `/api/predict` no valida `player_a == player_b`.** ✅ Resuelto (Fase 1): devuelve 400.
- [ ] **M3 · `custom_tree.py` no se usa en el pipeline.** Mover a `archive/` si es legado.
- [ ] **M4 · Feature importance Gini engaña con features correladas** (diff_elo/diff_rank). Añadir `permutation_importance` o **SHAP** (también habilita explicación por-predicción en la UI).
- [ ] **M5 · `test_label_balanced` flojo** (`0.4<ratio<0.6`). Seed fijo + assert exacto.

---

## Épica — Comparación multi-modelo (interactividad)

Registry de modelos + endpoints de comparación. Detalle en sección dedicada del Plan.

- [ ] **E1 ·** `src/train.py` entrena N modelos (LogReg baseline, RandomForest, GBM, XGBoost), cada uno calibrado, mismo split.
- [ ] **E2 ·** Exportar `modelos_atp.pkl` (`{nombre: modelo}`) + `metrics_atp.pkl` (`{nombre: {accuracy, log_loss, brier, auc}}`).
- [ ] **E3 ·** API: `GET /api/models` (lista + métricas), `?model=` en `/api/predict`, `GET /api/predict_all` (probas de los N para el mismo partido).
- [ ] **E4 ·** Frontend: dropdown de modelo, modo "comparar" (barras lado a lado → desacuerdo = incertidumbre), tabla de métricas test 2026.
- [ ] **E5 ·** Ensemble soft-voting como modelo extra.

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
