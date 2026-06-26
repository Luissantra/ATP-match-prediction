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
7. **⏸ E4 · Frontend multi-modelo** — diferido hasta rediseño UI (N2).
8. **⏸ E5 · Ensemble soft-voting** — diferido junto a E4.
9. **▶️ N2 · Rediseño UI** — en espera de propuesta visual del usuario. Luego G1 (dropdown `tourney_level`).
10. **M4 (SHAP), N1 (notebook), M1/M3/M5** — siguiente tanda.

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

- [ ] **G1 · Frontend no envía `tourney_level`** → la UI siempre usa el default (1). La API ya lo acepta. *Decisión: diferir* — es trabajo de UI (dropdown en `index.html` + `script.js`) y requiere verificación en navegador; no bloquea la calidad del modelo. Cerrar tras las fases P0.
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
- [ ] **I5 · Jugador desconocido cae a defaults silenciosos** (1500/999/26). Marcar `"unknown": true` en la respuesta y avisar en UI.
- [ ] **I6 · CSV con columnas faltantes → `KeyError` crudo** (`src/elo.py:163`). Validar columnas requeridas por archivo con error claro.

### Código
- [x] **I7 · Lógica ELO híbrido duplicada 3×** ✅ Resuelto (Fase 1). `elo_hibrido()` única en `src/features.py`, usada por `elo.py` y `app.py`. `LEVEL_MAP` también centralizado allí.
- [ ] **I8 · `crear_dataset_visual` reimplementa simetrización con `iterrows`** (lento, diverge de la versión vectorizada, seed global con efecto colateral). Unificar o borrar.
- [~] **I9 · Tests no cubren lo crítico.** Parcial (Fase 1): `tests/test_features.py` (orden/longitud vector) + `tests/test_app_features.py` (h2h/form reales, mismo jugador, `tourney_level` default, jugadores desconocidos). Falta: test del endpoint `/api/predict` vía test_client (superficie inválida, 400s) y determinismo de simetrización.

### Viz
- [x] **I10 · Faltan 3 plots clave:** ✅ Resuelto. `graficar_reliability_diagram` e `graficar_histograma_probas` añadidas a `src/evaluate.py`; `graficar_learning_curve` ya existía. Las 3 se invocan desde `main.py`. 8 tests en `tests/test_evaluate.py`.
- [ ] **N1 · Notebook didáctico** (`notebooks/atp_resumen.ipynb`) que resuma los puntos importantes del proyecto **sin tocar la parte web**: matemática del ELO híbrido (logística + actualización), simetrización del dataset (anti-leakage), las 6 features, CV temporal con embargo, y la lectura honesta de métricas (AUC/log-loss/Brier vs accuracy + learning curve). Objetivo portafolio/aprendizaje: narrativa + celdas ejecutables reusando `src/`.
- [ ] **N2 · Actualizar los visuales de la web** (`templates/index.html`, `static/style.css`, `static/script.js`): refrescar el diseño/estética de la SPA y mostrar las nuevas señales ya disponibles en la API (`diff_h2h`, `diff_form`, `tourney_level_num` en `features_debug`) que hoy el frontend no pinta.

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

- [x] **E1 ·** ✅ `entrenar_todos_los_modelos(X, y, dates)` en `src/train.py`: LogReg, RF, GBM, XGBoost — cada uno con GridSearchCV(neg_log_loss) + CV temporal purgado + `calibrar_modelo`. 4 tests. `xgboost==3.2.0` en requirements.txt.
- [x] **E2 ·** ✅ `main.py` exporta `modelos_atp.pkl` (`{nombre: modelo_calibrado}`) + `metrics_atp.pkl` (`{nombre: {accuracy, log_loss, brier, auc}}`). `modelo_atp.pkl` (GBM calibrado) se mantiene para compatibilidad con `app.py`.
- [x] **E3 ·** ✅ API multi-modelo: `GET /api/models` (lista ordenada por log-loss), `?model=` en `/api/predict` (valida nombre, 400 si inválido), `GET /api/predict_all` (probas de los 4 modelos para el mismo partido). Helper `_predecir_con()` evita duplicar lógica de features. 13 tests nuevos en `tests/test_api_endpoints.py`. 77 tests total.
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
