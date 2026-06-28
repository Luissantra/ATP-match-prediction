# ROADMAP — ATP Match Prediction

> Backlog derivado de la revisión técnica 2026-06-24. **Backlog cerrado** (2026-06-26).
> **Poda de minimalismo** (2026-06-26): ver sección abajo. 124 tests (pytest) + 4 (node).
> **Épica deploy + visual (2026-06-27): RESUELTA.** Plan en `docs/superpowers/plans/2026-06-27-visual-polish-then-hf-deploy.md`.
> **Desplegado en HuggingFace Spaces (2026-06-28):** https://luissantra-atp-prediction.hf.space — deploy vía `scripts/deploy-hf.sh` (migración LFS efímera de `.pkl`) + auto-sync GitHub→HF (`.github/workflows/sync-to-hf.yml`).
> **Nueva épica abierta (2026-06-28): refinamiento post-deploy.** Ver abajo.

## Próximo — Épica refinamiento post-deploy (2026-06-28)

App en producción. Pulir, ampliar funcionalidad y honestidad sobre los datos.

### R1 — Detalles técnicos
- Quitar el warning `JSONArgsRecommended` del Dockerfile (CMD en shell-form por `$PORT`): evaluar `ENTRYPOINT`/exec-form con `sh -c` o script de arranque.
- `notebooks/atp_resumen.ipynb` sigue en el modelo viejo (8 features): actualizar al de 5 o archivar.
- Mantener `requirements-serve.txt` en sync con `requirements.txt` (hoy manual; ¿test que lo verifique?).
- Revisar `metrics_atp.pkl` (4 KB) y demás artefactos: confirmar que todo lo servido es mínimo.

### R2 — Detalles visuales
- Estado de carga / errores del Space en frío (HF duerme el Space inactivo: primer request lento). Mensaje de "despertando modelo".
- Afinar intensidad de texturas si en producción se ven flojas (clay/grass sutiles).
- Pulir responsive en tablets (760–1024px), no solo móvil.
- Posible: enlace/footer a HF + GitHub, favicon, og:image para compartir.
- **Explicar por qué el modelo supera al ELO solo**: el modelo (AUC 0.709) supera al baseline ELO-híbrido (AUC 0.694) gracias a ranking, edad e `is_unranked`; el ELO puro no captura la señal de jugadores sin ranking ni la diferencia de edad. Añadir nota explicativa en "Detalle del modelo" (o tooltip) que lo articule al usuario.
- **Reconsiderar el orden superficie → predicción**: el usuario elige superficie antes de ver los resultados, pero la app ya muestra ELO en las tres superficies independientemente. Evaluar si tiene más sentido mostrar primero la predicción general y que la superficie sea un filtro secundario, o mantener el flujo actual y justificar que la superficie sí afecta al modelo (las features `diff_elo_sup` e `is_unranked` dependen de ella).

### R3 — Funcionalidad: simular torneo
- Simular un cuadro completo de un torneo actual (ATP 250/500/Masters/Grand Slam): el usuario elige torneo + superficie + lista de participantes (o seed real), y el sistema propaga probabilidades ronda a ronda hasta el campeón.
- Decisiones de diseño: ¿árbol de eliminatorias manual o draws reales? ¿probabilidad de título por Montecarlo sobre el bracket? ¿endpoint nuevo `/api/tournament`?
- Empezar simple: bracket de 8 con probabilidades por par y % de título vía simulación.

### R4 — Disclaimer de vigencia del modelo ✅ (2026-06-28)
- ✅ Banner `#model-disclaimer` en el frontend tras el header: "Modelo entrenado con datos hasta 2024 (test 2025). Las predicciones no reflejan lesiones, retiradas ni forma reciente fuera del ELO."
- ✅ Fecha de corte servida por el backend (`trained_through`/`tested_on` en `/api/model`, constantes `TRAINED_THROUGH`/`TESTED_ON` en `app.py`); `loadDisclaimer()` la lee, no se hardcodea en el HTML.

### R6 — Visualizaciones de rendimiento del modelo en la UI

`src/evaluate.py` ya genera estos plots en tiempo de entrenamiento (y `main.py` los guarda), pero no se exponen en el frontend:

- **Matriz de confusión** — muestra falsos positivos/negativos reales sobre test 2025 (n=2861). Insight directo: ¿en qué dirección falla el modelo?
- **Reliability diagram / calibration curve** — ya generado por `graficar_reliability_diagram`. Muestra si el 70% predicho ocurre ~70% de las veces. Clave para honestidad ante el usuario.
- **Histograma de probabilidades** — distribución de confianza del modelo (¿predice muchos 50-55% o llega a extremos?). Complementa el reliability diagram.
- **Scatter plot predicción vs resultado** — útil para ver si errores se concentran en partidos cercanos (prob ~50%) o si falla sistemáticamente en algún rango.

Plan de implementación sugerido:
1. `main.py` ya exporta plots a `static/plots/` (o nuevo directorio). Si no, añadir guardado PNG.
2. Flask: endpoint `GET /api/plots` devuelve lista, o servir directamente desde `static/`.
3. Frontend: subpanel "Rendimiento del modelo" (colapsable, igual que "Detalle del modelo") con las imágenes.

No requiere reentrenar — los plots se regeneran con `python main.py`. El scatter predicción-vs-resultado sí requiere guardar `y_prob` del test en `metrics_atp.pkl` si no está ya.

### R7 — UX: navegación por teclado en selector de jugador

Problema actual: el usuario escribe en el input de búsqueda pero debe hacer clic con el ratón para seleccionar de la lista desplegable. No hay navegación ↑↓ + Enter.

Cambios en `templates/index.html` / `static/script.js`:
- Tecla ↓ en input abre dropdown y mueve foco al primer ítem.
- ↑↓ navegan entre ítems de la lista (con `aria-activedescendant` actualizado).
- Enter sobre ítem seleccionado lo confirma (igual que clic).
- Escape cierra dropdown y devuelve foco al input.
- Tab fuera del combo cierra dropdown.
- `role="combobox"` + `aria-expanded` + `aria-autocomplete="list"` para accesibilidad.

Bajo coste, alto impacto en usabilidad (especialmente en desktop sin trackpad).

### R5 — Actualización de datos de entrenamiento (DECISIÓN ABIERTA)
Tensión real: añadir 2025 al train mejora la vigencia pero **sacrifica el test ciego honesto** (hoy 2025, n=2861). Opciones a evaluar:
- **Rolling window**: train hasta año N−1, test año N; reentrenar cada temporada. Mantiene evaluación honesta; el n del test varía año a año.
- **Walk-forward / backtesting**: evaluar sobre varios años out-of-sample encadenados. Más robusto, más trabajo.
- El test 2026 (n≈137) es hoy solo referencial por tamaño; a fin de temporada 2026 será un test válido sin tocar el split actual.
- **No** mover 2025 al train sin sustituir el test por uno igual de honesto. Decidir antes de tocar `main.py`.

## Resuelto — Épica deploy + visual (2026-06-27)

Orden de ejecución: `V2 → V1 → V3 → D1 → D2`.

- **V2** — Fix barras OR: clamp a 50% del track (`* 100` → `* 50`). (`overflow: hidden` descartado: recortaba el overhang del eje central; el clamp ya impide el desbordamiento.)
- **V1** — Texturas de fondo por superficie: rejilla ortogonal (hard), trama diagonal cruzada (clay), franjas de césped segado (grass).
- **V3** — Gráfica ELO multi-superficie (Hard/Clay/Grass × 2 jugadores) en panel de resultados; backend expone `elo_surfaces` en `/api/predict`, frontend `renderEloChart`.
- **D1** — `Dockerfile` + port configurable (`PORT` env var, default 8000 local / 7860 HF).
- **D2** — `README.md` con header YAML de HF Spaces (`sdk: docker`).

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
