# ROADMAP — ATP Match Prediction

> Backlog derivado de la revisión técnica 2026-06-24. **Backlog cerrado** (2026-06-26).
> **Poda de minimalismo** (2026-06-26 + re-poda 2026-06-29): ver sección abajo. 148 tests (pytest) + 4 (node).
> **Épica deploy + visual (2026-06-27): RESUELTA.** Plan archivado en `archive/docs/superpowers/plans/`.
> **Desplegado en HuggingFace Spaces (2026-06-28):** https://luissantra-atp-prediction.hf.space — deploy vía `scripts/deploy-hf.sh` (migración LFS efímera de `.pkl`) + auto-sync GitHub→HF (`.github/workflows/sync-to-hf.yml`).
> **Visual polish (2026-06-29):** trophy SVG, winner scoreboard, block-head accent borders, cmp-winner highlight.
> **Épica simulador de torneos (2026-06-30): RESUELTA.** `src/draw.py` + `src/simulator.py` (Monte Carlo) + endpoints `/api/tournaments|info|simulate` + bracket view en UI. Ver R3.
> **Bracket "camino al título" (2026-06-30):** rediseño del bracket como campo de probabilidad (rieles por slot, favorito marcado, campeón en oro con token `--gold`). Assets v17. Ver R2.
> **R5 + R6 RESUELTOS (2026-06):** R6 visualizaciones de rendimiento (4 plots Plotly desde `plots_data`) y R5 actualización de datos (Opción C: validación 2020–24/test 2025 + refit de producción 2020–25). Ver secciones abajo.
> **Estado del backlog (2026-06-30):** casi todo cerrado. Solo quedan ítems residuales de R2 (favicon/og:image, afinar texturas, decisión flujo superficie→predicción).

## Próximo — Épica refinamiento post-deploy

App en producción con simulador de torneos, visualizaciones de rendimiento y refit de producción funcionando. Lo que queda es pulido fino y una decisión de UX.

### R2 — Detalles visuales (pendiente residual)
- Afinar intensidad de texturas si en producción se ven flojas (clay/grass sutiles). Subjetivo; revisar en el Space desplegado.
- **Reconsiderar el flujo superficie → predicción** (decisión, no código): la superficie afecta al modelo (`diff_elo_sup`), pero el selector antes de ver resultados puede confundir. Evaluar si mostrar primero predicción general y que la superficie sea filtro secundario, o mantener y explicar mejor en la UI.
- ✅ **Favicon + meta Open Graph** — favicon SVG inline (pelota de tenis con costura, sin asset externo) + `theme-color` + tags OG/Twitter en `templates/index.html` (assets v19).
- ✅ **Precisión de los conectores del bracket** — codos simétricos exactos (impar baja / par sube media banda real, medida con `getBoundingClientRect`); conteo de partidos por ronda con `ceil` (fin del "7.5 Partidos"). Alineación exacta en draws completos potencia-de-2 (32/64/128); en draws parciales/impares (p.ej. Wimbledon en curso, 60 jugadores) queda un desfase residual ~12px en la ronda de conteo impar. Ver **Limitación conocida** abajo.
- ✅ **Estado de carga al despertar HF Space** — `#slow-load-warning` + timer en `static/script.js` avisa del primer request lento.
- ✅ **Responsive en tablets (761–1024 px)** — media query dedicada en `static/style.css`.
- ✅ **Footer** — `.app-footer` con créditos del modelo + TML-Database.
- ✅ **Bracket "camino al título"** — rediseño del bracket de torneo como campo de probabilidad: riel por slot (ancho = prob. de supervivencia Monte Carlo), favorito de cada cruce marcado con tick accent, campeón como marcador en oro (token `--gold`, armoniza en las 3 superficies a diferencia del `#f59e0b` hardcodeado anterior). La ronda 1 queda como draw factual; el campo se enciende tras simular (2026-06-30, assets v17).

### Limitación conocida — Bracket en draws no potencia-de-2 / parciales
El render del bracket y la simulación (`updateBracketWithSimulation`, `step = 2^k` sobre `initialDrawOrder`) asumen un cuadro **potencia-de-2**. Con draws parciales o impares (torneo en curso con byes, p.ej. 60 jugadores) aparecen: nombres de ronda nominales que no cuadran con el conteo real (columnas "1 Partido" de más) y un desfase de ~12px en la unión de la ronda con conteo impar. Solución correcta (pendiente, **no trivial**): renderizar sobre el tamaño padded a potencia-de-2 con byes explícitos, lo que **acopla con el endpoint de simulación** (que ya hace ese padding server-side) y el contrato de datos de `/api/tournament/info`. Mientras tanto los conectores son exactos para los cuadros completos (32/64/128).

### Idea P2 — Módulo "edge vs mercado" (apuestas, educativo)
El modelo ya da probabilidad **calibrada** (reliability diagram lo confirma), insumo natural para *value betting*: comparar `prob_victory` contra la prob. implícita de la cuota (`1/cuota` menos margen) y marcar edge positivo. **Veredicto de viabilidad:** viable como ejercicio de honestidad estadística (backtest con cuotas históricas para ver si el edge es positivo o ~cero tras margen), **no** como herramienta para ganar dinero — AUC 0.71 es un modelo interpretable modesto frente a los del mercado, y el feed de cuotas (histórico + live, cobertura ATP) es el cuello de botella real. Enfocarlo como módulo de backtest, nunca como recomendador. Requiere fuente de odds.

## Resuelto — R6 Visualizaciones de rendimiento en la UI (2026-06)

Subpanel colapsable "Rendimiento del modelo" en el frontend con 4 gráficos **Plotly interactivos** (no PNGs estáticos):
- **Matriz de confusión** — FP/FN reales sobre test 2025.
- **Reliability diagram** — calibración (¿el 70% predicho ocurre ~70% de las veces?).
- **Curva ROC** — con AUC.
- **Histograma de probabilidades** — distribución de confianza por clase.

`main.py` exporta `metrics['plots_data']` (matrices/series ya muestreadas) en `metrics_atp.pkl`; `/api/model` lo sirve y `renderPlots()` en `static/script.js` los dibuja. No usa `static/plots/` ni reentrena: los datos salen del test de validación 2025.

## Resuelto — R5 Actualización de datos (Opción C: validación + refit de producción)

Split en dos fases en `main.py`, resuelve la tensión "vigencia vs test ciego honesto":
- **Modelo de validación** — train 2020–2024, **test ciego 2025** (n=2861) intacto. De aquí salen TODAS las métricas reportadas y los `plots_data`.
- **Modelo de producción (refit)** — reentrenado sobre 2020–2025 completo (`PROD_TRAIN_END_YEAR=2026`); es el que se exporta a `modelos_atp.pkl` y sirve la app. Más vigente sin contaminar la evaluación.
- Eval secundaria 2026 (n≈137) sobre el modelo de producción: solo referencial.
- A fin de temporada 2026, 2026 será un test válido por tamaño sin tocar el split.

## Resuelto — Épica simulador de torneos (2026-06-28–30)

- **`src/draw.py`** — descarga `ongoing_tourneys.csv` de TML, lista torneos por nivel (G > A > 500 > 250), excluye Davis Cup.
- **`src/simulator.py`** — motor Monte Carlo: bracket potencia-de-2, caché de matchups, 10 000 iteraciones, devuelve `DataFrame` de % por ronda por jugador.
- **`app.py`** endpoints:
  - `GET /api/tournaments` — lista torneos en curso (caché en memoria).
  - `GET /api/tournament/info` — draw con metadatos de jugadores (rank, ELO).
  - `GET /api/tournament/simulate` — simulación MC; padding a potencia de 2 con `None`.
- **Frontend** — modal con selector de torneos poblado dinámicamente, vista lista (% por ronda) y vista bracket "camino al título" (campo de probabilidad: rieles por slot, favorito marcado, campeón en oro).
- **Tests** — `tests/test_draw.py`, `tests/test_backtest_simulator.py`. 148/148 verdes.
- `scripts/fetch_simulate.py` y `scripts/simulate_tournament.py` para uso CLI.

## Resuelto — UX y residuos (2026-06-30)

- **R7 — Navegación por teclado en el selector de jugador.** ↑↓ navegan el dropdown, Enter confirma, Escape cierra; `role="combobox"` + `aria-activedescendant`. Implementado en `static/script.js`.
- **D-RES3 — Notebook `atp_resumen.ipynb`.** Archivado en `archive/` (decisión: no mantener un notebook didáctico desacoplado del modelo de producción).

## Resuelto — Épica deploy + visual (2026-06-27)

- **V2** — Fix barras OR: clamp a 50% del track.
- **V1** — Texturas de fondo por superficie (hard/clay/grass).
- **V3** — Gráfica ELO multi-superficie en panel de resultados; backend expone `elo_surfaces`.
- **D1** — `Dockerfile` + port configurable (`PORT` env var). CMD exec-form: `sh -c "exec gunicorn ..."` (gunicorn recibe SIGTERM como PID 1; sin warning `JSONArgsRecommended`).
- **D2** — `README.md` con header YAML de HF Spaces.
- **R4** — Banner `#model-disclaimer` con fecha de corte servida por backend (`trained_through`/`tested_on`).
- **R3 (partial)** — Visual polish: trophy SVG, winner scoreboard como marcador, block-head accent borders, cmp-winner highlight. Assets v15.
- **Card ML vs ELO** — explica el lift al usuario.

## Estado final (post re-poda 2026-06-29, 5 features)

- LogReg calibrada: AUC=0.7093, log-loss=0.6225, Brier=0.217, accuracy=65.2%
- IC95% AUC ≈ ±0.009
- Gap CV→test Δ=+0.007 (prácticamente nulo)
- Supera baseline ELO-híbrido (AUC 0.709 vs 0.694, log-loss 0.6225 vs 0.6318, fuera del IC)
- Modelo único: LogReg iguala a GBM/RF/XGBoost

## Poda de minimalismo (2026-06-26)

- **Features 8 → 5.** Podadas `diff_h2h`, `diff_form`, `tourney_level_num`: permutation importance < 0.001, ablación dentro del IC95%.
- **Modelos 4 → 1.** LogReg calibrada único. GBM/RF/XGBoost/ensemble retirados.
- **Explicabilidad.** `coeficientes_modelo()` (odds-ratio), `/api/model`, panel "Detalle del modelo".
- **Fixes.** `is_unranked` desde máscara NaN real; baseline ELO híbrido honesto; calibración Platt.

## Re-poda + fix skew (2026-06-29)

- **Features 7 → 5.** `diff_matches_played` ruido puro; `diff_tb_ratio` significativa estadísticamente pero aporte trivial (+0.002 AUC) → podada por relevancia práctica.
- **Limpieza.** −135 líneas netas; `calcular_elos_historicos` vuelve a return de 3 valores.
- **Fix train/serve skew `is_unranked`.** Ahora se sirve el flag del pkl exportado (fallback a `rank>=999` para pkl antiguos).

## Resuelto (histórico)

### P0 — Críticos
- **C1** — Train/serve skew: `tourney_level_num=3` constante en inferencia → `src/features.py` fuente única.
- **C2** — Métrica equivocada `accuracy` en GridSearch → `neg_log_loss`.
- **C3** — Gap CV/test diagnosticado: distribution shift 2026, no overfit. CV con embargo temporal.
- **C4** — Servidor Werkzeug → `gunicorn`.

### P1 — Importantes
- **I1** — Calibración sigmoid/isotonic automática por log-loss.
- **I2** — `RANK_CAP=250`, feature `is_unranked`.
- **I3** — ELO separado general/superficie. AUC 0.693→0.707.
- **I4** — `verificar_version_sklearn`. `validar_metadata_pkl`.
- **I5** — Jugador desconocido → `"unknown": true`.
- **I6** — CSV con columnas faltantes → `ValueError` descriptivo.
- **I7** — `elo_hibrido()` centralizada en `src/features.py`.
- **I8** — `crear_dataset_visual` vectorizado.
- **I9** — Tests coherencia simetrización, determinismo, endpoints.
- **I10** — Reliability diagram, histograma probabilidades, learning curve.

### P2 — Menores
- **M1–M5** — ELO full-precision, validación `player_a == player_b`, `custom_tree.py` archivado, permutation importance, test_label_balanced exacto.

### Calidad estadística
- **Q1–Q6** — IC95% bootstrap, ELO con MOV + K-schedule, diagnóstico gap CV/test, calibración, notebook, RNG local.
