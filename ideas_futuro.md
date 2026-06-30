# Ideas a Futuro — Predicción ATP

> **Limpiado 2026-06-30** contra el estado real del proyecto: LogReg calibrada única, 5 features
> (`diff_elo_general`, `diff_elo_sup`, `diff_rank`, `is_unranked`, `diff_age`), AUC 0.709 en test
> ciego 2025 (n=2861). Criterio del proyecto: una feature entra **solo si aporta relevancia
> práctica**, no basta la significancia estadística. Las ideas del plan inicial ya evaluadas están
> en "Evaluadas y cerradas" para no re-litigarlas.

---

## 1. Implementable con los datos actuales

### ELO de servicio / resto (descomposición del ELO)
Los CSV ya traen stats por partido: `w_svpt`, `w_1stIn`, `w_1stWon`, `w_2ndWon`, `w_SvGms`,
`w_bpSaved`, `w_bpFaced` (y los `l_*`). Permite mantener, igual que el ELO general, **dos ratings
pre-partido por jugador** — habilidad al saque y al resto — actualizados cronológicamente desde el
% de puntos de saque ganados / puntos de resto ganados de cada partido (sin leakage, mismo patrón
que `src/elo.py`). La expectativa cruzaría ELO-saque de A vs ELO-resto de B.

- **Por qué tiene sentido:** es señal **nueva**, no complejidad de modelo. (El proyecto ya demostró
  que GBM/RF/XGBoost no superan a LogReg; esto es distinto: información que el modelo aún no ve.)
- **Caveat honesto:** desconocido si supera 0.709. Riesgo real: las stats de saque/resto
  correlacionan fuerte con el ELO general → puede no añadir lift incremental. Entra solo si lo aporta.
- **Coste:** medio. Nuevo cómputo en `src/elo.py` + 1–2 features en `src/features.py` + reentreno.
  TDD por fase. **Candidato nº1 a implementar.**

---

## 2. Idea a futuro (requiere datos externos)

### Módulo "edge vs mercado" / ROI (apuestas, educativo)
Backtest comparando la probabilidad **calibrada** del modelo vs cuotas históricas reales
(Pinnacle/Bet365), marcando edge positivo (`prob_victory` > prob. implícita tras margen) y
dimensionando con criterio de **Kelly**. Cruz-ref: idea P2 del `docs/ROADMAP.md`.
- **Veredicto de viabilidad:** viable como ejercicio de honestidad estadística (ver si el edge es
  positivo o ~cero tras margen), **no** como herramienta para ganar dinero — AUC 0.709 es un modelo
  interpretable modesto frente a los del mercado.
- **Cuello de botella:** feed de cuotas histórico + live con cobertura ATP; casi ninguno gratis. No
  implementar sin esa fuente. Enfocarlo como módulo de backtest, nunca como recomendador.

---

## 3. Evaluadas y cerradas (no reabrir sin datos/argumento nuevo)

- **Partidos jugados, rendimiento en finales, eficacia en tie-break (`diff_tb_ratio`), h2h, forma,
  nivel de torneo** → **podadas**: permutation importance ~0 o aporte trivial (`diff_tb_ratio`:
  significativa por bootstrap pero +0.002 AUC). Significancia ≠ relevancia práctica.
- **LSTM / redes recurrentes** sobre la secuencia de partidos → **descartado**: el proyecto concluyó
  que la complejidad no añade señal (LogReg iguala a GBM/RF/XGBoost). Un LSTM es justo más
  complejidad sin evidencia de lift.
- **Calibración Platt / isotónica** → **hecho** (selección automática por log-loss).
- **Log-loss como métrica** → **hecho** (scoring del GridSearch + reportada con IC95%).
- **Fatiga (días de descanso, jet-lag)** → **inviable con datos actuales**: `tourney_date` es por
  torneo, no por partido (no se puede calcular descanso intra-torneo); el jet-lag necesita
  coordenadas de sede, que no están en los CSV. (`minutes` sí existe; fatiga por minutos acumulados
  en el torneo sería lo único parcialmente calculable, pero de valor previsiblemente dudoso.)
- **ELO contra estilos (zurdo/diestro, sacador/fondo)** → `winner_hand`/`winner_ht` existen;
  zurdo/diestro es factible pero de señal previsiblemente marginal; el "estilo de juego" requiere
  una clasificación no disponible. Baja prioridad.
