# Rediseño Frontend — ATP Match Prediction

Fecha: 2026-06-26
Alcance: solo frontend (`templates/index.html`, `static/style.css`, `static/script.js`).
El backend (`app.py`) NO cambia. Se aprovechan endpoints ya existentes.

## Objetivos (roadmap N2 + G1 + E4)

1. **G1** — Selector de nivel de torneo en el formulario (la API ya acepta `tourney_level`; el frontend no lo enviaba).
2. **N2** — Mostrar `features_debug` como breakdown visual ("por qué gana quién").
3. **E4** — Panel "comparar modelos" con las 4 probabilidades (`/api/predict_all`) + tabla de métricas (`/api/models`).
4. **E4** — Marcar visualmente jugadores desconocidos (`unknown: true`).
5. **N2** — Diseño más cuidado: jerarquía tipográfica, identidad por superficie.

## Bug encontrado (se corrige de paso)

`static/script.js:269` lee `res.features_debug.diff_elo`, campo que **ya no existe**: la API lo renombró a `diff_elo_general` y `diff_elo_sup` (commit de las 8 features). Hoy el factor "Diferencia ELO" muestra `undefined`. El rediseño reescribe esta sección.

## Decisiones de diseño

- **Stack: vanilla JS.** Sin build step, sin React/CDN. App pequeña (3 endpoints, 1 vista). Se refactoriza `script.js` separando funciones puras a `static/format.js`.
- **Breakdown: barras divergentes con signo.** Cada factor es una barra horizontal centrada en 0 que crece hacia A (izquierda) o hacia B (derecha) según el signo de la diferencia.
- **Comparar modelos: sección colapsable** (`<details>`) bajo el resultado principal. No satura la vista.
- **Modelo principal: gbm fijo.** El formulario predice siempre con gbm (mejor log-loss). La comparación multi-modelo vive en el panel colapsable. Sin dropdown de modelo en el formulario.
- **Tests: `node --test`** (runner nativo, cero dependencias). Node v26 disponible.

## Honestidad de la visualización

Las barras divergentes muestran **diferencias de feature crudas**, NO contribuciones SHAP ponderadas por el modelo. Una diferencia grande en una feature no implica que el modelo la pondere mucho. La etiqueta de la sección es **"Diferencias por factor — a quién favorece cada uno"**, evitando palabras como "contribución" o "peso" que implicarían atribución del modelo.

## Arquitectura del frontend

### Archivos

- `templates/index.html` — estructura. Añade: selector nivel torneo, badges unknown, sección breakdown, `<details>` comparar modelos. Carga `format.js` antes de `script.js`.
- `static/format.js` — **funciones puras** (sin DOM). Patrón dual browser+node: expone en `window` y vía `module.exports` si existe.
- `static/script.js` — estado, fetch, render (usa `format.js`).
- `static/style.css` — refinado; nuevos componentes.
- `tests/format.test.mjs` — tests de `format.js` con `node --test`.

### Funciones puras (`static/format.js`)

| Función | Entrada | Salida | Notas |
|---|---|---|---|
| `normalizeFactor(value, feature)` | número, nombre de feature | `[-1, 1]` | Clampa según escala típica por feature. Para el ancho/dirección de la barra. |
| `formatDiff(value, unit)` | número, unidad | `"+52.0 pts"` | Signo explícito; decimales según unidad. |
| `formatRank(rank)` | int o `"Sin Ranking"` | string | Maneja ambos tipos que devuelve la API. |
| `mergeModels(predictAll, modelsMetrics)` | dos respuestas API | array de filas | Fusiona probas + métricas por nombre de modelo, para la tabla. |

**Escalas de `normalizeFactor`** (denominador para clamp a `[-1,1]`):
- `diff_elo_general`, `diff_elo_sup`: ±300
- `diff_rank`: ±250 (= RANK_CAP)
- `diff_age`: ±10
- `diff_h2h`: ±1
- `diff_form`: ±1

`is_unranked` y `tourney_level_num` NO se grafican (no son comparativos jugador-vs-jugador en el mismo sentido). `is_unranked` alimenta el badge unknown; `tourney_level_num` viene del selector.

### Nivel de torneo (G1)

Selector tipo pill (mismo patrón visual que superficie). Mapeo a keys de `LEVEL_MAP`:

| Etiqueta UI | Valor enviado | LEVEL_MAP |
|---|---|---|
| ATP 250 | `250` | 1 |
| ATP 500 | `500` | 2 |
| Masters 1000 | `M` | 4 |
| Grand Slam | `G` | 5 |

Default: `250`. Se añade `&tourney_level=<valor>` a las llamadas `/api/predict` y `/api/predict_all`.

### Jugadores desconocidos (E4)

Si `player.unknown === true`: badge "⚠ Sin datos — ELO base 1500" en la quick-card y en la tarjeta de resultados, con aviso de baja fiabilidad. Estilo de advertencia (color ámbar).

### Comparar modelos (E4)

`<details>` colapsable bajo los resultados. Al expandir (o al renderizar resultados), una llamada a `/api/predict_all` + una a `/api/models`. `mergeModels` fusiona por nombre. Tabla: fila por modelo con proba A / proba B / ganador + AUC / log-loss / Brier / accuracy. Se resalta la fila `gbm` (modelo principal). Modelos sin métricas (p. ej. si solo existe el principal) muestran "—".

## Flujo de datos

```
DOMContentLoaded
  → fetchPlayers()                    GET /api/players
usuario elige superficie + nivel + 2 jugadores
  → validateInputs() habilita botón
click Calcular
  → GET /api/predict?...&tourney_level=...   (modelo gbm)
  → renderResults(res)
      barra proba, ganador, breakdown (format.js), comparativa, badges unknown
  → GET /api/predict_all + GET /api/models   (para el panel colapsable)
  → renderModelComparison(mergeModels(...))
```

## Manejo de errores

- Fetch falla → mensaje inline en el panel (no `alert()` como hoy).
- `predict_all`/`models` fallan → el panel colapsable muestra "No disponible"; el resultado principal sigue visible.
- Jugador sin datos → no es error: badge unknown + predicción con ELO base.

## Testing

- **Unit (`node --test`):** `normalizeFactor` (signo, clamp, por feature), `formatDiff` (signo, decimales), `formatRank` (int vs string), `mergeModels` (fusión, modelo sin métricas).
- **Manual navegador** (`python app.py` → http://localhost:8000): predicción real, cambio superficie/nivel, breakdown correcto, panel modelos, jugador desconocido, responsive.

## Fuera de alcance (YAGNI)

- No dropdown de modelo en el formulario.
- No persistencia de preferencias.
- No gráficas SHAP reales (requeriría exponer importancias por el backend).
- No cambios en `app.py`.

## Cierre

Actualizar `docs/ROADMAP.md`: marcar N2, G1, E4 como hechos.
