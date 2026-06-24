# Design: ML Features Expansion + TimeSeriesSplit
**Date:** 2026-06-24  
**Scope:** Pasos 1-2 del plan de mejoras — nuevas features y validación cruzada temporal correcta

---

## Objetivo

Pasar de 3 features (`diff_elo`, `diff_rank`, `diff_age`) a 6 features añadiendo información de contexto real del partido (H2H, forma reciente, nivel del torneo), y corregir la validación cruzada para respetar la estructura temporal de los datos.

Accuracy esperado: de ~65% a >68-70%.

---

## Features Actuales vs. Nuevas

| Feature | Estado | Descripción |
|---------|--------|-------------|
| `diff_elo` | Existente | Diferencia ELO híbrido (50% general + 50% superficie) |
| `diff_rank` | Existente | Diferencia de ranking ATP |
| `diff_age` | Existente | Diferencia de edad |
| `diff_h2h` | **Nueva** | Diferencia de ratio H2H histórico pre-partido |
| `diff_form` | **Nueva** | Diferencia de win-ratio en últimos 10 partidos |
| `tourney_level_num` | **Nueva** | Nivel del torneo como ordinal (G=5, M=4, F/O=3, 500/A=2, 250/D=1) |

---

## Arquitectura de Cómputo

### Sin leakage — cálculo pre-partido

H2H y forma se calculan **durante el mismo loop cronológico** en `src/elo.py::calcular_elos_historicos`. Para cada partido:

1. Se consulta el estado **previo** al match (H2H actual, forma actual)
2. Se guarda ese estado como columnas en `df_completo`
3. Se actualiza el estado **tras** el match

Esto garantiza que el modelo solo ve información disponible en el momento de la predicción.

### Estructuras de datos nuevas en `src/elo.py`

```python
# H2H: victorias del jugador A sobre B (clave = frozenset o tuple ordenada)
h2h: dict[tuple, dict]  # h2h[(p1, p2)] = {'p1_wins': int, 'total': int}

# Forma reciente: últimos N resultados binarios (1=victoria, 0=derrota)
from collections import deque
form: dict[str, deque]  # form[player_name] = deque([1,0,1,...], maxlen=10)
```

### Columnas nuevas en `df_completo`

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `h2h_winner_ratio` | float | Ratio de victorias del ganador vs el perdedor (antes del partido) |
| `h2h_loser_ratio` | float | Ratio de victorias del perdedor vs el ganador (= 1 - h2h_winner_ratio si hay historial, 0.5 si no) |
| `form_winner` | float | Win-ratio del ganador en sus últimos 10 partidos |
| `form_loser` | float | Win-ratio del perdedor en sus últimos 10 partidos |

### Encoding de `tourney_level` en `src/data_processing.py`

```python
LEVEL_MAP = {
    'G': 5,    # Grand Slam
    'M': 4,    # Masters 1000
    'F': 3,    # Tour Finals / season-ending
    'O': 3,    # Olympics
    '500': 2, 'A': 2,   # ATP 500
    '250': 1, 'D': 1,   # ATP 250 / Davis Cup
}
```

`tourney_level_num` es **absoluta** (no diferencia), porque ambos jugadores juegan el mismo torneo. Indica qué tipo de partido importa.

---

## Cambios por Archivo

### `src/elo.py`

Función `calcular_elos_historicos`:
- Añadir `h2h = {}` y `form = {}` antes del loop
- En cada iteración, extraer ratios previos → append a listas → actualizar dicts post-match
- Añadir 4 columnas a `df_completo`: `h2h_winner_ratio`, `h2h_loser_ratio`, `form_winner`, `form_loser`
- Signature de retorno sin cambios (`df_completo, elo_general, elo_superficie`)

### `src/data_processing.py`

Función `preparar_datos_entrenamiento`:
- Leer `h2h_winner_ratio`, `h2h_loser_ratio`, `form_winner`, `form_loser` del DataFrame
- Calcular `diff_h2h` y `diff_form` con la simetrización existente (shuffle_mask)
- Añadir `tourney_level_num` via `LEVEL_MAP` (misma para A y B)
- `features.append(...)` incluye las 3 nuevas

### `main.py`

- Importar `TimeSeriesSplit` de `sklearn.model_selection`
- Cambiar `cv=3` → `TimeSeriesSplit(n_splits=5)` en `GridSearchCV`
- Actualizar `FEATURES` list y `X_train`/`X_test` slicing
- Actualizar gráfico de feature importance con los 6 nombres

---

## Validación Cruzada Temporal

`TimeSeriesSplit(n_splits=5)` sobre `X_train` (2020-2025):
- Fold 1: train en meses 1-10, val en 11-12
- Fold 2: train en meses 1-20, val en 21-22
- ... etc.

Los datos ya están en orden cronológico (`sort_values('tourney_date')`), así que TimeSeriesSplit opera correctamente sobre el índice.

---

## Criterio de Éxito

- Accuracy en test ciego 2026 ≥ el anterior (~65%)
- Feature importance de `diff_h2h` y `diff_form` > 0% (contribuyen algo)
- `tourney_level_num` con algún peso (aunque sea pequeño)
- Sin errores de KeyError en nuevas columnas
