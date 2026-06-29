# Diseño: Datos para Simulación de Torneos en Curso

**Fecha:** 2026-06-29  
**Estado:** Aprobado  
**Contexto:** La sección "Simular Torneo" del frontend y los endpoints `/api/tournament/info` y `/api/tournament/simulate` ya están implementados, pero fallan con 404 porque esperan un archivo local `data/ongoing_tourneys.csv` que nunca se descarga. El frontend tiene el select hardcodeado a "Australian Open 2026".

---

## Objetivo

Conectar la infraestructura existente con TML-Database para que la sección "Simular Torneo" funcione con datos reales y actualizados del torneo en curso de mayor categoría.

---

## Arquitectura

Tres capas de cambios sobre lo existente:

```
TML-Database (HTTPS)
    → src/draw.py: descargar_ongoing() + listar_torneos()
    → app.py: cache en memoria + GET /api/tournaments + parches en endpoints existentes
    → static/script.js: poblar <select> dinámicamente al abrir el modal
```

---

## Componentes

### 1. `src/draw.py` (nuevo)

Módulo de lógica pura, sin efectos secundarios sobre disco.

**`descargar_ongoing(url=TML_ONGOING_URL) -> pd.DataFrame`**
- Descarga `https://stats.tennismylife.org/data/ongoing_tourneys.csv` con `requests.get`
- Timeout 10s, raises `RuntimeError` si falla
- Devuelve DataFrame con las ~35 columnas del esquema TML

**`listar_torneos(df) -> list[dict]`**
- Agrupa por `tourney_id`, deduplica
- Ordena por prioridad de `tourney_level`: `'G' > 'A' > '500' > '250'` (Grand Slam > Masters 1000 > ATP 500 > ATP 250). Davis Cup (`'D'`) se excluye.
- Devuelve lista de dicts: `[{name, surface, level, draw_size, tourney_id}]`
- Si `df` está vacío, devuelve `[]`

Constante: `TML_ONGOING_URL = "https://stats.tennismylife.org/data/ongoing_tourneys.csv"`

### 2. `app.py` — cambios

**Cache en memoria** (junto a las otras variables globales de solo lectura):
```python
_ongoing_cache = {'df': None, 'fetched_at': None}
ONGOING_CACHE_TTL = 3600  # segundos
```

**Función privada `_get_ongoing_df()`**:
- Si cache válido (fetched_at no None y `time.time() - fetched_at < TTL`): devuelve cache
- Si no: llama `draw.descargar_ongoing()`, actualiza cache, devuelve df
- Propaga `RuntimeError` si la descarga falla

**Nuevo endpoint `GET /api/tournaments`**:
- Llama `_get_ongoing_df()`, luego `draw.listar_torneos(df)`
- Devuelve `{"tournaments": [{name, surface, level, draw_size}]}`
- Error 503 si TML no responde

**Modificar `/api/tournament/info`**:
- Reemplazar lectura de archivo local por `_get_ongoing_df()`
- El resto de la lógica sin cambios

**Modificar `/api/tournament/simulate`**:
- Reemplazar lectura de archivo local por `_get_ongoing_df()`
- El resto de la lógica sin cambios

### 3. `static/script.js` — un cambio

En `setupTournamentModal()`, al abrir el modal (listener del botón `open-tournament-btn`), antes de llamar `loadTournamentInfo()`:
- Fetch `GET /api/tournaments`
- Limpiar y repoblar el `<select id="tournament-select">` con las opciones devueltas
- Primera opción = torneo de mayor prioridad (auto-seleccionado, ya lo ordena el backend)
- Si el fetch falla: dejar el select como está (fallback al hardcodeado)
- Una vez poblado: llamar `loadTournamentInfo()` con el torneo seleccionado

### 4. `scripts/fetch_simulate.py` — revisado (ya existe, corregir bug)

El script actual tiene un bug: desempaqueta 4 valores de `calcular_elos_historicos` que devuelve 3. Reescribir para:
- Usar `draw.descargar_ongoing()` en lugar de leer archivo local
- Auto-seleccionar torneo con `draw.listar_torneos()` (prioridad GS > A > 500 > 250)
- Usar `modelos_atp.pkl` existente (sin reentrenar)
- Fix del unpack: `df_pre, elo_gen, elo_sup = calcular_elos_historicos(...)`
- Imprimir tabla Markdown en consola

---

## Flujo de datos completo

```
Usuario abre modal "Simular Torneo"
    → JS fetch GET /api/tournaments
    → _get_ongoing_df() → TML HTTPS (o cache)
    → listar_torneos() → lista ordenada
    → select poblado, primer torneo seleccionado
    → JS llama loadTournamentInfo()
    → GET /api/tournament/info?tournament=<nombre>
    → _get_ongoing_df() (cache hit)
    → df filtrado por tourney_name, primera ronda, matchups
    → frontend renderiza tab "Cuadro Inicial"

Usuario pulsa "Iniciar Simulación"
    → GET /api/tournament/simulate?tournament=<nombre>&simulations=5000
    → _get_ongoing_df() (cache hit)
    → reconstruct initial_draw
    → simular_torneo_montecarlo() con pkl existente
    → JSON con resultados por ronda
    → frontend renderiza tabla de probabilidades
```

---

## Manejo de errores

| Situación | Comportamiento |
|-----------|---------------|
| TML no responde (timeout) | 503 + mensaje "No se pudo conectar con la fuente de datos" |
| No hay torneos en curso | `{"tournaments": []}` + frontend muestra "No hay torneos en curso" |
| Jugador no en `stats_jugadores.pkl` | ELO=1500, rank=999, `is_unranked=True` (comportamiento existente) |
| `tourney_level` desconocido | Prioridad mínima (al final de la lista) |

---

## Tests

**`tests/test_draw.py`** (nuevo):
- `test_listar_torneos_prioridad`: DataFrame mock con G + M + 250 → orden correcto
- `test_listar_torneos_vacio`: df vacío → lista vacía
- `test_descargar_ongoing_timeout`: mock de requests con timeout → RuntimeError

**`tests/test_api_endpoints.py`** (extender existente):
- `test_api_tournaments`: mock de `descargar_ongoing` → endpoint devuelve lista válida
- `test_api_tournament_info_live`: mock de `_get_ongoing_df` con df fixture → 200 + campos esperados
- `test_api_tournament_simulate_live`: mock de `_get_ongoing_df` → simulación completa → 200

**`scripts/fetch_simulate.py`**: no tiene test unitario (es script de CLI); se verifica manualmente.

---

## Fuera de scope

- Actualización automática periódica del cache (cron, threading): innecesario, el TTL de 1h es suficiente
- Descarga de `2026.csv` para reentrenar el modelo: fase posterior independiente
- Soporte Challenger: `challenger_ongoing_tourneys.csv` — fase posterior
- Persistencia del cache a disco: el servidor se reinicia con cada deploy en HF Spaces de todas formas
