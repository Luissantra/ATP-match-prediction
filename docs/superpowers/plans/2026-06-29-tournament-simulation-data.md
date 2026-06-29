# Tournament Simulation Data — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Conectar la sección "Simular Torneo" del frontend con TML-Database para que muestre datos reales del torneo ATP en curso de mayor categoría, sin reentrenar el modelo.

**Architecture:** Se crea `src/draw.py` con lógica pura de descarga y priorización. `app.py` añade un cache en memoria (TTL 1h), un nuevo endpoint `/api/tournaments` y se parchean los dos endpoints existentes para usar el cache en vez de leer un archivo local. El frontend puebla el `<select>` dinámicamente al abrir el modal.

**Tech Stack:** Python 3 + requests + pandas + Flask (existente). Vanilla JS (existente).

## Global Constraints

- Python: compatible con versiones existentes del proyecto (ver `requirements.txt`)
- `requests` debe añadirse a `requirements.txt` con versión pineada `requests==2.32.4`
- Los endpoints existentes `/api/tournament/info` y `/api/tournament/simulate` mantienen su contrato de respuesta (sin cambiar campos del JSON)
- `tourney_level` en TML: `'G'` = Grand Slam, `'A'` = Masters 1000, `'500'` = ATP 500, `'250'` = ATP 250, `'D'` = Davis Cup (excluir)
- Sin escritura a disco en la ruta de producción — el cache vive solo en memoria
- TDD estricto: test primero, luego implementación
- Commits frecuentes, uno por tarea mínimo
- No `Co-Authored-By` en commits (instrucción global del proyecto)

---

## File Map

| Archivo | Acción | Responsabilidad |
|---------|--------|----------------|
| `src/draw.py` | Crear | Descarga y priorización de torneos desde TML |
| `tests/test_draw.py` | Crear | Tests unitarios de `src/draw.py` |
| `requirements.txt` | Modificar | Añadir `requests==2.32.4` |
| `app.py` | Modificar | Cache + `/api/tournaments` + parches en info/simulate |
| `tests/test_api_endpoints.py` | Modificar | Tests para nuevo endpoint y comportamiento mockeado |
| `static/script.js` | Modificar | Poblar select dinámicamente al abrir modal |
| `scripts/fetch_simulate.py` | Modificar | Fix bug 4-valores + usar draw module + usar pkl existente |

---

## Task 1: `src/draw.py` — descarga y priorización

**Files:**
- Create: `src/draw.py`
- Create: `tests/test_draw.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces:
  - `TML_ONGOING_URL: str` — constante con la URL del CSV
  - `descargar_ongoing(url=TML_ONGOING_URL) -> pd.DataFrame` — descarga y parsea el CSV; lanza `RuntimeError` si falla
  - `listar_torneos(df: pd.DataFrame) -> list[dict]` — devuelve lista de dicts ordenada por prioridad; cada dict: `{name: str, surface: str, level: str, draw_size: int, tourney_id: str}`

- [ ] **Step 1: Añadir `requests` a requirements.txt**

Editar `requirements.txt` añadiendo al final:
```
requests==2.32.4
```

Instalar en el entorno:
```bash
source venv/bin/activate && pip install requests==2.32.4
```

- [ ] **Step 2: Escribir tests que fallan — `tests/test_draw.py`**

Crear `tests/test_draw.py` con este contenido:

```python
import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import draw


def _df_multi():
    """DataFrame con torneos de distintos niveles."""
    return pd.DataFrame([
        {'tourney_id': '2026-580', 'tourney_name': 'Wimbledon',   'surface': 'Grass', 'tourney_level': 'G',   'winner_name': 'A', 'loser_name': 'B'},
        {'tourney_id': '2026-580', 'tourney_name': 'Wimbledon',   'surface': 'Grass', 'tourney_level': 'G',   'winner_name': 'C', 'loser_name': 'D'},
        {'tourney_id': '2026-422', 'tourney_name': 'Halle',       'surface': 'Grass', 'tourney_level': '250', 'winner_name': 'E', 'loser_name': 'F'},
        {'tourney_id': '2026-500', 'tourney_name': 'Queen Club',  'surface': 'Grass', 'tourney_level': '500', 'winner_name': 'G', 'loser_name': 'H'},
        {'tourney_id': '2026-999', 'tourney_name': 'Davis Cup',   'surface': 'Hard',  'tourney_level': 'D',   'winner_name': 'I', 'loser_name': 'J'},
    ])


def test_listar_torneos_orden_prioridad():
    """GS > 500 > 250; Davis Cup excluido."""
    result = draw.listar_torneos(_df_multi())
    names = [t['name'] for t in result]
    assert names[0] == 'Wimbledon'
    assert names[1] == 'Queen Club'
    assert names[2] == 'Halle'
    assert 'Davis Cup' not in names


def test_listar_torneos_campos():
    """Cada torneo tiene los campos requeridos."""
    result = draw.listar_torneos(_df_multi())
    for t in result:
        assert set(t.keys()) >= {'name', 'surface', 'level', 'draw_size', 'tourney_id'}


def test_listar_torneos_draw_size():
    """draw_size es el número de jugadores únicos del torneo."""
    result = draw.listar_torneos(_df_multi())
    wimbledon = next(t for t in result if t['name'] == 'Wimbledon')
    assert wimbledon['draw_size'] == 4  # A, B, C, D


def test_listar_torneos_vacio():
    """DataFrame vacío devuelve lista vacía."""
    assert draw.listar_torneos(pd.DataFrame()) == []


def test_listar_torneos_solo_davis_cup():
    """Si solo hay Davis Cup, resultado es lista vacía."""
    df = pd.DataFrame([
        {'tourney_id': '2026-999', 'tourney_name': 'Davis Cup', 'surface': 'Hard',
         'tourney_level': 'D', 'winner_name': 'A', 'loser_name': 'B'},
    ])
    assert draw.listar_torneos(df) == []


def test_descargar_ongoing_timeout(monkeypatch):
    """Si requests lanza Timeout, descargar_ongoing lanza RuntimeError."""
    import requests as req_lib

    def mock_get(*args, **kwargs):
        raise req_lib.exceptions.Timeout("timeout simulado")

    monkeypatch.setattr(req_lib, 'get', mock_get)
    with pytest.raises(RuntimeError, match="No se pudo descargar"):
        draw.descargar_ongoing()


def test_descargar_ongoing_http_error(monkeypatch):
    """Si requests devuelve 500, descargar_ongoing lanza RuntimeError."""
    import requests as req_lib

    class MockResponse:
        status_code = 500
        def raise_for_status(self):
            raise req_lib.exceptions.HTTPError("500 Server Error")

    monkeypatch.setattr(req_lib, 'get', lambda *a, **kw: MockResponse())
    with pytest.raises(RuntimeError, match="No se pudo descargar"):
        draw.descargar_ongoing()
```

- [ ] **Step 3: Ejecutar tests para verificar que fallan**

```bash
source venv/bin/activate && python -m pytest tests/test_draw.py -v
```

Esperado: `ERROR` o `ImportError` porque `src/draw.py` no existe.

- [ ] **Step 4: Implementar `src/draw.py`**

Crear `src/draw.py`:

```python
from io import StringIO

import pandas as pd
import requests

TML_ONGOING_URL = "https://stats.tennismylife.org/data/ongoing_tourneys.csv"

_LEVEL_PRIORITY = {'G': 0, 'A': 1, '500': 2, '250': 3}


def descargar_ongoing(url=TML_ONGOING_URL) -> pd.DataFrame:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"No se pudo descargar ongoing_tourneys.csv: {e}")
    return pd.read_csv(StringIO(resp.text))


def listar_torneos(df: pd.DataFrame) -> list:
    if df.empty or 'tourney_level' not in df.columns:
        return []

    df = df[df['tourney_level'] != 'D'].copy()
    if df.empty:
        return []

    valid_surfaces = {'Hard', 'Clay', 'Grass'}
    torneos = []
    for tourney_id, grupo in df.groupby('tourney_id'):
        row = grupo.iloc[0]
        surface = row['surface'] if row['surface'] in valid_surfaces else 'Hard'
        level = str(row.get('tourney_level', '250'))
        jugadores = pd.concat([grupo['winner_name'], grupo['loser_name']]).nunique()
        torneos.append({
            'name': row['tourney_name'],
            'surface': surface,
            'level': level,
            'draw_size': int(jugadores),
            'tourney_id': str(tourney_id),
        })

    torneos.sort(key=lambda t: _LEVEL_PRIORITY.get(t['level'], 99))
    return torneos
```

- [ ] **Step 5: Ejecutar tests para verificar que pasan**

```bash
source venv/bin/activate && python -m pytest tests/test_draw.py -v
```

Esperado: 7 tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add src/draw.py tests/test_draw.py requirements.txt
git commit -m "feat: src/draw.py — descarga TML y priorización de torneos"
```

---

## Task 2: Cache + `/api/tournaments` en `app.py`

**Files:**
- Modify: `app.py` (añadir imports, variables de cache, función `_get_ongoing_df`, endpoint `/api/tournaments`)
- Modify: `tests/test_api_endpoints.py` (añadir tests del nuevo endpoint)

**Interfaces:**
- Consumes: `draw.descargar_ongoing()`, `draw.listar_torneos()` de Task 1
- Produces:
  - `_ongoing_cache: dict` — `{'df': None, 'fetched_at': None}`
  - `_get_ongoing_df() -> pd.DataFrame` — devuelve cache o descarga
  - `GET /api/tournaments` → `{'tournaments': [{name, surface, level, draw_size, tourney_id}]}`; 503 si TML no responde

- [ ] **Step 1: Escribir tests que fallan**

Añadir al final de `tests/test_api_endpoints.py`:

```python
# ── GET /api/tournaments ─────────────────────────────────────────────────────

import pandas as pd
from unittest.mock import patch


def _df_ongoing_mock():
    return pd.DataFrame([
        {'tourney_id': '2026-580', 'tourney_name': 'Wimbledon',
         'surface': 'Grass', 'tourney_level': 'G',
         'winner_name': 'Sinner', 'loser_name': 'Alcaraz',
         'match_num': 1, 'round': 'R64'},
        {'tourney_id': '2026-422', 'tourney_name': 'Halle',
         'surface': 'Grass', 'tourney_level': '250',
         'winner_name': 'Zverev', 'loser_name': 'Ruud',
         'match_num': 1, 'round': 'R32'},
    ])


def test_api_tournaments_devuelve_200(client):
    with patch('app._get_ongoing_df', return_value=_df_ongoing_mock()):
        r = client.get('/api/tournaments')
    assert r.status_code == 200
    data = r.get_json()
    assert 'tournaments' in data
    assert len(data['tournaments']) == 2


def test_api_tournaments_prioridad(client):
    """Wimbledon (G) debe aparecer primero."""
    with patch('app._get_ongoing_df', return_value=_df_ongoing_mock()):
        data = client.get('/api/tournaments').get_json()
    assert data['tournaments'][0]['name'] == 'Wimbledon'


def test_api_tournaments_503_si_tml_falla(client):
    with patch('app._get_ongoing_df', side_effect=RuntimeError("timeout")):
        r = client.get('/api/tournaments')
    assert r.status_code == 503
```

- [ ] **Step 2: Ejecutar tests para verificar que fallan**

```bash
source venv/bin/activate && python -m pytest tests/test_api_endpoints.py::test_api_tournaments_devuelve_200 tests/test_api_endpoints.py::test_api_tournaments_prioridad tests/test_api_endpoints.py::test_api_tournaments_503_si_tml_falla -v
```

Esperado: `FAILED` con `404` o `AttributeError` porque el endpoint no existe.

- [ ] **Step 3: Añadir cache y endpoint a `app.py`**

En `app.py`, después de las líneas de imports existentes (antes de `app = Flask(...)`), añadir:

```python
import time
from src import draw
```

Después de las variables globales existentes (`stats_jugadores = {}`), añadir:

```python
_ongoing_cache = {'df': None, 'fetched_at': None}
ONGOING_CACHE_TTL = 3600
```

Añadir la función privada antes de la primera ruta (`@app.route`):

```python
def _get_ongoing_df():
    now = time.time()
    if (_ongoing_cache['df'] is not None
            and _ongoing_cache['fetched_at'] is not None
            and now - _ongoing_cache['fetched_at'] < ONGOING_CACHE_TTL):
        return _ongoing_cache['df']
    df = draw.descargar_ongoing()
    _ongoing_cache['df'] = df
    _ongoing_cache['fetched_at'] = now
    return df
```

Añadir el endpoint `/api/tournaments` antes de `/api/tournament/info`:

```python
@app.route('/api/tournaments')
def list_tournaments():
    try:
        df = _get_ongoing_df()
        torneos = draw.listar_torneos(df)
        return jsonify({'tournaments': torneos})
    except RuntimeError as e:
        return jsonify({'detail': str(e)}), 503
```

- [ ] **Step 4: Ejecutar tests para verificar que pasan**

```bash
source venv/bin/activate && python -m pytest tests/test_api_endpoints.py::test_api_tournaments_devuelve_200 tests/test_api_endpoints.py::test_api_tournaments_prioridad tests/test_api_endpoints.py::test_api_tournaments_503_si_tml_falla -v
```

Esperado: 3 tests `PASSED`.

- [ ] **Step 5: Verificar que la suite completa sigue verde**

```bash
source venv/bin/activate && python -m pytest tests/test_api_endpoints.py -v
```

Esperado: todos `PASSED` (los tests existentes de `tournament/info` y `tournament/simulate` pueden fallar porque aún leen el archivo local — se arreglan en Task 3).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_api_endpoints.py
git commit -m "feat: GET /api/tournaments + cache en memoria desde TML"
```

---

## Task 3: Parchar `/api/tournament/info` y `/api/tournament/simulate`

**Files:**
- Modify: `app.py` (reemplazar lectura de archivo local por `_get_ongoing_df()` en ambos endpoints)
- Modify: `tests/test_api_endpoints.py` (añadir mock de `_get_ongoing_df` en tests existentes de info/simulate)

**Interfaces:**
- Consumes: `_get_ongoing_df()` de Task 2
- Produces: mismos contratos JSON que antes; ahora los datos vienen de TML en vez del archivo local

- [ ] **Step 1: Escribir tests actualizados que fallan correctamente**

Los tests `test_simulate_tournament_devuelve_200` y `test_tournament_info_devuelve_200` actuales NO mockean el acceso a disco, por lo que solo pasan si existe `data/ongoing_tourneys.csv` localmente. Añadir versiones mockeadas al final de `tests/test_api_endpoints.py`:

```python
# ── Versiones mockeadas de info y simulate ───────────────────────────────────

def _df_ao_mock():
    """8 filas de R64 del Australian Open — potencia de 2 para el simulador."""
    rows = []
    players = [('Sinner', 'Alcaraz'), ('Djokovic', 'Ruud'), ('Medvedev', 'Rublev'),
               ('Zverev', 'Fritz'), ('Paul', 'Tiafoe'), ('Hurkacz', 'Norrie'),
               ('Khachanov', 'Mmoh'), ('Shapovalov', 'Musetti')]
    for i, (w, l) in enumerate(players, 1):
        rows.append({
            'tourney_id': '2026-580', 'tourney_name': 'Australian Open',
            'surface': 'Hard', 'tourney_level': 'G',
            'match_num': i, 'round': 'R64',
            'winner_name': w, 'loser_name': l,
            'winner_rank': float(i), 'loser_rank': float(i + 8),
            'winner_age': 24.0, 'loser_age': 25.0,
        })
    return pd.DataFrame(rows)


def test_tournament_info_con_mock_devuelve_200(client):
    with patch('app._get_ongoing_df', return_value=_df_ao_mock()):
        r = client.get('/api/tournament/info?tournament=Australian Open')
    assert r.status_code == 200
    data = r.get_json()
    assert data['tournament'] == 'Australian Open'
    assert len(data['matchups']) == 8
    assert 'player_a' in data['matchups'][0]


def test_tournament_info_con_mock_torneo_invalido(client):
    with patch('app._get_ongoing_df', return_value=_df_ao_mock()):
        r = client.get('/api/tournament/info?tournament=TorneoInexistente')
    assert r.status_code == 404


def test_simulate_tournament_con_mock_devuelve_200(client):
    with patch('app._get_ongoing_df', return_value=_df_ao_mock()):
        r = client.get('/api/tournament/simulate?tournament=Australian Open&simulations=10')
    assert r.status_code == 200
    data = r.get_json()
    assert data['tournament'] == 'Australian Open'
    assert data['simulations'] == 10
    assert len(data['results']) > 0
    assert 'probabilities' in data['results'][0]


def test_simulate_tournament_con_mock_torneo_invalido(client):
    with patch('app._get_ongoing_df', return_value=_df_ao_mock()):
        r = client.get('/api/tournament/simulate?tournament=TorneoInexistente&simulations=5')
    assert r.status_code == 404
```

- [ ] **Step 2: Ejecutar tests para verificar que fallan**

```bash
source venv/bin/activate && python -m pytest tests/test_api_endpoints.py::test_tournament_info_con_mock_devuelve_200 tests/test_api_endpoints.py::test_simulate_tournament_con_mock_devuelve_200 -v
```

Esperado: `FAILED` porque los endpoints todavía leen el archivo local (el mock de `_get_ongoing_df` no tiene efecto mientras el endpoint no lo llame).

- [ ] **Step 3: Parchar `/api/tournament/info` en `app.py`**

Localizar el inicio del endpoint `/api/tournament/info` (línea ~253). Reemplazar estas líneas:

```python
    import pandas as pd
    ongoing_path = os.path.join("data", "ongoing_tourneys.csv")
    if not os.path.exists(ongoing_path):
        return jsonify({"detail": f"No se encontró el archivo de torneos en curso: {ongoing_path}"}), 404

    try:
        df_ongoing = pd.read_csv(ongoing_path)
```

por:

```python
    try:
        df_ongoing = _get_ongoing_df()
```

- [ ] **Step 4: Parchar `/api/tournament/simulate` en `app.py`**

Localizar el inicio del endpoint `/api/tournament/simulate` (línea ~333). Reemplazar estas líneas:

```python
    import pandas as pd
    from src.simulator import simular_torneo_montecarlo

    ongoing_path = os.path.join("data", "ongoing_tourneys.csv")
    if not os.path.exists(ongoing_path):
        return jsonify({"detail": f"No se encontró el archivo de torneos en curso: {ongoing_path}"}), 404

    try:
        df_ongoing = pd.read_csv(ongoing_path)
```

por:

```python
    from src.simulator import simular_torneo_montecarlo

    try:
        df_ongoing = _get_ongoing_df()
```

- [ ] **Step 5: Ejecutar tests parcheados**

```bash
source venv/bin/activate && python -m pytest tests/test_api_endpoints.py -v -k "mock"
```

Esperado: los 4 tests nuevos `PASSED`.

- [ ] **Step 6: Suite completa**

```bash
source venv/bin/activate && python -m pytest tests/ -v
```

Esperado: todos `PASSED`. (Los tests originales sin mock — `test_simulate_tournament_devuelve_200`, `test_tournament_info_devuelve_200` — ahora harán la llamada real a TML o fallarán si no hay red; son tests de integración que se pueden marcar como `@pytest.mark.integration` o simplemente dejar que fallen en CI sin red.)

- [ ] **Step 7: Commit**

```bash
git add app.py tests/test_api_endpoints.py
git commit -m "fix: info/simulate usan _get_ongoing_df() en vez de archivo local"
```

---

## Task 4: Frontend — poblar `<select>` dinámicamente

**Files:**
- Modify: `static/script.js` (añadir `populateSelect` al abrir modal)

**Interfaces:**
- Consumes: `GET /api/tournaments` de Task 2

*Nota: no hay test unitario automatizado para esta tarea — es UI pura. La verificación es manual.*

- [ ] **Step 1: Localizar el punto de inyección en `static/script.js`**

En `setupTournamentModal()`, buscar el bloque que responde al click de `open-tournament-btn`. Actualmente está alrededor de la línea 754:

```javascript
openBtn.addEventListener('click', () => {
    modal.classList.remove('hidden');
    modal.setAttribute('aria-hidden', 'false');
    switchTab('draw');
    loadTournamentInfo();
});
```

- [ ] **Step 2: Añadir `populateSelect` y llamarla antes de `loadTournamentInfo`**

Reemplazar ese bloque por:

```javascript
const populateSelect = async () => {
    try {
        const r = await fetch('/api/tournaments');
        if (!r.ok) return;
        const data = await r.json();
        if (!data.tournaments || data.tournaments.length === 0) return;
        const surfaceLabel = { Hard: 'Dura', Clay: 'Tierra', Grass: 'Césped' };
        select.innerHTML = '';
        data.tournaments.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t.name;
            opt.textContent = `${t.name} (${surfaceLabel[t.surface] || t.surface})`;
            select.appendChild(opt);
        });
    } catch (_) {
        // fallback: mantener la opción hardcodeada existente
    }
};

openBtn.addEventListener('click', async () => {
    modal.classList.remove('hidden');
    modal.setAttribute('aria-hidden', 'false');
    switchTab('draw');
    await populateSelect();
    loadTournamentInfo();
});
```

- [ ] **Step 3: Incrementar el cache-busting en `templates/index.html`**

En `templates/index.html`, las líneas de `<script>`:
```html
<script src="/static/format.js?v=15"></script>
<script src="/static/script.js?v=15"></script>
```
Cambiar `v=15` a `v=16` en ambas líneas.

- [ ] **Step 4: Verificar manualmente**

```bash
source venv/bin/activate && python app.py
```

Abrir http://localhost:8000, pulsar "Simular Torneo". El `<select>` debe mostrar los torneos en curso ordenados por prioridad (el de mayor nivel primero). Si no hay torneos activos en TML, se mantiene la opción hardcodeada.

- [ ] **Step 5: Commit**

```bash
git add static/script.js templates/index.html
git commit -m "feat: select de torneos poblado dinámicamente desde /api/tournaments"
```

---

## Task 5: Corregir `scripts/fetch_simulate.py`

**Files:**
- Modify: `scripts/fetch_simulate.py` (fix bug unpack + usar draw module + cargar pkl existente)

*Nota: script CLI sin tests unitarios. Verificación manual.*

- [ ] **Step 1: Reescribir `scripts/fetch_simulate.py`**

Reemplazar el contenido completo por:

```python
#!/usr/bin/env python
"""
Script CLI: Simulación del torneo ATP en curso de mayor categoría.
Descarga el draw desde TML-Database y simula con el modelo pkl existente.
"""

import os
import sys
import pickle

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.draw import descargar_ongoing, listar_torneos
from src.simulator import simular_torneo_montecarlo


def main():
    print("=== SIMULADOR DE TORNEOS MONTE CARLO ===\n")

    # 1. Cargar modelos existentes
    print("[1/4] Cargando modelos desde models/...")
    with open("models/modelos_atp.pkl", "rb") as f:
        modelo = pickle.load(f)
    with open("models/stats_jugadores.pkl", "rb") as f:
        metadata = pickle.load(f)

    elo_general = metadata['elo_general']
    elo_superficie = metadata['elo_superficie']
    stats_jugadores = metadata['stats']
    print(f"  Modelo cargado. {len(elo_general)} jugadores con ELO.")

    # 2. Descargar torneos en curso
    print("\n[2/4] Descargando ongoing_tourneys.csv desde TML-Database...")
    df_ongoing = descargar_ongoing()
    print(f"  {len(df_ongoing)} partidos descargados.")

    torneos = listar_torneos(df_ongoing)
    if not torneos:
        print("  No hay torneos ATP en curso. Abortando.")
        sys.exit(0)

    torneo = torneos[0]
    print(f"  Torneo seleccionado: {torneo['name']} (nivel {torneo['level']}, {torneo['surface']})")

    # 3. Reconstruir draw inicial
    print("\n[3/4] Reconstruyendo cuadro inicial...")
    df_torney = df_ongoing[df_ongoing['tourney_name'] == torneo['name']].copy()

    for r in ['R128', 'R64', 'R32', 'R16', 'QF']:
        if r in df_torney['round'].values:
            first_round = r
            break
    else:
        first_round = df_torney['round'].iloc[0]

    df_first = df_torney[df_torney['round'] == first_round].sort_values('match_num')
    initial_draw = []
    for _, row in df_first.iterrows():
        initial_draw.append(row['winner_name'])
        initial_draw.append(row['loser_name'])

    print(f"  Ronda inicial: {first_round} — {len(initial_draw)} jugadores.")

    # 4. Simulación Monte Carlo
    n_sims = 10000
    print(f"\n[4/4] Ejecutando {n_sims} simulaciones...")
    surface = torneo['surface']
    df_prob = simular_torneo_montecarlo(
        initial_draw, surface, modelo, elo_general, elo_superficie, stats_jugadores,
        n_simulaciones=n_sims, seed=42
    )

    # Reporte
    rondas = list(df_prob.columns)
    header_cols = ['ELO Gen', 'Rank'] + rondas
    print(f"\n{'Jugador':<26} {'ELO Gen':>7} {'Rank':>5}  " + '  '.join(f"{r:>6}" for r in rondas))
    print("=" * (26 + 10 + 8 * len(rondas)))

    for player, row in df_prob.head(16).iterrows():
        elo = elo_general.get(player, 1500.0)
        rank = stats_jugadores.get(player, {}).get('rank', 999.0)
        rank_str = str(int(rank)) if rank < 999 else 'S/R'
        probs = '  '.join(f"{row[r]:>5.1f}%" for r in rondas)
        print(f"{player:<26} {elo:>7.1f} {rank_str:>5}  {probs}")

    print("\nSimulación completada.")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verificar manualmente**

```bash
source venv/bin/activate && python scripts/fetch_simulate.py
```

Esperado: descarga CSV, selecciona torneo de mayor prioridad, imprime tabla de probabilidades en consola. Si no hay torneos activos, imprime "No hay torneos ATP en curso."

- [ ] **Step 3: Commit**

```bash
git add scripts/fetch_simulate.py
git commit -m "fix: fetch_simulate usa draw module, pkl existente, fix bug unpack 4 valores"
```

---

## Self-Review

**Spec coverage:**

| Requisito spec | Tarea |
|---------------|-------|
| `src/draw.py` con `descargar_ongoing()` + `listar_torneos()` | Task 1 ✓ |
| Prioridad G > A > 500 > 250, excluir D | Task 1 ✓ |
| Cache en memoria TTL 1h en `app.py` | Task 2 ✓ |
| `GET /api/tournaments` | Task 2 ✓ |
| 503 si TML no responde | Task 2 ✓ |
| Parchar `/api/tournament/info` | Task 3 ✓ |
| Parchar `/api/tournament/simulate` | Task 3 ✓ |
| Frontend: select dinámico | Task 4 ✓ |
| `scripts/fetch_simulate.py` fix + usar pkl | Task 5 ✓ |
| `requests` en requirements.txt | Task 1 ✓ |
| Jugadores fuera de pkl → ELO=1500, rank=999, is_unranked=True | Ya en `src/simulator.py` existente ✓ |

**Placeholders:** ninguno encontrado.

**Consistencia de tipos:**
- `_get_ongoing_df()` devuelve `pd.DataFrame` — consistente en Tasks 2, 3
- `listar_torneos()` devuelve `list[dict]` con claves `{name, surface, level, draw_size, tourney_id}` — consistente en Tasks 1, 2, 4
- `descargar_ongoing()` lanza `RuntimeError` — capturado en Task 2 y en Task 5
