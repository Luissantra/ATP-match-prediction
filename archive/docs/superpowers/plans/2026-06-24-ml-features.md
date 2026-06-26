# ML Features Expansion + TimeSeriesSplit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Añadir H2H, forma reciente y nivel de torneo como features, y sustituir la validación cruzada estándar por TimeSeriesSplit.

**Architecture:** Extender el loop cronológico de `src/elo.py` para computar H2H y forma pre-partido sin leakage. `src/data_processing.py` consume las nuevas columnas y añade encoding de `tourney_level`. `main.py` actualiza el feature set y el CV.

**Tech Stack:** Python 3.11, pandas 3.0, numpy 2.4, scikit-learn (GradientBoostingClassifier, GridSearchCV, TimeSeriesSplit). Venv en `/Users/luissantra/Projects/ATP Prediction/venv`.

## Global Constraints

- Sin leakage: toda feature nueva se calcula con datos **anteriores** al partido actual.
- Python del venv: `/Users/luissantra/Projects/ATP Prediction/venv/bin/python`.
- Tests en `tests/` en la raíz del worktree.
- Worktree: `/Users/luissantra/Projects/ATP Prediction/.claude/worktrees/vigilant-chandrasekhar-7c303b`.
- Commits frecuentes, uno por tarea.

---

## Mapa de Archivos

| Archivo | Acción | Responsabilidad |
|---------|--------|-----------------|
| `src/elo.py` | Modificar | Añadir H2H + forma en el loop cronológico |
| `src/data_processing.py` | Modificar | Usar nuevas columnas, encoding tourney_level |
| `main.py` | Modificar | Feature list, TimeSeriesSplit, labels del gráfico |
| `tests/test_elo.py` | Crear | Tests unitarios de las nuevas columnas |
| `tests/test_data_processing.py` | Crear | Tests de nuevas features en el dataset simétrico |
| `tests/__init__.py` | Crear | Vacío, marca el módulo |

---

### Task 1: Tests unitarios para H2H y forma en `src/elo.py`

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_elo.py`
- Modify: `src/elo.py`

**Interfaces:**
- Consumes: `calcular_elos_historicos(data_dir, años)` existente
- Produces: `df_completo` con 4 columnas nuevas: `h2h_winner_ratio` (float 0-1), `h2h_loser_ratio` (float 0-1), `form_winner` (float 0-1), `form_loser` (float 0-1)

- [ ] **Step 1: Crear `tests/__init__.py`**

```bash
touch /Users/luissantra/Projects/ATP\ Prediction/.claude/worktrees/vigilant-chandrasekhar-7c303b/tests/__init__.py
```

- [ ] **Step 2: Escribir los tests fallidos**

Crear `tests/test_elo.py`:

```python
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.elo import calcular_elos_historicos


def _make_df():
    """Tres partidos mínimos: A vence a B dos veces, B vence a C una vez."""
    return pd.DataFrame([
        {'tourney_date': 20240101, 'match_num': 1, 'winner_name': 'A', 'loser_name': 'B',
         'surface': 'Hard', 'winner_rank': 10, 'loser_rank': 20,
         'winner_age': 25.0, 'loser_age': 27.0, 'tourney_level': 'G'},
        {'tourney_date': 20240102, 'match_num': 1, 'winner_name': 'A', 'loser_name': 'B',
         'surface': 'Clay', 'winner_rank': 10, 'loser_rank': 20,
         'winner_age': 25.0, 'loser_age': 27.0, 'tourney_level': 'M'},
        {'tourney_date': 20240103, 'match_num': 1, 'winner_name': 'B', 'loser_name': 'C',
         'surface': 'Grass', 'winner_rank': 20, 'loser_rank': 50,
         'winner_age': 27.0, 'loser_age': 30.0, 'tourney_level': '250'},
    ])


def test_new_columns_exist(tmp_path):
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, _, _ = calcular_elos_historicos(str(tmp_path), [2024])
    assert 'h2h_winner_ratio' in result.columns
    assert 'h2h_loser_ratio' in result.columns
    assert 'form_winner' in result.columns
    assert 'form_loser' in result.columns


def test_h2h_default_when_no_history(tmp_path):
    """Primer partido entre A y B: H2H debe ser 0.5 para ambos."""
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, _, _ = calcular_elos_historicos(str(tmp_path), [2024])
    # El primer partido (índice 0) no tiene historial previo
    assert result.iloc[0]['h2h_winner_ratio'] == 0.5
    assert result.iloc[0]['h2h_loser_ratio'] == 0.5


def test_h2h_updates_after_first_match(tmp_path):
    """Segundo partido A vs B: A ganó el primero, ratio debe ser 1.0 para A."""
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, _, _ = calcular_elos_historicos(str(tmp_path), [2024])
    # Segundo partido A vs B (índice 1)
    assert result.iloc[1]['h2h_winner_ratio'] == 1.0
    assert result.iloc[1]['h2h_loser_ratio'] == 0.0


def test_form_default_when_no_history(tmp_path):
    """Primer partido de A: forma debe ser 0.5."""
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, _, _ = calcular_elos_historicos(str(tmp_path), [2024])
    assert result.iloc[0]['form_winner'] == 0.5
    assert result.iloc[0]['form_loser'] == 0.5


def test_form_updates_after_matches(tmp_path):
    """Tercer partido (B vs C): B ganó 1 de 2 partidos previos → form=0.5. C debut → form=0.5."""
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, _, _ = calcular_elos_historicos(str(tmp_path), [2024])
    # B es el ganador en el tercer partido (índice 2)
    # B había ganado 0 de 2 previos → form_winner debe ser 0.0
    assert result.iloc[2]['form_winner'] == 0.0
    # C debuta → form_loser debe ser 0.5
    assert result.iloc[2]['form_loser'] == 0.5


def test_ratios_in_valid_range(tmp_path):
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, _, _ = calcular_elos_historicos(str(tmp_path), [2024])
    for col in ['h2h_winner_ratio', 'h2h_loser_ratio', 'form_winner', 'form_loser']:
        assert result[col].between(0.0, 1.0).all(), f"{col} out of range"
```

- [ ] **Step 3: Ejecutar tests — deben fallar**

```bash
cd "/Users/luissantra/Projects/ATP Prediction" && \
  venv/bin/python -m pytest .claude/worktrees/vigilant-chandrasekhar-7c303b/tests/test_elo.py -v 2>&1 | tail -20
```

Esperado: `AttributeError` o `KeyError` porque las columnas no existen aún.

- [ ] **Step 4: Implementar H2H y forma en `src/elo.py`**

En el worktree, editar `src/elo.py`. Añadir el import al inicio del archivo (después de `import numpy as np`):

```python
from collections import deque
```

En `calcular_elos_historicos`, añadir estas estructuras justo después de `lista_dfs = []`:

```python
h2h = {}   # {(p_menor, p_mayor): {player: wins}}
form = {}  # {player: deque(maxlen=10)}
```

Añadir estas listas justo antes del `for idx, row in df_completo.iterrows():`:

```python
h2h_winner_list = []
h2h_loser_list = []
form_winner_list = []
form_loser_list = []
```

Dentro del loop `for idx, row in df_completo.iterrows():`, justo **antes** de la línea `g_general = elo_general.get(ganador, 1500.0)`, añadir:

```python
        # --- H2H pre-partido ---
        h2h_key = tuple(sorted([ganador, perdedor]))
        if h2h_key not in h2h:
            h2h[h2h_key] = {ganador: 0, perdedor: 0}
        h2h_record = h2h[h2h_key]
        total_h2h = h2h_record.get(ganador, 0) + h2h_record.get(perdedor, 0)
        if total_h2h == 0:
            ratio_h2h_winner = 0.5
            ratio_h2h_loser = 0.5
        else:
            ratio_h2h_winner = h2h_record.get(ganador, 0) / total_h2h
            ratio_h2h_loser = h2h_record.get(perdedor, 0) / total_h2h
        h2h_winner_list.append(ratio_h2h_winner)
        h2h_loser_list.append(ratio_h2h_loser)

        # --- Forma pre-partido ---
        dq_winner = form.get(ganador)
        form_w = (sum(dq_winner) / len(dq_winner)) if dq_winner else 0.5
        dq_loser = form.get(perdedor)
        form_l = (sum(dq_loser) / len(dq_loser)) if dq_loser else 0.5
        form_winner_list.append(form_w)
        form_loser_list.append(form_l)
```

Dentro del mismo loop, justo **después** de `elo_superficie[superficie][perdedor] = nuevo_p_sup`, añadir:

```python
        # --- Actualizar H2H y forma post-partido ---
        h2h[h2h_key][ganador] = h2h[h2h_key].get(ganador, 0) + 1
        if ganador not in form:
            form[ganador] = deque(maxlen=10)
        form[ganador].append(1)
        if perdedor not in form:
            form[perdedor] = deque(maxlen=10)
        form[perdedor].append(0)
```

Después de `df_completo['elo_loser'] = elo_perdedor_previo`, añadir:

```python
    df_completo['h2h_winner_ratio'] = h2h_winner_list
    df_completo['h2h_loser_ratio'] = h2h_loser_list
    df_completo['form_winner'] = form_winner_list
    df_completo['form_loser'] = form_loser_list
```

- [ ] **Step 5: Ejecutar tests — deben pasar**

```bash
cd "/Users/luissantra/Projects/ATP Prediction" && \
  venv/bin/python -m pytest .claude/worktrees/vigilant-chandrasekhar-7c303b/tests/test_elo.py -v 2>&1 | tail -20
```

Esperado: `6 passed`.

- [ ] **Step 6: Commit**

```bash
cd "/Users/luissantra/Projects/ATP Prediction/.claude/worktrees/vigilant-chandrasekhar-7c303b" && \
  git add tests/__init__.py tests/test_elo.py src/elo.py && \
  git commit -m "feat: add H2H and recent form tracking in ELO loop"
```

---

### Task 2: Tests y feature engineering en `src/data_processing.py`

**Files:**
- Create: `tests/test_data_processing.py`
- Modify: `src/data_processing.py`

**Interfaces:**
- Consumes: `df_completo` con columnas `h2h_winner_ratio`, `h2h_loser_ratio`, `form_winner`, `form_loser`, `tourney_level` (de Task 1)
- Produces: `preparar_datos_entrenamiento(df)` retorna DataFrame con 6 features: `diff_elo`, `diff_rank`, `diff_age`, `diff_h2h`, `diff_form`, `tourney_level_num`

- [ ] **Step 1: Escribir tests fallidos**

Crear `tests/test_data_processing.py`:

```python
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_processing import preparar_datos_entrenamiento, LEVEL_MAP


def _make_df_with_elo():
    """DataFrame mínimo con todas las columnas que espera preparar_datos_entrenamiento."""
    return pd.DataFrame([
        {
            'tourney_date': 20240101, 'surface': 'Hard', 'tourney_level': 'G',
            'winner_rank': 10.0, 'loser_rank': 20.0,
            'winner_age': 25.0, 'loser_age': 27.0,
            'elo_winner': 1600.0, 'elo_loser': 1500.0,
            'h2h_winner_ratio': 0.75, 'h2h_loser_ratio': 0.25,
            'form_winner': 0.8, 'form_loser': 0.4,
        },
        {
            'tourney_date': 20240102, 'surface': 'Clay', 'tourney_level': 'M',
            'winner_rank': 5.0, 'loser_rank': 30.0,
            'winner_age': 22.0, 'loser_age': 32.0,
            'elo_winner': 1700.0, 'elo_loser': 1450.0,
            'h2h_winner_ratio': 0.5, 'h2h_loser_ratio': 0.5,
            'form_winner': 0.6, 'form_loser': 0.3,
        },
    ])


def test_new_feature_columns_exist():
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    assert 'diff_h2h' in result.columns
    assert 'diff_form' in result.columns
    assert 'tourney_level_num' in result.columns


def test_total_features_count():
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    expected = {'year', 'surface', 'diff_elo', 'diff_rank', 'diff_age',
                'diff_h2h', 'diff_form', 'tourney_level_num', 'label'}
    assert set(result.columns) == expected


def test_level_map_grand_slam():
    assert LEVEL_MAP['G'] == 5


def test_level_map_masters():
    assert LEVEL_MAP['M'] == 4


def test_level_map_500():
    assert LEVEL_MAP['500'] == 2
    assert LEVEL_MAP['A'] == 2


def test_level_map_250():
    assert LEVEL_MAP['250'] == 1
    assert LEVEL_MAP['D'] == 1


def test_tourney_level_encoded_correctly():
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    # El primer partido es Grand Slam (G=5)
    assert result.iloc[0]['tourney_level_num'] == 5
    # El segundo es Masters (M=4)
    assert result.iloc[1]['tourney_level_num'] == 4


def test_diff_h2h_symmetry():
    """Con shuffle_mask fijo (seed=42), diff_h2h debe ser ±(ratio_winner - ratio_loser)."""
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    # El valor absoluto de diff_h2h debe ser 0.5 para la primera fila (0.75-0.25=0.5)
    assert abs(result.iloc[0]['diff_h2h']) == 0.5


def test_diff_form_symmetry():
    """Valor absoluto de diff_form primera fila: |0.8 - 0.4| = 0.4"""
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    assert abs(result.iloc[0]['diff_form']) == pytest.approx(0.4, abs=1e-9)


def test_label_balanced():
    """Con suficientes filas, el label debe estar ~balanceado (seed=42)."""
    rows = [
        {
            'tourney_date': 20240101 + i, 'surface': 'Hard', 'tourney_level': 'G',
            'winner_rank': 10.0, 'loser_rank': 20.0,
            'winner_age': 25.0, 'loser_age': 27.0,
            'elo_winner': 1600.0, 'elo_loser': 1500.0,
            'h2h_winner_ratio': 0.5, 'h2h_loser_ratio': 0.5,
            'form_winner': 0.5, 'form_loser': 0.5,
        }
        for i in range(100)
    ]
    df = pd.DataFrame(rows)
    result = preparar_datos_entrenamiento(df)
    ratio = result['label'].mean()
    assert 0.4 < ratio < 0.6
```

Añadir al inicio del archivo después de los imports:

```python
import pytest
```

- [ ] **Step 2: Ejecutar tests — deben fallar**

```bash
cd "/Users/luissantra/Projects/ATP Prediction" && \
  venv/bin/python -m pytest .claude/worktrees/vigilant-chandrasekhar-7c303b/tests/test_data_processing.py -v 2>&1 | tail -20
```

Esperado: `ImportError` o `AssertionError` porque `LEVEL_MAP` y las nuevas features no existen.

- [ ] **Step 3: Implementar en `src/data_processing.py`**

Añadir `LEVEL_MAP` justo antes de la función `preparar_datos_entrenamiento`:

```python
LEVEL_MAP = {
    'G': 5,
    'M': 4,
    'F': 3,
    'O': 3,
    '500': 2, 'A': 2,
    '250': 1, 'D': 1,
}
```

Dentro de `preparar_datos_entrenamiento`, en el loop `for i in range(len(df)):`, después de extraer `w_elo, l_elo, w_rank, l_rank, w_age, l_age`, añadir:

```python
        h2h_w = row.get('h2h_winner_ratio', 0.5)
        h2h_l = row.get('h2h_loser_ratio', 0.5)
        form_w = row.get('form_winner', 0.5)
        form_l = row.get('form_loser', 0.5)
        tourney_level_num = LEVEL_MAP.get(str(row.get('tourney_level', '250')), 1)
```

En el bloque `if shuffle_mask[i]:`, añadir después de `label = 1`:

```python
            diff_h2h = h2h_w - h2h_l
            diff_form = form_w - form_l
```

En el bloque `else:`, añadir después de `label = 0`:

```python
            diff_h2h = h2h_l - h2h_w
            diff_form = form_l - form_w
```

En `features.append({...})`, añadir las tres nuevas claves antes de `'label': label`:

```python
            'diff_h2h': diff_h2h,
            'diff_form': diff_form,
            'tourney_level_num': tourney_level_num,
```

- [ ] **Step 4: Ejecutar tests — deben pasar**

```bash
cd "/Users/luissantra/Projects/ATP Prediction" && \
  venv/bin/python -m pytest .claude/worktrees/vigilant-chandrasekhar-7c303b/tests/test_data_processing.py -v 2>&1 | tail -20
```

Esperado: `9 passed`.

- [ ] **Step 5: Commit**

```bash
cd "/Users/luissantra/Projects/ATP Prediction/.claude/worktrees/vigilant-chandrasekhar-7c303b" && \
  git add tests/test_data_processing.py src/data_processing.py && \
  git commit -m "feat: add H2H, form, and tourney level features in data_processing"
```

---

### Task 3: Actualizar `main.py` con nuevas features y TimeSeriesSplit

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes: `preparar_datos_entrenamiento` ahora retorna `diff_h2h`, `diff_form`, `tourney_level_num` (Task 2)
- Consumes: `calcular_elos_historicos` retorna `df_completo` con 4 nuevas columnas (Task 1)
- Produces: modelo serializado en `modelo_atp.pkl` entrenado con 6 features

- [ ] **Step 1: Actualizar imports y constante FEATURES en `main.py`**

Reemplazar la línea:

```python
from sklearn.model_selection import GridSearchCV
```

Por:

```python
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
```

Añadir después de los imports (justo antes de `def imprimir_seccion`):

```python
FEATURES = ['diff_elo', 'diff_rank', 'diff_age', 'diff_h2h', 'diff_form', 'tourney_level_num']
```

- [ ] **Step 2: Actualizar el slicing de X_train y X_test**

Reemplazar:

```python
    X_train = df_train[['diff_elo', 'diff_rank', 'diff_age']]
    y_train = df_train['label']
    
    X_test = df_test[['diff_elo', 'diff_rank', 'diff_age']]
    y_test = df_test['label']
```

Por:

```python
    X_train = df_train[FEATURES]
    y_train = df_train['label']

    X_test = df_test[FEATURES]
    y_test = df_test['label']
```

- [ ] **Step 3: Sustituir cv=3 por TimeSeriesSplit**

Reemplazar en la llamada a `GridSearchCV`:

```python
        cv=3, 
```

Por:

```python
        cv=TimeSeriesSplit(n_splits=5),
```

- [ ] **Step 4: Actualizar etiquetas del gráfico de feature importance**

Reemplazar:

```python
    features_list = ['Diferencia ELO', 'Diferencia Ranking', 'Diferencia Edad']
```

Por:

```python
    features_list = ['Diferencia ELO', 'Diferencia Ranking', 'Diferencia Edad',
                     'H2H Histórico', 'Forma Reciente', 'Nivel de Torneo']
```

- [ ] **Step 5: Smoke test — ejecutar el pipeline completo**

```bash
cd "/Users/luissantra/Projects/ATP Prediction" && \
  venv/bin/python .claude/worktrees/vigilant-chandrasekhar-7c303b/main.py 2>&1 | tail -30
```

Esperado: pipeline completo sin errores, muestra accuracy final y línea `"Modelo y metadatos exportados con éxito"`.

- [ ] **Step 6: Verificar que los PKL están en el worktree (no en la raíz del proyecto)**

Los `.pkl` se generan en el CWD de ejecución. Si el script se ejecutó desde la raíz del proyecto, mover:

```bash
# Si están en la raíz del proyecto, copiarlos al worktree para que app.py los encuentre
cp "/Users/luissantra/Projects/ATP Prediction/modelo_atp.pkl" \
   "/Users/luissantra/Projects/ATP Prediction/.claude/worktrees/vigilant-chandrasekhar-7c303b/"
cp "/Users/luissantra/Projects/ATP Prediction/stats_jugadores.pkl" \
   "/Users/luissantra/Projects/ATP Prediction/.claude/worktrees/vigilant-chandrasekhar-7c303b/"
```

- [ ] **Step 7: Ejecutar todos los tests**

```bash
cd "/Users/luissantra/Projects/ATP Prediction" && \
  venv/bin/python -m pytest .claude/worktrees/vigilant-chandrasekhar-7c303b/tests/ -v 2>&1 | tail -20
```

Esperado: todos los tests pasan.

- [ ] **Step 8: Commit**

```bash
cd "/Users/luissantra/Projects/ATP Prediction/.claude/worktrees/vigilant-chandrasekhar-7c303b" && \
  git add main.py && \
  git commit -m "feat: use 6-feature set and TimeSeriesSplit in GridSearchCV"
```
