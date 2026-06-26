# I2 + I3: rank cap + ELO separado como dos features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminar el outlier rank=999 (I2) y dejar que el GBM aprenda el peso óptimo entre ELO general y ELO de superficie en lugar de fijarlo a 50/50 (I3), cambiando FEATURES de 6 a 8 y reentrenando todos los modelos.

**Architecture:** Todas las modificaciones parten de `src/features.py` (fuente única de verdad). Los cambios se propagan hacia abajo: `src/elo.py` emite columnas separadas por tipo de ELO, `src/data_processing.py` las consume para construir el nuevo vector, y `app.py` los reconstruye en inferencia con la misma semántica. Un único reentrenamiento final valida que los pkl son consistentes con los nuevos FEATURES.

**Tech Stack:** Python 3.11, scikit-learn, pandas, numpy, pytest, Flask, XGBoost.

## Global Constraints

- TDD estricto: test que falla → implementación mínima → test verde → commit.
- Un commit por tarea.
- No reordenar FEATURES sin reentrenar.
- No crear Pull Requests — merge a main localmente + push directo.
- `RANK_CAP = 250` — valor exacto, no cambiar sin consenso.
- Workflow: `source venv/bin/activate && python -m pytest -q` para correr la suite completa.
- Entorno: `venv/` en la raíz del proyecto.

---

## Nuevo vector de features (8 features)

```python
FEATURES = [
    'diff_elo_general', 'diff_elo_sup',   # I3: GBM aprende el peso
    'diff_rank', 'is_unranked',            # I2: cap a 250 + indicador wildcard
    'diff_age', 'diff_h2h', 'diff_form', 'tourney_level_num',
]
RANK_CAP = 250
```

`is_unranked` = `int(rank_a_raw >= 999) - int(rank_b_raw >= 999)` ∈ {-1, 0, 1}.

---

## Archivos modificados

| Archivo | Qué cambia |
|---|---|
| `src/features.py` | FEATURES (8), añadir RANK_CAP, quitar elo_hibrido del comentario de fuente |
| `src/elo.py` | Emite 4 columnas nuevas: `elo_winner_general`, `elo_loser_general`, `elo_winner_sup`, `elo_loser_sup` |
| `src/data_processing.py` | Usa las 4 columnas nuevas para `diff_elo_general`, `diff_elo_sup`; capea rank; añade `is_unranked` |
| `app.py` | `construir_features` devuelve las 8 features; `_predecir_con` actualiza `features_debug`; importa RANK_CAP |
| `tests/test_features.py` | Actualiza `_feat()` helper y el test de longitud |
| `tests/test_data_processing.py` | Actualiza `_make_df_with_elo`, `test_total_feature_columns`, añade tests de cap e `is_unranked` |
| `tests/test_app_features.py` | Reemplaza `test_diff_elo_usa_elo_hibrido` por dos tests; actualiza el test de jugadores desconocidos |
| `tests/test_elo.py` | Añade test de que las 4 columnas nuevas existen |

---

## Task 1: Actualizar FEATURES y RANK_CAP en src/features.py

**Files:**
- Modify: `src/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produce: `FEATURES` (lista de 8 strings), `RANK_CAP = 250` — consumidos por todas las tareas siguientes.

- [ ] **Step 1: Escribir el test que falla**

En `tests/test_features.py`, reemplaza el helper `_feat` y añade el test de longitud:

```python
# tests/test_features.py  — versión completa tras el cambio

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.features import FEATURES, RANK_CAP, elo_hibrido, vector_from_features


class TestEloHibrido:
    def test_default_5050(self):
        assert elo_hibrido(1600.0, 1400.0) == pytest.approx(1500.0)

    def test_peso_general_total(self):
        assert elo_hibrido(1600.0, 1400.0, w=1.0) == pytest.approx(1600.0)

    def test_peso_superficie_total(self):
        assert elo_hibrido(1600.0, 1400.0, w=0.0) == pytest.approx(1400.0)


class TestVectorFromFeatures:
    def _feat(self):
        return {
            'diff_elo_general': 100.0, 'diff_elo_sup': 200.0,
            'diff_rank': -5.0, 'is_unranked': 0,
            'diff_age': 2.0, 'diff_h2h': 0.3,
            'diff_form': 0.1, 'tourney_level_num': 4,
        }

    def test_orden_coincide_con_FEATURES(self):
        feat = self._feat()
        vec = vector_from_features(feat)
        assert vec == [feat[name] for name in FEATURES]

    def test_longitud_igual_a_FEATURES(self):
        assert len(vector_from_features(self._feat())) == len(FEATURES)

    def test_falta_clave_lanza_error(self):
        feat = self._feat()
        del feat['diff_h2h']
        with pytest.raises(KeyError):
            vector_from_features(feat)

    def test_features_tiene_8_elementos(self):
        assert len(FEATURES) == 8

    def test_rank_cap_es_250(self):
        assert RANK_CAP == 250
```

- [ ] **Step 2: Confirmar que el test falla**

```bash
cd "/Users/luissantra/Projects/ATP Prediction"
source venv/bin/activate
python -m pytest tests/test_features.py -v
```

Esperado: ImportError o AssertionError (RANK_CAP no existe, FEATURES tiene 6 elementos).

- [ ] **Step 3: Implementar los cambios en src/features.py**

```python
# src/features.py — versión completa

"""
Fuente de verdad única del vector de características del modelo.
==============================================================

Tanto el entrenamiento (`src/data_processing.py`, `main.py`) como la inferencia
(`app.py`) construyen el vector de features a partir de aquí. Centralizarlo evita
el *train/serve skew*: que el modelo se entrene con una representación y se sirva
con otra distinta (orden de columnas, pesos del ELO híbrido, defaults, etc.).
"""

# Orden canónico de las features que consume el modelo. NO reordenar sin reentrenar.
FEATURES = [
    'diff_elo_general', 'diff_elo_sup',   # I3: GBM aprende el peso óptimo (antes 50/50 fijo)
    'diff_rank', 'is_unranked',            # I2: rank capeado a 250 + indicador wildcard/qualifier
    'diff_age', 'diff_h2h', 'diff_form', 'tourney_level_num',
]

# Ranking máximo considerado. Por encima de este umbral el jugador se considera sin ranking.
RANK_CAP = 250

# Codificación ordinal del nivel de torneo (mayor = más importante).
LEVEL_MAP = {
    'G': 5,
    'M': 4,
    'F': 3,
    'O': 3,
    '500': 2, 'A': 2,
    '250': 1, 'D': 1,
}

# Nivel por defecto cuando no se conoce el torneo: 1 (ATP 250), el más común del
# circuito. Antes se usaba 3 (Finals/Olympics), lo que introducía sesgo sistemático.
DEFAULT_LEVEL_NUM = 1


def elo_hibrido(elo_general, elo_superficie, w=0.5):
    """ELO híbrido: w*general + (1-w)*superficie. Se sigue usando en elo.py para actualizar ratings."""
    return w * elo_general + (1 - w) * elo_superficie


def vector_from_features(feat):
    """
    Construye la lista de features en el orden canónico de FEATURES.

    Garantiza que entrenamiento e inferencia produzcan el mismo orden/longitud.
    Lanza KeyError si falta alguna feature (falla ruidoso, no silencioso).
    """
    return [feat[name] for name in FEATURES]
```

- [ ] **Step 4: Confirmar que los tests pasan**

```bash
python -m pytest tests/test_features.py -v
```

Esperado: todos PASS.

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat(I2+I3): nuevo FEATURES (8) — diff_elo_general, diff_elo_sup, is_unranked, RANK_CAP=250"
```

---

## Task 2: Emitir columnas ELO separadas desde src/elo.py

**Files:**
- Modify: `src/elo.py:156-234`
- Test: `tests/test_elo.py`

**Interfaces:**
- Consume: nada de las tareas anteriores (elo.py es independiente).
- Produce: columnas `elo_winner_general`, `elo_loser_general`, `elo_winner_sup`, `elo_loser_sup` en `df_completo` — consumidas por `data_processing.py` (Task 3).

- [ ] **Step 1: Escribir el test que falla**

Añadir al final de `tests/test_elo.py`:

```python
def test_columnas_elo_separado_existen(tmp_path):
    """Tras I3, el df debe exponer ELO general y de superficie por separado."""
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, *_ = calcular_elos_historicos(str(tmp_path), [2024])
    for col in ['elo_winner_general', 'elo_loser_general',
                'elo_winner_sup', 'elo_loser_sup']:
        assert col in result.columns, f"Falta columna {col}"


def test_elo_general_y_sup_son_distintos(tmp_path):
    """Los ratings general y de superficie divergen a medida que se juegan partidos."""
    df = _make_df()
    df.to_csv(tmp_path / "2024.csv", index=False)
    result, *_ = calcular_elos_historicos(str(tmp_path), [2024])
    # En el segundo partido (Clay), A tiene historial en Hard pero no en Clay:
    # general habrá cambiado, superficie Clay empieza en 1500
    row1 = result.iloc[1]
    assert row1['elo_winner_general'] != row1['elo_winner_sup']
```

- [ ] **Step 2: Confirmar que los tests fallan**

```bash
python -m pytest tests/test_elo.py -v
```

Esperado: FAIL en `test_columnas_elo_separado_existen` y `test_elo_general_y_sup_son_distintos`.

- [ ] **Step 3: Implementar en src/elo.py**

Localiza las listas que se inicializan antes del loop (línea ~156) y añade las 4 nuevas:

```python
    elo_ganador_previo = []
    elo_perdedor_previo = []
    elo_ganador_gen_previo = []    # nuevo I3
    elo_perdedor_gen_previo = []   # nuevo I3
    elo_ganador_sup_previo = []    # nuevo I3
    elo_perdedor_sup_previo = []   # nuevo I3
    h2h_winner_list = []
    h2h_loser_list = []
    form_winner_list = []
    form_loser_list = []
```

Dentro del loop, justo donde se appende `elo_ganador_previo` (línea ~208), añade:

```python
        elo_ganador_previo.append(elo_final_g)
        elo_perdedor_previo.append(elo_final_p)
        elo_ganador_gen_previo.append(g_general)      # nuevo I3
        elo_perdedor_gen_previo.append(p_general)      # nuevo I3
        elo_ganador_sup_previo.append(g_superficie)    # nuevo I3
        elo_perdedor_sup_previo.append(p_superficie)   # nuevo I3
```

Al final, donde se asignan las columnas al DataFrame (línea ~229), añade:

```python
    df_completo['elo_winner'] = elo_ganador_previo
    df_completo['elo_loser'] = elo_perdedor_previo
    df_completo['elo_winner_general'] = elo_ganador_gen_previo   # nuevo I3
    df_completo['elo_loser_general'] = elo_perdedor_gen_previo   # nuevo I3
    df_completo['elo_winner_sup'] = elo_ganador_sup_previo       # nuevo I3
    df_completo['elo_loser_sup'] = elo_perdedor_sup_previo       # nuevo I3
```

- [ ] **Step 4: Confirmar que los tests pasan**

```bash
python -m pytest tests/test_elo.py -v
```

Esperado: todos PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elo.py tests/test_elo.py
git commit -m "feat(I3): elo.py emite elo_winner/loser_general y elo_winner/loser_sup por separado"
```

---

## Task 3: Actualizar data_processing.py para construir las nuevas features

**Files:**
- Modify: `src/data_processing.py`
- Test: `tests/test_data_processing.py`

**Interfaces:**
- Consume: columnas `elo_winner_general`, `elo_loser_general`, `elo_winner_sup`, `elo_loser_sup` del DataFrame producido por `elo.py`; `RANK_CAP` de `src/features.py`.
- Produce: DataFrame con columnas `diff_elo_general`, `diff_elo_sup`, `diff_rank` (capeado), `is_unranked`.

- [ ] **Step 1: Escribir los tests que fallan**

Reemplaza el contenido de `tests/test_data_processing.py` con:

```python
import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_processing import preparar_datos_entrenamiento, LEVEL_MAP


def _make_df_with_elo():
    return pd.DataFrame([
        {
            'tourney_date': 20240101, 'surface': 'Hard', 'tourney_level': 'G',
            'winner_rank': 10.0, 'loser_rank': 20.0,
            'winner_age': 25.0, 'loser_age': 27.0,
            'elo_winner': 1650.0, 'elo_loser': 1500.0,        # híbrido (backward compat)
            'elo_winner_general': 1600.0, 'elo_loser_general': 1500.0,
            'elo_winner_sup': 1700.0, 'elo_loser_sup': 1500.0,
            'h2h_winner_ratio': 0.75, 'h2h_loser_ratio': 0.25,
            'form_winner': 0.8, 'form_loser': 0.4,
        },
        {
            'tourney_date': 20240102, 'surface': 'Clay', 'tourney_level': 'M',
            'winner_rank': 5.0, 'loser_rank': 30.0,
            'winner_age': 22.0, 'loser_age': 32.0,
            'elo_winner': 1575.0, 'elo_loser': 1475.0,
            'elo_winner_general': 1700.0, 'elo_loser_general': 1450.0,
            'elo_winner_sup': 1450.0, 'elo_loser_sup': 1500.0,
            'h2h_winner_ratio': 0.5, 'h2h_loser_ratio': 0.5,
            'form_winner': 0.6, 'form_loser': 0.3,
        },
    ])


def test_columnas_nuevas_existen():
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    assert 'diff_elo_general' in result.columns
    assert 'diff_elo_sup' in result.columns
    assert 'is_unranked' in result.columns


def test_columna_diff_elo_vieja_no_existe():
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    assert 'diff_elo' not in result.columns


def test_total_feature_columns():
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    expected = {
        'year', 'tourney_date', 'surface',
        'diff_elo_general', 'diff_elo_sup',
        'diff_rank', 'is_unranked',
        'diff_age', 'diff_h2h', 'diff_form',
        'tourney_level_num', 'label',
    }
    assert set(result.columns) == expected


def test_rank_cap_limita_outlier():
    """rank=999 (wildcard) debe capar a RANK_CAP=250 en diff_rank."""
    df = pd.DataFrame([{
        'tourney_date': 20240101, 'surface': 'Hard', 'tourney_level': 'G',
        'winner_rank': 1.0, 'loser_rank': 999.0,          # outlier
        'winner_age': 25.0, 'loser_age': 27.0,
        'elo_winner': 1650.0, 'elo_loser': 1500.0,
        'elo_winner_general': 1700.0, 'elo_loser_general': 1500.0,
        'elo_winner_sup': 1600.0, 'elo_loser_sup': 1500.0,
        'h2h_winner_ratio': 0.5, 'h2h_loser_ratio': 0.5,
        'form_winner': 0.5, 'form_loser': 0.5,
    }])
    result = preparar_datos_entrenamiento(df)
    # |diff_rank| debe ser ≤ RANK_CAP - 1 = 249 (cap 250 - cap 1 = 249)
    assert abs(result.iloc[0]['diff_rank']) <= 249


def test_is_unranked_detecta_wildcard():
    """Cuando un jugador tiene rank 999, is_unranked != 0."""
    df = pd.DataFrame([{
        'tourney_date': 20240101, 'surface': 'Hard', 'tourney_level': 'G',
        'winner_rank': 1.0, 'loser_rank': 999.0,
        'winner_age': 25.0, 'loser_age': 27.0,
        'elo_winner': 1650.0, 'elo_loser': 1500.0,
        'elo_winner_general': 1700.0, 'elo_loser_general': 1500.0,
        'elo_winner_sup': 1600.0, 'elo_loser_sup': 1500.0,
        'h2h_winner_ratio': 0.5, 'h2h_loser_ratio': 0.5,
        'form_winner': 0.5, 'form_loser': 0.5,
    }])
    result = preparar_datos_entrenamiento(df)
    assert result.iloc[0]['is_unranked'] != 0


def test_is_unranked_cero_cuando_ambos_rankeados():
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    assert result.iloc[0]['is_unranked'] == 0
    assert result.iloc[1]['is_unranked'] == 0


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
    assert result.iloc[0]['tourney_level_num'] == 5  # G
    assert result.iloc[1]['tourney_level_num'] == 4  # M


def test_diff_h2h_absolute_value():
    """Valor absoluto de diff_h2h primera fila: |0.75 - 0.25| = 0.5"""
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    assert abs(result.iloc[0]['diff_h2h']) == pytest.approx(0.5, abs=1e-9)


def test_diff_form_absolute_value():
    """Valor absoluto de diff_form primera fila: |0.8 - 0.4| = 0.4"""
    df = _make_df_with_elo()
    result = preparar_datos_entrenamiento(df)
    assert abs(result.iloc[0]['diff_form']) == pytest.approx(0.4, abs=1e-9)


def test_label_balanced():
    rows = [
        {
            'tourney_date': 20240101 + i, 'surface': 'Hard', 'tourney_level': 'G',
            'winner_rank': 10.0, 'loser_rank': 20.0,
            'winner_age': 25.0, 'loser_age': 27.0,
            'elo_winner': 1600.0, 'elo_loser': 1500.0,
            'elo_winner_general': 1600.0, 'elo_loser_general': 1500.0,
            'elo_winner_sup': 1600.0, 'elo_loser_sup': 1500.0,
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

- [ ] **Step 2: Confirmar que los tests fallan**

```bash
python -m pytest tests/test_data_processing.py -v
```

Esperado: varios FAIL (columnas nuevas no existen, diff_elo sigue existiendo).

- [ ] **Step 3: Implementar en src/data_processing.py**

Reemplaza la función `preparar_datos_entrenamiento` con:

```python
def preparar_datos_entrenamiento(df_con_elo):
    from src.features import RANK_CAP  # importar aquí para no crear dependencia circular
    df = df_con_elo.copy()

    # Imputar nulos de ranking y edad
    df['winner_rank'] = df['winner_rank'].fillna(999)
    df['loser_rank'] = df['loser_rank'].fillna(999)
    mediana_winner_age = df['winner_age'].median()
    mediana_loser_age = df['loser_age'].median()
    df['winner_age'] = df['winner_age'].fillna(mediana_winner_age)
    df['loser_age'] = df['loser_age'].fillna(mediana_loser_age)

    np.random.seed(42)
    shuffle = np.random.rand(len(df)) > 0.5

    # Ranks crudos (para is_unranked)
    rank_a_raw = np.where(shuffle, df['winner_rank'], df['loser_rank'])
    rank_b_raw = np.where(shuffle, df['loser_rank'], df['winner_rank'])

    # ELO general y superficie separados (I3)
    elo_gen_a = np.where(shuffle, df['elo_winner_general'], df['elo_loser_general'])
    elo_gen_b = np.where(shuffle, df['elo_loser_general'], df['elo_winner_general'])
    elo_sup_a = np.where(shuffle, df['elo_winner_sup'], df['elo_loser_sup'])
    elo_sup_b = np.where(shuffle, df['elo_loser_sup'], df['elo_winner_sup'])

    # Simetrización del resto de features
    age_a  = np.where(shuffle, df['winner_age'],      df['loser_age'])
    age_b  = np.where(shuffle, df['loser_age'],       df['winner_age'])
    h2h_a  = np.where(shuffle, df.get('h2h_winner_ratio', 0.5), df.get('h2h_loser_ratio', 0.5))
    h2h_b  = np.where(shuffle, df.get('h2h_loser_ratio', 0.5),  df.get('h2h_winner_ratio', 0.5))
    form_a = np.where(shuffle, df.get('form_winner', 0.5),       df.get('form_loser', 0.5))
    form_b = np.where(shuffle, df.get('form_loser', 0.5),        df.get('form_winner', 0.5))

    level_col = df['tourney_level'].astype(str) if 'tourney_level' in df.columns else pd.Series(['250'] * len(df))
    tourney_level_num = level_col.map(lambda x: LEVEL_MAP.get(x, 1))

    return pd.DataFrame({
        'year':             df['tourney_date'].astype(str).str[:4].astype(int).values,
        'tourney_date':     df['tourney_date'].values,
        'surface':          df['surface'].values if 'surface' in df.columns else 'Hard',
        'diff_elo_general': elo_gen_a - elo_gen_b,                             # I3
        'diff_elo_sup':     elo_sup_a - elo_sup_b,                             # I3
        'diff_rank':        np.minimum(rank_a_raw, RANK_CAP) - np.minimum(rank_b_raw, RANK_CAP),  # I2 cap
        'is_unranked':      (rank_a_raw >= 999).astype(int) - (rank_b_raw >= 999).astype(int),    # I2 flag
        'diff_age':         age_a - age_b,
        'diff_h2h':         h2h_a - h2h_b,
        'diff_form':        form_a - form_b,
        'tourney_level_num': tourney_level_num.values,
        'label':            np.where(shuffle, 1, 0),
    })
```

Nota: el import de `LEVEL_MAP` al inicio del archivo ya viene de `src.features`; añade `RANK_CAP` al mismo import:

```python
from src.features import LEVEL_MAP, RANK_CAP  # fuente única
```

(Si el import está dentro de la función, muévelo fuera.)

- [ ] **Step 4: Confirmar que los tests pasan**

```bash
python -m pytest tests/test_data_processing.py -v
```

Esperado: todos PASS.

- [ ] **Step 5: Correr suite completa para detectar regresiones**

```bash
python -m pytest -q
```

Algunos tests de `test_app_features.py` pueden fallar porque `construir_features` devuelve `diff_elo` (viejo). Es aceptable — se corrigen en Task 4.

- [ ] **Step 6: Commit**

```bash
git add src/data_processing.py tests/test_data_processing.py
git commit -m "feat(I2+I3): data_processing usa ELO separado, capea rank y añade is_unranked"
```

---

## Task 4: Actualizar app.py (inferencia) para construir las nuevas features

**Files:**
- Modify: `app.py:8-113, 139-168`
- Test: `tests/test_app_features.py`

**Interfaces:**
- Consume: `RANK_CAP` de `src/features.py`.
- Produce: `construir_features` devuelve las 8 features en el mismo orden que `FEATURES`.

- [ ] **Step 1: Escribir los tests que fallan**

Reemplaza el contenido de `tests/test_app_features.py` con:

```python
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import app
from src.features import FEATURES, DEFAULT_LEVEL_NUM, RANK_CAP


@pytest.fixture(autouse=True)
def _stub_state(monkeypatch):
    monkeypatch.setattr(app, 'elo_general', {'A': 1600.0, 'B': 1500.0})
    monkeypatch.setattr(app, 'elo_superficie', {'Hard': {'A': 1700.0, 'B': 1500.0}})
    monkeypatch.setattr(app, 'stats_jugadores', {
        'A': {'rank': 5.0, 'age': 24.0},
        'B': {'rank': 20.0, 'age': 30.0},
    })
    monkeypatch.setattr(app, 'h2h', {('A', 'B'): {'A': 3, 'B': 1}})
    monkeypatch.setattr(app, 'form_final', {'A': 0.8, 'B': 0.4})


def test_devuelve_todas_las_features():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    assert set(feat.keys()) == set(FEATURES)


def test_diff_elo_general():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    # A: elo_general=1600 ; B: elo_general=1500
    assert feat['diff_elo_general'] == pytest.approx(100.0)


def test_diff_elo_sup():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    # A: elo_sup_Hard=1700 ; B: elo_sup_Hard=1500
    assert feat['diff_elo_sup'] == pytest.approx(200.0)


def test_diff_rank_usa_rank_real():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    # A rank=5, B rank=20 → diff = 5-20 = -15
    assert feat['diff_rank'] == pytest.approx(-15.0)


def test_is_unranked_cero_cuando_ambos_rankeados():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    assert feat['is_unranked'] == 0


def test_diff_h2h_usa_historial_real_no_cero():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    assert feat['diff_h2h'] == pytest.approx(0.5)


def test_diff_form_usa_forma_real_no_cero():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    assert feat['diff_form'] == pytest.approx(0.8 - 0.4)


def test_tourney_level_se_mapea_desde_parametro():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level='G')
    assert feat['tourney_level_num'] == 5


def test_tourney_level_desconocido_usa_default_no_3():
    feat = app.construir_features('A', 'B', 'Hard', tourney_level=None)
    assert feat['tourney_level_num'] == DEFAULT_LEVEL_NUM


def test_jugadores_desconocidos_son_neutros():
    feat = app.construir_features('X', 'Z', 'Hard', tourney_level='G')
    assert feat['diff_elo_general'] == pytest.approx(0.0)
    assert feat['diff_elo_sup'] == pytest.approx(0.0)
    assert feat['diff_h2h'] == pytest.approx(0.0)
    assert feat['diff_form'] == pytest.approx(0.0)
```

- [ ] **Step 2: Confirmar que los tests fallan**

```bash
python -m pytest tests/test_app_features.py -v
```

Esperado: fallan `test_devuelve_todas_las_features`, `test_diff_elo_general`, `test_diff_elo_sup`, etc.

- [ ] **Step 3: Actualizar el import en app.py**

En la línea de imports de `app.py`, añade `RANK_CAP`:

```python
from src.features import (
    FEATURES, LEVEL_MAP, DEFAULT_LEVEL_NUM, RANK_CAP, elo_hibrido, vector_from_features,
)
```

- [ ] **Step 4: Actualizar construir_features en app.py**

Reemplaza la función `construir_features` (líneas 72-113):

```python
def construir_features(player_a, player_b, surface, tourney_level=None):
    """
    Construye el dict de features para inferencia con la MISMA semántica que el
    entrenamiento (sin train/serve skew). H2H y forma se reconstruyen del historial
    real persistido; el nivel de torneo se mapea del parámetro (default = ATP 250).
    """
    gen_a = elo_general.get(player_a, 1500.0)
    gen_b = elo_general.get(player_b, 1500.0)
    sup_a = elo_superficie.get(surface, {}).get(player_a, 1500.0)
    sup_b = elo_superficie.get(surface, {}).get(player_b, 1500.0)

    rank_a = stats_jugadores.get(player_a, {}).get('rank', 999.0)
    rank_b = stats_jugadores.get(player_b, {}).get('rank', 999.0)
    age_a = stats_jugadores.get(player_a, {}).get('age', 26.0)
    age_b = stats_jugadores.get(player_b, {}).get('age', 26.0)

    # H2H real: ratio de victorias en enfrentamientos previos (0.5/0.5 si no hay historial)
    record = h2h.get(tuple(sorted([player_a, player_b])))
    if record:
        total = record.get(player_a, 0) + record.get(player_b, 0)
        if total > 0:
            ratio_a = record.get(player_a, 0) / total
            ratio_b = record.get(player_b, 0) / total
        else:
            ratio_a = ratio_b = 0.5
    else:
        ratio_a = ratio_b = 0.5

    form_a = form_final.get(player_a, 0.5)
    form_b = form_final.get(player_b, 0.5)

    level_num = LEVEL_MAP.get(str(tourney_level), DEFAULT_LEVEL_NUM)

    return {
        'diff_elo_general': gen_a - gen_b,
        'diff_elo_sup':     sup_a - sup_b,
        'diff_rank':        min(rank_a, RANK_CAP) - min(rank_b, RANK_CAP),
        'is_unranked':      int(rank_a >= 999) - int(rank_b >= 999),
        'diff_age':         age_a - age_b,
        'diff_h2h':         ratio_a - ratio_b,
        'diff_form':        form_a - form_b,
        'tourney_level_num': level_num,
    }
```

- [ ] **Step 5: Actualizar features_debug en _predecir_con (líneas 159-166)**

```python
        "features_debug": {
            "diff_elo_general": round(feat['diff_elo_general'], 1),
            "diff_elo_sup":     round(feat['diff_elo_sup'], 1),
            "diff_rank":        int(feat['diff_rank']),
            "is_unranked":      int(feat['is_unranked']),
            "diff_age":         round(feat['diff_age'], 2),
            "diff_h2h":         round(feat['diff_h2h'], 3),
            "diff_form":        round(feat['diff_form'], 3),
            "tourney_level_num": feat['tourney_level_num'],
        },
```

- [ ] **Step 6: Confirmar que los tests pasan**

```bash
python -m pytest tests/test_app_features.py -v
```

Esperado: todos PASS.

- [ ] **Step 7: Correr suite completa**

```bash
python -m pytest -q
```

Esperado: todos PASS (puede haber warnings de sklearn; son ok).

- [ ] **Step 8: Commit**

```bash
git add app.py tests/test_app_features.py
git commit -m "feat(I2+I3): app.py — construir_features usa ELO separado, rank capeado, is_unranked"
```

---

## Task 5: Reentrenar y actualizar roadmap

**Files:**
- Run: `python main.py`
- Modify: `docs/ROADMAP.md`

**Interfaces:**
- Consume: todos los cambios de Tasks 1-4.
- Produce: pkl actualizados en la raíz del proyecto.

- [ ] **Step 1: Ejecutar el pipeline de entrenamiento**

```bash
source venv/bin/activate
python main.py
```

Esperado: el pipeline completa sin errores. Las métricas del test ciego 2026 aparecen para los 4 modelos. Anotar los valores de log-loss y AUC del GBM para comparar con las métricas anteriores (log-loss ~0.683, AUC ~0.615).

- [ ] **Step 2: Verificar que los pkl se actualizaron**

```bash
ls -lh modelo_atp.pkl modelos_atp.pkl metrics_atp.pkl stats_jugadores.pkl
```

Esperado: todos tienen timestamps recientes.

- [ ] **Step 3: Smoke test de la API**

```bash
python -c "
import app, json
from unittest.mock import patch
# No hace falta cargar pkl reales; solo verificar que el import no explota
print('Import OK')
print('FEATURES:', app.FEATURES)
"
```

Esperado: `Import OK` y FEATURES con 8 elementos.

- [ ] **Step 4: Suite completa**

```bash
python -m pytest -q
```

Esperado: todos PASS.

- [ ] **Step 5: Actualizar roadmap**

En `docs/ROADMAP.md`, marca I2 e I3 como resueltos y actualiza los "Próximos pasos".

```markdown
- [x] **I2 · `rank=999` genera outliers en `diff_rank`** ✅ Resuelto. Cap a RANK_CAP=250 en `data_processing.py` y `app.py`. Añadida feature `is_unranked` (∈ {-1,0,1}) para que el modelo distinga wildcards. FEATURES pasa de 6 a 8 elementos.
- [x] **I3 · ELO híbrido 50/50 arbitrario.** ✅ Resuelto. `elo.py` emite `elo_winner/loser_general` y `elo_winner/loser_sup` como columnas separadas. `diff_elo` reemplazada por `diff_elo_general` + `diff_elo_sup`; el GBM aprende el peso. Reentrenado con los 4 modelos.
```

- [ ] **Step 6: Commit final**

```bash
git add modelo_atp.pkl modelos_atp.pkl metrics_atp.pkl stats_jugadores.pkl docs/ROADMAP.md
git commit -m "feat(I2+I3): reentrenamiento con 8 features — ELO separado, rank cap, is_unranked"
```

---

## Self-Review

**Spec coverage:**
- I2 (rank=999): ✅ RANK_CAP=250 en Task 1/3/4; `is_unranked` en Task 1/3/4.
- I3 (ELO aprendido): ✅ `diff_elo_general`+`diff_elo_sup` en FEATURES Task 1; columnas en elo.py Task 2; data_processing Task 3; app.py Task 4.
- Reentrenamiento: ✅ Task 5.
- Tests TDD: ✅ cada tarea tiene test-fail → implement → test-pass → commit.

**Placeholder scan:** Ninguno detectado.

**Type consistency:**
- `RANK_CAP` definido en Task 1, consumido en Task 3 y 4 ✅
- `diff_elo_general`, `diff_elo_sup`, `is_unranked` definidos en FEATURES Task 1, usados en Tasks 3/4 ✅
- `elo_winner_general`, `elo_loser_general`, `elo_winner_sup`, `elo_loser_sup` producidos en Task 2, consumidos en Task 3 ✅
