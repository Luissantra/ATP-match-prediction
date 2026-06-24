# Design: Engineering Improvements (Pasos 3-7)
**Date:** 2026-06-24  
**Scope:** Vectorización, Flask, requirements.txt, modularización main.py, tests ELO math

---

## Paso 3 — Vectorizar `data_processing.py`

Reemplazar el loop `for i in range(len(df))` con operaciones vectorizadas numpy/pandas.
`shuffle_mask` ya es un array booleano — `np.where` aplica directamente sobre columnas completas.

```python
elo_a  = np.where(shuffle, df['elo_winner'],        df['elo_loser'])
elo_b  = np.where(shuffle, df['elo_loser'],         df['elo_winner'])
rank_a = np.where(shuffle, df['winner_rank'],       df['loser_rank'])
rank_b = np.where(shuffle, df['loser_rank'],        df['winner_rank'])
age_a  = np.where(shuffle, df['winner_age'],        df['loser_age'])
age_b  = np.where(shuffle, df['loser_age'],         df['winner_age'])
h2h_a  = np.where(shuffle, df['h2h_winner_ratio'],  df['h2h_loser_ratio'])
h2h_b  = np.where(shuffle, df['h2h_loser_ratio'],   df['h2h_winner_ratio'])
form_a = np.where(shuffle, df['form_winner'],       df['form_loser'])
form_b = np.where(shuffle, df['form_loser'],        df['form_winner'])
```

Los 16 tests existentes verifican el output — no cambian.

---

## Paso 4 — Migrar `app.py` a Flask

Dependencias: `flask`, `flask-cors`.

Estructura resultante:
- `cargar_modelo()` se llama una vez al arrancar la app via `with app.app_context()`
- `/api/players` → `@app.get('/api/players')`
- `/api/predict` → `@app.get('/api/predict')`
- Estáticos servidos por `send_from_directory('static', filename)` y `render_template('index.html')`
- CORS con `flask_cors.CORS(app)`
- Arranque: `app.run(port=8000)`

La clase `ATPPredictHandler` desaparece completamente. El estado global `modelo`, `elo_general`, etc. pasa a variables de módulo cargadas en startup (comportamiento idéntico al actual pero thread-safe con Flask dev server).

---

## Paso 5 — `requirements.txt`

Pin exacto de las dependencias del venv usadas en producción:

```
flask
flask-cors
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Con versiones exactas extraídas del venv (`pip freeze`).

---

## Paso 6 — Modularizar `main.py`

`main.py` pasa a ser un orquestador de ~45 líneas. Dos módulos nuevos:

**`src/train.py`**
```python
def entrenar_modelo(X_train, y_train, param_grid) -> GradientBoostingClassifier
```
Recibe datos y param_grid, retorna el mejor estimador de GridSearchCV con TimeSeriesSplit.

**`src/evaluate.py`**
```python
def evaluar_y_graficar(modelo, X_test, y_test, df_test, features) -> float
```
Genera los 3 plots (matriz confusión, feature importance, precisión por superficie), retorna accuracy.

`main.py` queda como: cargar datos → preparar features → llamar `entrenar_modelo` → llamar `evaluar_y_graficar` → exportar pkl.

---

## Paso 7 — Tests unitarios matemática ELO

Nuevos tests en `tests/test_elo_math.py` para `calcular_expectativa` y `actualizar_ratings`:

- Ratings iguales → expectativa = 0.5
- Diferencia +400 → expectativa ≈ 0.909
- Diferencia −400 → expectativa ≈ 0.091
- Suma cero: cambio_ganador + cambio_perdedor = 0
- Upset: ganador inesperado recibe más puntos que favorito esperado
- K=0 → ratings no cambian

---

## Archivos modificados/creados

| Archivo | Acción |
|---------|--------|
| `src/data_processing.py` | Vectorizar loop |
| `app.py` | Reescribir con Flask |
| `requirements.txt` | Crear |
| `src/train.py` | Crear |
| `src/evaluate.py` | Crear |
| `main.py` | Refactorizar como orquestador |
| `tests/test_elo_math.py` | Crear |
