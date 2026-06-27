# HuggingFace Deploy + Visual Polish — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Desplegar la app Flask en HuggingFace Spaces y pulir tres aspectos visuales: texturas de fondo por superficie, corrección del desbordamiento de barras OR, y nueva gráfica ELO multi-superficie.

**Architecture:** Dos subsistemas independientes. (A) Deploy: Dockerfile + metadata HF + port configurable. (B) Visual: dos fixes en CSS/JS y una nueva gráfica que requiere exponer más datos en el backend.

**Tech Stack:** Python/Flask, gunicorn, Docker, CSS custom properties, SVG vanilla JS.

## Global Constraints

- Python 3.11+ (imagen base Docker)
- HuggingFace Spaces: puerto 7860, variable `PORT` en env
- Sin frameworks JS: todo vanilla, patrón existente del proyecto
- Versionado de assets con `?v=N` — incrementar `v` en cada cambio de CSS/JS
- Commits atómicos por tarea; no PRs (merge directo a main)
- Tests: `python -m pytest -q` debe pasar tras cada tarea
- No añadir `Co-Authored-By` en commits

---

## Subsistema A — HuggingFace Deployment

### Task 1: Puerto configurable en `app.py` + `Dockerfile`

**Files:**
- Modify: `app.py:229-230`
- Create: `Dockerfile`
- Modify: `README.md` (añadir header YAML de HF Spaces)

**Interfaces:**
- Produces: imagen Docker funcional en puerto 7860; app sigue arrancando en 8000 con `python app.py` localmente

- [ ] **Step 1: Verificar tamaño de los modelos (no requieren LFS)**

```bash
du -sh models/*.pkl
```
Esperado: `modelos_atp.pkl` < 10 MB. Si supera, necesitaría Git LFS — descartado por ahora (son ~350 KB, seguro).

- [ ] **Step 2: Modificar `app.py` para leer puerto de env var**

En `app.py` líneas 229-230, reemplazar:
```python
if __name__ == '__main__':
    app.run(port=8000, debug=False)
```
por:
```python
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

- [ ] **Step 3: Crear `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

CMD gunicorn -w 2 -b 0.0.0.0:$PORT app:app
```

- [ ] **Step 4: Verificar build local**

```bash
docker build -t atp-test . && docker run --rm -p 7860:7860 atp-test
curl -s http://localhost:7860/api/players | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d), 'jugadores')"
```
Esperado: `880 jugadores` (o similar).

- [ ] **Step 5: Añadir header YAML a `README.md`**

Prepend al `README.md` existente:
```yaml
---
title: ATP Match Forecast
emoji: 🎾
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
```

- [ ] **Step 6: Verificar tests**

```bash
python -m pytest -q
```
Esperado: todos en verde (el cambio de puerto no afecta tests).

- [ ] **Step 7: Commit**

```bash
git add app.py Dockerfile README.md
git commit -m "feat: HuggingFace Spaces deployment (Docker, port configurable)"
```

---

## Subsistema B — Visual Polish

### Task 2: Texturas de fondo por superficie

**Files:**
- Modify: `static/style.css`

**Interfaces:**
- Produces: `.court-bg` con patrón distinto para `body.surface-hard`, `body.surface-clay`, `body.surface-grass`

Contexto: `.court-bg` es un `div` fixed de z-index 0 con background CSS actual (solo gradiente radial + líneas horizontales uniformes). Se reemplaza con patrones diferenciados por superficie, manteniendo la misma paleta de variables.

- [ ] **Step 1: Localizar el bloque `.court-bg` en style.css**

Líneas 53-63 aprox. El selector actual:
```css
.court-bg {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background:
        radial-gradient(120% 80% at 50% -10%, color-mix(in srgb, var(--accent) 18%, transparent) 0%, transparent 55%),
        repeating-linear-gradient(180deg, transparent 0 119px, var(--line-faint) 119px 120px);
    transition: background var(--t);
    opacity: 0.9;
}
```

- [ ] **Step 2: Reemplazar con base neutra + tres overrides**

Sustituir el bloque por:
```css
/* Base neutra (solo gradiente radial, sin líneas) */
.court-bg {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background: radial-gradient(120% 80% at 50% -10%, color-mix(in srgb, var(--accent) 18%, transparent) 0%, transparent 55%);
    transition: background var(--t);
    opacity: 0.9;
}

/* Hard: rejilla ortogonal tenue (pista dura pintada) */
body.surface-hard .court-bg {
    background:
        radial-gradient(120% 80% at 50% -10%, color-mix(in srgb, #2E9BE6 18%, transparent) 0%, transparent 55%),
        repeating-linear-gradient(90deg,  transparent 0 79px, rgba(244,244,236,0.045) 79px 80px),
        repeating-linear-gradient(180deg, transparent 0 79px, rgba(244,244,236,0.045) 79px 80px);
}

/* Clay: trama diagonal cruzada (textura de arena/arcilla) */
body.surface-clay .court-bg {
    background:
        radial-gradient(120% 80% at 50% -10%, color-mix(in srgb, #E0703A 20%, transparent) 0%, transparent 55%),
        repeating-linear-gradient(135deg, transparent 0 3px, rgba(224,112,58,0.07) 3px 6px),
        repeating-linear-gradient(45deg,  transparent 0 3px, rgba(224,112,58,0.05) 3px 6px);
}

/* Grass: franjas alternas (césped segado, efecto mower stripes) */
body.surface-grass .court-bg {
    background:
        radial-gradient(120% 80% at 50% -10%, color-mix(in srgb, #5BB85B 15%, transparent) 0%, transparent 55%),
        repeating-linear-gradient(90deg,
            transparent          0  24px,
            rgba(20,80,30,0.10) 24px 48px);
}
```

- [ ] **Step 3: Incrementar versión de asset en `index.html`**

Cambiar `?v=5` → `?v=6` en la línea de `style.css`:
```html
<link rel="stylesheet" href="/static/style.css?v=6">
```

- [ ] **Step 4: Test visual en el servidor**

Con el servidor corriendo (ya arrancado), recargar `http://localhost:8000` y cambiar superficie entre Dura/Tierra/Césped. Verificar que el fondo cambia visiblemente.

- [ ] **Step 5: Tests automáticos**

```bash
python -m pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add static/style.css templates/index.html
git commit -m "feat(ui): texturas de fondo diferenciadas por superficie (rejilla/trama/franjas)"
```

---

### Task 3: Corregir desbordamiento de barras OR en panel de coeficientes

**Files:**
- Modify: `static/script.js:283-294`

**Contexto del bug:** En `renderModelInfo()`, `pct = (Math.abs(v.coef) / maxAbs) * 100`. Las barras `.ffill.toward-a` tienen `right: 50%; width: {pct}%`. Con pct=100, el elemento tiene `right: 50%` y `width: 100%` del contenedor → su borde izquierdo queda en `50% - 100% = -50%` (FUERA del `.ftrack`). El factor bars usa correctamente `* 50` (ancho máximo = mitad del track).

**Interfaces:**
- Produces: barras OR que nunca superan la línea central del track

- [ ] **Step 1: Localizar el cálculo en `script.js`**

Buscar en `renderModelInfo` la línea:
```javascript
const pct = (Math.abs(v.coef) / maxAbs) * 100;
```

- [ ] **Step 2: Cambiar multiplicador a 50**

```javascript
const pct = (Math.abs(v.coef) / maxAbs) * 50;
```

- [ ] **Step 3: Añadir `overflow: hidden` al `.ftrack` como salvaguarda**

En `style.css`, en el selector `.fbar .ftrack`:
```css
.fbar .ftrack {
    position: relative;
    height: 22px;
    background: var(--line-faint);
    border-radius: 4px;
    overflow: hidden;  /* ← añadir */
}
```

- [ ] **Step 4: Incrementar versión JS en `index.html`**

Cambiar `?v=6` → `?v=7` en la línea de `script.js` (si ya se subió en Task 2, se aplica sobre esa). En realidad se debe subir AMBOS scripts a la misma versión. Si Task 2 ya subió style.css a v=6:
```html
<link rel="stylesheet" href="/static/style.css?v=6">
...
<script src="/static/format.js?v=6"></script>
<script src="/static/script.js?v=6"></script>
```

- [ ] **Step 5: Test visual**

En `http://localhost:8000`, predecir un partido, abrir "Detalle del modelo". Verificar que las barras OR no sobrepasan la línea central.

- [ ] **Step 6: Tests**

```bash
python -m pytest -q
```

- [ ] **Step 7: Commit**

```bash
git add static/script.js static/style.css templates/index.html
git commit -m "fix(ui): barras OR clamped a 50% del track (evita desbordamiento)"
```

---

### Task 4: Backend — exponer ELO por las 3 superficies en `/api/predict`

**Files:**
- Modify: `app.py:131-161` (función `_predecir_con`)

**Interfaces:**
- Produces: `player_a.elo_surfaces` y `player_b.elo_surfaces` en la respuesta JSON:
  ```json
  "elo_surfaces": {"Hard": 1850.1, "Clay": 1780.5, "Grass": 1900.2}
  ```

- [ ] **Step 1: Localizar `_predecir_con` en `app.py`**

Líneas 106-160. El dict de respuesta por jugador tiene actualmente `elo_general`, `elo_surface` (solo de la superficie elegida), `elo_hybrid`.

- [ ] **Step 2: Añadir `elo_surfaces` al dict de respuesta de cada jugador**

Dentro de `_predecir_con`, antes de construir el return, añadir:
```python
SURFACES = ('Hard', 'Clay', 'Grass')
elo_surfaces_a = {s: round(elo_superficie.get(s, {}).get(player_a, 1500.0), 1) for s in SURFACES}
elo_surfaces_b = {s: round(elo_superficie.get(s, {}).get(player_b, 1500.0), 1) for s in SURFACES}
```

Y en el return dict, dentro de `"player_a": {...}` añadir:
```python
"elo_surfaces": elo_surfaces_a,
```
Y dentro de `"player_b": {...}`:
```python
"elo_surfaces": elo_surfaces_b,
```

- [ ] **Step 3: Verificar con curl**

```bash
curl -s "http://localhost:8000/api/predict?player_a=Jannik+Sinner&player_b=Carlos+Alcaraz&surface=Clay" | python3 -m json.tool | grep -A 5 elo_surfaces
```
Esperado: tres superficies con valores ELO distintos para cada jugador.

- [ ] **Step 4: Tests**

```bash
python -m pytest -q
```
Si hay tests de `/api/predict` que comparan el JSON exactamente, pueden necesitar actualización para incluir `elo_surfaces`. Revisar `tests/`.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat(api): exponer elo_surfaces (Hard/Clay/Grass) en /api/predict"
```

---

### Task 5: Frontend — gráfica ELO por superficie

**Files:**
- Modify: `templates/index.html` (añadir contenedor de la gráfica en `#results`)
- Modify: `static/script.js` (función `renderEloChart`)
- Modify: `static/style.css` (estilos de la gráfica)

**Interfaces:**
- Consumes: `res.player_a.elo_surfaces`, `res.player_b.elo_surfaces` (producido por Task 4)
- Produces: gráfica HTML/CSS de barras agrupadas (3 superficies × 2 jugadores), integrada en la sección de resultados entre la cancha y el bloque "Diferencias por factor"

**Diseño:** Barras horizontales agrupadas. Por cada superficie (Hard / Tierra / Césped): dos barras cortas, A (blanco) y B (color de acento de esa superficie). Escala relativa al rango min-max entre todos los valores mostrados. Datos como etiquetas al lado derecho. Sin librerías externas.

- [ ] **Step 1: Añadir contenedor en `index.html`**

En la sección `<!-- RESULTADOS -->`, después del bloque `<!-- Ganador -->` y antes de `<!-- Por qué: diferencias por factor -->`, añadir:

```html
<!-- Gráfica ELO por superficie -->
<div class="elo-chart">
    <div class="block-head">
        <h3>ELO por superficie</h3>
        <p class="block-sub">Comparativa de ambos jugadores en las tres superficies del circuito.</p>
    </div>
    <div id="elo-chart-body"></div>
</div>
```

- [ ] **Step 2: Añadir estilos en `style.css`**

```css
/* ---- Gráfica ELO por superficie ---- */
.elo-chart { /* usa .block-head ya existente */ }

.elo-surface-group {
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin-bottom: 18px;
}
.elo-surface-label {
    font-family: var(--mono);
    font-size: 0.66rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-faint);
    margin-bottom: 4px;
}
.elo-bar-row {
    display: grid;
    grid-template-columns: 60px 1fr 70px;
    align-items: center;
    gap: 10px;
}
.elo-bar-name {
    font-size: 0.78rem;
    color: var(--text-mut);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.elo-bar-track {
    height: 14px;
    background: var(--line-faint);
    border-radius: 3px;
    overflow: hidden;
}
.elo-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.7s var(--ease);
}
.elo-bar-fill.player-a { background: var(--line); opacity: 0.75; }
.elo-bar-fill.player-b { background: var(--accent); opacity: 0.9; transition: width 0.7s var(--ease), background var(--t); }
.elo-bar-val {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-mut);
    text-align: right;
    white-space: nowrap;
}
```

- [ ] **Step 3: Añadir función `renderEloChart` en `script.js`**

Añadir DENTRO del IIFE, justo antes del cierre `})();`, la función:

```javascript
function renderEloChart(a, b) {
    const container = document.getElementById('elo-chart-body');
    container.innerHTML = '';

    const SURF_META = [
        { key: 'Hard',  label: 'DURA',    accent: '#2E9BE6' },
        { key: 'Clay',  label: 'TIERRA',  accent: '#E0703A' },
        { key: 'Grass', label: 'CÉSPED',  accent: '#5BB85B' },
    ];

    // Rango global para escala común
    const allVals = SURF_META.flatMap(s => [a.elo_surfaces[s.key], b.elo_surfaces[s.key]]);
    const minElo = Math.min(...allVals) - 50;
    const maxElo = Math.max(...allVals) + 50;
    const range = maxElo - minElo || 1;

    SURF_META.forEach(({ key, label, accent }) => {
        const eloA = a.elo_surfaces[key];
        const eloB = b.elo_surfaces[key];
        const pctA = ((eloA - minElo) / range) * 100;
        const pctB = ((eloB - minElo) / range) * 100;

        const group = document.createElement('div');
        group.className = 'elo-surface-group';
        group.innerHTML = `
            <div class="elo-surface-label">${label}</div>
            <div class="elo-bar-row">
                <span class="elo-bar-name">${a.name.split(' ').slice(-1)[0]}</span>
                <div class="elo-bar-track">
                    <div class="elo-bar-fill player-a" style="width:${pctA.toFixed(1)}%"></div>
                </div>
                <span class="elo-bar-val">${eloA.toFixed(0)}</span>
            </div>
            <div class="elo-bar-row">
                <span class="elo-bar-name">${b.name.split(' ').slice(-1)[0]}</span>
                <div class="elo-bar-track">
                    <div class="elo-bar-fill player-b" style="width:${pctB.toFixed(1)}%; background:${accent}"></div>
                </div>
                <span class="elo-bar-val">${eloB.toFixed(0)}</span>
            </div>`;
        container.appendChild(group);
    });
}
```

- [ ] **Step 4: Llamar a `renderEloChart` desde `renderResults`**

En `renderResults(res)`, después de `fillCompare('b', b)` y antes de `markUnknown(...)`, añadir:
```javascript
renderEloChart(res.player_a, res.player_b);
```

- [ ] **Step 5: Actualizar versión de assets en `index.html`**

`style.css?v=6` → `?v=7`, `format.js?v=6` → `?v=7`, `script.js?v=6` → `?v=7`.

(Ajustar el número si las tareas anteriores ya incrementaron la versión.)

- [ ] **Step 6: Test visual**

Predecir `Sinner vs Alcaraz` en `Clay`. Verificar que la gráfica ELO por superficie aparece entre el bloque "Favorito" y "Diferencias por factor", con tres grupos de dos barras cada uno, y que el acento de color de cada barra de jugador B varía por superficie.

- [ ] **Step 7: Tests**

```bash
python -m pytest -q
node --test tests/format.test.mjs
```

- [ ] **Step 8: Commit**

```bash
git add templates/index.html static/script.js static/style.css
git commit -m "feat(ui): gráfica ELO multi-superficie en panel de resultados"
```

---

## Post-implementación: merge a main

```bash
git checkout main
git merge --no-ff claude/agitated-wilbur-22f75d -m "feat: HF deploy + visual polish (texturas, OR fix, gráfica ELO)"
git push origin main
```

---

## Self-Review

**Spec coverage:**
- HuggingFace deploy → Task 1 ✓
- Texturas por superficie → Task 2 ✓  
- OR no se salen de la barra → Task 3 ✓
- Gráfica relevante (ELO multi-superficie) → Tasks 4+5 ✓

**Dependencias entre tareas:**
- Task 5 depende de Task 4 (necesita `elo_surfaces` en respuesta API)
- Tasks 2, 3, 1 son independientes entre sí y respecto a Tasks 4-5
- Orden sugerido: 1 → 2 → 3 → 4 → 5

**Riesgos:**
- Si tests de `/api/predict` (en `tests/`) comparan estructura JSON exacta, Task 4 puede romperlos. Revisar antes de commitear.
- El número de versión de assets (`?v=N`) debe ser consistente entre `style.css`, `format.js` y `script.js` en index.html.
- Docker: si HF Spaces requiere un usuario no-root, añadir `USER nobody` al Dockerfile.
