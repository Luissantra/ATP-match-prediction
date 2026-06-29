# Visual Polish + HuggingFace Deploy — Implementation Plan (orden visual-first)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pulir tres aspectos visuales de la app Flask y luego desplegarla en HuggingFace Spaces. Orden visual-first: cerrar el producto presentable antes de publicarlo.

**Architecture:** Dos subsistemas. (A) Visual: dos fixes CSS/JS y una nueva gráfica que requiere exponer más datos en el backend. (B) Deploy: Dockerfile + metadata HF + port configurable.

**Tech Stack:** Python/Flask, gunicorn, Docker, CSS custom properties, vanilla JS.

## Orden de ejecución (sugerido)

`V2 → V1 → V3 → D1 → D2`

| Tarea | Qué | Riesgo | Por qué aquí |
|-------|-----|--------|--------------|
| **V2** (Task 1) | Fix barras OR (clamp 50%) | nulo | bug visual real, fix chico, primero |
| **V1** (Task 2) | Texturas de fondo por superficie | bajo | cosmético puro |
| **V3** (Task 3+4) | Gráfica ELO multi-superficie | medio | toca backend + frontend |
| **D1** (Task 5a) | Dockerfile + port configurable | bajo | deploy tras producto pulido |
| **D2** (Task 5b) | README.md header YAML HF | nulo | metadata final |

## Global Constraints

- Python 3.11+ (imagen base Docker)
- HuggingFace Spaces: puerto 7860, variable `PORT` en env
- Sin frameworks JS: todo vanilla, patrón existente del proyecto
- Versionado de assets con `?v=N` — base actual `v=5`. V2→`v6`, V1→`v7`, V3→`v8`. Subir SIEMPRE los tres (style.css, format.js, script.js) a la misma versión.
- Commits atómicos por tarea; no PRs (merge directo a main)
- Tests: `python -m pytest -q` debe pasar tras cada tarea
- No añadir `Co-Authored-By` en commits

---

## Subsistema A — Visual Polish

### Task 1 (V2): Corregir desbordamiento de barras OR en panel de coeficientes

**Files:**
- Modify: `static/script.js` (cálculo `pct` en `renderModelInfo`)
- Modify: `static/style.css` (`.fbar .ftrack`)
- Modify: `templates/index.html` (asset version → `v=6`)

**Contexto del bug:** En `renderModelInfo()`, `pct = (Math.abs(v.coef) / maxAbs) * 100`. Las barras `.ffill.toward-a` tienen `right: 50%; width: {pct}%`. Con pct=100 el borde izquierdo queda en `50% - 100% = -50%` (FUERA del `.ftrack`). Las factor bars ya usan `* 50` (ancho máximo = mitad del track).

**Interfaces:**
- Produces: barras OR que nunca superan la línea central del track

- [ ] **Step 1: Localizar el cálculo en `script.js`**

Buscar en `renderModelInfo`:
```javascript
const pct = (Math.abs(v.coef) / maxAbs) * 100;
```

- [ ] **Step 2: Cambiar multiplicador a 50**

```javascript
const pct = (Math.abs(v.coef) / maxAbs) * 50;
```

- [ ] **Step 3: Añadir `overflow: hidden` al `.ftrack` como salvaguarda**

En `style.css`, selector `.fbar .ftrack`:
```css
.fbar .ftrack {
    position: relative;
    height: 22px;
    background: var(--line-faint);
    border-radius: 4px;
    overflow: hidden;  /* ← añadir */
}
```

- [ ] **Step 4: Subir versión de assets a `v=6` en `index.html`**

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

### Task 2 (V1): Texturas de fondo por superficie

**Files:**
- Modify: `static/style.css`
- Modify: `templates/index.html` (asset version → `v=7`)

**Interfaces:**
- Produces: `.court-bg` con patrón distinto para `body.surface-hard`, `body.surface-clay`, `body.surface-grass`

Contexto: `.court-bg` es un `div` fixed z-index 0 con background CSS actual (gradiente radial + líneas horizontales uniformes). Se reemplaza con patrones diferenciados, manteniendo la paleta de variables.

- [ ] **Step 1: Localizar el bloque `.court-bg` en style.css**

Selector actual (~líneas 53-63):
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

/* Clay: trama diagonal cruzada (textura de arcilla) */
body.surface-clay .court-bg {
    background:
        radial-gradient(120% 80% at 50% -10%, color-mix(in srgb, #E0703A 20%, transparent) 0%, transparent 55%),
        repeating-linear-gradient(135deg, transparent 0 3px, rgba(224,112,58,0.07) 3px 6px),
        repeating-linear-gradient(45deg,  transparent 0 3px, rgba(224,112,58,0.05) 3px 6px);
}

/* Grass: franjas alternas (mower stripes) */
body.surface-grass .court-bg {
    background:
        radial-gradient(120% 80% at 50% -10%, color-mix(in srgb, #5BB85B 15%, transparent) 0%, transparent 55%),
        repeating-linear-gradient(90deg,
            transparent          0  24px,
            rgba(20,80,30,0.10) 24px 48px);
}
```

- [ ] **Step 3: Subir versión de assets a `v=7` en `index.html`**

```html
<link rel="stylesheet" href="/static/style.css?v=7">
...
<script src="/static/format.js?v=7"></script>
<script src="/static/script.js?v=7"></script>
```

- [ ] **Step 4: Test visual**

Servidor corriendo, recargar `http://localhost:8000`, cambiar superficie Dura/Tierra/Césped. Verificar que el fondo cambia visiblemente.

- [ ] **Step 5: Tests**

```bash
python -m pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add static/style.css templates/index.html
git commit -m "feat(ui): texturas de fondo diferenciadas por superficie (rejilla/trama/franjas)"
```

---

### Task 3 (V3-backend): exponer ELO por las 3 superficies en `/api/predict`

**Files:**
- Modify: `app.py` (función `_predecir_con`)

**Interfaces:**
- Produces: `player_a.elo_surfaces` y `player_b.elo_surfaces` en la respuesta JSON:
  ```json
  "elo_surfaces": {"Hard": 1850.1, "Clay": 1780.5, "Grass": 1900.2}
  ```

- [ ] **Step 1: Localizar `_predecir_con` en `app.py`** (~líneas 106-160). El dict por jugador tiene `elo_general`, `elo_surface` (solo de la superficie elegida), `elo_hybrid`.

- [ ] **Step 2: Añadir `elo_surfaces` al dict de cada jugador**

Antes del return:
```python
SURFACES = ('Hard', 'Clay', 'Grass')
elo_surfaces_a = {s: round(elo_superficie.get(s, {}).get(player_a, 1500.0), 1) for s in SURFACES}
elo_surfaces_b = {s: round(elo_superficie.get(s, {}).get(player_b, 1500.0), 1) for s in SURFACES}
```
En `"player_a": {...}` añadir `"elo_surfaces": elo_surfaces_a,` y en `"player_b": {...}` `"elo_surfaces": elo_surfaces_b,`.

- [ ] **Step 3: Verificar con curl**

```bash
curl -s "http://localhost:8000/api/predict?player_a=Jannik+Sinner&player_b=Carlos+Alcaraz&surface=Clay" | python3 -m json.tool | grep -A 5 elo_surfaces
```
Esperado: tres superficies con valores ELO distintos por jugador.

- [ ] **Step 4: Tests** — revisar `tests/` por aserciones de estructura JSON exacta de `/api/predict`; actualizar si comparan el dict completo.

```bash
python -m pytest -q
```

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat(api): exponer elo_surfaces (Hard/Clay/Grass) en /api/predict"
```

---

### Task 4 (V3-frontend): gráfica ELO por superficie

**Files:**
- Modify: `templates/index.html` (contenedor + asset version → `v=8`)
- Modify: `static/script.js` (`renderEloChart`)
- Modify: `static/style.css` (estilos)

**Interfaces:**
- Consumes: `res.player_a.elo_surfaces`, `res.player_b.elo_surfaces` (Task 3)
- Produces: barras agrupadas (3 superficies × 2 jugadores) entre "Favorito" y "Diferencias por factor"

**Diseño:** Barras horizontales agrupadas. Por superficie: dos barras, A (blanco) y B (acento de esa superficie). Escala relativa al rango min-max global. Sin librerías externas.

- [ ] **Step 1: Añadir contenedor en `index.html`**

En `<!-- RESULTADOS -->`, tras `<!-- Ganador -->` y antes de `<!-- Por qué: diferencias por factor -->`:
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

- [ ] **Step 3: Añadir `renderEloChart` en `script.js`** (DENTRO del IIFE, antes del cierre `})();`)

```javascript
function renderEloChart(a, b) {
    const container = document.getElementById('elo-chart-body');
    container.innerHTML = '';

    const SURF_META = [
        { key: 'Hard',  label: 'DURA',    accent: '#2E9BE6' },
        { key: 'Clay',  label: 'TIERRA',  accent: '#E0703A' },
        { key: 'Grass', label: 'CÉSPED',  accent: '#5BB85B' },
    ];

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

- [ ] **Step 4: Llamar desde `renderResults(res)`** — tras `fillCompare('b', b)` y antes de `markUnknown(...)`:
```javascript
renderEloChart(res.player_a, res.player_b);
```

- [ ] **Step 5: Subir versión de assets a `v=8` en `index.html`**

```html
<link rel="stylesheet" href="/static/style.css?v=8">
...
<script src="/static/format.js?v=8"></script>
<script src="/static/script.js?v=8"></script>
```

- [ ] **Step 6: Test visual** — `Sinner vs Alcaraz` en `Clay`. Verificar gráfica entre "Favorito" y "Diferencias por factor", tres grupos de dos barras, acento de B variando por superficie.

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

## Subsistema B — HuggingFace Deployment

### Task 5 (D1 + D2): Dockerfile + port configurable + README HF

**Files:**
- Modify: `app.py` (bloque `__main__`, ~líneas 229-230)
- Create: `Dockerfile`
- Modify: `README.md` (header YAML HF Spaces)

**Interfaces:**
- Produces: imagen Docker funcional en puerto 7860; app sigue arrancando en 8000 con `python app.py` local

- [ ] **Step 1 (D1): Verificar tamaño de modelos (no requieren LFS)**

```bash
du -sh models/*.pkl
```
Esperado: `modelos_atp.pkl` < 10 MB (son ~350 KB, seguro). Si supera, necesitaría Git LFS.

- [ ] **Step 2 (D1): `app.py` lee puerto de env var**

Reemplazar:
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
(Verificar que `os` ya está importado.)

- [ ] **Step 3 (D1): Crear `Dockerfile`**

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

- [ ] **Step 4 (D1): Verificar build local**

```bash
docker build -t atp-test . && docker run --rm -p 7860:7860 atp-test
curl -s http://localhost:7860/api/players | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d), 'jugadores')"
```

- [ ] **Step 5 (D2): Header YAML en `README.md`** (prepend al existente)

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

- [ ] **Step 6: Tests**

```bash
python -m pytest -q
```

- [ ] **Step 7: Commit**

```bash
git add app.py Dockerfile README.md
git commit -m "feat: HuggingFace Spaces deployment (Docker, port configurable)"
```

---

## Post-implementación: merge a main

```bash
git checkout main
git merge --no-ff claude/great-hoover-3f1a50 -m "feat: visual polish + HF deploy (OR fix, texturas, gráfica ELO, Docker)"
git push origin main
```

---

## Self-Review

**Spec coverage:**
- OR no se salen de la barra (V2) → Task 1 ✓
- Texturas por superficie (V1) → Task 2 ✓
- Gráfica ELO multi-superficie (V3) → Tasks 3+4 ✓
- HuggingFace deploy (D1+D2) → Task 5 ✓

**Dependencias:**
- Task 4 depende de Task 3 (necesita `elo_surfaces` en respuesta API)
- Tasks 1, 2, 5 independientes entre sí y de 3-4
- Orden ejecución: `V2 (1) → V1 (2) → V3 (3→4) → D1/D2 (5)`

**Riesgos:**
- Tests de `/api/predict` que comparen JSON exacto pueden romper en Task 3. Revisar antes de commit.
- Versión de assets (`?v=N`) consistente entre style.css, format.js, script.js. Base `v=5` → V2 `v6` → V1 `v7` → V3 `v8`.
- Docker: si HF Spaces exige usuario no-root, añadir `USER nobody` al Dockerfile.
