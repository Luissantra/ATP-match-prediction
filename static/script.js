// =========================================================================
// ATP MATCH FORECAST — capa de UI (estado, fetch, render).
// Funciones puras de formato en static/format.js (window.ATPFormat).
// IIFE: aísla el scope. format.js expone normalizeFactor/etc como funciones
// globales; sin este wrapper, el destructuring de abajo colisionaría con esos
// identificadores globales y el script entero sería rechazado.
// =========================================================================
(function () {
'use strict';
const { normalizeFactor, formatDiff, formatRank } = window.ATPFormat;

// Factores a graficar como barras divergentes. `dir` = +1 si un valor positivo
// (A−B) favorece a A; -1 si lo favorece B (ranking: nº menor es mejor; edad:
// más joven se toma como ventaja). Heurística de orientación, no peso del modelo.
const FACTORS = [
    { key: 'diff_elo_general', name: 'ELO general',    unit: 'pts',  dec: 1, dir: 1 },
    { key: 'diff_elo_sup',     name: 'ELO superficie', unit: 'pts',  dec: 1, dir: 1 },
    { key: 'diff_rank',        name: 'Ranking',        unit: 'pos',  dec: 0, dir: -1 },
    { key: 'diff_age',         name: 'Edad',           unit: 'años', dec: 1, dir: -1 },
];

// ----- Estado -----
let players = [];
let selA = null, selB = null;
let surface = 'Hard';

// ----- DOM -----
const body = document.body;
const inputA = document.getElementById('player-a-input');
const inputB = document.getElementById('player-b-input');
const listA = document.getElementById('player-a-list');
const listB = document.getElementById('player-b-list');
const cardA = document.getElementById('card-a');
const cardB = document.getElementById('card-b');
const btn = document.getElementById('predict-btn');
const results = document.getElementById('results');
const formError = document.getElementById('form-error');

function init() {
    fetchPlayers();
    setupPills('surface-group', (v) => {
        surface = v;
        body.className = `surface-${v.toLowerCase()}`;
    });
    setupSearch(inputA, listA, 'A');
    setupSearch(inputB, listB, 'B');
    btn.addEventListener('click', runPrediction);
    loadDisclaimer();
    setupTournamentModal();
}

// Banner de vigencia (R4): fecha de corte servida por /api/model, no hardcodeada.
async function loadDisclaimer() {
    const el = document.getElementById('model-disclaimer');
    if (!el) return;
    try {
        const info = await fetch('/api/model').then((r) => r.json());
        if (info.trained_through == null || info.tested_on == null) return;
        el.textContent =
            `Modelo entrenado con datos hasta ${info.trained_through} ` +
            `(test ${info.tested_on}). Las predicciones no reflejan lesiones, ` +
            `retiradas ni forma reciente fuera del ELO.`;
        el.hidden = false;
    } catch (e) {
        // Sin red: el banner queda oculto (no bloquea la app).
    }
}

// El DOM ya puede estar listo cuando corre este script (al final del body):
// si esperamos a DOMContentLoaded y ya disparó, init nunca se ejecutaría.
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

async function fetchPlayers() {
    try {
        const r = await fetch('/api/players');
        if (!r.ok) throw new Error('players');
        players = await r.json();
    } catch (e) {
        console.error('Error cargando jugadores:', e);
    }
}

// ----- Selectores pill (radio) -----
function setupPills(groupId, onChange) {
    const group = document.getElementById(groupId);
    group.querySelectorAll('input[type="radio"]').forEach((radio) => {
        radio.addEventListener('change', () => {
            group.querySelectorAll('.pill').forEach((p) => p.classList.remove('active'));
            radio.closest('.pill').classList.add('active');
            onChange(radio.value);
        });
    });
}

// ----- Buscador con autocompletado -----
function setupSearch(input, list, side) {
    let activeIndex = -1;

    function closeList() {
        list.style.display = 'none';
        input.setAttribute('aria-expanded', 'false');
        input.removeAttribute('aria-activedescendant');
        activeIndex = -1;
    }

    input.addEventListener('input', () => {
        const q = input.value.trim().toLowerCase();
        list.innerHTML = '';
        activeIndex = -1;
        input.removeAttribute('aria-activedescendant');

        if (q.length < 2) { closeList(); clearSelection(side); return; }

        const matches = players.filter((p) => p.name.toLowerCase().includes(q)).slice(0, 8);
        if (matches.length === 0) {
            list.innerHTML = '<div class="ac-item no-hover" style="cursor:default"><span class="ac-name" style="color:var(--text-faint)">Sin resultados</span></div>';
            list.style.display = 'block';
            input.setAttribute('aria-expanded', 'true');
            return;
        }
        matches.forEach((p, idx) => {
            const item = document.createElement('div');
            item.className = 'ac-item';
            item.id = `ac-item-${side}-${idx}`;
            item.setAttribute('role', 'option');
            item.innerHTML = `<span class="ac-name">${p.name}</span>
                <span class="ac-meta">${formatRank(p.rank)} · ELO ${Math.round(p.elo)}</span>`;
            item.addEventListener('click', () => {
                input.value = p.name;
                closeList();
                selectPlayer(p, side);
            });
            list.appendChild(item);
        });
        list.style.display = 'block';
        input.setAttribute('aria-expanded', 'true');
    });

    input.addEventListener('focus', () => {
        if (input.value.trim().length >= 2 && list.children.length) {
            list.style.display = 'block';
            input.setAttribute('aria-expanded', 'true');
        }
    });

    input.addEventListener('keydown', (e) => {
        const items = list.querySelectorAll('.ac-item:not(.no-hover)');
        if (list.style.display !== 'block' || items.length === 0) {
            if (e.key === 'ArrowDown' && input.value.trim().length >= 2 && list.children.length) {
                list.style.display = 'block';
                input.setAttribute('aria-expanded', 'true');
            }
            return;
        }

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            activeIndex = (activeIndex + 1) % items.length;
            updateActiveItem(items);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            activeIndex = (activeIndex - 1 + items.length) % items.length;
            updateActiveItem(items);
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (activeIndex > -1 && items[activeIndex]) {
                items[activeIndex].click();
            }
        } else if (e.key === 'Escape') {
            closeList();
        }
    });

    function updateActiveItem(items) {
        items.forEach((item, idx) => {
            if (idx === activeIndex) {
                item.classList.add('ac-active');
                item.setAttribute('aria-selected', 'true');
                input.setAttribute('aria-activedescendant', item.id);
                item.scrollIntoView({ block: 'nearest' });
            } else {
                item.classList.remove('ac-active');
                item.removeAttribute('aria-selected');
            }
        });
    }

    document.addEventListener('click', (e) => {
        if (e.target !== input) closeList();
    });
}

function selectPlayer(p, side) {
    if (side === 'A') selA = p; else selB = p;
    fillCard(side === 'A' ? cardA : cardB, p);
    validate();
}

function clearSelection(side) {
    if (side === 'A') selA = null; else selB = null;
    resetCard(side === 'A' ? cardA : cardB);
    validate();
}

function fillCard(card, p) {
    card.classList.add('filled');
    card.classList.remove('unknown');
    card.querySelector('.mini-name').textContent = p.name;
    card.querySelector('.rank-val').textContent = formatRank(p.rank);
    card.querySelector('.age-val').textContent = `${p.age}`;
    card.querySelector('.elo-val').textContent = Math.round(p.elo);
    card.querySelector('.unknown-flag').hidden = true;
}

function resetCard(card) {
    card.classList.remove('filled', 'unknown');
    card.querySelector('.mini-name').textContent = '—';
    card.querySelectorAll('.rank-val,.age-val,.elo-val').forEach((el) => (el.textContent = '—'));
    card.querySelector('.unknown-flag').hidden = true;
}

function validate() {
    const ok = selA && selB && selA.name !== selB.name;
    btn.disabled = !ok;
    formError.hidden = true;
    if (selA && selB && selA.name === selB.name) {
        formError.hidden = false;
        formError.textContent = 'Elige dos jugadores distintos.';
    }
}

// ----- Predicción -----
async function runPrediction() {
    if (!selA || !selB) return;
    btn.classList.add('loading');
    btn.disabled = true;

    const warnEl = document.getElementById('slow-load-warning');
    if (warnEl) {
        warnEl.hidden = true;
        warnEl.textContent = '';
    }

    let slowTimer = setTimeout(() => {
        if (warnEl) {
            warnEl.textContent = 'El servidor de Hugging Face está despertando de su inactividad (esto puede tomar entre 15 y 20 segundos)...';
            warnEl.hidden = false;
        }
    }, 1500);

    const params = `player_a=${encodeURIComponent(selA.name)}&player_b=${encodeURIComponent(selB.name)}&surface=${surface}`;
    try {
        const r = await fetch(`/api/predict?${params}`);
        if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'Fallo en la predicción'); }
        const data = await r.json();
        renderResults(data);
        loadModelInfo();
    } catch (e) {
        formError.hidden = false;
        formError.textContent = e.message;
    } finally {
        clearTimeout(slowTimer);
        if (warnEl) warnEl.hidden = true;
        btn.classList.remove('loading');
        validate();
    }
}

function renderResults(res) {
    results.classList.remove('hidden');
    const a = res.player_a, b = res.player_b;

    // Cancha (firma): territorio = probabilidad, red según prob A
    const winA = res.predicted_winner === a.name;

    const cnA = document.getElementById('court-name-a');
    const cnB = document.getElementById('court-name-b');
    cnA.textContent = a.name;
    cnB.textContent = b.name;
    cnA.classList.toggle('cn-winner', winA);
    cnB.classList.toggle('cn-winner', !winA);

    const halfA = document.getElementById('half-a');
    const halfB = document.getElementById('half-b');
    halfA.style.width = `${a.prob_victory}%`;
    halfB.style.width = `${b.prob_victory}%`;
    halfA.classList.toggle('is-winner', winA);
    halfA.classList.toggle('is-loser', !winA);
    halfB.classList.toggle('is-winner', !winA);
    halfB.classList.toggle('is-loser', winA);

    document.getElementById('court-net').style.left = `${a.prob_victory}%`;

    const pctA = document.getElementById('court-pct-a');
    const pctB = document.getElementById('court-pct-b');
    pctA.textContent = `${a.prob_victory}%`;
    pctB.textContent = `${b.prob_victory}%`;
    pctA.classList.toggle('is-winner', winA);
    pctA.classList.toggle('is-loser', !winA);
    pctB.classList.toggle('is-winner', !winA);
    pctB.classList.toggle('is-loser', winA);

    // Ganador
    document.getElementById('winner-name').textContent = res.predicted_winner;
    const conf = winA ? a.prob_victory : b.prob_victory;
    document.getElementById('winner-conf').textContent = `Probabilidad estimada ${conf}% · modelo ${res.model_used || 'logreg'}`;

    // Eje + barras de factores
    document.getElementById('axis-name-a').textContent = a.name;
    document.getElementById('axis-name-b').textContent = b.name;
    renderFactors(res.features_debug);

    // Comparativa numérica
    fillCompare('a', a);
    fillCompare('b', b);

    // Gráfica ELO por superficie
    renderEloChart(res.player_a, res.player_b);

    // Marcas de desconocido
    markUnknown(cardA, a.unknown);
    markUnknown(cardB, b.unknown);

    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderFactors(dbg) {
    const wrap = document.getElementById('factor-bars');
    wrap.innerHTML = '';
    FACTORS.forEach((f) => {
        const raw = dbg[f.key];
        const advA = normalizeFactor(raw, f.key) * f.dir; // [-1,1], + = hacia A
        const pct = Math.abs(advA) * 50;
        const towardA = advA >= 0;
        const label = formatDiff(raw, f.unit, f.dec).trim();

        const row = document.createElement('div');
        row.className = 'fbar';
        row.innerHTML = `
            <span class="fname">${f.name}</span>
            <div class="ftrack">
                <div class="ffill ${towardA ? 'toward-a' : 'toward-b'}" style="width:${pct}%"></div>
                <span class="fval ${towardA ? 'on-a' : 'on-b'}">${label}</span>
            </div>`;
        wrap.appendChild(row);
    });
}

function fillCompare(side, p) {
    document.getElementById(`cmp-name-${side}`).textContent = p.name;
    document.getElementById(`cmp-hyb-${side}`).textContent = p.elo_hybrid.toFixed(1);
    document.getElementById(`cmp-gen-${side}`).textContent = p.elo_general.toFixed(1);
    document.getElementById(`cmp-sup-${side}`).textContent = p.elo_surface.toFixed(1);
    document.getElementById(`cmp-rank-${side}`).textContent = formatRank(p.rank);
    document.getElementById(`cmp-age-${side}`).textContent = `${p.age} años`;
}

function markUnknown(card, unknown) {
    card.classList.toggle('unknown', Boolean(unknown));
    card.querySelector('.unknown-flag').hidden = !unknown;
}

// ----- Detalle del modelo: métricas + coeficientes (explicabilidad) -----
const COEF_LABELS = {
    diff_elo_general: 'ELO general',
    diff_elo_sup: 'ELO superficie',
    diff_rank: 'Ranking',
    is_unranked: 'Sin ranking',
    diff_age: 'Edad',
};

async function loadModelInfo() {
    const bodyEl = document.getElementById('models-body');
    bodyEl.innerHTML = '<p class="models-loading">Cargando…</p>';
    const plotsLoading = document.getElementById('plots-loading');
    if (plotsLoading) plotsLoading.style.display = 'block';
    
    try {
        const info = await fetch('/api/model').then((r) => r.json());
        renderModelInfo(info);
        if (info.plots_data) {
            if (plotsLoading) plotsLoading.style.display = 'none';
            renderPlots(info.plots_data);
        }
    } catch (e) {
        bodyEl.innerHTML = '<p class="models-loading">No disponible.</p>';
        if (plotsLoading) plotsLoading.textContent = 'Gráficos no disponibles.';
    }
}

function renderPlots(data) {
    if (!window.Plotly) return;
    const fontColor = '#C0C0C0';
    const gridColor = 'rgba(255,255,255,0.1)';
    const bgColor = 'rgba(0,0,0,0)';

    const layoutBase = {
        paper_bgcolor: bgColor,
        plot_bgcolor: bgColor,
        font: { color: fontColor, family: 'Inter, system-ui, sans-serif', size: 12 },
        margin: { l: 50, r: 20, t: 50, b: 50 }
    };

    // 1. Matriz de confusión (Heatmap)
    if (data.confusion_matrix) {
        const cm = data.confusion_matrix;
        const z = [[cm[1][0], cm[1][1]], [cm[0][0], cm[0][1]]]; // Invertir Y para visualización correcta
        const traceCM = {
            z: z,
            x: ['Pred. Derrota A', 'Pred. Victoria A'],
            y: ['Real Victoria A', 'Real Derrota A'],
            type: 'heatmap',
            colorscale: [[0, 'rgba(46, 155, 230, 0.05)'], [1, 'rgba(46, 155, 230, 0.85)']],
            showscale: false,
            hoverinfo: 'z',
            hoverlabel: { bgcolor: '#2c3e50', font: { color: 'white' } }
        };
        const layoutCM = {
            ...layoutBase,
            title: { text: 'Matriz de Confusión', font: { color: '#FFFFFF', size: 16 } },
            xaxis: { title: 'Predicción del Modelo', showgrid: false },
            yaxis: { title: 'Estado Real', showgrid: false },
            annotations: []
        };
        for(let i=0; i<2; i++) {
            for(let j=0; j<2; j++) {
                const val = z[i][j];
                const total = z[i][0] + z[i][1];
                const pct = (val / total * 100).toFixed(1);
                layoutCM.annotations.push({
                    x: traceCM.x[j],
                    y: traceCM.y[i],
                    text: `<b>${val}</b><br>(${pct}%)`,
                    font: { color: val > 800 ? '#FFFFFF' : '#E0E0E0', size: 14 },
                    showarrow: false
                });
            }
        }
        Plotly.newPlot('plot-confusion', [traceCM], layoutCM, {displayModeBar: false, responsive: true});
    }

    // 2. Reliability Diagram (Scatter/Line)
    if (data.reliability) {
        const traceRel = {
            x: data.reliability.prob_pred,
            y: data.reliability.prob_true,
            mode: 'lines+markers',
            name: 'Modelo',
            line: { color: '#2E9BE6', width: 2 },
            marker: { size: 8, line: { color: '#FFFFFF', width: 1 } },
            hoverlabel: { bgcolor: '#2E9BE6', font: { color: 'white' } }
        };
        const tracePerf = {
            x: [0, 1],
            y: [0, 1],
            mode: 'lines',
            name: 'Perfecta',
            line: { color: 'rgba(255,255,255,0.3)', dash: 'dash', width: 2 },
            hoverinfo: 'skip'
        };
        const layoutRel = {
            ...layoutBase,
            title: { text: 'Calibración (Reliability Diagram)', font: { color: '#FFFFFF', size: 16 } },
            xaxis: { title: 'Probabilidad Predicha', range: [0, 1], gridcolor: gridColor, zerolinecolor: gridColor },
            yaxis: { title: 'Tasa Real Positivos', range: [0, 1], gridcolor: gridColor, zerolinecolor: gridColor },
            showlegend: true,
            legend: { orientation: 'h', x: 0.5, y: -0.25, xanchor: 'center' }
        };
        Plotly.newPlot('plot-reliability', [traceRel, tracePerf], layoutRel, {displayModeBar: false, responsive: true});
    }

    // 3. Curva ROC (Scatter/Line)
    if (data.roc_curve) {
        const traceROC = {
            x: data.roc_curve.fpr,
            y: data.roc_curve.tpr,
            mode: 'lines',
            name: `LogReg (AUC = ${data.roc_curve.auc.toFixed(4)})`,
            line: { color: '#27ae60', width: 2.5 },
            hoverlabel: { bgcolor: '#27ae60', font: { color: 'white' } }
        };
        const traceDiag = {
            x: [0, 1],
            y: [0, 1],
            mode: 'lines',
            name: 'Azar (AUC = 0.5000)',
            line: { color: 'rgba(255,255,255,0.3)', dash: 'dash', width: 1.5 },
            hoverinfo: 'skip'
        };
        const layoutROC = {
            ...layoutBase,
            title: { text: 'Curva ROC (Discriminación)', font: { color: '#FFFFFF', size: 16 } },
            xaxis: { title: 'Tasa de Falsos Positivos (FPR)', range: [0, 1], gridcolor: gridColor, zerolinecolor: gridColor },
            yaxis: { title: 'Tasa de Verdaderos Positivos (TPR)', range: [0, 1.05], gridcolor: gridColor, zerolinecolor: gridColor },
            showlegend: true,
            legend: { orientation: 'h', x: 0.5, y: -0.25, xanchor: 'center' }
        };
        Plotly.newPlot('plot-roc', [traceROC, traceDiag], layoutROC, {displayModeBar: false, responsive: true});
    }

    // 4. Histograma de Probabilidades
    if (data.histogram) {
        const trace0 = {
            x: data.histogram.class_0,
            type: 'histogram',
            name: 'Derrota A',
            marker: { color: '#E0703A' },
            opacity: 0.75,
            xbins: { start: 0, end: 1, size: 0.05 },
            hoverlabel: { bgcolor: '#E0703A', font: { color: 'white' } }
        };
        const trace1 = {
            x: data.histogram.class_1,
            type: 'histogram',
            name: 'Victoria A',
            marker: { color: '#5BB85B' },
            opacity: 0.75,
            xbins: { start: 0, end: 1, size: 0.05 },
            hoverlabel: { bgcolor: '#5BB85B', font: { color: 'white' } }
        };
        const layoutHist = {
            ...layoutBase,
            title: { text: 'Distribución de Probabilidades', font: { color: '#FFFFFF', size: 16 } },
            barmode: 'overlay',
            xaxis: { title: 'P(victoria A)', range: [0, 1], gridcolor: gridColor, zerolinecolor: gridColor },
            yaxis: { title: 'Frecuencia', gridcolor: gridColor, zerolinecolor: gridColor },
            showlegend: true,
            legend: { orientation: 'h', x: 0.5, y: -0.25, xanchor: 'center' },
            shapes: [{
                type: 'line', x0: 0.5, x1: 0.5, y0: 0, y1: 1, yref: 'paper',
                line: { color: 'rgba(255,255,255,0.5)', dash: 'dash', width: 2 }
            }]
        };
        Plotly.newPlot('plot-histogram', [trace0, trace1], layoutHist, {displayModeBar: false, responsive: true});
    }
}

function renderModelInfo(info) {
    const bodyEl = document.getElementById('models-body');
    const m = info.metrics || {};
    const fmt = (x, d = 3) => (x === null || x === undefined ? '—' : Number(x).toFixed(d));

    const metricsTable = `
        <table class="models-table">
            <thead><tr><th>AUC</th><th>LogLoss</th><th>Brier</th><th>Accuracy</th></tr></thead>
            <tbody><tr>
                <td>${fmt(m.auc)}</td><td>${fmt(m.log_loss)}</td>
                <td>${fmt(m.brier)}</td><td>${fmt(m.accuracy)}</td>
            </tr></tbody>
        </table>`;

    // Coeficientes ordenados por magnitud: odds-ratio interpretable por feature.
    const coefs = Object.entries(info.coeficientes || {})
        .sort((a, b) => Math.abs(b[1].coef) - Math.abs(a[1].coef));
    const maxAbs = coefs.reduce((mx, [, v]) => Math.max(mx, Math.abs(v.coef)), 1e-9);
    const coefRows = coefs.map(([k, v]) => {
        const pct = (Math.abs(v.coef) / maxAbs) * 50;
        const towardA = v.coef >= 0;
        return `<div class="fbar">
            <span class="fname">${COEF_LABELS[k] || k}</span>
            <div class="ftrack">
                <div class="ffill ${towardA ? 'toward-a' : 'toward-b'}" style="width:${pct}%"></div>
                <span class="fval ${towardA ? 'on-a' : 'on-b'}">OR ${v.odds_ratio.toFixed(2)}</span>
            </div></div>`;
    }).join('');

    bodyEl.innerHTML = `
        ${metricsTable}
        <p class="models-note">Regresión logística calibrada · test ciego 2025 (n=2861). Menor log-loss = mejor.</p>
        <div class="coef-head"><strong>Coeficientes (odds-ratio por +1 desviación estándar)</strong></div>
        <div class="factor-bars">${coefRows}</div>
        <p class="models-note">OR &gt; 1 (verde) inclina hacia el Jugador A; OR &lt; 1 (rojo) hacia el B. Es el peso real del modelo, no la diferencia del partido.</p>
        
        <div class="ml-vs-elo-card">
            <h4>
                <span>ML vs. ELO Puro</span>
                <span class="badge">¿Por qué importa un +0.015 de AUC?</span>
            </h4>
            <p>
                El rating ELO es un estimador robusto del rendimiento histórico, pero adolece de inercia y es ciego a factores extra-deportivos. Nuestro modelo de Machine Learning (LogReg calibrado) actúa sobre el ELO corrigiendo sus limitaciones mediante características demográficas y contextuales clave:
            </p>
            <div class="ml-vs-elo-features">
                <div class="ml-feature-item">
                    <strong>Diferencia de Edad</strong>
                    <span>Modula el desgaste y declive físico en veteranos, o el progreso acelerado en tenistas jóvenes, compensando la inercia del ELO.</span>
                </div>
                <div class="ml-feature-item">
                    <strong>Ranking ATP</strong>
                    <span>Incorpora la presión por defender puntos y la consistencia en el circuito profesional durante la temporada actual.</span>
                </div>
                <div class="ml-feature-item">
                    <strong>Jugadores sin Ranking</strong>
                    <span>Ajusta la alta incertidumbre de jugadores sin ranking formal (debido a lesiones largas o invitaciones de torneos).</span>
                </div>
            </div>
            <div class="ml-vs-elo-footer">
                En modelos probabilísticos deportivos, superar la barrera del ELO en <strong>+1.5% de AUC</strong> y reducir el LogLoss ($0.631 \to 0.622$) representa la diferencia matemática para obtener una ventaja predictiva consistente a largo plazo.
            </div>
        </div>`;
}

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

        const winA = eloA >= eloB;
        const fillW = `background:${accent}; opacity:0.9`;
        const fillL = `background:var(--line); opacity:0.4`;
        const group = document.createElement('div');
        group.className = 'elo-surface-group';
        group.innerHTML = `
            <div class="elo-surface-label">${label}</div>
            <div class="elo-bar-row${winA ? ' surf-winner' : ''}" style="--surf-accent:${accent}">
                <span class="elo-bar-name">${a.name.split(' ').slice(-1)[0]}</span>
                <div class="elo-bar-track">
                    <div class="elo-bar-fill" style="width:${pctA.toFixed(1)}%; ${winA ? fillW : fillL}"></div>
                </div>
                <span class="elo-bar-val">${eloA.toFixed(0)}</span>
            </div>
            <div class="elo-bar-row${!winA ? ' surf-winner' : ''}" style="--surf-accent:${accent}">
                <span class="elo-bar-name">${b.name.split(' ').slice(-1)[0]}</span>
                <div class="elo-bar-track">
                    <div class="elo-bar-fill" style="width:${pctB.toFixed(1)}%; ${!winA ? fillW : fillL}"></div>
                </div>
                <span class="elo-bar-val">${eloB.toFixed(0)}</span>
            </div>`;
        container.appendChild(group);
    });
}

// ----- Simulador de Torneos -----
function setupTournamentModal() {
    const openBtn = document.getElementById('open-tournament-btn');
    const closeBtn = document.getElementById('close-tournament-btn');
    const modal = document.getElementById('tournament-modal');
    const runBtn = document.getElementById('run-tournament-btn');
    const select = document.getElementById('tournament-select');
    const loadingMsg = document.getElementById('tournament-loading-msg');
    const errorMsg = document.getElementById('tournament-error-msg');
    const resultsWrap = document.getElementById('tournament-results-wrap');
    const tableBody = document.getElementById('tournament-table-body');
    const tableHead = modal.querySelector('.tournament-table thead tr');

    // Tabs
    const tabDrawBtn = document.getElementById('tab-draw-btn');
    const tabSimBtn = document.getElementById('tab-sim-btn');
    const viewDraw = document.getElementById('view-draw');
    const viewSim = document.getElementById('view-sim');
    const drawGrid = document.getElementById('draw-matchups-grid');
    const drawLoadingMsg = document.getElementById('draw-loading-msg');

    if (!openBtn || !closeBtn || !modal) return;

    let tournamentInfoLoaded = false;

    // Función para cambiar de vista (tabs)
    const switchTab = (tabName) => {
        if (tabName === 'draw') {
            tabDrawBtn.classList.add('active');
            tabSimBtn.classList.remove('active');
            viewDraw.classList.remove('hidden');
            viewSim.classList.add('hidden');
        } else {
            tabDrawBtn.classList.remove('active');
            tabSimBtn.classList.add('active');
            viewDraw.classList.add('hidden');
            viewSim.classList.remove('hidden');
        }
    };

    if (tabDrawBtn && tabSimBtn) {
        tabDrawBtn.addEventListener('click', () => switchTab('draw'));
        tabSimBtn.addEventListener('click', () => switchTab('sim'));
    }

    // Cargar información inicial del torneo (Participantes y Matchups)
    const loadTournamentInfo = async () => {
        const tournament = select.value;
        if (drawLoadingMsg) drawLoadingMsg.classList.remove('hidden');
        
        try {
            const r = await fetch(`/api/tournament/info?tournament=${encodeURIComponent(tournament)}`);
            if (!r.ok) {
                const errData = await r.json();
                throw new Error(errData.detail || 'Error cargando datos del torneo');
            }
            const data = await r.json();
            
            // Renderizar los enfrentamientos en el grid
            drawGrid.innerHTML = '';
            document.getElementById('draw-round-name').textContent = data.round;

            data.matchups.forEach(m => {
                const card = document.createElement('div');
                card.className = 'matchup-card';
                card.innerHTML = `
                    <div class="matchup-card-header">
                        <span>PARTIDO ${m.match_num}</span>
                        <span>1ª RONDA</span>
                    </div>
                    <div class="matchup-player-row">
                        <span class="matchup-player-name">${m.player_a.name}</span>
                        <span class="matchup-player-meta">#${m.player_a.rank} · ELO ${Math.round(m.player_a.elo)}</span>
                    </div>
                    <div class="matchup-vs">VS</div>
                    <div class="matchup-player-row">
                        <span class="matchup-player-name">${m.player_b.name}</span>
                        <span class="matchup-player-meta">#${m.player_b.rank} · ELO ${Math.round(m.player_b.elo)}</span>
                    </div>
                `;
                drawGrid.appendChild(card);
            });

            tournamentInfoLoaded = true;
        } catch (err) {
            drawGrid.innerHTML = `<div class="form-error" style="padding: 20px;">${err.message}</div>`;
        }
    };

    // Abrir modal
    openBtn.addEventListener('click', () => {
        modal.classList.remove('hidden');
        modal.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden'; // Evitar scroll de fondo
        
        // Reset tabs to default (first tab)
        switchTab('draw');

        // Cargar datos del torneo si no se han cargado aún
        loadTournamentInfo();
    });

    // Cambiar de torneo en el select
    select.addEventListener('change', () => {
        loadTournamentInfo();
        // Reset simulation results
        resultsWrap.classList.add('hidden');
        tableBody.innerHTML = '';
    });

    // Cerrar modal
    const closeModal = () => {
        modal.classList.add('hidden');
        modal.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
    };

    closeBtn.addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
            closeModal();
        }
    });

    // Correr simulación
    runBtn.addEventListener('click', async () => {
        const tournament = select.value;
        
        // UI reset
        runBtn.disabled = true;
        runBtn.classList.add('loading');
        loadingMsg.classList.remove('hidden');
        errorMsg.classList.add('hidden');
        errorMsg.textContent = '';
        resultsWrap.classList.add('hidden');
        tableBody.innerHTML = '';

        try {
            const r = await fetch(`/api/tournament/simulate?tournament=${encodeURIComponent(tournament)}&simulations=5000`);
            if (!r.ok) {
                const errData = await r.json();
                throw new Error(errData.detail || 'Error en la simulación');
            }
            const data = await r.json();
            
            // Actualizar etiquetas meta
            document.getElementById('tourney-surf-tag').textContent = data.surface;
            document.getElementById('tourney-sims-tag').textContent = `${data.simulations.toLocaleString()} sims`;

            // Construir cabeceras dinámicamente (omitir primera ronda que es 100%)
            const roundKeys = data.round_keys;
            const displayRounds = roundKeys.length > 1 ? roundKeys.slice(1) : roundKeys;

            tableHead.innerHTML = `
                <th>Jugador</th>
                <th>Rank</th>
                <th>ELO Gen</th>
                <th>ELO Sup</th>
                ${displayRounds.map(r => `<th>${r === 'Winner' ? '🏆 Campeón' : r}</th>`).join('')}
            `;

            // Renderizar filas (Top 20 favoritos por probabilidad de ser Campeón)
            const top20 = data.results.slice(0, 20);

            top20.forEach(player => {
                const tr = document.createElement('tr');
                
                // Formatear celdas de ronda
                const roundCellsHtml = displayRounds.map(rKey => {
                    const pct = player.probabilities[rKey] || 0.0;
                    const style = pct > 0 ? `style="background-color: color-mix(in srgb, var(--accent) ${Math.min(pct * 0.6, 60).toFixed(1)}%, transparent)"` : '';
                    return `<td ${style}>${pct.toFixed(1)}%</td>`;
                }).join('');

                tr.innerHTML = `
                    <td>${player.name}</td>
                    <td>${player.rank}</td>
                    <td>${Math.round(player.elo_general)}</td>
                    <td>${Math.round(player.elo_surface)}</td>
                    ${roundCellsHtml}
                `;
                tableBody.appendChild(tr);
            });

            resultsWrap.classList.remove('hidden');
        } catch (err) {
            errorMsg.textContent = err.message;
            errorMsg.classList.remove('hidden');
        } finally {
            runBtn.disabled = false;
            runBtn.classList.remove('loading');
            loadingMsg.classList.add('hidden');
        }
    });
}

})();
