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
    input.addEventListener('input', () => {
        const q = input.value.trim().toLowerCase();
        list.innerHTML = '';
        if (q.length < 2) { list.style.display = 'none'; clearSelection(side); return; }

        const matches = players.filter((p) => p.name.toLowerCase().includes(q)).slice(0, 8);
        if (matches.length === 0) {
            list.innerHTML = '<div class="ac-item"><span class="ac-name" style="color:var(--text-faint)">Sin resultados</span></div>';
            list.style.display = 'block';
            return;
        }
        matches.forEach((p) => {
            const item = document.createElement('div');
            item.className = 'ac-item';
            item.innerHTML = `<span class="ac-name">${p.name}</span>
                <span class="ac-meta">${formatRank(p.rank)} · ELO ${Math.round(p.elo)}</span>`;
            item.addEventListener('click', () => {
                input.value = p.name;
                list.style.display = 'none';
                selectPlayer(p, side);
            });
            list.appendChild(item);
        });
        list.style.display = 'block';
    });

    input.addEventListener('focus', () => {
        if (input.value.trim().length >= 2 && list.children.length) list.style.display = 'block';
    });
    document.addEventListener('click', (e) => {
        if (e.target !== input) list.style.display = 'none';
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
    try {
        const info = await fetch('/api/model').then((r) => r.json());
        renderModelInfo(info);
    } catch (e) {
        bodyEl.innerHTML = '<p class="models-loading">No disponible.</p>';
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
        <p class="models-note">OR &gt; 1 (verde) inclina hacia el Jugador A; OR &lt; 1 (rojo) hacia el B. Es el peso real del modelo, no la diferencia del partido.</p>`;
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

})();
