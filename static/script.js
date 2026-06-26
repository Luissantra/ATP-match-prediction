// =========================================================================
// ATP MATCH FORECAST — capa de UI (estado, fetch, render).
// Funciones puras de formato en static/format.js (window.ATPFormat).
// IIFE: aísla el scope. format.js expone normalizeFactor/etc como funciones
// globales; sin este wrapper, el destructuring de abajo colisionaría con esos
// identificadores globales y el script entero sería rechazado.
// =========================================================================
(function () {
'use strict';
const { normalizeFactor, formatDiff, formatRank, mergeModels } = window.ATPFormat;

// Factores a graficar como barras divergentes. `dir` = +1 si un valor positivo
// (A−B) favorece a A; -1 si lo favorece B (ranking: nº menor es mejor; edad:
// más joven se toma como ventaja). Heurística de orientación, no peso del modelo.
const FACTORS = [
    { key: 'diff_elo_general', name: 'ELO general',    unit: 'pts',  dec: 1, dir: 1 },
    { key: 'diff_elo_sup',     name: 'ELO superficie', unit: 'pts',  dec: 1, dir: 1 },
    { key: 'diff_rank',        name: 'Ranking',        unit: 'pos',  dec: 0, dir: -1 },
    { key: 'diff_age',         name: 'Edad',           unit: 'años', dec: 1, dir: -1 },
    { key: 'diff_h2h',         name: 'Head-to-head',   unit: '',     dec: 2, dir: 1 },
    { key: 'diff_form',        name: 'Forma reciente', unit: '',     dec: 2, dir: 1 },
];

// ----- Estado -----
let players = [];
let selA = null, selB = null;
let surface = 'Hard';
let level = '250';

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
    setupPills('level-group', (v) => { level = v; });
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
    const params = `player_a=${encodeURIComponent(selA.name)}&player_b=${encodeURIComponent(selB.name)}&surface=${surface}&tourney_level=${encodeURIComponent(level)}`;
    try {
        const r = await fetch(`/api/predict?${params}`);
        if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'Fallo en la predicción'); }
        const data = await r.json();
        renderResults(data);
        loadModelComparison(params);
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
    document.getElementById('court-name-a').textContent = a.name;
    document.getElementById('court-name-b').textContent = b.name;
    document.getElementById('half-a').style.width = `${a.prob_victory}%`;
    document.getElementById('half-b').style.width = `${b.prob_victory}%`;
    document.getElementById('court-net').style.left = `${a.prob_victory}%`;
    document.getElementById('court-pct-a').textContent = `${a.prob_victory}%`;
    document.getElementById('court-pct-b').textContent = `${b.prob_victory}%`;

    // Ganador
    const winA = res.predicted_winner === a.name;
    document.getElementById('winner-name').textContent = res.predicted_winner;
    const conf = winA ? a.prob_victory : b.prob_victory;
    document.getElementById('winner-conf').textContent = `Probabilidad estimada ${conf}% · modelo ${res.model_used || 'gbm'}`;

    // Eje + barras de factores
    document.getElementById('axis-name-a').textContent = a.name;
    document.getElementById('axis-name-b').textContent = b.name;
    renderFactors(res.features_debug);

    // Comparativa numérica
    fillCompare('a', a);
    fillCompare('b', b);

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

// ----- Comparar modelos -----
async function loadModelComparison(params) {
    const bodyEl = document.getElementById('models-body');
    bodyEl.innerHTML = '<p class="models-loading">Cargando modelos…</p>';
    try {
        const [pa, mm] = await Promise.all([
            fetch(`/api/predict_all?${params}`).then((r) => r.json()),
            fetch('/api/models').then((r) => r.json()),
        ]);
        renderModels(mergeModels(pa, mm));
    } catch (e) {
        bodyEl.innerHTML = '<p class="models-loading">No disponible.</p>';
    }
}

function renderModels(rows) {
    const bodyEl = document.getElementById('models-body');
    if (!rows.length) { bodyEl.innerHTML = '<p class="models-loading">Sin modelos.</p>'; return; }
    const fmt = (x, d = 3) => (x === null || x === undefined ? '—' : Number(x).toFixed(d));
    const trs = rows.map((r) => {
        const probs = r.error ? '<td colspan="2">error</td>'
            : `<td>${fmt(r.prob_a, 1)}%</td><td>${fmt(r.prob_b, 1)}%</td>`;
        return `<tr class="${r.name === 'gbm' ? 'is-main' : ''}">
            <td class="mname">${r.name}</td>${probs}
            <td>${fmt(r.auc)}</td><td>${fmt(r.log_loss)}</td>
            <td>${fmt(r.brier)}</td><td>${fmt(r.accuracy)}</td></tr>`;
    }).join('');
    bodyEl.innerHTML = `
        <table class="models-table">
            <thead><tr>
                <th>Modelo</th><th>Prob A</th><th>Prob B</th>
                <th>AUC</th><th>LogLoss</th><th>Brier</th><th>Acc</th>
            </tr></thead>
            <tbody>${trs}</tbody>
        </table>
        <p class="models-note">Probabilidades del partido actual · métricas del test ciego 2026. Menor log-loss = mejor.</p>`;
}

})();
