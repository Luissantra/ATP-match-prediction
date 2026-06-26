// Tests de las funciones puras de formato/normalización del frontend.
// Runner nativo: `node --test tests/format.test.mjs` (cero dependencias).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

// format.js usa el patrón dual browser+node (module.exports si existe).
const require = createRequire(import.meta.url);
const { normalizeFactor, formatDiff, formatRank, mergeModels } = require('../static/format.js');

test('normalizeFactor: clampa a [-1, 1] según la escala de cada feature', () => {
  // ELO escala ±300: 150 → 0.5
  assert.equal(normalizeFactor(150, 'diff_elo_general'), 0.5);
  assert.equal(normalizeFactor(-300, 'diff_elo_sup'), -1);
  // Por encima de la escala se clampa
  assert.equal(normalizeFactor(600, 'diff_elo_general'), 1);
  assert.equal(normalizeFactor(-600, 'diff_elo_general'), -1);
  // rank escala ±250
  assert.equal(normalizeFactor(125, 'diff_rank'), 0.5);
  // edad escala ±10
  assert.equal(normalizeFactor(-5, 'diff_age'), -0.5);
  // h2h / form escala ±1
  assert.equal(normalizeFactor(0.5, 'diff_h2h'), 0.5);
  assert.equal(normalizeFactor(1, 'diff_form'), 1);
  // cero → cero
  assert.equal(normalizeFactor(0, 'diff_elo_general'), 0);
});

test('normalizeFactor: feature desconocida usa escala 1', () => {
  assert.equal(normalizeFactor(0.3, 'desconocida'), 0.3);
  assert.equal(normalizeFactor(5, 'desconocida'), 1);
});

test('formatDiff: signo explícito y decimales', () => {
  assert.equal(formatDiff(52, 'pts'), '+52.0 pts');
  assert.equal(formatDiff(-12.34, 'pts'), '-12.3 pts');
  assert.equal(formatDiff(0, 'pts'), '+0.0 pts');
  // decimales 0 para puestos de ranking
  assert.equal(formatDiff(-2, 'puestos', 0), '-2 puestos');
  assert.equal(formatDiff(3, 'puestos', 0), '+3 puestos');
});

test('formatRank: maneja int, 999 y string de la API', () => {
  assert.equal(formatRank(2), '#2');
  assert.equal(formatRank(150), '#150');
  assert.equal(formatRank(999), 'Sin ranking');
  assert.equal(formatRank('Sin Ranking'), 'Sin Ranking');
});

test('mergeModels: fusiona probas y métricas por nombre', () => {
  const predictAll = {
    predictions: {
      gbm:          { prob_a: 60, prob_b: 40, predicted_winner: 'A' },
      logreg:       { prob_a: 55, prob_b: 45, predicted_winner: 'A' },
      randomforest: { error: 'boom' },
    },
  };
  const metrics = [
    { nombre: 'gbm',    accuracy: 0.57, log_loss: 0.68, brier: 0.24, auc: 0.61 },
    { nombre: 'logreg', accuracy: 0.55, log_loss: 0.70, brier: 0.25, auc: 0.59 },
  ];
  const rows = mergeModels(predictAll, metrics);
  assert.equal(rows.length, 3);

  const gbm = rows.find(r => r.name === 'gbm');
  assert.equal(gbm.prob_a, 60);
  assert.equal(gbm.auc, 0.61);

  // modelo con predicción pero sin métricas → métricas null
  const rf = rows.find(r => r.name === 'randomforest');
  assert.equal(rf.error, 'boom');
  assert.equal(rf.auc, null);
  assert.equal(rf.prob_a, null);
});

test('mergeModels: tolera entradas vacías', () => {
  assert.deepEqual(mergeModels({}, []), []);
  assert.deepEqual(mergeModels({ predictions: {} }, null), []);
});
