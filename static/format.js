// =========================================================================
// FUNCIONES PURAS DE FORMATO Y NORMALIZACIÓN (sin DOM)
// Fuente única para la capa de presentación; testeadas con `node --test`.
// Patrón dual: se exponen en window (browser) y en module.exports (node).
// =========================================================================

// Escala típica de cada feature para normalizar a [-1, 1] (ancho/dirección de
// las barras divergentes). NO es contribución del modelo, solo comparabilidad
// visual entre features de magnitudes muy distintas.
const FACTOR_SCALES = {
  diff_elo_general: 300,
  diff_elo_sup: 300,
  diff_rank: 250, // = RANK_CAP del backend
  diff_age: 10,
  diff_h2h: 1,
  diff_form: 1,
};

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

// Normaliza el valor de una feature a [-1, 1] según su escala. Feature
// desconocida → escala 1.
function normalizeFactor(value, feature) {
  const scale = FACTOR_SCALES[feature] || 1;
  return clamp(value / scale, -1, 1);
}

// Formatea una diferencia con signo explícito y unidad. `decimals` por defecto 1
// (pásalo 0 para enteros como puestos de ranking).
function formatDiff(value, unit, decimals = 1) {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(decimals)} ${unit}`;
}

// Formatea un ranking. La API devuelve int, 999 (sin rank), o el string
// "Sin Ranking" ya formateado.
function formatRank(rank) {
  if (typeof rank === 'string') return rank;
  if (rank === 999) return 'Sin ranking';
  return `#${rank}`;
}

// Fusiona la respuesta de /api/predict_all con las métricas de /api/models en un
// array de filas para la tabla comparativa, indexando por nombre de modelo.
// Modelos con predicción pero sin métricas → métricas a null. Modelos con error
// → probas a null y se conserva el campo `error`.
function mergeModels(predictAll, modelsMetrics) {
  const predictions = (predictAll && predictAll.predictions) || {};
  const metrics = modelsMetrics || [];
  const metricsByName = {};
  for (const m of metrics) {
    metricsByName[m.nombre] = m;
  }

  return Object.keys(predictions).map((name) => {
    const pred = predictions[name] || {};
    const met = metricsByName[name] || {};
    const hasError = Boolean(pred.error);
    return {
      name,
      prob_a: hasError ? null : (pred.prob_a ?? null),
      prob_b: hasError ? null : (pred.prob_b ?? null),
      predicted_winner: hasError ? null : (pred.predicted_winner ?? null),
      error: pred.error || null,
      accuracy: met.accuracy ?? null,
      log_loss: met.log_loss ?? null,
      brier: met.brier ?? null,
      auc: met.auc ?? null,
    };
  });
}

// Exposición dual.
const API = { normalizeFactor, formatDiff, formatRank, mergeModels, FACTOR_SCALES };
if (typeof module !== 'undefined' && module.exports) {
  module.exports = API;
}
if (typeof window !== 'undefined') {
  window.ATPFormat = API;
}
