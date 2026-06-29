# Revisión de calidad — modelado estadístico (2026-06-26)

> Crítica experta del estado del proyecto (modelo + notebook didáctico `notebooks/atp_resumen.ipynb`).
> Foco: rigor estadístico, no ingeniería. La ingeniería está por encima de la media; el modelado no está exprimido y las conclusiones no respetan el tamaño muestral.
> Severidad: 🔴 crítico · 🟠 importante · 🟡 menor.

## Resumen del veredicto

- Ingeniería: **8/10** (fuente única de features, embargo temporal, scoring log-loss, 92 tests).
- Modelado estadístico: **4/10** (ELO sin margin-of-victory ni K-schedule, sin baseline, conclusiones que ignoran n≈137).
- Notebook didáctico: **7/10** (pedagogía sólida, pero repite el pecado del tamaño muestral justo donde predica honestidad métrica).

---

## Proyecto — modelado

### 🔴 R1 · Test set n≈137 invalida la narrativa de mejora
2026 es año parcial (revisión a 26-jun) → ~137 partidos. IC95% del AUC sobre n=137 ≈ **±0.08–0.09**.
- "AUC 0.615→0.629 / log-loss 0.683→0.674" por I3 cae **dentro del ruido**: no es mejora demostrable.
- Rankear 4 modelos por log-loss en n=137 no distingue nada.
- Todo claim de mejora necesita IC o test (bootstrap / DeLong para AUC). Hoy el roadmap presenta ruido como señal.

### 🔴 R2 · Falta el baseline que importa: ¿el ML supera al ELO crudo?
`calcular_expectativa(diff_elo)` ya produce una probabilidad. Baseline obligatorio = log-loss/AUC de esa proba sola, frente al GBM con 8 features + GridSearch + calibración. Sin él no se sabe si todo el stack aporta sobre una resta y una logística. Sospecha fuerte: aporta poco (ver R3).

### 🟠 R3 · AUC 0.62 por debajo de literatura ELO-tenis (~0.66–0.70 surface-ELO)
No es techo del tenis: es ELO débil dejando señal en la mesa.
- **K=32 fijo**, sin K provisional para debutantes ni decay temporal.
- **Sin margin-of-victory** (sets/games): la señal más barata y potente del tenis, ausente por completo.
- **ELO superficie cold-start a 1500 con K=32**: muchos menos partidos/jugador → converge lento, el cold-start domina → `diff_elo_sup` más ruido que señal.
- **Inyección a 1500** de cada debutante infla/desinfla a sus rivales (rating drift).

Arreglar el ELO probablemente da más AUC que toda la épica multi-modelo junta.

### 🟠 R4 · Diagnóstico "distribution shift confirmado" es flojo
Gap CV 0.620 vs test 0.683 mezcla tres causas sin separar:
1. Selection bias: `best_score_` es optimista (GridSearch elige sobre él).
2. Muestra estacional sesgada (2026 parcial: mezcla torneo/superficie distinta a 2020–25).
3. Shift real.
"Confirmado" es exagerado; es *consistente con*, no probado.

### 🟡 R5 · Calibración isotonic en folds TS purgados pequeños
Isotonic necesita muchos datos; en folds chicos sobreajusta escalones → mete varianza. Para este n, **sigmoid (Platt)** es lo correcto. Comparar ambos con log-loss.

### 🟡 R6 · `np.random.seed(42)` global
En `preparar_datos_entrenamiento` y `crear_dataset_visual`: efecto colateral en el RNG global del proceso (ya señalado en I8). Migrar a `np.random.default_rng(42)`.

### Crédito (bien hecho)
Fuente única de features (anti train/serve skew), simetrización vectorizada correcta, embargo temporal bien implementado, `scoring=neg_log_loss`, `evaluar()` con 4 métricas, ELO pre-partido sin leakage, 92 tests. El problema no es el código — es el modelado estadístico y el respeto al tamaño muestral.

---

## Notebook `notebooks/atp_resumen.ipynb`

### 🔴 R7 · §7 "métricas honestas" no menciona n≈137
La sección predica honestidad métrica y omite la lección #1: con 137 partidos los 4 decimales son humo. Debe enseñar IC/bootstrap del AUC. Hoy enseña a leer ruido con cara seria — contradice su propio título.

### 🟠 R8 · Falta el baseline ELO-crudo en el notebook
La pregunta más instructiva del proyecto —"¿el ML supera a la resta de ELO?"— no se enseña. Una celda: `log_loss(y_test, calcular_expectativa(...))` vs modelo.

### 🟠 R9 · Métricas del notebook ≠ documentadas
Usa `AÑOS=[2022–2026]` y grid reducido → ELO e historia distintos a `main.py`. Los números no coinciden con CLAUDE.md/roadmap. Falta aviso explícito arriba ("estos números no son los de producción").

### 🟡 R10 · §7 habla de calibración pero no dibuja reliability diagram
`graficar_reliability_diagram` ya existe en `src/evaluate.py`. Concepto central mencionado y no mostrado. Añadirlo (como ya hace con learning_curve).

---

## Fixes por ROI (orden recomendado)

1. **R1 + R2** — baseline ELO-crudo + bootstrap IC en todas las métricas. Mata el "improvement theatre"; cambia cómo se interpreta el proyecto entero.
2. **R3** — ELO con margin-of-victory + K provisional/schedule. Sube AUC de verdad.
3. **R7 + R8 + R9 + R10** — notebook: celda IC/baseline en §7, aviso de no-coincidencia, reliability diagram.
4. **R4, R5, R6** — separar causas del gap, probar sigmoid vs isotonic, RNG local.
