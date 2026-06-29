import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance as _sk_permutation_importance
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    log_loss, brier_score_loss, roc_auc_score,
)


def evaluar(modelo, X, y):
    """
    Métricas de evaluación para un problema probabilístico binario.

    El producto es la *probabilidad* de victoria, así que accuracy por sí sola
    engaña: se reportan también log-loss, Brier y AUC. Reutilizable para comparar
    varios modelos con la misma interfaz (épica multi-modelo).

    Returns
    -------
    dict con 'accuracy', 'log_loss', 'brier', 'auc'.
    """
    preds = modelo.predict(X)
    proba = modelo.predict_proba(X)[:, 1]
    return {
        'accuracy': accuracy_score(y, preds),
        'log_loss': log_loss(y, proba, labels=[0, 1]),
        'brier':    brier_score_loss(y, proba),
        'auc':      roc_auc_score(y, proba),
    }


def bootstrap_ic95(y_true, proba, metric='auc', n_iter=1000, seed=42):
    """
    Bootstrap IC95% para una métrica escalar.
    metric ∈ {'auc', 'log_loss', 'brier'}.
    Con n≈137 el IC95% del AUC ≈ ±0.08-0.09; diferencias entre modelos
    suelen caer dentro del ruido → evitar claims de mejora sin IC.
    """
    _fns = {
        'auc':      lambda yt, yp: roc_auc_score(yt, yp),
        'log_loss': lambda yt, yp: log_loss(yt, yp, labels=[0, 1]),
        'brier':    lambda yt, yp: brier_score_loss(yt, yp),
    }
    fn = _fns[metric]
    y_arr = np.asarray(y_true)
    p_arr = np.asarray(proba)
    rng = np.random.default_rng(seed)
    n = len(y_arr)
    scores = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        scores.append(fn(y_arr[idx], p_arr[idx]))
    scores = np.array(scores)
    return {
        'mean':  float(np.mean(scores)),
        'lower': float(np.percentile(scores, 2.5)),
        'upper': float(np.percentile(scores, 97.5)),
    }


def evaluar_con_ic(modelo, X, y, n_iter=1000, seed=42):
    """
    Extiende evaluar() añadiendo IC95% bootstrap para AUC, log-loss y Brier.
    Con n≈137 las diferencias de AUC < 0.08 son ruido, no mejora demostrable.
    """
    proba = modelo.predict_proba(X)[:, 1]
    base = evaluar(modelo, X, y)
    base['auc_ic']      = bootstrap_ic95(y, proba, 'auc',      n_iter, seed)
    base['log_loss_ic'] = bootstrap_ic95(y, proba, 'log_loss', n_iter, seed)
    base['brier_ic']    = bootstrap_ic95(y, proba, 'brier',    n_iter, seed)
    return base


def evaluar_baseline_elo(df_test, y_true, n_iter=1000, seed=42):
    """
    Baseline obligatorio: ¿cuánto aporta el ML sobre ELO-crudo solo?

    Predictor = calcular_expectativa(diff_elo). Baseline HONESTO: usa el ELO híbrido
    (0.5*general + 0.5*superficie) cuando hay columna de superficie, de modo que reciba
    la misma información de superficie que el modelo. Usar solo el general infla
    artificialmente la ventaja del ML. Fallback a general si no hay diff_elo_sup.
    Referencia: si log-loss/AUC del baseline ≈ modelo → el stack ML no añade valor.
    """
    from src.elo import calcular_expectativa
    if 'diff_elo_sup' in df_test.columns:
        diff = 0.5 * (df_test['diff_elo_general'].values + df_test['diff_elo_sup'].values)
    else:
        diff = df_test['diff_elo_general'].values
    proba_baseline = np.array([calcular_expectativa(d, 0) for d in diff])
    y_arr = np.asarray(y_true)
    preds = (proba_baseline >= 0.5).astype(int)
    met = {
        'accuracy': float(accuracy_score(y_arr, preds)),
        'log_loss': float(log_loss(y_arr, proba_baseline, labels=[0, 1])),
        'brier':    float(brier_score_loss(y_arr, proba_baseline)),
        'auc':      float(roc_auc_score(y_arr, proba_baseline)),
    }
    met['auc_ic']      = bootstrap_ic95(y_arr, proba_baseline, 'auc',      n_iter, seed)
    met['log_loss_ic'] = bootstrap_ic95(y_arr, proba_baseline, 'log_loss', n_iter, seed)
    met['brier_ic']    = bootstrap_ic95(y_arr, proba_baseline, 'brier',    n_iter, seed)
    return met


def evaluar_y_graficar(modelo, X_test, y_test, df_test, features,
                       modelo_para_importancia=None):
    preds = modelo.predict(X_test)
    metricas = evaluar(modelo, X_test, y_test)
    accuracy = metricas['accuracy']

    print(f"\nMÉTRICAS TEST CIEGO:")
    print(f"  Accuracy : {metricas['accuracy']:.2%}")
    print(f"  Log-loss : {metricas['log_loss']:.4f}  (menor es mejor; azar = {0.6931:.4f})")
    print(f"  Brier    : {metricas['brier']:.4f}  (menor es mejor; azar = 0.25)")
    print(f"  AUC      : {metricas['auc']:.4f}  (0.5 = azar)")
    print(classification_report(y_test, preds, target_names=['Derrota A', 'Victoria A']))

    os.makedirs("plots", exist_ok=True)
    cm_data = _plot_confusion_matrix(y_test, preds, accuracy)
    # El Gini sólo aplica a modelos de árboles. Para el modelo lineal la explicabilidad
    # se cubre con coeficientes (graficar_coeficientes) y permutation importance.
    modelo_imp = modelo_para_importancia if modelo_para_importancia is not None else modelo
    if hasattr(modelo_imp, 'feature_importances_'):
        _plot_feature_importance(modelo_imp, features)
    _plot_accuracy_by_surface(df_test, preds, accuracy)

    return cm_data


def graficar_learning_curve(modelo, X_train, y_train, cv):
    """
    Diagnostica el gap CV/test: dibuja log-loss de train vs validación según crece
    el tamaño de entrenamiento. Si val_loss >> train_loss y no convergen → sobreajuste;
    si ambas son altas y planas → underfitting / falta de señal.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        clone(modelo), X_train, y_train, cv=cv, scoring='neg_log_loss',
        train_sizes=np.linspace(0.2, 1.0, 6), n_jobs=-1,
    )
    train_loss = -train_scores.mean(axis=1)
    val_loss = -val_scores.mean(axis=1)

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_loss, 'o-', color="#27ae60", label="Train log-loss")
    plt.plot(train_sizes, val_loss, 'o-', color="#c0392b", label="Validación log-loss")
    plt.axhline(0.6931, color='#7f8c8d', linestyle='--', alpha=0.7, label='Azar (0.693)')
    plt.title("Curva de Aprendizaje (CV temporal con embargo)\n¿Sobreajuste o falta de señal?",
              fontsize=12, pad=15, weight='bold')
    plt.xlabel("Tamaño de entrenamiento", fontsize=11)
    plt.ylabel("Log-loss (menor = mejor)", fontsize=11)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    sns.despine()
    plt.tight_layout()
    plt.savefig("plots/learning_curve.png", dpi=300)
    plt.close()


def graficar_reliability_diagram(modelo, X, y, n_bins=10):
    """
    Muestra si las probabilidades predichas son honestas: si el modelo dice 70%,
    ¿gana el jugador A ~70% de las veces en el test? Una diagonal perfecta = calibración perfecta.
    """
    proba = modelo.predict_proba(X)[:, 1]
    prob_true, prob_pred = calibration_curve(y, proba, n_bins=n_bins, strategy='uniform')

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, 'o-', color="#2980b9", linewidth=2, label="Modelo calibrado")
    plt.plot([0, 1], [0, 1], '--', color="#7f8c8d", alpha=0.7, label="Calibración perfecta")
    plt.title("Reliability Diagram (Calibración)\n¿Las probabilidades predichas son honestas?",
              fontsize=12, pad=15, weight='bold')
    plt.xlabel("Probabilidad predicha", fontsize=11)
    plt.ylabel("Fracción de positivos reales", fontsize=11)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    sns.despine()
    plt.tight_layout()
    plt.savefig("plots/reliability_diagram.png", dpi=300)
    plt.close()
    return {"prob_pred": prob_pred.tolist(), "prob_true": prob_true.tolist()}


def graficar_histograma_probas(modelo, X, y, bins=20):
    """
    Distribución de las probabilidades predichas por clase.
    Un modelo discriminativo separa bien las dos distribuciones (poca superposición).
    """
    proba = modelo.predict_proba(X)[:, 1]
    y_arr = np.asarray(y)

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(proba[y_arr == 0], bins=bins, alpha=0.6, color="#c0392b",
             label="Clase 0 (derrota A)", edgecolor='none')
    plt.hist(proba[y_arr == 1], bins=bins, alpha=0.6, color="#27ae60",
             label="Clase 1 (victoria A)", edgecolor='none')
    plt.axvline(0.5, color='#7f8c8d', linestyle='--', alpha=0.7, label='Umbral 0.5')
    plt.title("Distribución de Probabilidades Predichas\n¿Separa el modelo las dos clases?",
              fontsize=12, pad=15, weight='bold')
    plt.xlabel("P(victoria jugador A)", fontsize=11)
    plt.ylabel("Frecuencia", fontsize=11)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    sns.despine()
    plt.tight_layout()
    plt.savefig("plots/histograma_probas.png", dpi=300)
    plt.close()
    return {"class_0": proba[y_arr == 0].tolist(), "class_1": proba[y_arr == 1].tolist()}


def _plot_confusion_matrix(y_test, preds, accuracy):
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Pred. Derrota A', 'Pred. Victoria A'],
                yticklabels=['Real Derrota A', 'Real Victoria A'],
                annot_kws={"size": 14, "weight": "bold"})
    for i in range(2):
        for j in range(2):
            total = np.sum(cm[i, :])
            pct = (cm[i, j] / total) * 100
            plt.text(j + 0.5, i + 0.7, f"({pct:.1f}%)", ha="center", va="center",
                     color="darkblue" if cm[i, j] < cm.max() / 2 else "white", fontsize=11)
    plt.title(f"Matriz de Confusión (Test Ciego)\nATP Tennis Prediction Model", fontsize=13, pad=15, weight='bold')
    plt.ylabel("Estado Real", fontsize=12)
    plt.xlabel("Predicción del Modelo", fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/matriz_confusion.png", dpi=300)
    plt.close()
    return cm.tolist()


def _plot_feature_importance(modelo, features):
    feature_labels = {
        'diff_elo_general':  'ELO General',
        'diff_elo_sup':      'ELO Superficie',
        'diff_rank':         'Ranking',
        'is_unranked':       'Sin Ranking',
        'diff_age':          'Edad',
    }
    labels = [feature_labels.get(f, f) for f in features]
    importancias = modelo.feature_importances_
    orden = np.argsort(importancias)

    plt.figure(figsize=(8, 4.5))
    plt.barh(range(len(importancias)), importancias[orden], color="#34495e", edgecolor="#2c3e50", height=0.6)
    plt.yticks(range(len(importancias)), [labels[i] for i in orden], fontsize=11)
    plt.xlabel("Importancia (Gini Impurity Decrease)", fontsize=11)
    plt.title("Importancia Relativa de las Variables\n¿Qué factores deciden la victoria?", fontsize=12, pad=15, weight='bold')
    for idx, val in enumerate(importancias[orden]):
        plt.text(val + 0.005, idx, f"{val:.1%}", va='center', ha='left', fontweight='bold', color="#2c3e50")
    plt.xlim(0, max(importancias) + 0.1)
    sns.despine()
    plt.tight_layout()
    plt.savefig("plots/importancia_variables.png", dpi=300)
    plt.close()


def _plot_accuracy_by_surface(df_test, preds, accuracy_global):
    df_test = df_test.copy()
    df_test['pred'] = preds
    colores = {'Clay': '#e67e22', 'Grass': '#2ecc71', 'Hard': '#3498db'}
    res = {}
    for sup in ['Hard', 'Clay', 'Grass']:
        sub = df_test[df_test['surface'] == sup]
        if len(sub) > 0:
            res[sup] = (accuracy_score(sub['label'], sub['pred']), len(sub))

    if not res:
        return

    names = list(res.keys())
    values = [res[k][0] for k in names]
    counts = [res[k][1] for k in names]
    colors = [colores.get(n, '#7f8c8d') for n in names]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, values, color=colors, edgecolor='none', width=0.55)
    plt.axhline(accuracy_global, color='#7f8c8d', linestyle='--', alpha=0.7,
                label=f'Precisión Global ({accuracy_global:.1%})')
    plt.title("Precisión por Superficie (Test Ciego)\n¿En qué canchas es más predecible el tenis?",
              fontsize=12, pad=15, weight='bold')
    plt.ylabel("Precisión (Accuracy)", fontsize=11)
    plt.ylim(0, 0.85)
    plt.legend(loc='lower center', frameon=True, facecolor='white', edgecolor='none')
    for bar, val, count in zip(bars, values, counts):
        plt.text(bar.get_x() + bar.get_width() / 2.0, val + 0.02,
                 f"{val:.1%}\n(n={count})", ha='center', va='bottom',
                 fontweight='bold', fontsize=10, color='#2c3e50')
    sns.despine()
    plt.tight_layout()
    plt.savefig("plots/precision_por_superficie.png", dpi=300)
    plt.close()


def permutation_importancia(modelo, X, y, features, n_repeats=30, seed=42):
    """
    Importancia por permutación (scoring=neg_log_loss).
    Más robusta que Gini para features correladas (e.g. diff_elo_general / diff_elo_sup).
    Un valor > 0 indica que permutar esa feature empeora el log-loss → feature útil.

    Returns
    -------
    dict {feature_name: {'mean': float, 'std': float}}, ordenado de mayor a menor importancia.
    """
    result = _sk_permutation_importance(
        modelo, X, y,
        n_repeats=n_repeats,
        scoring='neg_log_loss',
        random_state=seed,
    )
    importancias = {
        feat: {'mean': float(result.importances_mean[i]), 'std': float(result.importances_std[i])}
        for i, feat in enumerate(features)
    }
    return dict(sorted(importancias.items(), key=lambda kv: kv[1]['mean'], reverse=True))


def graficar_permutation_importance(modelo, X, y, features, n_repeats=30, seed=42):
    """
    Grafica la importancia por permutación con barras de error (±1 std).
    Reemplaza/complementa el Gini plot para comparación honesta entre features correladas.
    """
    imp = permutation_importancia(modelo, X, y, features, n_repeats=n_repeats, seed=seed)
    feature_labels = {
        'diff_elo_general':  'ELO General',
        'diff_elo_sup':      'ELO Superficie',
        'diff_rank':         'Ranking',
        'is_unranked':       'Sin Ranking',
        'diff_age':          'Edad',
        'diff_matches_played': 'Experiencia',
        'diff_tb_ratio':      'Tie-breaks',
    }
    feats_sorted = list(imp.keys())
    means = np.array([imp[f]['mean'] for f in feats_sorted])
    stds  = np.array([imp[f]['std']  for f in feats_sorted])
    labels = [feature_labels.get(f, f) for f in feats_sorted]

    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#27ae60" if m > 0 else "#c0392b" for m in means]
    ax.barh(range(len(means)), means, xerr=stds, color=colors, edgecolor='none',
            height=0.6, capsize=4, error_kw={'elinewidth': 1.2, 'ecolor': '#7f8c8d'})
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.axvline(0, color='#7f8c8d', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Importancia por permutación (Δ neg-log-loss, ±1 std)", fontsize=11)
    ax.set_title(
        "Importancia por Permutación\n¿Qué factores importan realmente (sin sesgo por correlación)?",
        fontsize=12, pad=15, weight='bold',
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig("plots/permutation_importance.png", dpi=300)
    plt.close()


def graficar_coeficientes(coefs, features=None):
    """
    Explicabilidad del modelo lineal: coeficientes estandarizados (log-odds por +1 std).
    Barras divergentes: positivo (verde) favorece a A, negativo (rojo) favorece a B.
    `coefs` = dict {feature: {'coef': float, 'odds_ratio': float}} (de coeficientes_modelo).
    """
    feature_labels = {
        'diff_elo_general':  'ELO General',
        'diff_elo_sup':      'ELO Superficie',
        'diff_rank':         'Ranking',
        'is_unranked':       'Sin Ranking',
        'diff_age':          'Edad',
    }
    items = sorted(coefs.items(), key=lambda kv: kv[1]['coef'])
    labels = [feature_labels.get(f, f) for f, _ in items]
    valores = [v['coef'] for _, v in items]
    colores = ["#27ae60" if c > 0 else "#c0392b" for c in valores]

    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(range(len(valores)), valores, color=colores, edgecolor='none', height=0.6)
    ax.set_yticks(range(len(valores)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.axvline(0, color='#7f8c8d', linewidth=0.8, linestyle='--')
    for i, (_, v) in enumerate(items):
        ax.text(v['coef'], i, f"  OR={v['odds_ratio']:.2f}", va='center',
                ha='left' if v['coef'] >= 0 else 'right', fontsize=9, color='#2c3e50')
    ax.set_xlabel("Coeficiente LogReg (log-odds por +1 std)", fontsize=11)
    ax.set_title("Coeficientes del Modelo (Explicabilidad)\n+ favorece al Jugador A · − favorece al Jugador B",
                 fontsize=12, pad=15, weight='bold')
    sns.despine()
    plt.tight_layout()
    plt.savefig("plots/coeficientes_modelo.png", dpi=300)
    plt.close()


def diagnosticar_gap_cv_test(cv_best_score: float, test_log_loss: float, n_test: int) -> str:
    """
    Análisis textual del gap CV/test. No usa la palabra 'confirmado':
    el gap mezcla optimismo de GridSearch (selection bias), sesgo estacional
    y shift real — no se puede separar sin nested CV.
    """
    gap = test_log_loss - cv_best_score
    lines = [
        f"Gap CV→test: {cv_best_score:.4f} → {test_log_loss:.4f} (Δ={gap:+.4f})",
        f"n_test={n_test} → IC95% log-loss ≈ ±{1.0 / (2 * (n_test**0.5)):.3f} aprox.",
        "",
        "Causas posibles (no separadas sin nested CV):",
        "  1. Selection bias: best_score_ de GridSearch es optimista (elige sobre el mismo CV).",
        "  2. Sesgo estacional: test=2026 parcial (mix torneo/superficie ≠ 2020-25).",
        "  3. Distribution shift real: el circuito 2026 puede ser distinto.",
        "",
        "Diagnóstico: gap consistente con distribution shift + optimismo de GridSearch.",
        "No se puede afirmar cuál domina sin nested CV o análisis de distribución train/test.",
    ]
    return "\n".join(lines)
