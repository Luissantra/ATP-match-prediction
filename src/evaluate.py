import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
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


def evaluar_y_graficar(modelo, X_test, y_test, df_test, features):
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
    _plot_confusion_matrix(y_test, preds, accuracy)
    _plot_feature_importance(modelo, features)
    _plot_accuracy_by_surface(df_test, preds, accuracy)

    return accuracy


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


def _plot_feature_importance(modelo, features):
    feature_labels = {
        'diff_elo': 'Diferencia ELO',
        'diff_rank': 'Diferencia Ranking',
        'diff_age': 'Diferencia Edad',
        'diff_h2h': 'H2H Histórico',
        'diff_form': 'Forma Reciente',
        'tourney_level_num': 'Nivel de Torneo',
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
