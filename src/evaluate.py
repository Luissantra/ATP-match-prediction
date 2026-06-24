import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluar_y_graficar(modelo, X_test, y_test, df_test, features):
    preds = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    print(f"\nPRECISIÓN FINAL (ACCURACY): {accuracy:.2%}")
    print(classification_report(y_test, preds, target_names=['Derrota A', 'Victoria A']))

    os.makedirs("plots", exist_ok=True)
    _plot_confusion_matrix(y_test, preds, accuracy)
    _plot_feature_importance(modelo, features)
    _plot_accuracy_by_surface(df_test, preds, accuracy)

    return accuracy


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
