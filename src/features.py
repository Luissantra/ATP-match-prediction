"""
Fuente de verdad única del vector de características del modelo.
==============================================================

Tanto el entrenamiento (`src/data_processing.py`, `main.py`) como la inferencia
(`app.py`) construyen el vector de features a partir de aquí. Centralizarlo evita
el *train/serve skew*: que el modelo se entrene con una representación y se sirva
con otra distinta (orden de columnas, pesos del ELO híbrido, defaults, etc.).
"""

# Orden canónico de las features que consume el modelo. NO reordenar sin reentrenar.
#
# Reducido a 5 features tras dos rondas de ablación con bootstrap sobre test ciego
# 2025 (n=2861):
#   - 2026-06-26: diff_h2h, diff_form y tourney_level_num tenían perm. imp. ~0 y su
#     ablación movía el AUC dentro del IC95% (±0.009). El ELO ya absorbe la forma
#     reciente; el H2H es débil tras controlar por ELO.
#   - 2026-06-29: diff_matches_played (ruido puro, IC de ΔAUC cruzaba 0) y diff_tb_ratio
#     (significativa por bootstrap pareado pero aporte trivial, +0.002 AUC) se podaron
#     por minimalismo: solo se mantienen features con relevancia práctica.
# Todo el lift sobre el baseline ELO (0.694→0.709) viene de rank/age/unranked.
FEATURES = [
    'diff_elo_general', 'diff_elo_sup',   # ELO separado general/superficie (el modelo aprende el peso)
    'diff_rank', 'is_unranked',           # rank capeado a 250 + indicador wildcard/qualifier (rank ausente)
    'diff_age',
]

# Ranking máximo considerado. Por encima de este umbral se satura la diferencia de rank.
RANK_CAP = 250


def elo_hibrido(elo_general, elo_superficie, w=0.5):
    """ELO híbrido: w*general + (1-w)*superficie. Se sigue usando en elo.py para actualizar ratings."""
    return w * elo_general + (1 - w) * elo_superficie


def vector_from_features(feat):
    """
    Construye la lista de features en el orden canónico de FEATURES.

    Garantiza que entrenamiento e inferencia produzcan el mismo orden/longitud.
    Lanza KeyError si falta alguna feature (falla ruidoso, no silencioso).
    """
    return [feat[name] for name in FEATURES]
