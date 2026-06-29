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
# Reducido de 8 a 5 features (2026-06-26) tras estudio de permutation importance +
# ablación sobre test ciego 2025 (n=2861): diff_h2h, diff_form y tourney_level_num
# tenían importancia ~0 (perm. imp. < 0.001) y su ablación movía el AUC dentro del
# IC95% (±0.009). El ELO ya absorbe la forma reciente; el H2H es débil tras controlar
# por ELO. Todo el lift sobre el baseline ELO (0.694→0.711) viene de rank/age/unranked.
FEATURES = [
    'diff_elo_general', 'diff_elo_sup',   # ELO separado general/superficie (el modelo aprende el peso)
    'diff_rank', 'is_unranked',           # rank capeado a 250 + indicador wildcard/qualifier (rank ausente)
    'diff_age',
    'diff_matches_played',                # Experiencia (partidos profesionales jugados acumulados)
    'diff_tb_ratio',                      # Tasa suavizada de tie-breaks ganados (temple bajo presión)
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
