"""
Fuente de verdad única del vector de características del modelo.
==============================================================

Tanto el entrenamiento (`src/data_processing.py`, `main.py`) como la inferencia
(`app.py`) construyen el vector de features a partir de aquí. Centralizarlo evita
el *train/serve skew*: que el modelo se entrene con una representación y se sirva
con otra distinta (orden de columnas, pesos del ELO híbrido, defaults, etc.).
"""

# Orden canónico de las features que consume el modelo. NO reordenar sin reentrenar.
FEATURES = [
    'diff_elo_general', 'diff_elo_sup',   # I3: GBM aprende el peso óptimo (antes 50/50 fijo)
    'diff_rank', 'is_unranked',            # I2: rank capeado a 250 + indicador wildcard/qualifier
    'diff_age', 'diff_h2h', 'diff_form', 'tourney_level_num',
]

# Ranking máximo considerado. Por encima de este umbral el jugador se considera sin ranking.
RANK_CAP = 250

# Codificación ordinal del nivel de torneo (mayor = más importante).
LEVEL_MAP = {
    'G': 5,
    'M': 4,
    'F': 3,
    'O': 3,
    '500': 2, 'A': 2,
    '250': 1, 'D': 1,
}

# Nivel por defecto cuando no se conoce el torneo: 1 (ATP 250), el más común del
# circuito. Antes se usaba 3 (Finals/Olympics), lo que introducía sesgo sistemático.
DEFAULT_LEVEL_NUM = 1


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
