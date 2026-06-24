import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from src.features import (
    FEATURES, LEVEL_MAP, DEFAULT_LEVEL_NUM, elo_hibrido, vector_from_features,
)

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

modelo = None
elo_general = {}
elo_superficie = {}
stats_jugadores = {}
h2h = {}
form_final = {}


def cargar_modelo():
    global modelo, elo_general, elo_superficie, stats_jugadores, h2h, form_final
    try:
        with open("modelo_atp.pkl", "rb") as f:
            modelo = pickle.load(f)
        with open("stats_jugadores.pkl", "rb") as f:
            metadata = pickle.load(f)
        elo_general = metadata['elo_general']
        elo_superficie = metadata['elo_superficie']
        stats_jugadores = metadata['stats']
        h2h = metadata.get('h2h', {})
        form_final = metadata.get('form', {})
        print("Modelo y estadísticas cargados.")
        return True
    except Exception as e:
        print(f"Advertencia: no se pudieron cargar los pkl: {e}")
        return False


def construir_features(player_a, player_b, surface, tourney_level=None):
    """
    Construye el dict de features para inferencia con la MISMA semántica que el
    entrenamiento (sin train/serve skew). H2H y forma se reconstruyen del historial
    real persistido; el nivel de torneo se mapea del parámetro (default = ATP 250).
    """
    gen_a = elo_general.get(player_a, 1500.0)
    gen_b = elo_general.get(player_b, 1500.0)
    sup_a = elo_superficie.get(surface, {}).get(player_a, 1500.0)
    sup_b = elo_superficie.get(surface, {}).get(player_b, 1500.0)

    rank_a = stats_jugadores.get(player_a, {}).get('rank', 999.0)
    rank_b = stats_jugadores.get(player_b, {}).get('rank', 999.0)
    age_a = stats_jugadores.get(player_a, {}).get('age', 26.0)
    age_b = stats_jugadores.get(player_b, {}).get('age', 26.0)

    # H2H real: ratio de victorias en enfrentamientos previos (0.5/0.5 si no hay historial)
    record = h2h.get(tuple(sorted([player_a, player_b])))
    if record:
        total = record.get(player_a, 0) + record.get(player_b, 0)
        if total > 0:
            ratio_a = record.get(player_a, 0) / total
            ratio_b = record.get(player_b, 0) / total
        else:
            ratio_a = ratio_b = 0.5
    else:
        ratio_a = ratio_b = 0.5

    # Forma real: media de los últimos resultados (0.5 si jugador desconocido)
    form_a = form_final.get(player_a, 0.5)
    form_b = form_final.get(player_b, 0.5)

    level_num = LEVEL_MAP.get(str(tourney_level), DEFAULT_LEVEL_NUM)

    return {
        'diff_elo':  elo_hibrido(gen_a, sup_a) - elo_hibrido(gen_b, sup_b),
        'diff_rank': rank_a - rank_b,
        'diff_age':  age_a - age_b,
        'diff_h2h':  ratio_a - ratio_b,
        'diff_form': form_a - form_b,
        'tourney_level_num': level_num,
    }


with app.app_context():
    cargar_modelo()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/api/players')
def players():
    if not elo_general:
        cargar_modelo()
    players_sorted = sorted(elo_general.items(), key=lambda x: x[1], reverse=True)
    result = [
        {
            "name": name,
            "elo": elo,
            "rank": stats_jugadores.get(name, {}).get('rank', 999),
            "age": round(stats_jugadores.get(name, {}).get('age', 26.0), 1),
        }
        for name, elo in players_sorted
    ]
    return jsonify(result)


@app.route('/api/predict')
def predict():
    if modelo is None:
        if not cargar_modelo():
            return jsonify({"detail": "Modelo no encontrado. Ejecuta python main.py primero."}), 500

    player_a = request.args.get('player_a')
    player_b = request.args.get('player_b')
    surface = request.args.get('surface', 'Hard')

    if not player_a or not player_b:
        return jsonify({"detail": "Faltan parámetros 'player_a' o 'player_b'."}), 400

    if surface not in ('Hard', 'Clay', 'Grass'):
        return jsonify({"detail": "La superficie debe ser Hard, Clay o Grass."}), 400

    if player_a == player_b:
        return jsonify({"detail": "Los jugadores deben ser distintos."}), 400

    tourney_level = request.args.get('tourney_level')  # ej. 'G','M','500','250'; None → default

    gen_a = elo_general.get(player_a, 1500.0)
    gen_b = elo_general.get(player_b, 1500.0)
    sup_a = elo_superficie.get(surface, {}).get(player_a, 1500.0)
    sup_b = elo_superficie.get(surface, {}).get(player_b, 1500.0)
    elo_hybrid_a = elo_hibrido(gen_a, sup_a)
    elo_hybrid_b = elo_hibrido(gen_b, sup_b)

    rank_a = stats_jugadores.get(player_a, {}).get('rank', 999.0)
    rank_b = stats_jugadores.get(player_b, {}).get('rank', 999.0)
    age_a = stats_jugadores.get(player_a, {}).get('age', 26.0)
    age_b = stats_jugadores.get(player_b, {}).get('age', 26.0)

    # Vector de inferencia con la misma semántica que el entrenamiento (sin train/serve skew)
    feat = construir_features(player_a, player_b, surface, tourney_level)
    diff_elo = feat['diff_elo']
    diff_rank = feat['diff_rank']
    diff_age = feat['diff_age']

    try:
        features = pd.DataFrame([vector_from_features(feat)], columns=FEATURES)
        probs = modelo.predict_proba(features)[0]
        prob_a = float(probs[1])
        prob_b = float(probs[0])
    except Exception as e:
        return jsonify({"detail": f"Error al predecir: {e}"}), 500

    return jsonify({
        "player_a": {
            "name": player_a,
            "elo_general": round(gen_a, 1),
            "elo_surface": round(sup_a, 1),
            "elo_hybrid": round(elo_hybrid_a, 1),
            "rank": int(rank_a) if rank_a != 999 else "Sin Ranking",
            "age": round(age_a, 1),
            "prob_victory": round(prob_a * 100, 1),
        },
        "player_b": {
            "name": player_b,
            "elo_general": round(gen_b, 1),
            "elo_surface": round(sup_b, 1),
            "elo_hybrid": round(elo_hybrid_b, 1),
            "rank": int(rank_b) if rank_b != 999 else "Sin Ranking",
            "age": round(age_b, 1),
            "prob_victory": round(prob_b * 100, 1),
        },
        "surface": surface,
        "features_debug": {
            "diff_elo": round(diff_elo, 1),
            "diff_rank": int(diff_rank),
            "diff_age": round(diff_age, 2),
            "diff_h2h": round(feat['diff_h2h'], 3),
            "diff_form": round(feat['diff_form'], 3),
            "tourney_level_num": feat['tourney_level_num'],
        },
        "predicted_winner": player_a if prob_a > prob_b else player_b,
    })


if __name__ == '__main__':
    app.run(port=8000, debug=False)
