import os
import pickle
import sklearn
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from src.features import (
    FEATURES, LEVEL_MAP, DEFAULT_LEVEL_NUM, RANK_CAP, elo_hibrido, vector_from_features,
)

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

modelo = None
todos_modelos = {}    # {nombre: modelo_calibrado} — multi-modelo (E3)
metrics_todos = {}    # {nombre: {accuracy, log_loss, brier, auc}} — E3
elo_general = {}
elo_superficie = {}
stats_jugadores = {}
h2h = {}
form_final = {}


def verificar_version_sklearn(saved_version):
    """
    Compara la versión de sklearn con la que serializó el modelo. Cargar un pkl con
    otra versión puede romper silenciosamente. Devuelve un aviso (str) o None.
    """
    if saved_version is None:
        return None  # pkl antiguo sin la clave: no podemos comprobar
    if saved_version != sklearn.__version__:
        return (f"Aviso: el modelo se entrenó con scikit-learn {saved_version} "
                f"pero está cargado con {sklearn.__version__}. Reentrena con 'python main.py'.")
    return None


def cargar_modelo():
    global modelo, todos_modelos, metrics_todos
    global elo_general, elo_superficie, stats_jugadores, h2h, form_final
    try:
        with open("models/stats_jugadores.pkl", "rb") as f:
            metadata = pickle.load(f)
        elo_general = metadata['elo_general']
        elo_superficie = metadata['elo_superficie']
        stats_jugadores = metadata['stats']
        h2h = metadata.get('h2h', {})
        form_final = metadata.get('form', {})
        aviso = verificar_version_sklearn(metadata.get('sklearn_version'))
        if aviso:
            print(aviso)

        with open("models/modelos_atp.pkl", "rb") as f:
            todos_modelos = pickle.load(f)
        with open("models/metrics_atp.pkl", "rb") as f:
            metrics_todos = pickle.load(f)
        modelo = todos_modelos['gbm']
        print(f"Modelos cargados: {list(todos_modelos.keys())}")
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
        'diff_elo_general': gen_a - gen_b,
        'diff_elo_sup':     sup_a - sup_b,
        'diff_rank':        min(rank_a, RANK_CAP) - min(rank_b, RANK_CAP),
        'is_unranked':      int(rank_a >= 999) - int(rank_b >= 999),
        'diff_age':         age_a - age_b,
        'diff_h2h':         ratio_a - ratio_b,
        'diff_form':        form_a - form_b,
        'tourney_level_num': level_num,
    }


def _predecir_con(modelo_usado, player_a, player_b, surface, tourney_level):
    """
    Ejecuta la predicción para un modelo dado. Reutilizado por /api/predict y
    /api/predict_all para evitar duplicar la lógica de features.
    """
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

    feat = construir_features(player_a, player_b, surface, tourney_level)
    features = pd.DataFrame([vector_from_features(feat)], columns=FEATURES)
    probs = modelo_usado.predict_proba(features)[0]
    prob_a = float(probs[1])
    prob_b = float(probs[0])

    unknown_a = player_a not in elo_general
    unknown_b = player_b not in elo_general

    return {
        "player_a": {
            "name": player_a,
            "elo_general": round(gen_a, 1),
            "elo_surface": round(sup_a, 1),
            "elo_hybrid": round(elo_hybrid_a, 1),
            "rank": int(rank_a) if rank_a != 999 else "Sin Ranking",
            "age": round(age_a, 1),
            "prob_victory": round(prob_a * 100, 1),
            "unknown": unknown_a,
        },
        "player_b": {
            "name": player_b,
            "elo_general": round(gen_b, 1),
            "elo_surface": round(sup_b, 1),
            "elo_hybrid": round(elo_hybrid_b, 1),
            "rank": int(rank_b) if rank_b != 999 else "Sin Ranking",
            "age": round(age_b, 1),
            "prob_victory": round(prob_b * 100, 1),
            "unknown": unknown_b,
        },
        "surface": surface,
        "features_debug": {
            "diff_elo_general": round(feat['diff_elo_general'], 1),
            "diff_elo_sup":     round(feat['diff_elo_sup'], 1),
            "diff_rank":        int(feat['diff_rank']),
            "is_unranked":      int(feat['is_unranked']),
            "diff_age":         round(feat['diff_age'], 2),
            "diff_h2h":         round(feat['diff_h2h'], 3),
            "diff_form":        round(feat['diff_form'], 3),
            "tourney_level_num": feat['tourney_level_num'],
        },
        "predicted_winner": player_a if prob_a > prob_b else player_b,
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


@app.route('/api/models')
def models():
    """Lista los modelos disponibles con sus métricas de test ciego 2026."""
    resultado = []
    for nombre, metricas in metrics_todos.items():
        resultado.append({
            'nombre': nombre,
            'accuracy': metricas.get('accuracy'),
            'log_loss': metricas.get('log_loss'),
            'brier':    metricas.get('brier'),
            'auc':      metricas.get('auc'),
        })
    resultado.sort(key=lambda x: (x['log_loss'] is None, x['log_loss']))
    return jsonify(resultado)


@app.route('/api/predict')
def predict():
    if modelo is None:
        if not cargar_modelo():
            return jsonify({"detail": "Modelo no encontrado. Ejecuta python main.py primero."}), 500

    player_a = request.args.get('player_a')
    player_b = request.args.get('player_b')
    surface = request.args.get('surface', 'Hard')
    model_name = request.args.get('model', 'gbm')
    tourney_level = request.args.get('tourney_level')

    if not player_a or not player_b:
        return jsonify({"detail": "Faltan parámetros 'player_a' o 'player_b'."}), 400
    if surface not in ('Hard', 'Clay', 'Grass'):
        return jsonify({"detail": "La superficie debe ser Hard, Clay o Grass."}), 400
    if player_a == player_b:
        return jsonify({"detail": "Los jugadores deben ser distintos."}), 400

    # Selección de modelo: prioriza todos_modelos, fallback a modelo principal
    if todos_modelos:
        if model_name not in todos_modelos:
            return jsonify({
                "detail": f"Modelo '{model_name}' no disponible. Opciones: {list(todos_modelos.keys())}"
            }), 400
        modelo_usado = todos_modelos[model_name]
    else:
        modelo_usado = modelo

    try:
        resultado = _predecir_con(modelo_usado, player_a, player_b, surface, tourney_level)
        resultado['model_used'] = model_name
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"detail": f"Error al predecir: {e}"}), 500


@app.route('/api/predict_all')
def predict_all():
    """Devuelve las probabilidades de todos los modelos para el mismo partido."""
    if modelo is None:
        if not cargar_modelo():
            return jsonify({"detail": "Modelo no encontrado. Ejecuta python main.py primero."}), 500

    player_a = request.args.get('player_a')
    player_b = request.args.get('player_b')
    surface = request.args.get('surface', 'Hard')
    tourney_level = request.args.get('tourney_level')

    if not player_a or not player_b:
        return jsonify({"detail": "Faltan parámetros 'player_a' o 'player_b'."}), 400
    if surface not in ('Hard', 'Clay', 'Grass'):
        return jsonify({"detail": "La superficie debe ser Hard, Clay o Grass."}), 400
    if player_a == player_b:
        return jsonify({"detail": "Los jugadores deben ser distintos."}), 400

    modelos_a_usar = todos_modelos if todos_modelos else {'gbm': modelo}
    predictions = {}
    for nombre, m in modelos_a_usar.items():
        try:
            res = _predecir_con(m, player_a, player_b, surface, tourney_level)
            predictions[nombre] = {
                'prob_a': res['player_a']['prob_victory'],
                'prob_b': res['player_b']['prob_victory'],
                'predicted_winner': res['predicted_winner'],
            }
        except Exception as e:
            predictions[nombre] = {'error': str(e)}

    return jsonify({
        'player_a': player_a,
        'player_b': player_b,
        'surface': surface,
        'predictions': predictions,
    })


if __name__ == '__main__':
    app.run(port=8000, debug=False)
