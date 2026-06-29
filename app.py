import os
import pickle
import sklearn
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from src.features import RANK_CAP, elo_hibrido, vector_from_features

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Vigencia del modelo (R4): ventana de datos del artefacto servido. Refleja
# main.py. Inicializado con fallbacks y actualizado dinámicamente al cargar stats_jugadores.pkl.
TRAINED_THROUGH = 2024   # último año incluido en el entrenamiento
TESTED_ON = 2025         # año del test ciego principal

modelo = None
metrics = {}          # {accuracy, log_loss, brier, auc} del test ciego 2025
coeficientes = {}     # {feature: {coef, odds_ratio}} — explicabilidad del modelo lineal
elo_general = {}
elo_superficie = {}
stats_jugadores = {}


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


def validar_metadata_pkl(metadata):
    """
    Valida la estructura del dict cargado de stats_jugadores.pkl.
    Devuelve un mensaje de error (str) si hay claves ausentes o tipo incorrecto, None si OK.
    """
    if not isinstance(metadata, dict):
        return f"stats_jugadores.pkl corrupto: se esperaba dict, se obtuvo {type(metadata).__name__}"
    claves_requeridas = {'elo_general', 'elo_superficie', 'stats'}
    faltantes = claves_requeridas - metadata.keys()
    if faltantes:
        return f"stats_jugadores.pkl corrupto: claves ausentes {sorted(faltantes)}"
    return None


def cargar_modelo():
    global modelo, metrics, coeficientes
    global elo_general, elo_superficie, stats_jugadores, TRAINED_THROUGH, TESTED_ON
    try:
        with open("models/stats_jugadores.pkl", "rb") as f:
            metadata = pickle.load(f)
        error_estructura = validar_metadata_pkl(metadata)
        if error_estructura:
            print(error_estructura)
            return False
        elo_general = metadata['elo_general']
        elo_superficie = metadata['elo_superficie']
        stats_jugadores = metadata['stats']
        coeficientes = metadata.get('coeficientes', {})
        TRAINED_THROUGH = metadata.get('trained_through', 2024)
        TESTED_ON = metadata.get('tested_on', 2025)
        aviso = verificar_version_sklearn(metadata.get('sklearn_version'))
        if aviso:
            print(aviso)

        with open("models/modelos_atp.pkl", "rb") as f:
            modelo = pickle.load(f)
        with open("models/metrics_atp.pkl", "rb") as f:
            metrics = pickle.load(f)
        print("Modelo LogReg cargado.")
        return True
    except Exception as e:
        print(f"Advertencia: no se pudieron cargar los pkl: {e}")
        return False


def construir_features(player_a, player_b, surface):
    """
    Construye el dict de features para inferencia con la MISMA semántica que el
    entrenamiento (sin train/serve skew). 5 features: ELO general/superficie, ranking
    capeado, indicador sin-ranking y diferencia de edad.
    """
    gen_a = elo_general.get(player_a, 1500.0)
    gen_b = elo_general.get(player_b, 1500.0)
    sup_a = elo_superficie.get(surface, {}).get(player_a, 1500.0)
    sup_b = elo_superficie.get(surface, {}).get(player_b, 1500.0)

    rank_a = stats_jugadores.get(player_a, {}).get('rank', 999.0)
    rank_b = stats_jugadores.get(player_b, {}).get('rank', 999.0)
    age_a = stats_jugadores.get(player_a, {}).get('age', 26.0)
    age_b = stats_jugadores.get(player_b, {}).get('age', 26.0)

    # is_unranked: jugador sin ranking real conocido (desconocido o sentinela 999).
    unranked_a = int(player_a not in stats_jugadores or rank_a >= 999)
    unranked_b = int(player_b not in stats_jugadores or rank_b >= 999)

    # Experiencia
    mp_a = stats_jugadores.get(player_a, {}).get('matches_played', 0)
    mp_b = stats_jugadores.get(player_b, {}).get('matches_played', 0)
    
    # Tie-breaks
    tb_w_a = stats_jugadores.get(player_a, {}).get('tb_wins', 0)
    tb_p_a = stats_jugadores.get(player_a, {}).get('tb_played', 0)
    tb_w_b = stats_jugadores.get(player_b, {}).get('tb_wins', 0)
    tb_p_b = stats_jugadores.get(player_b, {}).get('tb_played', 0)

    # Suavizado bayesiano Beta(2, 2)
    tb_ratio_a = (tb_w_a + 2.0) / (tb_p_a + 4.0)
    tb_ratio_b = (tb_w_b + 2.0) / (tb_p_b + 4.0)

    return {
        'diff_elo_general': gen_a - gen_b,
        'diff_elo_sup':     sup_a - sup_b,
        'diff_rank':        min(rank_a, RANK_CAP) - min(rank_b, RANK_CAP),
        'is_unranked':      unranked_a - unranked_b,
        'diff_age':         age_a - age_b,
        'diff_matches_played': mp_a - mp_b,
        'diff_tb_ratio':      tb_ratio_a - tb_ratio_b,
    }


def _predecir_con(modelo_usado, player_a, player_b, surface):
    """Ejecuta la predicción para un modelo dado y arma la respuesta de la API."""
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

    feat = construir_features(player_a, player_b, surface)
    # El modelo se entrena sobre numpy; servir numpy evita el warning de feature-names.
    features = np.array([vector_from_features(feat)])
    probs = modelo_usado.predict_proba(features)[0]
    prob_a = float(probs[1])
    prob_b = float(probs[0])

    unknown_a = player_a not in elo_general
    unknown_b = player_b not in elo_general

    SURFACES = ('Hard', 'Clay', 'Grass')
    elo_surfaces_a = {s: round(elo_superficie.get(s, {}).get(player_a, 1500.0), 1) for s in SURFACES}
    elo_surfaces_b = {s: round(elo_superficie.get(s, {}).get(player_b, 1500.0), 1) for s in SURFACES}

    return {
        "player_a": {
            "name": player_a,
            "elo_general": round(gen_a, 1),
            "elo_surface": round(sup_a, 1),
            "elo_surfaces": elo_surfaces_a,
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
            "elo_surfaces": elo_surfaces_b,
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
            "diff_matches_played": int(feat['diff_matches_played']),
            "diff_tb_ratio":      round(feat['diff_tb_ratio'], 4),
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


@app.route('/api/model')
def model_info():
    """Métricas del modelo (test ciego 2025) y coeficientes para explicabilidad."""
    # Extraer plots_data si existe para no mezclarlo con las métricas puras
    metrics_only = {k: v for k, v in metrics.items() if k != 'plots_data'}
    plots_data = metrics.get('plots_data', {})
    
    return jsonify({
        'nombre': 'logreg',
        'metrics': metrics_only,
        'coeficientes': coeficientes,
        'plots_data': plots_data,
        'trained_through': TRAINED_THROUGH,
        'tested_on': TESTED_ON,
    })


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

    try:
        resultado = _predecir_con(modelo, player_a, player_b, surface)
        resultado['model_used'] = 'logreg'
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"detail": f"Error al predecir: {e}"}), 500


@app.route('/api/tournament/info')
def tournament_info():
    tourney = request.args.get('tournament', 'Australian Open')

    import pandas as pd
    ongoing_path = os.path.join("data", "ongoing_tourneys.csv")
    if not os.path.exists(ongoing_path):
        return jsonify({"detail": f"No se encontró el archivo de torneos en curso: {ongoing_path}"}), 404

    try:
        df_ongoing = pd.read_csv(ongoing_path)
        df_tourney = df_ongoing[df_ongoing['tourney_name'].str.lower() == tourney.lower()].copy()

        if len(df_tourney) == 0:
            return jsonify({"detail": f"No hay datos para el torneo '{tourney}' en ongoing_tourneys.csv"}), 404

        # Determinar la primera ronda
        rondas_disponibles = df_tourney['round'].unique()
        for r in ['R128', 'R64', 'R32', 'R16']:
            if r in rondas_disponibles:
                first_round = r
                break
        else:
            first_round = rondas_disponibles[0]

        df_first_round = df_tourney[df_tourney['round'] == first_round].copy()
        df_first_round = df_first_round.sort_values(by='match_num')

        # Extraer partidos del cuadro inicial
        partidos = []
        for _, row in df_first_round.iterrows():
            player_a = row['winner_name']
            player_b = row['loser_name']
            
            p_rank_a = stats_jugadores.get(player_a, {}).get('rank', 999.0)
            p_rank_b = stats_jugadores.get(player_b, {}).get('rank', 999.0)
            p_elo_a = elo_general.get(player_a, 1500.0)
            p_elo_b = elo_general.get(player_b, 1500.0)
            
            partidos.append({
                "match_num": int(row['match_num']),
                "player_a": {
                    "name": player_a,
                    "rank": int(p_rank_a) if p_rank_a < 999 else "S/R",
                    "elo": round(p_elo_a, 1)
                },
                "player_b": {
                    "name": player_b,
                    "rank": int(p_rank_b) if p_rank_b < 999 else "S/R",
                    "elo": round(p_elo_b, 1)
                }
            })

        # Obtener lista única de participantes ordenada por ELO general
        participantes = []
        seen = set()
        for p in partidos:
            for side in ['player_a', 'player_b']:
                name = p[side]['name']
                if name not in seen:
                    seen.add(name)
                    participantes.append({
                        "name": name,
                        "rank": p[side]['rank'],
                        "elo": p[side]['elo']
                    })
        participantes.sort(key=lambda x: x['elo'], reverse=True)

        return jsonify({
            "tournament": tourney,
            "surface": df_first_round['surface'].iloc[0] if 'surface' in df_first_round.columns else 'Hard',
            "draw_size": len(participantes),
            "round": first_round,
            "participants": participantes,
            "matchups": partidos
        })
    except Exception as e:
        return jsonify({"detail": f"Error al obtener info del torneo: {str(e)}"}), 500


@app.route('/api/tournament/simulate')
def simulate_tournament():
    if modelo is None:
        if not cargar_modelo():
            return jsonify({"detail": "Modelo no encontrado. Ejecuta python main.py primero."}), 500

    tourney = request.args.get('tournament', 'Australian Open')
    n_sims = request.args.get('simulations', default=5000, type=int)

    # Validar simulations
    if n_sims <= 0 or n_sims > 10000:
        return jsonify({"detail": "El número de simulaciones debe estar entre 1 y 10000."}), 400

    import pandas as pd
    from src.simulator import simular_torneo_montecarlo

    ongoing_path = os.path.join("data", "ongoing_tourneys.csv")
    if not os.path.exists(ongoing_path):
        return jsonify({"detail": f"No se encontró el archivo de torneos en curso: {ongoing_path}"}), 404

    try:
        df_ongoing = pd.read_csv(ongoing_path)
        # Filtrar por torneo
        df_tourney = df_ongoing[df_ongoing['tourney_name'].str.lower() == tourney.lower()].copy()

        if len(df_tourney) == 0:
            return jsonify({"detail": f"No hay datos para el torneo '{tourney}' en ongoing_tourneys.csv"}), 404

        # Determinar la primera ronda
        rondas_disponibles = df_tourney['round'].unique()
        # Orden de prioridad de rondas iniciales
        for r in ['R128', 'R64', 'R32', 'R16']:
            if r in rondas_disponibles:
                first_round = r
                break
        else:
            first_round = rondas_disponibles[0]  # fallback

        df_first_round = df_tourney[df_tourney['round'] == first_round].copy()
        df_first_round = df_first_round.sort_values(by='match_num')

        initial_draw = []
        for _, row in df_first_round.iterrows():
            initial_draw.append(row['winner_name'])
            initial_draw.append(row['loser_name'])

        # Determinar superficie
        surface = df_first_round['surface'].iloc[0] if 'surface' in df_first_round.columns else 'Hard'
        if surface not in ('Hard', 'Clay', 'Grass'):
            surface = 'Hard'

        # Ejecutar simulación
        df_prob = simular_torneo_montecarlo(
            initial_draw, surface, modelo, elo_general, elo_superficie, stats_jugadores,
            n_simulaciones=n_sims, seed=42
        )

        # Convertir a formato amigable para el frontend
        resultados = []
        for player_name, row in df_prob.iterrows():
            p_elo_gen = elo_general.get(player_name, 1500.0)
            p_elo_sup = elo_superficie.get(surface, {}).get(player_name, 1500.0)
            p_rank = stats_jugadores.get(player_name, {}).get('rank', 999.0)
            
            prob_rondas = row.to_dict()
            
            resultados.append({
                "name": player_name,
                "elo_general": round(p_elo_gen, 1),
                "elo_surface": round(p_elo_sup, 1),
                "rank": int(p_rank) if p_rank < 999 else "S/R",
                "probabilities": {k: round(v, 2) for k, v in prob_rondas.items()}
            })

        return jsonify({
            "tournament": tourney,
            "surface": surface,
            "round_keys": list(df_prob.columns),
            "simulations": n_sims,
            "results": resultados
        })
    except Exception as e:
        return jsonify({"detail": f"Error al simular torneo: {str(e)}"}), 500




if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
