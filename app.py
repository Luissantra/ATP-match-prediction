import os
import sys
import pickle
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

PORT = 8000

# Variables globales para modelo y metadatos
modelo = None
elo_general = {}
elo_superficie = {}
stats_jugadores = {}

def cargar_modelo():
    global modelo, elo_general, elo_superficie, stats_jugadores
    try:
        with open("modelo_atp.pkl", "rb") as f:
            modelo = pickle.load(f)
        with open("stats_jugadores.pkl", "rb") as f:
            metadata = pickle.load(f)
        elo_general = metadata['elo_general']
        elo_superficie = metadata['elo_superficie']
        stats_jugadores = metadata['stats']
        print("✅ Modelo y estadísticas cargados con éxito (Pickle).")
        return True
    except Exception as e:
        print(f"⚠️ Advertencia: No se pudieron cargar los archivos pickle: {e}")
        return False

# Intentar cargar al iniciar
cargar_modelo()

class ATPPredictHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Permitir CORS para desarrollo local
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        # 1. API: Obtener lista de jugadores
        if path == "/api/players":
            if not elo_general:
                cargar_modelo()
            
            # Devolver los jugadores ordenados por su ELO general decreciente
            players_sorted = sorted(elo_general.items(), key=lambda x: x[1], reverse=True)
            res = [
                {
                    "name": name,
                    "elo": elo,
                    "rank": stats_jugadores.get(name, {}).get('rank', 999),
                    "age": round(stats_jugadores.get(name, {}).get('age', 26.0), 1)
                }
                for name, elo in players_sorted
            ]
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode('utf-8'))
            return

        # 2. API: Predicción de partido
        elif path == "/api/predict":
            if modelo is None:
                if not cargar_modelo():
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"detail": "Modelo predictor no entrenado o no encontrado. Ejecuta python main.py primero."}).encode())
                    return
            
            # Obtener parámetros
            player_a_list = query.get("player_a")
            player_b_list = query.get("player_b")
            surface_list = query.get("surface", ["Hard"])

            if not player_a_list or not player_b_list:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"detail": "Faltan parámetros 'player_a' o 'player_b'."}).encode())
                return

            player_a = player_a_list[0]
            player_b = player_b_list[0]
            surface = surface_list[0]

            if surface not in ["Hard", "Clay", "Grass"]:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"detail": "La superficie debe ser Hard, Clay o Grass."}).encode())
                return

            # ELOs y diferencias
            gen_a = elo_general.get(player_a, 1500.0)
            gen_b = elo_general.get(player_b, 1500.0)
            
            sup_a = elo_superficie.get(surface, {}).get(player_a, 1500.0)
            sup_b = elo_superficie.get(surface, {}).get(player_b, 1500.0)
            
            elo_hybrid_a = 0.5 * gen_a + 0.5 * sup_a
            elo_hybrid_b = 0.5 * gen_b + 0.5 * sup_b
            diff_elo = elo_hybrid_a - elo_hybrid_b
            
            # Rankings
            rank_a = stats_jugadores.get(player_a, {}).get('rank', 999.0)
            rank_b = stats_jugadores.get(player_b, {}).get('rank', 999.0)
            diff_rank = rank_a - rank_b
            
            # Edades
            age_a = stats_jugadores.get(player_a, {}).get('age', 26.0)
            age_b = stats_jugadores.get(player_b, {}).get('age', 26.0)
            diff_age = age_a - age_b

            # Ejecutar modelo
            try:
                input_features = np.array([[diff_elo, diff_rank, diff_age]])
                probs = modelo.predict_proba(input_features)[0]
                prob_a = probs[1]
                prob_b = probs[0]
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"detail": f"Error al ejecutar predicción: {str(e)}"}).encode())
                return

            res = {
                "player_a": {
                    "name": player_a,
                    "elo_general": round(gen_a, 1),
                    "elo_surface": round(sup_a, 1),
                    "elo_hybrid": round(elo_hybrid_a, 1),
                    "rank": int(rank_a) if rank_a != 999 else "Sin Ranking (999)",
                    "age": round(age_a, 1),
                    "prob_victory": round(prob_a * 100, 1)
                },
                "player_b": {
                    "name": player_b,
                    "elo_general": round(gen_b, 1),
                    "elo_surface": round(sup_b, 1),
                    "elo_hybrid": round(elo_hybrid_b, 1),
                    "rank": int(rank_b) if rank_b != 999 else "Sin Ranking (999)",
                    "age": round(age_b, 1),
                    "prob_victory": round(prob_b * 100, 1)
                },
                "surface": surface,
                "features_debug": {
                    "diff_elo": round(diff_elo, 1),
                    "diff_rank": int(diff_rank),
                    "diff_age": round(diff_age, 2)
                },
                "predicted_winner": player_a if prob_a > prob_b else player_b
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode('utf-8'))
            return

        # 3. Servir Frontend (Estáticos y HTML)
        else:
            # Normalizar la ruta del archivo
            if path == "/":
                file_path = "templates/index.html"
            else:
                # Quitar el prefijo '/' para buscar localmente
                file_path = path.lstrip("/")

            # Prevenir ataques de Directory Traversal
            abs_file_path = os.path.abspath(file_path)
            abs_cwd = os.path.abspath(os.getcwd())
            if not abs_file_path.startswith(abs_cwd):
                self.send_error(403, "Acceso no permitido.")
                return

            if os.path.exists(file_path) and os.path.isfile(file_path):
                # Determinar Content-Type
                content_type = "text/plain"
                if file_path.endswith(".html"):
                    content_type = "text/html; charset=utf-8"
                elif file_path.endswith(".css"):
                    content_type = "text/css"
                elif file_path.endswith(".js"):
                    content_type = "application/javascript"
                elif file_path.endswith(".png"):
                    content_type = "image/png"
                elif file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
                    content_type = "image/jpeg"
                elif file_path.endswith(".ico"):
                    content_type = "image/x-icon"

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                
                # Leer y responder en binario
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "Archivo no encontrado.")
                return

def run(server_class=HTTPServer, handler_class=ATPPredictHandler):
    server_address = ('', PORT)
    httpd = server_class(server_address, handler_class)
    print(f"🚀 Servidor HTTP ATP Prediction corriendo en http://localhost:{PORT} ...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrando el servidor...")
        httpd.server_close()

if __name__ == "__main__":
    run()
