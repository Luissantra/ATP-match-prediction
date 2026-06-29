from io import StringIO

import pandas as pd
import requests

TML_ONGOING_URL = "https://stats.tennismylife.org/data/ongoing_tourneys.csv"

_LEVEL_PRIORITY = {'G': 0, 'A': 1, '500': 2, '250': 3}


def descargar_ongoing(url=TML_ONGOING_URL) -> pd.DataFrame:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"No se pudo descargar ongoing_tourneys.csv: {e}")
    return pd.read_csv(StringIO(resp.text))


def listar_torneos(df: pd.DataFrame) -> list:
    if df.empty or 'tourney_level' not in df.columns:
        return []

    df = df[df['tourney_level'] != 'D'].copy()
    if df.empty:
        return []

    valid_surfaces = {'Hard', 'Clay', 'Grass'}
    torneos = []
    for tourney_id, grupo in df.groupby('tourney_id'):
        row = grupo.iloc[0]
        surface = row['surface'] if row['surface'] in valid_surfaces else 'Hard'
        level = str(row.get('tourney_level', '250'))
        jugadores = pd.concat([grupo['winner_name'], grupo['loser_name']]).nunique()
        torneos.append({
            'name': row['tourney_name'],
            'surface': surface,
            'level': level,
            'draw_size': int(jugadores),
            'tourney_id': str(tourney_id),
        })

    torneos.sort(key=lambda t: _LEVEL_PRIORITY.get(t['level'], 99))
    return torneos
