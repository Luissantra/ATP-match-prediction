import sys
import os

import sklearn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import app


def test_version_coincidente_no_avisa():
    assert app.verificar_version_sklearn(sklearn.__version__) is None


def test_version_distinta_devuelve_aviso():
    aviso = app.verificar_version_sklearn("0.0.1-vieja")
    assert aviso is not None
    assert "0.0.1-vieja" in aviso
    assert sklearn.__version__ in aviso


def test_version_ausente_no_revienta():
    # pkl antiguo sin la clave → None (sin aviso, no excepción)
    assert app.verificar_version_sklearn(None) is None


def test_validar_metadata_pkl_estructura_correcta():
    metadata = {
        'elo_general': {'Djokovic': 1800.0},
        'elo_superficie': {'Djokovic': {'Hard': 1820.0}},
        'stats': {'Djokovic': {'rank': 1.0, 'age': 36.0}},
        'h2h': {},
        'form': {},
        'sklearn_version': '1.9.0',
    }
    assert app.validar_metadata_pkl(metadata) is None


def test_validar_metadata_pkl_clave_faltante():
    metadata = {'elo_general': {}}  # faltan claves obligatorias
    error = app.validar_metadata_pkl(metadata)
    assert error is not None
    assert 'elo_superficie' in error or 'stats' in error


def test_validar_metadata_pkl_no_es_dict():
    error = app.validar_metadata_pkl("corrupted")
    assert error is not None
