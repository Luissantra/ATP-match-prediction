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
