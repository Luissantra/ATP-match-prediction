import os

def parse_requirements(filepath):
    """Parsea un archivo requirements.txt y devuelve un dict de pkg -> version."""
    dependencies = {}
    if not os.path.exists(filepath):
        return dependencies
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignorar comentarios y líneas vacías
            if not line or line.startswith('#'):
                continue
            # Parsear pkg==version o pkg>=version, etc.
            if '==' in line:
                pkg, val = line.split('==', 1)
                dependencies[pkg.strip().lower()] = val.strip()
    return dependencies


def test_requirements_sync():
    """Verifica que las dependencias comunes en requirements-serve.txt estén en sync con requirements.txt."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    req_file = os.path.join(base_dir, 'requirements.txt')
    serve_file = os.path.join(base_dir, 'requirements-serve.txt')

    reqs = parse_requirements(req_file)
    serve_reqs = parse_requirements(serve_file)

    assert len(reqs) > 0, "requirements.txt no debería estar vacío"
    assert len(serve_reqs) > 0, "requirements-serve.txt no debería estar vacío"

    # Cada dependencia en requirements-serve.txt debe existir en requirements.txt y coincidir exactamente
    for pkg, version in serve_reqs.items():
        assert pkg in reqs, f"La dependencia '{pkg}' está en requirements-serve.txt pero falta en requirements.txt"
        assert reqs[pkg] == version, (
            f"Desajuste de versión para '{pkg}': "
            f"requirements.txt tiene '{reqs[pkg]}' pero requirements-serve.txt tiene '{version}'"
        )
