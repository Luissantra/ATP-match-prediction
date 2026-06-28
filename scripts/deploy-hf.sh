#!/usr/bin/env bash
# Despliega main a HuggingFace Spaces convirtiendo los .pkl a Git LFS.
#
# La conversión a LFS se hace sobre un CLON EFÍMERO: la historia de tu repo
# (origin / GitHub) NO se toca. Solo el repo del Space queda con los .pkl en LFS.
#
# Uso:
#   export HF_TOKEN=hf_xxx        # token de escritura: https://huggingface.co/settings/tokens
#   ./scripts/deploy-hf.sh
#
# Requisitos: git-lfs instalado (brew install git-lfs).

set -euo pipefail

HF_USER="Luissantra"
HF_SPACE="ATP-Prediction"
SOURCE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: exporta HF_TOKEN con un token de escritura de HuggingFace." >&2
  echo "  https://huggingface.co/settings/tokens (tipo Write)" >&2
  exit 1
fi

command -v git-lfs >/dev/null || { echo "ERROR: git-lfs no instalado (brew install git-lfs)." >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "→ Clonando main a un workspace temporal…"
git clone --quiet --single-branch --branch main "$SOURCE_REPO" "$TMP/repo"
cd "$TMP/repo"

echo "→ Convirtiendo *.pkl a Git LFS (solo en este clon)…"
git lfs install --local
git lfs migrate import --include="*.pkl" --everything --yes

echo "→ Empujando a HuggingFace Space ($HF_USER/$HF_SPACE)…"
git push --force "https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${HF_USER}/${HF_SPACE}" HEAD:main

echo "✓ Desplegado. El Space construirá la imagen Docker en unos minutos:"
echo "  https://huggingface.co/spaces/${HF_USER}/${HF_SPACE}"
