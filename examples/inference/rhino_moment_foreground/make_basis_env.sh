#!/bin/sh
# Recreate the scratch venv that basis.py runs in: SyncMoments 0.4.0 pins
# jax 0.10.x, the twin's venv has jax 0.11.0, so the two cannot share one
# interpreter. Usage: sh make_basis_env.sh <directory>; then
# <directory>/bin/python basis.py
set -e
DIR=${1:?usage: make_basis_env.sh <venv directory>}
uv venv --python 3.12 "$DIR"
uv pip install --python "$DIR/bin/python" "syncmoments==0.4.0" numpy scipy matplotlib
uv pip list --python "$DIR/bin/python" | grep -i -E "^(syncmoments|jax|jaxlib|numpy|scipy|equinox) "
