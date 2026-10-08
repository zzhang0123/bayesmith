#!/bin/sh
# The whole Demo C pipeline. Usage: sh run.sh <basis venv python> [--quick]
# Environment: RHEPLICANT (the rheplicant checkout; default below).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
BASIS_PY=${1:?usage: run.sh <python of the SyncMoments venv> [--quick]}
shift
QUICK=${1:-}
RHEPLICANT=${RHEPLICANT:-/Users/zzhang/projects/rheplicant}
TWIN_PY="$RHEPLICANT/.venv/bin/python"
export PYTHONPATH="$RHEPLICANT/examples"
"$BASIS_PY" "$HERE/basis.py"
"$TWIN_PY" "$HERE/recover.py" $QUICK
"$TWIN_PY" "$HERE/identify.py" $QUICK
"$TWIN_PY" "$HERE/model.py" --sample 10
"$TWIN_PY" "$HERE/figures.py" $QUICK
python3 "$HERE/report.py" $QUICK
