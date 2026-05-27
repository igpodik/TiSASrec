#!/usr/bin/env bash
# Usage:
#   ./run.sh train [-v] [--cpu-only] [--data-dir PATH]
#   ./run.sh predict [-v] [--cpu-only] [--out submission.csv] [--data-dir PATH]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"
PIP="${ROOT}/.venv/bin/pip"

if [[ ! -x "$PYTHON" ]]; then
  echo "Creating venv..."
  python3 -m venv .venv
  "$PIP" install -r requirements.txt
fi

cmd="${1:-}"
shift || true

case "$cmd" in
  train)
    exec "$PYTHON" train.py "$@"
    ;;
  predict)
    exec "$PYTHON" predict.py "$@"
    ;;
  *)
    echo "Usage: $0 train [train.py args...]"
    echo "       $0 predict [predict.py args...]"
    echo ""
    echo "Examples:"
    echo "  $0 train -v"
    echo "  $0 train --cpu-only -v"
    echo "  $0 predict -v --out submission.csv"
    exit 1
    ;;
esac
