#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: ./scripts/run.sh <command> [args...]"
  exit 1
fi

if [[ -f ".venv/Scripts/python.exe" ]]; then
  PY=".venv/Scripts/python.exe"
elif [[ -f ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

"$PY" -m chatbot.sft_ops "$@"
