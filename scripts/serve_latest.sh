#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-room_lora_qwen25_3b_base}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
CONFIG_SFT="${CONFIG_SFT:-configs/sft.yaml}"
ENV_PATH="${ENV_PATH:-.env}"

if [[ -x ".venv/Scripts/python.exe" ]]; then
  PY_EXE=".venv/Scripts/python.exe"
elif [[ -x ".venv/bin/python" ]]; then
  PY_EXE=".venv/bin/python"
else
  PY_EXE="python"
fi

pick_adapter_for_run() {
  local run="$1"
  local ckpt=""
  ckpt="$(ls -d "checkpoints_lora/$run"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "$ckpt" ]]; then
    echo "$ckpt"
    return 0
  fi
  if [[ -d "checkpoints_lora/$run/adapter_latest" ]]; then
    echo "checkpoints_lora/$run/adapter_latest"
    return 0
  fi
  if [[ -d "checkpoints_lora/$run/adapter_best" ]]; then
    echo "checkpoints_lora/$run/adapter_best"
    return 0
  fi
  return 1
}

pick_latest_run() {
  local latest=""
  latest="$(
    find checkpoints_lora -mindepth 1 -maxdepth 1 -type d ! -name "_backup" -printf '%T@ %f\n' 2>/dev/null \
      | sort -nr \
      | awk '{print $2}'
  )"
  while IFS= read -r run; do
    [[ -z "$run" ]] && continue
    if pick_adapter_for_run "$run" >/dev/null; then
      echo "$run"
      return 0
    fi
  done <<< "$latest"
  return 1
}

SELECTED_RUN="$RUN_NAME"
ADAPTER="$(pick_adapter_for_run "$SELECTED_RUN" || true)"

if [[ -z "$ADAPTER" && "$RUN_NAME" != *_cpt ]]; then
  CPT_RUN="${RUN_NAME}_cpt"
  ADAPTER="$(pick_adapter_for_run "$CPT_RUN" || true)"
  if [[ -n "$ADAPTER" ]]; then
    SELECTED_RUN="$CPT_RUN"
    echo "[warn] run '$RUN_NAME' not found; using '$SELECTED_RUN' instead."
  fi
fi

if [[ -z "$ADAPTER" ]]; then
  AUTO_RUN="$(pick_latest_run || true)"
  if [[ -n "$AUTO_RUN" ]]; then
    ADAPTER="$(pick_adapter_for_run "$AUTO_RUN")"
    SELECTED_RUN="$AUTO_RUN"
    echo "[warn] run '$RUN_NAME' not found; auto-selected latest run '$SELECTED_RUN'."
  fi
fi

if [[ -z "$ADAPTER" ]]; then
  echo "[error] no adapter found."
  echo "[hint] set RUN_NAME manually, e.g. RUN_NAME=room_lora_qwen25_3b_base_cpt bash scripts/serve_latest.sh"
  exit 1
fi

echo "[serve] run=$SELECTED_RUN"
echo "[serve] adapter=$ADAPTER"
echo "[serve] endpoint=http://$HOST:$PORT/v1/chat"

exec "$PY_EXE" -m chatbot.sft_ops serve \
  --host "$HOST" \
  --port "$PORT" \
  --config_sft "$CONFIG_SFT" \
  --env_path "$ENV_PATH" \
  --adapter "$ADAPTER"
