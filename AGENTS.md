# AGENTS.md

## Purpose
Operational handoff for Codex sessions in `kakaotalk-trained-ai-chatbot`.

## Critical Rules
- Never commit private data: `data/raw/*`, `data/processed/*`, `checkpoints/*`.
- Publish at most one artifact file: `artifacts/model_latest.pt` or `.enc`.
- Bridge stays dry-run unless user explicitly asks `--send`.
- Do not delete/archive user data unless requested.

## Main Commands (keep stable)
- `python -m chatbot.ops archive`
- `python -m chatbot.ops organize`
- `python -m chatbot.ops preprocess`
- `python -m chatbot.ops train`
- `python -m chatbot.ops smoke`
- `python -m chatbot.ops reply "..."`
- `python -m chatbot.ops chat`
- `python -m chatbot.ops bridge --dry`

## Current Training Design (v2)
- Run name default: `room_v2_context`
- Preprocess mode: `context_windows`
- Multi-turn context samples from `configs/paths.yaml > processed.preprocess`
- Loss mode: `response_only`
- Preprocess generates aligned masks:
  - `train_loss_mask.bin`
  - `val_loss_mask.bin`
- Trainer applies masked CE over response tokens only.

## Key Files
- Preprocess: `src/chatbot/preprocess.py`
- Trainer: `src/chatbot/train.py`
- Inference: `src/chatbot/inference.py`
- Ops CLI: `src/chatbot/ops.py`
- Archive utility: `src/chatbot/archive_state.py`
- Smoke test: `src/chatbot/smoke.py`

## Config Ownership
- `configs/paths.yaml`: raw/processed paths + preprocess constants
- `configs/train.yaml`: model/optimization/objective constants
- `configs/gen.yaml`: inference/dialogue/output/smoke/security constants

Rule: prefer config constants edits over adding CLI knobs.

## Resume Behavior
- Auto-resume from `checkpoints/<run_name>/latest.pt`
- Stop via Ctrl+C or STOP file
- Keep `latest.pt`, `best.pt`, snapshots
- Resume safety checks for model/tokenizer compatibility remain enabled

## Minimal Validation
```bash
python -m compileall src
python -m chatbot.ops preprocess
python -m chatbot.ops train
python -m chatbot.ops smoke
python -m chatbot.ops reply "테스트"
```
