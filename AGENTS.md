# AGENTS.md

## Purpose
Operational guide for Codex sessions in `kakaotalk-trained-ai-chatbot`.
Primary goal: keep training/inference reproducible while protecting private chat data.

## Critical Safety Rules
- Never commit private chat data or derived private datasets.
- Keep these out of git: `data/raw/*`, `data/processed/*`, `checkpoints/*`.
- Only publish one model artifact: `artifacts/model_latest.pt` or `artifacts/model_latest.enc`.
- Bridge must remain dry-run unless user explicitly passes `--send`.
- Do not delete user data/checkpoints unless explicitly requested.

## Entry Points
- Unified CLI: `python -m chatbot.ops <command>`
- PowerShell wrapper: `scripts/run.ps1 <command>`
- Preprocess: `src/chatbot/preprocess.py`
- Train: `src/chatbot/train.py`
- Inference: `src/chatbot/inference.py`
- Bridge: `src/chatbot/kakao_bridge.py`
- Archive/reset: `src/chatbot/archive_state.py`

## Command Set (keep stable)
- `organize`
- `archive`
- `preprocess`
- `train`
- `reply`
- `chat`
- `bridge`
- `publish`
- `encrypt-model`
- `decrypt-model`

## Current Training Design
- Default preprocess mode is `context_windows`.
- Conversation data is split into sessions per source file and time gap.
- Each training example uses multiple previous turns (`context_turns`) + target turn.
- Low-signal/too-short messages can be filtered by preprocess config.

Config location:
- `configs/paths.yaml > processed.preprocess`

## Resume Behavior
- Auto-resume from `checkpoints/<run_name>/latest.pt`.
- Graceful stop:
  - `Ctrl+C`
  - create `checkpoints/<run_name>/STOP`
- Immutable resume checks:
  - model architecture mismatch
  - tokenizer content/hash mismatch
- Mutable overrides are logged.

## Archive/Cleanup Workflow
- Dry-run: `python -m chatbot.ops archive --dry`
- Apply: `python -m chatbot.ops archive`
- Default archives:
  - `data/processed`
  - `checkpoints`
  - `artifacts/model_latest*` / `model_info.json`
- Archive output root: `data/archive/<timestamp>_train_only/`

## Quick Validation
```bash
python -m compileall src
python -m chatbot.ops archive --dry
python -m chatbot.ops preprocess
python -m chatbot.ops train --max_steps 20
python -m chatbot.ops reply "테스트"
```

## Change Principles
- Keep existing CLI UX and command names intact.
- Prefer minimal, targeted edits over full rewrites.
- Write configs/logs/status for reproducibility.
- Keep Bash + Windows PowerShell execution paths both usable.
