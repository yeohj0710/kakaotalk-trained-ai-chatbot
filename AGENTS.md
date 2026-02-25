# AGENTS.md

## Purpose
Operational guide for future Codex sessions on `kakaotalk-trained-ai-chatbot`.
Goal: keep pipeline reproducible and safe for sensitive KakaoTalk exports.

## Critical Rules
- Never commit sensitive chat exports or derived private data.
- Keep `data/raw/*`, `data/processed/*`, `checkpoints/*` out of git.
- Only `artifacts/model_latest.pt` or `artifacts/model_latest.enc` may be intentionally published.
- Bridge must stay dry-run by default; sending requires explicit `--send`.
- Do not remove user checkpoints/data unless explicitly requested.

## Main Entry Points
- Unified CLI: `python -m chatbot.ops ...`
- PowerShell wrapper: `.\scripts\run.ps1 <command> [args]`
- Train engine: `src/chatbot/train.py`
- Inference loader: `src/chatbot/inference.py`
- Raw organizer: `src/chatbot/organize_raw.py`
- Security gate: `src/chatbot/security.py`
- Encryption utilities: `src/chatbot/crypto_utils.py`

## Config Files
- `configs/paths.yaml`: data/checkpoint/artifact paths
- `configs/train.yaml`: train/resume/hyperparameters
- `configs/gen.yaml`: generation/bridge/security defaults
- `.env`: local secrets (not committed)
- `.env.example`: template

## Standard Workflow
1. `.\scripts\run.ps1 organize`
2. `.\scripts\run.ps1 preprocess`
3. `.\scripts\run.ps1 train`
4. `.\scripts\run.ps1 reply "테스트 메시지"`
5. `.\scripts\run.ps1 chat`
6. `.\scripts\run.ps1 bridge --dry`
7. Optional publish: `.\scripts\publish_model.ps1 -Encrypt`

## Resume Behavior
- Training auto-resumes from `checkpoints/<run_name>/latest.pt` by default.
- Graceful stop:
  - `Ctrl+C` (latest checkpoint saved)
  - Create `checkpoints/<run_name>/STOP` (picked up during training loop)
- Immutable resume checks:
  - model architecture and vocab/tokenizer content mismatch => hard error.
- Mutable overrides:
  - optimization/logging/runtime knobs are allowed and logged.

## Validation Checklist
- `python -m compileall src`
- `python -m chatbot.ops organize --dry`
- `python -m chatbot.ops preprocess`
- `python -m chatbot.ops train --max_steps 20`
- `python -m chatbot.ops reply "테스트"`
- `python -m chatbot.ops publish`

## Change Principles
- Prefer extending existing modules over rewrites.
- Keep CLI backward-compatible where practical.
- Write config snapshots and run metadata for reproducibility.
- Preserve Windows-first usability (`scripts/run.ps1` stays primary).
