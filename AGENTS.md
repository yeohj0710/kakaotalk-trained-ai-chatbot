# AGENTS.md

## Purpose
Handoff guide for Codex sessions in this repository.
Current architecture is **LoRA SFT on a pretrained LLM** (not from-scratch GPT).

## Non-negotiable Rules
- Never commit private chat content or derived private datasets.
- Keep these out of git: `data/raw/*`, `data/sft/*`, `checkpoints_lora/*`.
- Bridge/send actions must be explicit; dry-run first.
- Do not delete/overwrite user checkpoints unless explicitly requested.

## Primary Commands
- `python -m chatbot.sft_ops archive`
- `python -m chatbot.sft_ops organize`
- `python -m chatbot.sft_ops preprocess`
- `python -m chatbot.sft_ops train`
- `python -m chatbot.sft_ops reply "..."`
- `python -m chatbot.sft_ops chat`
- `python -m chatbot.sft_ops smoke`

## Key Files
- SFT config: `configs/sft.yaml`
- SFT preprocess: `src/chatbot/sft_preprocess.py`
- SFT training: `src/chatbot/sft_train.py`
- SFT inference engine: `src/chatbot/sft_infer.py`
- Chat CLI: `src/chatbot/sft_chat.py`
- Smoke: `src/chatbot/sft_smoke.py`
- Unified entrypoint: `src/chatbot/sft_ops.py`

## Pipeline Summary
1. Parse/clean Kakao txt logs.
2. Build context->response SFT JSONL (`train.jsonl`, `val.jsonl`).
3. Fine-tune LoRA adapter on pretrained base model.
4. Auto-resume from latest checkpoint.
5. Keep `adapter_best` and `adapter_latest` for inference.

## Resume/Stop Behavior
- Resume source: `checkpoints_lora/<run_name>/checkpoint-*`.
- Stop by `Ctrl+C` or `STOP` file in run dir.
- Training writes `status.json` with best/latest adapter paths.

## Validation Checklist
```bash
python -m compileall src
python -m chatbot.sft_ops preprocess
python -m chatbot.sft_ops train
python -m chatbot.sft_ops smoke
python -m chatbot.sft_ops reply "테스트"
```

## Change Principles
- Keep `configs/sft.yaml` as single source of training constants.
- Prefer practical quality improvements (data filtering, context design, decoding stability) over cosmetic refactors.
- Maintain one-line, low-friction command UX.
