# AGENTS.md

## Purpose
Handoff guide for Codex sessions in this repository.
Current architecture is **Base LLM + LoRA 2-stage training (CPT -> SFT)**.

Current best practical conversational model is a separate 1:1 persona-chat track:
- Run: `room_chat_qwen25_7b_instruct_pht_v1_refine`
- Runtime config: `configs/sft.chat7b.pht.chat.yaml`
- Adapter: `checkpoints_lora/room_chat_qwen25_7b_instruct_pht_v1_refine/adapter_best`
- Handoff doc: `docs/BEST_MODEL_CHAT7B.md`

## Non-negotiable Rules
- Never commit private chat content or derived private datasets.
- Keep these out of git: `data/raw/*`, `data/sft/*`, `checkpoints_lora/*`.
- Do not delete/overwrite user checkpoints unless explicitly requested.

## Primary Commands
- `python -m chatbot.sft_ops organize`
- `python -m chatbot.sft_ops preprocess`
- `python -m chatbot.sft_ops train`
- `python -m chatbot.sft_ops reply "..."`
- `python -m chatbot.sft_ops chat`
- `python -m chatbot.sft_ops smoke`
- `python -m chatbot.sft_ops serve`

Current best model quick commands:
- `python -m chatbot.sft_ops chat --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one`
- `python -m chatbot.sft_ops smoke --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one`

## Key Files
- Config: `configs/sft.yaml`
- Best-model handoff: `docs/BEST_MODEL_CHAT7B.md`
- Best-model train configs:
  - `configs/sft.chat7b.pht.yaml`
  - `configs/sft.chat7b.pht.refine.yaml`
- Best-model runtime config: `configs/sft.chat7b.pht.chat.yaml`
- Preprocess: `src/chatbot/sft_preprocess.py`
- CPT train: `src/chatbot/sft_cpt_train.py`
- SFT train: `src/chatbot/sft_train.py`
- Pipeline train orchestrator: `src/chatbot/sft_train_pipeline.py`
- Inference engine: `src/chatbot/sft_infer.py`
- Chat CLI: `src/chatbot/sft_chat.py`
- Unified entrypoint: `src/chatbot/sft_ops.py`
- Legacy archive: `legacy/v1_from_scratch/`

## Pipeline Summary
1. Parse/clean Kakao txt logs.
2. Build SFT dataset (`train.jsonl`, `val.jsonl`) and CPT dataset (`cpt_train.jsonl`, `cpt_val.jsonl`).
3. Run CPT stage with auto-resume.
4. Bootstrap SFT from CPT adapter (if SFT checkpoint does not exist).
5. Run SFT stage with auto-resume.

## Resume/Stop Behavior
- Resume source: `checkpoints_lora/<run_name>/checkpoint-*`.
- Stop by `Ctrl+C` or STOP file in run directory.
- Same command resumes from latest valid checkpoint.

## Validation Checklist
```bash
python -m compileall src/chatbot
python -m chatbot.sft_ops preprocess --config_sft configs/sft.yaml --env_path .env
python -m chatbot.sft_ops train --config_sft configs/sft.yaml --env_path .env
python -m chatbot.sft_ops smoke --config_sft configs/sft.yaml --env_path .env --mode one_on_one
python -m chatbot.sft_ops reply "test" --config_sft configs/sft.yaml --env_path .env --mode one_on_one
```

## Change Principles
- Keep `configs/sft.yaml` as single source of constants.
- Preserve one-command UX for train and test.
- Prefer practical quality gains (data shaping + stage design + decoding) over cosmetic refactors.

## Current Session Guidance
- If the user says \"use the best model\", use `room_chat_qwen25_7b_instruct_pht_v1_refine` with `configs/sft.chat7b.pht.chat.yaml`.
- Do not treat `answer_refine`, `direct_refine`, or `closer_refine` as production defaults unless explicitly re-evaluating experiments.
- For the full reasoning, lineage, and reproduction steps, start from `docs/BEST_MODEL_CHAT7B.md`.
