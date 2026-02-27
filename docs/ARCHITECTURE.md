# ARCHITECTURE

## Goal
Train a chat model that behaves like a group-room member, while keeping training workflow safe and resumable.

## Pipeline
1. Parse and clean Kakao chat logs.
2. Build CPT and SFT datasets.
3. Run CPT with periodic eval/save.
4. Start SFT from CPT `adapter_best`.
5. Continue SFT from latest valid checkpoint.

## Training Safety Guards
- Invalid checkpoint auto-skip:
  - Resume logic ignores `checkpoint-*` directories missing required files.
- Fresh SFT guard:
  - `training.require_init_adapter_on_fresh_start=true`
  - Fresh SFT without `--init_adapter` is blocked unless `--allow_fresh_start` is set.
- Run lineage metadata:
  - Each run writes `run_meta.json` with `init_mode` and `init_adapter_path`.
- Pipeline abort guards:
  - Abort if CPT bootstrap adapter is missing.
  - Abort if existing SFT run was initialized as fresh LoRA when CPT-bootstrap is expected.

## Data Shaping
Configured in `configs/sft.yaml`:
- Context window: 8 turns.
- Mention-heavy messages filtered.
- Summary artifact messages filtered.
- URL masking enabled.

## Inference Modes
- `group`:
  - Reply gating enabled (`reply_or_skip`).
  - Rules include minimum user turns since last bot message and bot-turn cap in window.
- `one_on_one`:
  - No gating, always reply.

CLI mode switch:
- `--mode group`
- `--mode one_on_one`

## Important Paths
- Config: `configs/sft.yaml`
- Preprocess: `src/chatbot/sft_preprocess.py`
- CPT train: `src/chatbot/sft_cpt_train.py`
- SFT train: `src/chatbot/sft_train.py`
- Pipeline: `src/chatbot/sft_train_pipeline.py`
- Inference: `src/chatbot/sft_infer.py`
- Chat CLI: `src/chatbot/sft_chat.py`
