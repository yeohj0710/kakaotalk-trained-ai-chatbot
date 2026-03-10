# ARCHITECTURE

## Goal
Train a chat model that behaves like a KakaoTalk room member while keeping training workflow safe and resumable.

Current practical goal for the best model family:
- one-on-one chat
- room-member identity preserved through persona projection
- direct conversational reply over a short sliding window

## Legacy Pipeline
1. Parse and clean Kakao chat logs.
2. Build CPT and SFT datasets.
3. Run CPT with periodic eval and save.
4. Start SFT from CPT `adapter_best`.
5. Continue SFT from latest valid checkpoint.

## Current Best Chat-Family Pipeline
1. Parse and clean Kakao chat logs.
2. Build `projected_dialogue` SFT data.
3. Project the chosen persona speaker to `assistant`.
4. Project all other speakers to `user`.
5. Train a fresh LoRA on `Qwen/Qwen2.5-7B-Instruct`.
6. Run one refine stage from the base adapter.
7. Use a stricter inference config than the training config.

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
Configured in `configs/sft.yaml` for the legacy path:
- Context window: 8 turns.
- Mention-heavy messages filtered.
- Summary artifact messages filtered.
- URL masking enabled.

For the current best persona-chat family:
- `training_mode: projected_dialogue`
- `target_speakers: ["¹ÚÇöÅ¹"]`
- `project_user_names: true`
- `require_last_context_from_other_speaker: true`
- tokenizer chat template alignment during both training and inference

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
- Legacy config: `configs/sft.yaml`
- Best-model handoff: `docs/BEST_MODEL_CHAT7B.md`
- Best-model inference config: `configs/sft.chat7b.pht.chat.yaml`
- Best-model training configs:
  - `configs/sft.chat7b.pht.yaml`
  - `configs/sft.chat7b.pht.refine.yaml`
- Preprocess: `src/chatbot/sft_preprocess.py`
- CPT train: `src/chatbot/sft_cpt_train.py`
- SFT train: `src/chatbot/sft_train.py`
- Pipeline: `src/chatbot/sft_train_pipeline.py`
- Inference: `src/chatbot/sft_infer.py`
- Chat CLI: `src/chatbot/sft_chat.py`
