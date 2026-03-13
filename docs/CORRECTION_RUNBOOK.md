# Correction Runbook

This runbook is for the first correction-refine pass on top of the current best model.

Base model for correction:
- `room_chat_qwen25_7b_instruct_pht_v1_refine`

Correction output model:
- `room_chat_qwen25_7b_instruct_pht_v1_correction_v1`

## 1. Review file

The training script expects a reviewed JSONL file:

- `artifacts/failure_reviews/pht_v1_refine_seed_review01.jsonl`

Each row should contain:
- `prompt`
- optional `history`
- `reply`
- `corrected_reply`

Only rows with non-empty `corrected_reply` are used.

## 2. Build correction dataset

```bash
python -m chatbot.sft_correction_build --input artifacts/failure_reviews/pht_v1_refine_seed_review01.jsonl --train_output data/sft/chat7b_pht_correction/train.jsonl --val_output data/sft/chat7b_pht_correction/val.jsonl --preview_output data/sft/chat7b_pht_correction/preview.json
```

## 3. Train correction refine

```bash
python -m chatbot.sft_train --config_sft configs/sft.chat7b.pht.correction.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_correction_v1 --init_adapter checkpoints_lora/room_chat_qwen25_7b_instruct_pht_v1_refine/adapter_best
```

## 4. One-command script

```bash
powershell.exe -ExecutionPolicy Bypass -File "./scripts/train_chat7b_pht_correction.ps1"
```

## 5. Intended scope

This is not a full retrain.

It is a small correction pass intended to reduce:
- filler openings
- irrelevant jumps
- needless counter-questions
- answer drift after the first clause

## 6. Current Status

Implemented correction runs:
- `room_chat_qwen25_7b_instruct_pht_v1_correction_v1`
- `room_chat_qwen25_7b_instruct_pht_v1_correction_mix_v1`
- `room_chat_qwen25_7b_instruct_pht_v1_correction_mix_v2`

Outcome:
- correction tooling is now in place
- none of the correction runs beat `room_chat_qwen25_7b_instruct_pht_v1_refine` in repeated real prompt evaluation
- production default remains `room_chat_qwen25_7b_instruct_pht_v1_refine`

Promotion rule:
- do not promote based only on correction-set `eval_loss`
- require qualitative wins and no safety regressions
