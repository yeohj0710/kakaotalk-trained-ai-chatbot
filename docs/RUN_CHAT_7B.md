# Chat 7B Runbook

This is the canonical training path for the current best conversational family in this repository.

Target:
- one-on-one Korean chat
- persona fixed to `¹ÚÇöÅ¹`
- direct reply behavior over short recent context

Core design:
- base model: `Qwen/Qwen2.5-7B-Instruct`
- no CPT
- `projected_dialogue` dataset
- target persona projected to `assistant`
- other speakers projected to `user`
- tokenizer chat template used in both training and inference

Canonical configs:
- base training: `configs/sft.chat7b.pht.yaml`
- refine training: `configs/sft.chat7b.pht.refine.yaml`
- production inference: `configs/sft.chat7b.pht.chat.yaml`

Canonical training script:
- `scripts/train_chat7b_pht.ps1`

Current best output of this family:
- `room_chat_qwen25_7b_instruct_pht_v1_refine`

## 1) Build dataset

```bash
python -m chatbot.sft_ops preprocess --config_sft configs/sft.chat7b.pht.yaml --env_path .env
```

## 2) Train base chat model

```bash
python -m chatbot.sft_train --config_sft configs/sft.chat7b.pht.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1
```

## 3) Refine from base best adapter

```bash
python -m chatbot.sft_train --config_sft configs/sft.chat7b.pht.refine.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --init_adapter checkpoints_lora/room_chat_qwen25_7b_instruct_pht_v1/adapter_best
```

## 4) Use the best model

Use the production inference config, not the training config:

```bash
python -m chatbot.sft_ops chat --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one
```

## 5) One-command training

```bash
powershell.exe -ExecutionPolicy Bypass -File "./scripts/train_chat7b_pht.ps1"
```

## 6) Resume behavior

- stop with `Ctrl+C`
- rerun the same command
- the same `run_name` resumes from the latest valid checkpoint

## 7) Important note on later experiments

These runs exist but are not the production default:
- `room_chat_qwen25_7b_instruct_pht_v1_answer_refine`
- `room_chat_qwen25_7b_instruct_pht_v1_direct_refine`
- `room_chat_qwen25_7b_instruct_pht_v1_closer_refine`

They were useful as experiments, but they should not replace `room_chat_qwen25_7b_instruct_pht_v1_refine` unless a future session re-tests and proves otherwise.
