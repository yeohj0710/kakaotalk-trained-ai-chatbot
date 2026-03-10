# RUNBOOK

This runbook is for day-to-day training and testing.
It only covers commands in this repository.

Current recommended production target:
- run: `room_chat_qwen25_7b_instruct_pht_v1_refine`
- inference config: `configs/sft.chat7b.pht.chat.yaml`

If you are starting a new session and want the best known model, go to `docs/BEST_MODEL_CHAT7B.md` first.

## 1) Preflight
```bash
cd /c/dev/kakaotalk-trained-ai-chatbot
source .venv/Scripts/activate
```

Check env:
```bash
cat .env
```
You need at least:
- `CHATBOT_PASSWORD=...`

## 2) Best-Model Inference

Current best one-on-one chat:
```bash
python -m chatbot.sft_ops chat --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one
```

Quick smoke:
```bash
python -m chatbot.sft_ops smoke --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one
```

## 3) Best-Model Training Reproduction

Build dataset:
```bash
python -m chatbot.sft_ops preprocess --config_sft configs/sft.chat7b.pht.yaml --env_path .env
```

Base SFT:
```bash
python -m chatbot.sft_train --config_sft configs/sft.chat7b.pht.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1
```

Refine:
```bash
python -m chatbot.sft_train --config_sft configs/sft.chat7b.pht.refine.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --init_adapter checkpoints_lora/room_chat_qwen25_7b_instruct_pht_v1/adapter_best
```

One-command script:
```bash
powershell.exe -ExecutionPolicy Bypass -File "./scripts/train_chat7b_pht.ps1"
```

## 4) Stop and Resume
- Stop immediately: `Ctrl+C`
- Graceful stop: create `STOP` in run directory
- Resume: same command, same `run_name`

## 5) Legacy Pipeline

Dataset build:
```bash
python -m chatbot.sft_ops preprocess --config_sft configs/sft.yaml --env_path .env
```

Pipeline train:
```bash
python -m chatbot.sft_ops train --config_sft configs/sft.yaml --env_path .env
```

## 6) Validation
```bash
python -m compileall src/chatbot
python -m chatbot.sft_ops smoke --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one
```
