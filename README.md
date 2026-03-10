# kakaotalk-trained-ai-chatbot

LoRA training pipeline for KakaoTalk-style chat models.

This repository currently has two working tracks:
- Legacy/default pipeline: `Qwen/Qwen2.5-7B`, `CPT -> SFT`, `configs/sft.yaml`
- Current best conversational pipeline: persona-projected 1:1 chat on `Qwen/Qwen2.5-7B-Instruct`

## Current Best Model

Current recommended model for actual use:
- Run name: `room_chat_qwen25_7b_instruct_pht_v1_refine`
- Adapter: `checkpoints_lora/room_chat_qwen25_7b_instruct_pht_v1_refine/adapter_best`
- Runtime config: `configs/sft.chat7b.pht.chat.yaml`
- Mode: `one_on_one`

Run it with:
```bash
python -m chatbot.sft_ops chat --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one
```

Why this is the best current model:
- it is the strongest practical result among the 1:1 persona-chat runs
- later experimental refines narrowed the output distribution too hard and regressed in real chat quality
- production should stay on `pht_v1_refine` until a future session proves a better replacement

## Safety Rules
- Never commit private chat data.
- Keep these out of git:
  - `data/raw/*`
  - `data/sft/*`
  - `checkpoints_lora/*`

## Install
```bash
cd /c/dev/kakaotalk-trained-ai-chatbot
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m pip install -e .
cp .env.example .env
```

Required env:
- `CHATBOT_PASSWORD=...`

## Quick Start

### Legacy/default pipeline
1. Build datasets:
```bash
python -m chatbot.sft_ops preprocess --config_sft configs/sft.yaml --env_path .env
```

2. Run full pipeline (CPT then SFT, with resume):
```bash
python -m chatbot.sft_ops train --config_sft configs/sft.yaml --env_path .env
```

### Best current conversational model
Run the best current conversational model:
```bash
python -m chatbot.sft_ops chat --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one
```

## Manual Stage Commands
Run CPT only:
```bash
python -m chatbot.sft_cpt_train --config_sft configs/sft.yaml --env_path .env --run_name room_lora_qwen25_7b_group_v2_cpt
```

Start SFT from CPT best adapter (first SFT start only):
```bash
python -m chatbot.sft_train --config_sft configs/sft.yaml --env_path .env --run_name room_lora_qwen25_7b_group_v2 --init_adapter checkpoints_lora/room_lora_qwen25_7b_group_v2_cpt/adapter_best
```

Resume SFT:
```bash
python -m chatbot.sft_train --config_sft configs/sft.yaml --env_path .env --run_name room_lora_qwen25_7b_group_v2
```

## Test Inference
Single-turn:
```bash
python -m chatbot.sft_ops reply "test" --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one
```

Interactive chat:
```bash
python -m chatbot.sft_ops chat --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one
```

## Local API
```bash
python -m chatbot.sft_ops serve --host 127.0.0.1 --port 8000 --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one
```

## Stop and Resume
- Stop: `Ctrl+C` or create `STOP` in run directory.
- Resume: run the same train command again.
- Resume source: latest valid `checkpoint-*` under `checkpoints_lora/<run_name>/`.

## Documentation Map
- Best-model handoff: `docs/BEST_MODEL_CHAT7B.md`
- Persona-chat training runbook: `docs/RUN_CHAT_7B.md`
- Operations runbook: `docs/RUNBOOK.md`
- Command quick reference: `docs/COMMANDS.md`
- Architecture and safeguards: `docs/ARCHITECTURE.md`
- Cleanup record: `docs/CLEANUP.md`
