# kakaotalk-trained-ai-chatbot

LoRA training pipeline for KakaoTalk-style chat models.

Current default architecture:
- Base model: `Qwen/Qwen2.5-7B`
- Training stages: `CPT -> SFT`
- Main config: `configs/sft.yaml`
- Default run name: `room_lora_qwen25_7b_group_v2`

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
1. Build datasets:
```bash
python -m chatbot.sft_ops preprocess --config_sft configs/sft.yaml --env_path .env
```

2. Run full pipeline (CPT then SFT, with resume):
```bash
python -m chatbot.sft_ops train --config_sft configs/sft.yaml --env_path .env
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
python -m chatbot.sft_ops reply "test" --config_sft configs/sft.yaml --env_path .env --mode one_on_one
```

Interactive chat:
```bash
python -m chatbot.sft_ops chat --config_sft configs/sft.yaml --env_path .env --mode group
```

## Local API
```bash
python -m chatbot.sft_ops serve --host 127.0.0.1 --port 8000 --config_sft configs/sft.yaml --env_path .env --mode group
```

## Kakao Desktop Bridge (Optional)
Calibrate once, then poll chat area and auto-reply when model decides to speak:
```bash
python -m chatbot.sft_ops bridge --config_sft configs/sft.yaml --env_path .env --run_name room_lora_qwen25_7b_group_v2 --mode group --calibrate --save_calibration artifacts/kakao_bridge_points.json --bot_name "<내카톡닉네임>" --send
```

## Stop and Resume
- Stop: `Ctrl+C` or create `STOP` in run directory.
- Resume: run the same train command again.
- Resume source: latest valid `checkpoint-*` under `checkpoints_lora/<run_name>/`.

## Documentation Map
- Operations runbook: `docs/RUNBOOK.md`
- Command quick reference: `docs/COMMANDS.md`
- Architecture and safeguards: `docs/ARCHITECTURE.md`
- Cleanup record: `docs/CLEANUP.md`
