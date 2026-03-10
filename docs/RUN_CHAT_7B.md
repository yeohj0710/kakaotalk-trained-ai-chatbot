# Chat 7B Runbook

This runbook targets proper short-context conversation rather than generic room next-turn imitation.

Design:
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- No CPT by default
- SFT data uses `projected_dialogue`
- Target reply is always trained as `bot`
- Other speakers are projected into `user` turns, optionally with `[speaker]` tags
- Train/infer prompt format is aligned through tokenizer chat template

Configs:
- `configs/sft.chat7b.yaml`
- `configs/sft.chat7b.refine.yaml`

Optional identity tightening:
- Set `data.target_speakers` to your own speaker name aliases if you want the bot persona to follow a specific member rather than an averaged room-member persona.
- Or set `.env`: `CHATBOT_PERSONA_SPEAKERS=["홍길동","길동"]`

## 1) Build dataset
```bash
python -m chatbot.sft_ops preprocess --config_sft configs/sft.chat7b.yaml --env_path .env
```

## 2) Train base chat model
```bash
python -m chatbot.sft_train --config_sft configs/sft.chat7b.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_v1
```

## 3) Refine from best adapter
```bash
python -m chatbot.sft_train --config_sft configs/sft.chat7b.refine.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_v1_refine --init_adapter checkpoints_lora/room_chat_qwen25_7b_instruct_v1/adapter_best
```

## 4) Test
```bash
python -m chatbot.sft_ops chat --config_sft configs/sft.chat7b.refine.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_v1_refine --mode one_on_one
```

## Train only block
```bash
python -m chatbot.sft_ops preprocess --config_sft configs/sft.chat7b.yaml --env_path .env
python -m chatbot.sft_train --config_sft configs/sft.chat7b.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_v1
python -m chatbot.sft_train --config_sft configs/sft.chat7b.refine.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_v1_refine --init_adapter checkpoints_lora/room_chat_qwen25_7b_instruct_v1/adapter_best
```
