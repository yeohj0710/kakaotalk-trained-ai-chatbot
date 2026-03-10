# COMMANDS

## Current Best Model
```bash
python -m chatbot.sft_ops chat --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one
python -m chatbot.sft_ops smoke --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one
python -m chatbot.sft_ops reply "test" --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one
```

## Current Best Model Training
```bash
python -m chatbot.sft_ops preprocess --config_sft configs/sft.chat7b.pht.yaml --env_path .env
python -m chatbot.sft_train --config_sft configs/sft.chat7b.pht.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1
python -m chatbot.sft_train --config_sft configs/sft.chat7b.pht.refine.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --init_adapter checkpoints_lora/room_chat_qwen25_7b_instruct_pht_v1/adapter_best
```

## Current Best Model Training Script
```bash
powershell.exe -ExecutionPolicy Bypass -File "./scripts/train_chat7b_pht.ps1"
```

## Core Legacy Pipeline
```bash
python -m chatbot.sft_ops organize
python -m chatbot.sft_ops preprocess --config_sft configs/sft.yaml --env_path .env
python -m chatbot.sft_ops train --config_sft configs/sft.yaml --env_path .env
```

## Stage-specific Legacy Commands
```bash
python -m chatbot.sft_cpt_train --config_sft configs/sft.yaml --env_path .env --run_name room_lora_qwen25_7b_group_v2_cpt
python -m chatbot.sft_train --config_sft configs/sft.yaml --env_path .env --run_name room_lora_qwen25_7b_group_v2 --init_adapter checkpoints_lora/room_lora_qwen25_7b_group_v2_cpt/adapter_best
python -m chatbot.sft_train --config_sft configs/sft.yaml --env_path .env --run_name room_lora_qwen25_7b_group_v2
```

## API
```bash
python -m chatbot.sft_ops serve --host 127.0.0.1 --port 8000 --config_sft configs/sft.chat7b.pht.chat.yaml --env_path .env --run_name room_chat_qwen25_7b_instruct_pht_v1_refine --mode one_on_one
```
