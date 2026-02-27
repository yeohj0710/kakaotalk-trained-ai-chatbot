# COMMANDS

## Core
```bash
python -m chatbot.sft_ops organize
python -m chatbot.sft_ops preprocess --config_sft configs/sft.yaml --env_path .env
python -m chatbot.sft_ops train --config_sft configs/sft.yaml --env_path .env
```

## Stage-specific
```bash
python -m chatbot.sft_cpt_train --config_sft configs/sft.yaml --env_path .env --run_name room_lora_qwen25_7b_group_v2_cpt
python -m chatbot.sft_train --config_sft configs/sft.yaml --env_path .env --run_name room_lora_qwen25_7b_group_v2 --init_adapter checkpoints_lora/room_lora_qwen25_7b_group_v2_cpt/adapter_best
python -m chatbot.sft_train --config_sft configs/sft.yaml --env_path .env --run_name room_lora_qwen25_7b_group_v2
```

## Inference
```bash
python -m chatbot.sft_ops reply "test" --config_sft configs/sft.yaml --env_path .env --mode one_on_one
python -m chatbot.sft_ops chat --config_sft configs/sft.yaml --env_path .env --mode one_on_one
python -m chatbot.sft_ops chat --config_sft configs/sft.yaml --env_path .env --mode group
python -m chatbot.sft_ops smoke --config_sft configs/sft.yaml --env_path .env --mode one_on_one
```

## API
```bash
python -m chatbot.sft_ops serve --host 127.0.0.1 --port 8000 --config_sft configs/sft.yaml --env_path .env --mode group
```

## Checkpoint Shortcuts
Git Bash latest checkpoint:
```bash
RUN=room_lora_qwen25_7b_group_v2
CKPT=$(ls -d checkpoints_lora/$RUN/checkpoint-* 2>/dev/null | sort -V | tail -1)
```

PowerShell latest checkpoint:
```powershell
$RUN="room_lora_qwen25_7b_group_v2"
$CKPT=(Get-ChildItem "checkpoints_lora/$RUN" -Directory -Filter "checkpoint-*" | Sort-Object {[int]($_.Name -replace 'checkpoint-','')} -Descending | Select-Object -First 1).FullName
```
