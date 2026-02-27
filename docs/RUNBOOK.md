# RUNBOOK

This runbook is for day-to-day training and testing.
It only covers commands in this repository.

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

## 2) Dataset Build
```bash
python -m chatbot.sft_ops preprocess --config_sft configs/sft.yaml --env_path .env
```

Outputs:
- `data/sft/train.jsonl`
- `data/sft/val.jsonl`
- `data/sft/cpt_train.jsonl`
- `data/sft/cpt_val.jsonl`
- `data/sft/stats.json`

## 3) CPT Stage
Start or resume CPT:
```bash
python -m chatbot.sft_cpt_train --config_sft configs/sft.yaml --env_path .env --run_name room_lora_qwen25_7b_group_v2_cpt
```

Checkpoint path:
- `checkpoints_lora/room_lora_qwen25_7b_group_v2_cpt/`

## 4) SFT Stage
First SFT start must point to CPT adapter:
```bash
python -m chatbot.sft_train --config_sft configs/sft.yaml --env_path .env --run_name room_lora_qwen25_7b_group_v2 --init_adapter checkpoints_lora/room_lora_qwen25_7b_group_v2_cpt/adapter_best
```

Resume SFT:
```bash
python -m chatbot.sft_train --config_sft configs/sft.yaml --env_path .env --run_name room_lora_qwen25_7b_group_v2
```

Notes:
- Fresh SFT without `--init_adapter` is blocked by default.
- Use `--allow_fresh_start` only when you intentionally want fresh LoRA.

## 5) One-command Pipeline (Optional)
```bash
python -m chatbot.sft_ops train --config_sft configs/sft.yaml --env_path .env
```

Pipeline behavior:
1. Runs CPT first.
2. Requires CPT completion if enabled in config.
3. Bootstraps SFT from CPT adapter.
4. Aborts if bootstrap adapter is missing.

## 6) Inference Test
Chat test with explicit mode:
```bash
python -m chatbot.sft_ops chat --config_sft configs/sft.yaml --env_path .env --run_name room_lora_qwen25_7b_group_v2 --mode one_on_one
```

Group-mode test:
```bash
python -m chatbot.sft_ops chat --config_sft configs/sft.yaml --env_path .env --run_name room_lora_qwen25_7b_group_v2 --mode group
```

## 7) Latest Checkpoint Test
Git Bash:
```bash
RUN=room_lora_qwen25_7b_group_v2
CKPT=$(ls -d checkpoints_lora/$RUN/checkpoint-* 2>/dev/null | sort -V | tail -1)
python -m chatbot.sft_ops chat --config_sft configs/sft.yaml --env_path .env --adapter "$CKPT" --mode one_on_one
```

PowerShell:
```powershell
$RUN="room_lora_qwen25_7b_group_v2"
$CKPT=(Get-ChildItem "checkpoints_lora/$RUN" -Directory -Filter "checkpoint-*" | Sort-Object {[int]($_.Name -replace 'checkpoint-','')} -Descending | Select-Object -First 1).FullName
python -m chatbot.sft_ops chat --config_sft configs/sft.yaml --env_path .env --adapter "$CKPT" --mode one_on_one
```

## 8) Stop and Resume
- Stop immediately: `Ctrl+C`
- Graceful stop: create `STOP` in run directory
- Resume: same command, same `run_name`

## 9) Validation
```bash
python -m compileall src/chatbot
python -m chatbot.sft_ops smoke --config_sft configs/sft.yaml --env_path .env --mode one_on_one
```
