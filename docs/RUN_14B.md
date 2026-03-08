# 14B Runbook

This runbook keeps the current 7B design and hyperparameter style, and only upgrades base model scale to 14B.

Config files:
- `configs/sft.14b.yaml` (CPT + SFT)
- `configs/sft.14b.refine.yaml` (refine stage)

Preflight:
- 14B training in this repo assumes 4bit loading works (`bitsandbytes` installed).
- If 4bit is not available, training/inference can fall back to full precision and likely fail with OOM on consumer GPUs.
- Quick check:
```bash
python -c "import bitsandbytes, torch; print('bnb_ok', torch.cuda.is_available())"
```

Run names:
- CPT: `room_lora_qwen25_14b_group_v1_cpt`
- SFT: `room_lora_qwen25_14b_group_v1`
- Refine: `room_lora_qwen25_14b_group_v1_refine`

## 0) Preprocess (once)
```bash
python -m chatbot.sft_ops preprocess --config_sft configs/sft.14b.yaml --env_path .env
```

## 1) CPT (start/resume: same command)
```bash
python -m chatbot.sft_cpt_train --config_sft configs/sft.14b.yaml --env_path .env --run_name room_lora_qwen25_14b_group_v1_cpt
```

### CPT test
```bash
python -m chatbot.sft_ops reply "테스트" --config_sft configs/sft.14b.yaml --env_path .env --adapter checkpoints_lora/room_lora_qwen25_14b_group_v1_cpt/adapter_best --mode one_on_one
```

## 2) SFT (start/resume: same command)
```bash
python -m chatbot.sft_train --config_sft configs/sft.14b.yaml --env_path .env --run_name room_lora_qwen25_14b_group_v1 --init_adapter checkpoints_lora/room_lora_qwen25_14b_group_v1_cpt/adapter_best
```

### SFT test
```bash
python -m chatbot.sft_ops smoke --config_sft configs/sft.14b.yaml --env_path .env --run_name room_lora_qwen25_14b_group_v1 --mode one_on_one
python -m chatbot.sft_ops chat --config_sft configs/sft.14b.yaml --env_path .env --run_name room_lora_qwen25_14b_group_v1 --mode one_on_one
```

## 3) Refine (start/resume: same command)
```bash
python -m chatbot.sft_train --config_sft configs/sft.14b.refine.yaml --env_path .env --run_name room_lora_qwen25_14b_group_v1_refine --init_adapter checkpoints_lora/room_lora_qwen25_14b_group_v1/adapter_best
```

### Refine test
```bash
python -m chatbot.sft_ops smoke --config_sft configs/sft.14b.refine.yaml --env_path .env --run_name room_lora_qwen25_14b_group_v1_refine --mode one_on_one
python -m chatbot.sft_ops chat --config_sft configs/sft.14b.refine.yaml --env_path .env --run_name room_lora_qwen25_14b_group_v1_refine --mode one_on_one
```

## Optional one-command pipeline (CPT -> SFT only)
```bash
python -m chatbot.sft_ops train --config_sft configs/sft.14b.yaml --env_path .env
```

## Stop/Resume
- Stop now: `Ctrl+C`
- Graceful stop: create `STOP` file inside run directory.
- Resume: rerun the same command above.
