$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$trainIn = "data/sft/chat7b_pht/train.jsonl"
$valIn = "data/sft/chat7b_pht/val.jsonl"
$trainOut = "data/sft/chat7b_pht_auto_direct/train.jsonl"
$valOut = "data/sft/chat7b_pht_auto_direct/val.jsonl"
$previewOut = "data/sft/chat7b_pht_auto_direct/preview.json"

python -m chatbot.sft_ops auto_direct_dataset `
  --input_train $trainIn `
  --input_val $valIn `
  --train_output $trainOut `
  --val_output $valOut `
  --preview_output $previewOut
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m chatbot.sft_train `
  --config_sft configs/sft.chat7b.pht.auto_direct.yaml `
  --env_path .env `
  --run_name room_chat_qwen25_7b_instruct_pht_v1_auto_direct `
  --init_adapter checkpoints_lora/room_chat_qwen25_7b_instruct_pht_v1_refine/adapter_best
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
