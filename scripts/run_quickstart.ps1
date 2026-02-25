param(
  [string]$InputGlob = "*.txt",
  [string]$OutDataDir = "data/processed",
  [string]$OutCkptDir = "checkpoints/quick",
  [int]$MaxSteps = 1500
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
  python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

python -m chatbot.preprocess --input_glob $InputGlob --output_dir $OutDataDir --val_ratio 0.02 --mask_urls

python -m chatbot.train `
  --data_dir $OutDataDir `
  --out_dir $OutCkptDir `
  --block_size 256 `
  --n_layer 6 --n_head 6 --n_embd 384 `
  --batch_size 16 `
  --max_steps $MaxSteps `
  --eval_interval 200 `
  --save_interval 400 `
  --device auto --dtype auto
