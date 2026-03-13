Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-Python {
  if (Test-Path ".venv\\Scripts\\python.exe") {
    return ".venv\\Scripts\\python.exe"
  }
  return "python"
}

function Read-JsonFile {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    return $null
  }
  return Get-Content $Path -Raw | ConvertFrom-Json
}

function Test-StageCompleted {
  param([string]$RunName)
  $statusPath = Join-Path "checkpoints_lora" "$RunName\\status.json"
  $status = Read-JsonFile -Path $statusPath
  if ($null -eq $status) {
    return $false
  }
  if ($status.stopped) {
    return $false
  }
  $bestAdapterDir = [string]$status.best_adapter_dir
  if ([string]::IsNullOrWhiteSpace($bestAdapterDir)) {
    return $false
  }
  return (Test-Path $bestAdapterDir)
}

$pythonExe = Resolve-Python

& $pythonExe -c "import chatbot" 1>$null 2>$null
if ($LASTEXITCODE -ne 0) {
  & $pythonExe -m pip install -e .
}

$reviewPath = "artifacts/failure_reviews/pht_v1_refine_seed_review01.jsonl"
$trainJsonl = "data/sft/chat7b_pht_correction/train.jsonl"
$valJsonl = "data/sft/chat7b_pht_correction/val.jsonl"
$previewJson = "data/sft/chat7b_pht_correction/preview.json"
$configPath = "configs/sft.chat7b.pht.correction.yaml"
$runName = "room_chat_qwen25_7b_instruct_pht_v1_correction_v1"
$initAdapter = "checkpoints_lora/room_chat_qwen25_7b_instruct_pht_v1_refine/adapter_best"

if (-not (Test-Path $reviewPath)) {
  throw "Missing reviewed correction file: $reviewPath"
}

& $pythonExe -m chatbot.sft_correction_build --input $reviewPath --train_output $trainJsonl --val_output $valJsonl --preview_output $previewJson
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

if (-not (Test-Path $initAdapter)) {
  throw "Missing init adapter: $initAdapter"
}

if (-not (Test-StageCompleted -RunName $runName)) {
  & $pythonExe -m chatbot.sft_train --config_sft $configPath --env_path .env --run_name $runName --init_adapter $initAdapter
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}
