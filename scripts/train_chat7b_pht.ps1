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
  param(
    [string]$RunName
  )
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

$baseConfig = "configs/sft.chat7b.pht.yaml"
$refineConfig = "configs/sft.chat7b.pht.refine.yaml"
$baseRun = "room_chat_qwen25_7b_instruct_pht_v1"
$refineRun = "room_chat_qwen25_7b_instruct_pht_v1_refine"
$baseAdapter = "checkpoints_lora/$baseRun/adapter_best"

$requiredDataFiles = @(
  "data/sft/chat7b_pht/train.jsonl",
  "data/sft/chat7b_pht/val.jsonl",
  "data/sft/chat7b_pht/preview.json",
  "data/sft/chat7b_pht/stats.json"
)

$needsPreprocess = $false
foreach ($path in $requiredDataFiles) {
  if (-not (Test-Path $path)) {
    $needsPreprocess = $true
    break
  }
}

if ($needsPreprocess) {
  & $pythonExe -m chatbot.sft_ops preprocess --config_sft $baseConfig --env_path .env
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}

if (-not (Test-StageCompleted -RunName $baseRun)) {
  & $pythonExe -m chatbot.sft_train --config_sft $baseConfig --env_path .env --run_name $baseRun
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}

if (-not (Test-Path $baseAdapter)) {
  throw "Missing base adapter: $baseAdapter"
}

if (-not (Test-StageCompleted -RunName $refineRun)) {
  & $pythonExe -m chatbot.sft_train --config_sft $refineConfig --env_path .env --run_name $refineRun --init_adapter $baseAdapter
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}
