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

$configPath = "configs/sft.chat7b.pht.direct_refine.yaml"
$runName = "room_chat_qwen25_7b_instruct_pht_v1_direct_refine"
$initAdapter = "checkpoints_lora/room_chat_qwen25_7b_instruct_pht_v1_refine/adapter_best"

$requiredDataFiles = @(
  "data/sft/chat7b_pht_direct_refine/train.jsonl",
  "data/sft/chat7b_pht_direct_refine/val.jsonl",
  "data/sft/chat7b_pht_direct_refine/preview.json",
  "data/sft/chat7b_pht_direct_refine/stats.json"
)

$needsPreprocess = $false
foreach ($path in $requiredDataFiles) {
  if (-not (Test-Path $path)) {
    $needsPreprocess = $true
    break
  }
}

if ($needsPreprocess) {
  & $pythonExe -m chatbot.sft_ops preprocess --config_sft $configPath --env_path .env
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
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
