param(
  [string]$RunName = "quick",
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

python -m chatbot.ops organize
python -m chatbot.ops preprocess
python -m chatbot.ops train --run_name $RunName --max_steps $MaxSteps
