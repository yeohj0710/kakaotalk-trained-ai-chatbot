param(
  [Parameter(Mandatory = $true, Position = 0)]
  [string]$Command,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args,
  [switch]$Bootstrap
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-Python {
  if (Test-Path ".venv\\Scripts\\python.exe") {
    return ".venv\\Scripts\\python.exe"
  }
  return "python"
}

if ($Bootstrap -and -not (Test-Path ".venv\\Scripts\\python.exe")) {
  python -m venv .venv
}

$pythonExe = Resolve-Python

if ($Bootstrap) {
  & $pythonExe -m pip install --upgrade pip
  & $pythonExe -m pip install -r requirements.txt
  & $pythonExe -m pip install -e .
}

& $pythonExe -c "import chatbot" 1>$null 2>$null
if ($LASTEXITCODE -ne 0) {
  & $pythonExe -m pip install -e .
}

& $pythonExe -m chatbot.ops $Command @Args
exit $LASTEXITCODE
