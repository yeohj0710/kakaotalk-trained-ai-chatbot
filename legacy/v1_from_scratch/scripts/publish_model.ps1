param(
  [switch]$Encrypt,
  [string]$RunName = "",
  [string]$Source = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$pythonExe = if (Test-Path ".venv\\Scripts\\python.exe") { ".venv\\Scripts\\python.exe" } else { "python" }

$cmd = @("-m", "chatbot.ops", "publish")
if ($Encrypt) { $cmd += "--encrypt" }
if ($RunName) { $cmd += @("--run_name", $RunName) }
if ($Source) { $cmd += @("--source", $Source) }

& $pythonExe @cmd
exit $LASTEXITCODE
