$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\app.py"
