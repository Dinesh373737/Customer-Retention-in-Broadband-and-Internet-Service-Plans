$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    Write-Host 'Create .venv and install requirements first; see README.md.'
    exit 1
}
& $taskPython -m streamlit run app.py --server.address 127.0.0.1
