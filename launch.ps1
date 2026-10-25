param(
    [switch]$Install,
    [switch]$Legacy
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if ($Legacy) {
    if ($Install) {
        throw 'For the original Streamlit application, install requirements.txt in the root .venv first. See README.md.'
    }
    $taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) {
        throw 'Create the root .venv and install requirements.txt first; see README.md.'
    }
    Write-Host 'Original IBM Streamlit application: http://127.0.0.1:8501'
    & $taskPython -m streamlit run app.py --server.address 127.0.0.1
    return
}

$taskPlatformLauncher = Join-Path $PSScriptRoot 'platform\launch.ps1'
if (-not (Test-Path -LiteralPath $taskPlatformLauncher)) {
    throw 'The React platform files are missing. Pull the latest main branch, including platform/.'
}
& $taskPlatformLauncher -Install:$Install
