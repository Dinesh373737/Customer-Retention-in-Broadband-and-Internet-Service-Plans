param([switch]$Install)
$ErrorActionPreference = 'Stop'
$taskPlatform = $PSScriptRoot
$taskRoot = Split-Path -Parent $taskPlatform
$taskPython = Join-Path $taskPlatform '.venv\Scripts\python.exe'
Set-Location -LiteralPath $taskRoot
if ($Install) {
    if (-not (Test-Path -LiteralPath $taskPython)) {
        & python -m venv (Join-Path $taskPlatform '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
    }
    & $taskPython -m pip install -r (Join-Path $taskPlatform 'backend\requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Backend dependency installation failed.' }
    $taskNode = Get-Command node -ErrorAction SilentlyContinue
    $taskBundledNode = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin'
    if (-not $taskNode -and (Test-Path -LiteralPath (Join-Path $taskBundledNode 'node.exe'))) {
        $env:PATH = $taskBundledNode + ';' + $env:PATH
    }
    $taskPnpm = Get-Command pnpm -ErrorAction SilentlyContinue
    $taskBundledPnpm = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\fallback\pnpm.cmd'
    if (-not $taskPnpm -and (Test-Path -LiteralPath $taskBundledPnpm)) { $taskPnpm = $taskBundledPnpm }
    if (-not $taskPnpm) { throw 'Install Node.js 22+ and pnpm 11, then run this launcher again. See platform/README.md.' }
    Push-Location -LiteralPath (Join-Path $taskPlatform 'frontend')
    try {
        & $taskPnpm install --frozen-lockfile
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
        & $taskPnpm build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally { Pop-Location }
}
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run platform\launch.ps1 -Install or follow platform/README.md first.' }
if (-not (Test-Path -LiteralPath (Join-Path $taskPlatform 'frontend\dist\index.html'))) { throw 'Build the React frontend first: pnpm build in platform/frontend.' }
Write-Host 'Retention Lab 2.0: http://127.0.0.1:8000'
Write-Host 'API documentation: http://127.0.0.1:8000/docs'
Write-Host 'The original IBM Streamlit application runs independently on port 8501.'
& $taskPython -m uvicorn app.main:app --app-dir (Join-Path $taskPlatform 'backend') --host 127.0.0.1 --port 8000
