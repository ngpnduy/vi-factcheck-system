param([switch]$ReuseInstalledPackages)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if ($ReuseInstalledPackages) {
    python -m venv --system-site-packages .venv
} else {
    python -m venv .venv
}
if ($LASTEXITCODE -ne 0) { throw "Environment creation failed" }
if (-not $ReuseInstalledPackages) {
    & .\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed" }
}
& .\.venv\Scripts\python.exe -m pip install -e ".[test]"
if ($LASTEXITCODE -ne 0) { throw "Project installation failed" }
& .\.venv\Scripts\python.exe -c "import torch; print('CUDA available:', torch.cuda.is_available())"
