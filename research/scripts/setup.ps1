param([switch]$ReuseInstalledPackages, [string]$EnvironmentDirectory = '')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$environmentPath = if ($EnvironmentDirectory) { [IO.Path]::GetFullPath($EnvironmentDirectory) } else { Join-Path $projectRoot '.venv' }
Push-Location -LiteralPath $projectRoot
try {
    if (Test-Path -LiteralPath (Join-Path $projectRoot '.venv')) { throw 'research/.venv already exists; verify or remove it explicitly before setup.' }
    if (Test-Path -LiteralPath $environmentPath) { throw 'Environment directory already exists; choose a new directory.' }
    if ($ReuseInstalledPackages) {
        python -m venv --without-pip --system-site-packages $environmentPath
    } else {
        python -m venv $environmentPath
    }
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
    if ($EnvironmentDirectory -and $environmentPath -ne (Join-Path $projectRoot '.venv')) {
        New-Item -ItemType Junction -Path (Join-Path $projectRoot '.venv') -Target $environmentPath | Out-Null
    }
    if (-not $ReuseInstalledPackages) {
        & $pythonPath -m pip install -r requirements-tested.txt --extra-index-url https://download.pytorch.org/whl/cu124
        if ($LASTEXITCODE -ne 0) { throw 'Pinned dependency installation failed' }
        & $pythonPath -m pip install --no-deps --no-build-isolation -e .
        if ($LASTEXITCODE -ne 0) { throw 'Project installation failed' }
    }
    & $pythonPath (Join-Path $PSScriptRoot 'check_environment.py')
    if ($LASTEXITCODE -ne 0) { throw 'Installed packages differ from the validated research environment; no global packages were changed.' }
} finally { Pop-Location }
