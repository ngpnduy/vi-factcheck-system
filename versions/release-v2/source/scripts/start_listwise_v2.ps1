param(
    [string]$OutputDirectory = '',
    [string]$StatusDirectory = '',
    [string]$DataDirectory = '',
    [string]$ModelDirectory = '',
    [string]$SegmenterDirectory = '',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputDirectory) {
    $persistentBase = Join-Path ([Environment]::GetFolderPath('UserProfile')) 'AppData\Local\ViFactCheck'
    $OutputDirectory = Join-Path $persistentBase 'full-five-listwise-best5-v2'
}
if (-not $StatusDirectory) { $StatusDirectory = Join-Path $projectRoot 'runs\full-five-listwise-best5-v2' }
function Find-Asset([string]$explicitPath, [string]$relativePath) {
    if ($explicitPath) { return (Resolve-Path -LiteralPath $explicitPath).Path }
    $candidate = Join-Path $projectRoot $relativePath
    if (Test-Path -LiteralPath $candidate) { return (Resolve-Path -LiteralPath $candidate).Path }
    throw "Missing $relativePath; supply its directory explicitly."
}
$DataDirectory = Find-Asset $DataDirectory 'data\prepared'
$ModelDirectory = Find-Asset $ModelDirectory '.cache\phobert'
$SegmenterDirectory = Find-Asset $SegmenterDirectory '.cache\vncorenlp'
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
# Respect the physical path recorded by an earlier packaged-app launch.
$locationFile = Join-Path $StatusDirectory 'LOCATION.json'
if (-not $PSBoundParameters.ContainsKey('OutputDirectory') -and (Test-Path -LiteralPath $locationFile)) {
    $OutputDirectory = (Get-Content -LiteralPath $locationFile -Raw | ConvertFrom-Json).checkpoint_and_result_directory
}
$drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($OutputDirectory))
$freeGib = $drive.AvailableFreeSpace / 1GB
$hasCheckpoints = $false
if (Test-Path -LiteralPath $OutputDirectory) {
    $hasCheckpoints = $null -ne (Get-ChildItem -LiteralPath $OutputDirectory -Recurse -File -Filter '*.pt' | Select-Object -First 1)
}
$ready = $hasCheckpoints -or $freeGib -ge 35
New-Item -ItemType Directory -Path $StatusDirectory -Force | Out-Null
$preflight = [ordered]@{
    status = $(if ($ready) { 'ready' } else { 'needs_space' })
    protocol_version = 2
    checked_utc = [DateTime]::UtcNow.ToString('o')
    required_start_free_gib = 35
    free_disk_gib = $freeGib
    output = $OutputDirectory
    models = @('M0', 'M1', 'A1', 'M2', 'A2')
    seeds = @(42, 13, 2026)
    architecture = 'listwise'
    message = $(if ($ready) { 'Storage preflight passed; complete data/GPU checks run in the worker.' } else { 'Free at least 35 GiB on the output drive, then rerun this command. Existing checkpoints are preserved.' })
}
$json = $preflight | ConvertTo-Json -Depth 4
[IO.File]::WriteAllText((Join-Path $StatusDirectory 'storage_preflight.json'), $json, [Text.UTF8Encoding]::new($false))
if (-not $ready) {
    [IO.File]::WriteAllText((Join-Path $StatusDirectory 'status.json'), $json, [Text.UTF8Encoding]::new($false))
}
if ($CheckOnly) { $json; return }
if (-not $ready) { throw $preflight.message }
$tempDirectory = Join-Path (Split-Path -Parent $OutputDirectory) 'tmp-listwise-v2'
New-Item -ItemType Directory -Path $tempDirectory -Force | Out-Null
$env:TEMP = $tempDirectory
$env:TMP = $tempDirectory
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
& (Join-Path $PSScriptRoot 'start_full.ps1') -IncludeAblations -RerankerArchitecture listwise `
    -OutputDirectory $OutputDirectory -StatusDirectory $StatusDirectory `
    -DataDirectory $DataDirectory -ModelDirectory $ModelDirectory -SegmenterDirectory $SegmenterDirectory
