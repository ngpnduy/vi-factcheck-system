param(
    [string]$OutputDirectory = (Join-Path $env:LOCALAPPDATA 'ViFactCheck\full-m0-m1-m2'),
    [string]$StatusDirectory = '',
    [string]$DataDirectory = '',
    [string]$ModelDirectory = '',
    [string]$SegmenterDirectory = '',
    [switch]$IncludeAblations,
    [ValidateSet('pairwise', 'listwise')]
    [string]$RerankerArchitecture = 'pairwise'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create the project environment first.' }
if (-not $StatusDirectory) { $StatusDirectory = Join-Path $projectRoot 'runs\full-m0-m1-m2' }
if (-not $DataDirectory) { $DataDirectory = Join-Path $projectRoot 'data\prepared' }
if (-not $ModelDirectory) { $ModelDirectory = Join-Path $projectRoot '.cache\phobert' }
if (-not $SegmenterDirectory) { $SegmenterDirectory = Join-Path $projectRoot '.cache\vncorenlp' }
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $StatusDirectory -Force | Out-Null
$resolvedOutput = (Resolve-Path -LiteralPath $OutputDirectory).Path
$resolvedStatus = (Resolve-Path -LiteralPath $StatusDirectory).Path
# Windows packaged apps can redirect AppData paths. Resume the physical output
# recorded by the Python worker rather than creating a second experiment.
$locationPath = Join-Path $resolvedStatus 'LOCATION.json'
if (Test-Path -LiteralPath $locationPath) {
    $oldLocation = Get-Content -LiteralPath $locationPath -Raw | ConvertFrom-Json
    $recordedOutput = $oldLocation.checkpoint_and_result_directory
    if (-not (Test-Path -LiteralPath $recordedOutput)) { throw 'Recorded checkpoint directory is missing.' }
    $recordedResolved = (Resolve-Path -LiteralPath $recordedOutput).Path
    if ($resolvedOutput -ne $recordedResolved) {
        $launchPath = Join-Path $resolvedStatus 'launch.json'
        $oldLaunch = if (Test-Path -LiteralPath $launchPath) { Get-Content -LiteralPath $launchPath -Raw | ConvertFrom-Json } else { $null }
        if (-not $oldLaunch -or $resolvedOutput -ne $oldLaunch.output) {
            throw 'This status directory belongs to another output location; use its LOCATION.json or a new status directory.'
        }
        $resolvedOutput = $recordedResolved
    }
}
$statusPath = Join-Path $resolvedStatus 'status.json'
if (Test-Path -LiteralPath $statusPath) {
    $oldStatus = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    $oldProcess = if ($oldStatus.pid) { Get-Process -Id $oldStatus.pid -ErrorAction SilentlyContinue } else { $null }
    if ($oldStatus.status -eq 'running' -and $oldProcess -and $oldProcess.ProcessName -eq 'python') {
        throw "Full experiment already running as process $($oldStatus.pid)."
    }
}
if (Test-Path -LiteralPath (Join-Path $resolvedStatus 'STOP_REQUESTED')) {
    throw 'Remove the previous STOP_REQUESTED file before explicitly resuming.'
}
function Quote-PathArgument([string]$value) {
    if ($value.Contains('"')) { throw 'A path argument cannot contain a quote.' }
    return '"' + $value + '"'
}
$argumentList = @('-u', '-m', 'vifactcheck', 'full',
    '--output', (Quote-PathArgument $resolvedOutput),
    '--status-dir', (Quote-PathArgument $resolvedStatus),
    '--data', (Quote-PathArgument (Resolve-Path -LiteralPath $DataDirectory).Path),
    '--model-path', (Quote-PathArgument (Resolve-Path -LiteralPath $ModelDirectory).Path),
    '--segmenter-path', (Quote-PathArgument (Resolve-Path -LiteralPath $SegmenterDirectory).Path),
    '--reranker-architecture', $RerankerArchitecture)
if ($IncludeAblations) { $argumentList += '--include-ablations' }
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$consolePath = Join-Path $resolvedStatus "console-$stamp.log"
$errorPath = Join-Path $resolvedStatus "stderr-$stamp.log"
$job = Start-Process -FilePath $pythonPath -ArgumentList $argumentList -WorkingDirectory $projectRoot `
    -WindowStyle Hidden -RedirectStandardOutput $consolePath -RedirectStandardError $errorPath -PassThru
$launch = [ordered]@{ process_id = $job.Id; started_utc = [DateTime]::UtcNow.ToString('o');
    output = $resolvedOutput; status_directory = $resolvedStatus; stdout = $consolePath; stderr = $errorPath }
$launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $resolvedStatus 'launch.json') -Encoding utf8
$launch | ConvertTo-Json
