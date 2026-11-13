param(
    [string]$OutputDirectory = (Join-Path $env:LOCALAPPDATA 'ViFactCheck\full-m0-m1-m2'),
    [string]$StatusDirectory = '',
    [string]$DataDirectory = '',
    [string]$ModelDirectory = '',
    [string]$SegmenterDirectory = '',
    [switch]$IncludeAblations,
    [ValidateSet('pairwise', 'listwise')][string]$RerankerArchitecture = 'pairwise'
)
$ErrorActionPreference = 'Stop'
Write-Warning 'The standalone project moved to research/. This launcher forwards there.'
& (Join-Path $PSScriptRoot '..\..\research\scripts\start_full.ps1') @PSBoundParameters
