param(
    [string]$OutputDirectory = '',
    [string]$StatusDirectory = '',
    [string]$DataDirectory = '',
    [string]$ModelDirectory = '',
    [string]$SegmenterDirectory = '',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
Write-Warning 'The standalone project moved to research/. This launcher forwards there.'
& (Join-Path $PSScriptRoot '..\..\research\scripts\start_listwise_v2.ps1') @PSBoundParameters
