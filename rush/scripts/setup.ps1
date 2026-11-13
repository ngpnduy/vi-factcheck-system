param([switch]$ReuseInstalledPackages, [string]$EnvironmentDirectory = '')
$ErrorActionPreference = 'Stop'
Write-Warning 'The standalone project moved to research/. This launcher forwards there.'
& (Join-Path $PSScriptRoot '..\..\research\scripts\setup.ps1') @PSBoundParameters
