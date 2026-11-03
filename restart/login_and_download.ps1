$ErrorActionPreference='Stop'
$VccRoot = Split-Path -Parent $PSScriptRoot
$VccPython = Join-Path $VccRoot '.venv\Scripts\python.exe'
$VccCli = Join-Path $VccRoot '.venv\Scripts\vcc.exe'
Write-Host 'Open https://virtualcellchallenge.org/app/credentials in your signed-in Chrome.'
Write-Host 'Generate/copy your VCC API key, then paste it at the hidden prompt below. Do not post the key in chat.'
& $VccCli login --token-stdin
if ($LASTEXITCODE -ne 0) { throw 'VCC login did not succeed.' }
& $VccPython (Join-Path $PSScriptRoot 'get_official_data.py')
if ($LASTEXITCODE -ne 0) { throw 'Official data download did not complete. Run this script again to resume.' }
Write-Host 'Official final and validation input bundles are ready. No predictions have been submitted.'
