$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$VccRoot = Split-Path -Parent $PSScriptRoot
$VccPython = Join-Path $VccRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $VccPython)) { & 'C:\ProgramData\anaconda3\envs\bigbro\python.exe' -m venv --system-site-packages (Join-Path $VccRoot '.venv'); if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' } }
& $VccPython -m pip install --disable-pip-version-check 'vcc-cli>=0.3.0' requests h5py anndata scipy scikit-learn
if ($LASTEXITCODE -ne 0) { throw 'package installation failed' }
& (Join-Path $VccRoot '.venv\Scripts\vcc.exe') --version
& $VccPython -c "import torch,anndata,numpy,scipy; print({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'anndata':anndata.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__})"
& $VccPython -m pip freeze | Set-Content -Encoding UTF8 (Join-Path $PSScriptRoot 'environment.lock.txt')
'DONE'
