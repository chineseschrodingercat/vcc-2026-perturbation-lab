@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -u "restart\download_data.py" --root "%CD%" --manifest "restart\download_manifest.json" --workers 4
pause
