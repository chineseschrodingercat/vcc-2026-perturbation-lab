@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" "restart\project_status.py"
pause
