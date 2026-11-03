@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -u "restart\build_final.py" %*
set "VCC_EXIT=%ERRORLEVEL%"
if not "%VCC_EXIT%"=="0" echo Build stopped. Read the error above or the run log.
pause
exit /b %VCC_EXIT%
