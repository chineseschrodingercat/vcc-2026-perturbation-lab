@echo off
setlocal
cd /d "%~dp0"
echo Open https://virtualcellchallenge.org/app/credentials in your signed-in Chrome.
echo Paste your API key only into the hidden local prompt. Do not send it in chat.
".venv\Scripts\vcc.exe" login --token-stdin
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -u "restart\get_official_data.py"
if errorlevel 1 goto failed
echo Official input downloads finished. Next: BUILD_FINAL.cmd
pause
exit /b 0
:failed
echo Login or download did not complete. Read the error above, then run this file again.
pause
exit /b 1
