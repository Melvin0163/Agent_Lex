@echo off
cd /d "%~dp0"
title LexAgent - public link
echo.
echo  Share LexAgent with a public link
echo  Keep Ollama and the decision model (ollajev serve ...) running first.
echo.
set "LEXAGENT_ACCESS_CODE="
set /p "LEXAGENT_ACCESS_CODE=Choose an access code for visitors: "
if "%LEXAGENT_ACCESS_CODE%"=="" goto nocode
set "CF=cloudflared"
where /q cloudflared || set "CF=%ProgramFiles(x86)%\cloudflared\cloudflared.exe"
if not exist "%CF%" if not "%CF%"=="cloudflared" set "CF=%ProgramFiles%\cloudflared\cloudflared.exe"
start "LexAgent app" cmd /k .venv\Scripts\python.exe webapp\server.py --no-browser
echo.
echo  Starting the app. Waiting 45 seconds...
timeout /t 45 /nobreak >nul
echo.
echo  In a few seconds a line with  https://....trycloudflare.com  appears below.
echo  That is your public link. Share it together with the access code.
echo  To stop sharing, close this window and the "LexAgent app" window.
echo.
"%CF%" tunnel --no-autoupdate --url http://127.0.0.1:8600
goto end
:nocode
echo  You need an access code to share the app. Run this file again.
:end
pause
