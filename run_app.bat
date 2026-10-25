@echo off
cd /d "%~dp0"
echo Starting LexAgent. Keep Ollama and the decision model (ollajev serve ...) running.
".venv\Scripts\python.exe" webapp\server.py
pause
