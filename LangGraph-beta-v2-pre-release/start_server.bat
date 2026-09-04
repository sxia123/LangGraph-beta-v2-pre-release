@echo off
cd /d "%~dp0"
echo ===================================================
echo Starting LangGraph Multi-Agent Studio Server...
echo ===================================================
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" server.py
) else (
    python server.py
)
pause
