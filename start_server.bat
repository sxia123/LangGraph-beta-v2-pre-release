@echo off
cd /d "%~dp0"
echo ===================================================
echo Starting LangGraph Multi-Agent Studio Server...
echo ===================================================
if exist "LangGraph-beta-v2-pre-release\.venv\Scripts\python.exe" (
    "LangGraph-beta-v2-pre-release\.venv\Scripts\python.exe" server.py
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" server.py
) else (
    python server.py
)
pause
