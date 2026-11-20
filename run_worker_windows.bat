@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Chua co moi truong .venv.
  echo Hay tao moi truong va cai dat requirements.txt truoc.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" worker.py
pause
