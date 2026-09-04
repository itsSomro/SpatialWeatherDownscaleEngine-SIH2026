@echo off
cd /d "%~dp0"
title Spatial Weather API Backend - SIH 2026

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [*] Starting FastAPI Backend on http://127.0.0.1:8000 ...
"%PYTHON_EXE%" -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
pause
