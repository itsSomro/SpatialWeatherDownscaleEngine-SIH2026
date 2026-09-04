@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"
title Spatial Weather Downscaler - SIH 2026

echo =========================================================
echo   Spatial Weather Downscale Engine - SIH 2026
echo =========================================================
echo.

:: Locate Python executable
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "PYTHON_EXE=python"
    ) else (
        echo [ERROR] Python was not found! Please create or activate .venv.
        pause
        exit /b 1
    )
)

:: Check if FastAPI backend on port 8000 is running
powershell -Command "if ((Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue)) { exit 0 } else { exit 1 }"
if %ERRORLEVEL% neq 0 (
    echo [*] Starting FastAPI Backend on http://127.0.0.1:8000 ...
    start "Spatial Weather API Backend" /min "%PYTHON_EXE%" -m uvicorn api.app:app --host 127.0.0.1 --port 8000
    :: Brief pause to let backend spin up
    timeout /t 3 /nobreak >nul
) else (
    echo [OK] Backend API is already running on port 8000.
)

echo [*] Launching Streamlit UI on http://localhost:8501 ...
echo.
"%PYTHON_EXE%" -m streamlit run frontend\ui.py

pause
