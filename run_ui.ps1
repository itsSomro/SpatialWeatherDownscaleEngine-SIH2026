# PowerShell Run Script for Spatial Weather Downscaler
$ProjectRoot = $PSScriptRoot
Set-Location -Path $ProjectRoot

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "  Spatial Weather Downscale Engine - SIH 2026" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host ""

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction SilentlyContinue)?.Source
    if (-not $PythonExe) {
        Write-Error "Python executable not found. Make sure .venv exists."
        Exit 1
    }
}

# Check if Backend is running on port 8000
$BackendActive = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if (-not $BackendActive) {
    Write-Host "[*] Starting FastAPI Backend on http://127.0.0.1:8000 in background..." -ForegroundColor Yellow
    Start-Process -FilePath $PythonExe -ArgumentList "-m uvicorn api.app:app --host 127.0.0.1 --port 8000" -WindowStyle Minimized
    Start-Sleep -Seconds 3
} else {
    Write-Host "[OK] Backend API is already running on port 8000." -ForegroundColor Green
}

Write-Host "[*] Launching Streamlit UI on http://localhost:8501 ..." -ForegroundColor Green
& $PythonExe -m streamlit run frontend/ui.py
