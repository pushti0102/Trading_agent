# =============================================================================
# OliveTrade 2.0 — Automated Windows PowerShell Virtual Environment Setup Script
# =============================================================================

Write-Host "======================================================" -ForegroundColor Green
Write-Host " OliveTrade 2.0: Multi-Agent RL Research Platform Setup" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green

$ErrorActionPreference = "Stop"
$WorkspaceRoot = $PSScriptRoot

Set-Location $WorkspaceRoot
Write-Host "[1/4] Checking Python Installation..." -ForegroundColor Cyan

# Locate Python 3.11+
$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    $PotentialPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "C:\Program Files\Python311\python.exe",
        "C:\Python311\python.exe"
    )
    foreach ($path in $PotentialPaths) {
        if (Test-Path $path) {
            $PythonExe = $path
            break
        }
    }
}

if (-not $PythonExe) {
    Write-Error "Python 3.11+ was not detected. Please install Python 3.11 and ensure it is in your PATH."
    exit 1
}

Write-Host "Found Python executable at: $PythonExe" -ForegroundColor Green

# Create Virtual Environment if not present
$VenvDir = Join-Path $WorkspaceRoot ".venv"
if (-not (Test-Path $VenvDir)) {
    Write-Host "[2/4] Creating virtual environment (.venv)..." -ForegroundColor Cyan
    & $PythonExe -m venv $VenvDir
} else {
    Write-Host "[2/4] Existing virtual environment found at $VenvDir" -ForegroundColor Green
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

# Install Dependencies
Write-Host "[3/4] Installing / Verifying pinned dependencies from requirements.txt..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

# Execute Validation Unit Tests
Write-Host "[4/4] Executing verification test suite..." -ForegroundColor Cyan
& $VenvPython -m pytest tests/test_core_units.py tests/test_gym_env.py tests/test_phase_03_agents.py tests/test_continuous_rl.py tests/test_risk_overlay.py tests/test_ensemble_agents.py tests/test_financial_metrics.py tests/test_walk_forward.py tests/test_robustness.py tests/test_api_v2.py -v

Write-Host "`n======================================================" -ForegroundColor Green
Write-Host " Setup Complete! OliveTrade 2.0 is ready." -ForegroundColor Green
Write-Host " To start API Backend: .venv\Scripts\uvicorn src.api.main:app --reload --port 8000" -ForegroundColor Yellow
Write-Host " To start React UI:    cd frontend; npm run dev" -ForegroundColor Yellow
Write-Host " To run experiments:   .venv\Scripts\python.exe run_experiments.py --all" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Green
