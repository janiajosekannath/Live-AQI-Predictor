@echo off
title Regime-Aware AQI Prediction System - DM Project
echo ======================================================================
echo    REGIME-AWARE AIR QUALITY INDEX (AQI) PREDICTION & PATTERN MINING
echo    Data Mining Course Project Frontend
echo ======================================================================
echo.
echo Starting Flask backend on http://127.0.0.1:5000 ...
echo Opening web application in default browser...
echo.
timeout /t 2 >nul
start "" "http://127.0.0.1:5000"
py app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Notice] Python Flask encountered an issue or is not running.
    echo Opening standalone web interface directly in browser...
    start "" "index.html"
)
pause
