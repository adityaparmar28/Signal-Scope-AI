@echo off
title SignalScope v2.0 - High-Speed Forensic Suite
echo ========================================================
echo   SIGNALSCOPE v2.0 - AI FORENSIC DETECTION SUITE
echo   High-Speed HTML/CSS/JS Frontend + FastAPI Engine
echo ========================================================
echo.
echo [*] Starting Local Engine on http://localhost:8000 ...
echo [*] Model: EfficientNet-B0 + SRM Frequency Steganalysis
echo [*] Accuracy: 91.02%% ^| ROC-AUC: 0.9711
echo.

start "" "http://localhost:8000"

python -m uvicorn web_server:app --host 127.0.0.1 --port 8000

pause
