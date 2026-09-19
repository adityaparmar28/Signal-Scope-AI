@echo off
title SignalScope Multi-Dataset Training Pipeline
cd /d "%~dp0"

echo ========================================================
echo   SIGNALSCOPE - DEEP LEARNING MODEL TRAINING PIPELINE
echo   Architecture: EfficientNet-B0 + SRM Frequency Steganalysis
echo ========================================================
echo.
echo [*] Working Directory: %CD%
echo [*] Verifying PyTorch and dependencies...
python -c "import torch; print('[+] PyTorch Version:', torch.__version__)"
if %ERRORLEVEL% NEQ 0 (
    echo [!] PyTorch is not loading properly.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo [*] Starting Model Training on Multi-Generator Dataset...
echo ========================================================
echo [*] Training will run with early stopping and automatic best weights saving.
echo.

python src\train.py --epochs 10 --batch_size 16 --lr 3e-5 --patience 4 --data_dir data

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Training stopped with an error code: %ERRORLEVEL%
) else (
    echo.
    echo ========================================================
    echo [OK] Training completed successfully!
    echo [OK] Best model saved in: model\weights\best_model.pth
    echo ========================================================
)

echo.
pause
