@echo off
echo ===========================================
echo SignalScope - 99%% Accuracy Training Pipeline
echo ===========================================

echo [1/2] Running Data Pipeline (Download, Merge, Dedup)...
python src\data_pipeline.py

echo.
echo [2/2] Starting Model Training...
python src\train.py --epochs 15 --batch_size 32 --lr 3e-5 --patience 4

echo.
echo Pipeline Complete! Best model saved in model\weights\best_model.pth
pause
