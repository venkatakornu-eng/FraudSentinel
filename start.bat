@echo off
echo.
echo  ================================================
echo    FraudSentinel v1.0 - Real-Time Fraud Detection
echo  ================================================
echo.

echo [1/3] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Install from python.org
    pause
    exit /b 1
)
echo   OK: Python found

echo [2/3] Checking dependencies...
python -c "import flask, pandas, numpy, sklearn" >nul 2>&1
if errorlevel 1 (
    echo   Installing requirements...
    pip install flask pandas numpy scikit-learn --quiet
)
echo   OK: Dependencies ready

echo [3/3] Starting server...
echo.
echo   Open browser:  http://localhost:5000
echo   Stop server:   Ctrl+C
echo.
echo   Model training starts automatically (about 60 seconds)
echo   Dashboard unlocks when training is complete
echo.

timeout /t 3 /nobreak >nul
start "" http://localhost:5000
python app.py
pause
