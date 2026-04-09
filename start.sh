#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  FraudSentinel — Startup Script
#  Launches the Flask backend and opens the dashboard
# ═══════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║         🛡  FraudSentinel v1.0               ║"
echo "║   Real-Time Fraud Detection System           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌  Python 3 not found. Please install Python 3.9+"
    exit 1
fi

# Check pip packages
echo "[ 1/3 ] Checking dependencies..."
python3 -c "import flask, pandas, numpy, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "  Installing requirements..."
    pip3 install flask pandas numpy scikit-learn --quiet
fi
echo "  ✓  Dependencies OK"

# Check data file
echo "[ 2/3 ] Checking data..."
if [ ! -f "data/fraud_transactions.csv" ]; then
    echo "❌  data/fraud_transactions.csv not found"
    echo "    Please place the dataset in the data/ folder"
    exit 1
fi
echo "  ✓  Data file found (75,000 transactions)"

# Launch server
echo "[ 3/3 ] Starting FraudSentinel server..."
echo ""
echo "  URL:  http://localhost:5000"
echo "  Stop: Ctrl+C"
echo ""
echo "  ⏳  Model training starts automatically (~60 seconds)"
echo "  ✅  Dashboard will unlock once training completes"
echo ""

# Open browser after short delay (macOS / Linux)
(sleep 3 && python3 -c "
import subprocess, sys
try:
    if sys.platform == 'darwin':
        subprocess.run(['open', 'http://localhost:5000'], check=False)
    elif sys.platform == 'linux':
        subprocess.run(['xdg-open', 'http://localhost:5000'], check=False)
except: pass
") &

python3 app.py
