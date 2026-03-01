#!/bin/bash
# Script de démarrage persistant du bot Trading
cd /home/baptisteleloup/Trading

# Kill any existing instance
pkill -f "main.py" 2>/dev/null
sleep 2

# Launch
nohup .venv/bin/python3 main.py \
  --mode live \
  --strategy trend \
  --symbol BTC/USDC:USDC \
  >> logs/bear_trader.log 2>&1 &

echo "Bot démarré PID=$!"
