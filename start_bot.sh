#!/bin/bash
cd /home/baptisteleloup/Trading-HyperLiquid

PIDFILE="/tmp/hl_bot.pid"

# Check if already running via PID file
if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Bot HL déjà en cours (PID=$OLD_PID), rien à faire."
        exit 0
    else
        echo "PID file obsolète, nettoyage..."
        rm -f "$PIDFILE"
    fi
fi

# Also kill any stray instances just in case
pkill -f "Trading-HyperLiquid.*main.py" 2>/dev/null
sleep 1

HL_SKIP_CONFIRM=YES nohup .venv/bin/python3 main.py \
  --mode live --strategy trend bull --symbol BTC/USDC:USDC \
  >> logs/bear_trader.log 2>&1 &

BOT_PID=$!
echo "$BOT_PID" > "$PIDFILE"
echo "Bot HL lancé PID=$BOT_PID"
