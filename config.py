"""
Single source of truth for all system parameters.
No magic numbers should appear in logic files — import from here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Exchange credentials
# ---------------------------------------------------------------------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------

# Bear market regime filter (higher timeframe)
REGIME_TIMEFRAME = "4h"
REGIME_EMA_PERIOD = 200          # Price must be below this EMA

# Strategy 1 — Trend Following Short
TREND_TIMEFRAME = "1h"
TREND_EMA_FAST = 20
TREND_EMA_SLOW = 50
TREND_ADX_PERIOD = 14
TREND_ADX_THRESHOLD = 25         # Minimum ADX for a "strong" trend
TREND_RSI_PERIOD = 14
TREND_RSI_ENTRY_MAX = 50         # RSI must be below this to enter
TREND_RSI_OVERSOLD = 30          # Close position when RSI hits oversold

# Strategy 2 — Breakdown Short
BREAKDOWN_TIMEFRAME = "1h"
BREAKDOWN_SUPPORT_PERIOD = 20    # Rolling window to detect support
BREAKDOWN_VOLUME_MULTIPLIER = 1.5  # Volume at break >= 1.5× avg volume
BREAKDOWN_CONFIRM_CANDLES = 2    # Candles price must stay below support
BREAKDOWN_MACD_FAST = 12
BREAKDOWN_MACD_SLOW = 26
BREAKDOWN_MACD_SIGNAL = 9

# ATR for stop-loss calculation (shared)
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0             # SL = entry + ATR_MULTIPLIER × ATR

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------
RISK_PER_TRADE = 0.01            # 1% of equity per trade
LEVERAGE = 5                     # 5× leverage
MAX_OPEN_POSITIONS = 3
CIRCUIT_BREAKER_DRAWDOWN = 0.15  # Halt if equity drops 15% from session high
MIN_ORDER_USDT = 10.0            # Minimum Binance order size

# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
BACKTEST_START = "2022-06-01"
BACKTEST_END = "2023-01-01"
BACKTEST_INITIAL_CAPITAL = 10_000.0   # USDT
BACKTEST_TAKER_FEE = 0.0004          # 0.04%
BACKTEST_SLIPPAGE = 0.0005           # 5 bps
BACKTEST_FUNDING_RATE = 0.0001       # 0.01% per 8h

# ---------------------------------------------------------------------------
# Live trading
# ---------------------------------------------------------------------------
POLL_INTERVAL_SECONDS = 60
ORDER_TIMEOUT_SECONDS = 30           # Cancel limit order after this
LIMIT_ORDER_OFFSET_BPS = 0.0002      # 2 bps inside best bid/ask

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FILE = "logs/bear_trader.log"
TRADE_JOURNAL_FILE = "logs/trade_journal.csv"
