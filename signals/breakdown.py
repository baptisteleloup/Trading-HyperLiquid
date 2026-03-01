"""
Strategy 2: Breakdown Short

Entry conditions (ALL must be true):
  1. Bear market regime confirmed on 4h TF
  2. Price breaks and closes below 20-period support level
  3. Volume at breakdown >= 1.5× 20-period average volume
  4. MACD histogram is negative (bearish momentum)
  5. Price stays below support for 2 confirmation candles

Exit conditions:
  - 2R take profit (TP = entry − 2 × risk)
  - ATR stop loss above breakdown level (entry + 2×ATR)
  - MACD bullish crossover
"""

import numpy as np
import pandas as pd

import config
from indicators.technical import add_all_indicators
from signals.regime import get_regime_series, align_regime_to_df
from data.cache import load_ohlcv
from utils.logger import get_logger

logger = get_logger(__name__)

STRATEGY_NAME = "breakdown"


def generate_signals(
    symbol: str,
    start: str,
    end: str | None = None,
    testnet: bool = True,
) -> pd.DataFrame:
    """
    Generate entry/exit signals for Strategy 2.

    Returns a DataFrame with 1h OHLCV + indicators + signal columns:
        signal        : 1=enter short, 0=hold, -1=exit
        entry_price   : close of signal bar
        stop_loss     : entry + 2×ATR
        take_profit   : entry − 2×risk (2R)
    """
    df = load_ohlcv(
        symbol, config.BREAKDOWN_TIMEFRAME, start, end, testnet=testnet
    )
    df = add_all_indicators(df)

    # Align 4h regime signal
    regime = get_regime_series(symbol, start, end, testnet=testnet)
    df["bear_regime"] = align_regime_to_df(regime, df)

    # -------------------------------------------------------------------------
    # Breakdown detection
    # -------------------------------------------------------------------------
    # Condition 2: close breaks below rolling support (previous bar's support)
    support_prev = df["support_level"].shift(1)
    broke_support = df["close"] < support_prev

    # Condition 3: volume spike
    volume_spike = df["volume"] >= (
        config.BREAKDOWN_VOLUME_MULTIPLIER * df["volume_sma"]
    )

    # Condition 4: MACD histogram negative
    macd_bearish = df["macd_hist"] < 0

    # Initial breakdown bar
    initial_break = df["bear_regime"] & broke_support & volume_spike & macd_bearish

    # Condition 5: confirmation — price must stay below support for N candles
    # We require the NEXT N bars also close below support_prev of the break bar
    confirm_window = config.BREAKDOWN_CONFIRM_CANDLES
    confirmed = pd.Series(False, index=df.index)
    for i in range(confirm_window, len(df)):
        if not initial_break.iloc[i - confirm_window]:
            continue
        ref_support = support_prev.iloc[i - confirm_window]
        window_closes = df["close"].iloc[i - confirm_window + 1 : i + 1]
        if (window_closes < ref_support).all():
            confirmed.iloc[i] = True

    entry_mask = confirmed

    # -------------------------------------------------------------------------
    # Exit conditions
    # -------------------------------------------------------------------------
    # MACD bullish crossover: histogram crosses from negative to positive
    macd_bullish_cross = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)
    exit_mask = macd_bullish_cross

    # -------------------------------------------------------------------------
    # Build signal column
    # -------------------------------------------------------------------------
    df["signal"] = 0
    df.loc[entry_mask, "signal"] = 1
    df.loc[exit_mask & ~entry_mask, "signal"] = -1

    # -------------------------------------------------------------------------
    # Trade metadata
    # -------------------------------------------------------------------------
    risk = config.ATR_MULTIPLIER * df["atr"]
    df["entry_price"] = np.where(entry_mask, df["close"], np.nan)
    df["stop_loss"] = np.where(entry_mask, df["close"] + risk, np.nan)
    df["take_profit"] = np.where(entry_mask, df["close"] - 2 * risk, np.nan)

    logger.info(
        "[%s] %s: %d entry signals, %d exit signals",
        STRATEGY_NAME, symbol, entry_mask.sum(), exit_mask.sum(),
    )
    return df


def get_latest_signal(
    symbol: str,
    testnet: bool = True,
) -> dict:
    """Return the latest signal dict for live trading."""
    from datetime import datetime, timedelta

    end = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    # EMA-200 on 4h needs 200 bars = ~33 days; use 60 days for safe warmup
    start = (datetime.utcnow() - timedelta(days=60)).strftime("%Y-%m-%d")

    df = generate_signals(symbol, start=start, end=end, testnet=testnet)
    last = df.iloc[-1]

    return {
        "strategy": STRATEGY_NAME,
        "symbol": symbol,
        "timestamp": df.index[-1],
        "signal": int(last["signal"]),
        "entry_price": last["entry_price"],
        "stop_loss": last["stop_loss"],
        "take_profit": last["take_profit"],
        "bear_regime": bool(last["bear_regime"]),
        "macd_hist": last["macd_hist"],
    }
