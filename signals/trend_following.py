"""
Strategy 1: Trend Following Short

Entry conditions (ALL must be true):
  1. Bear market regime confirmed on 4h TF (price below EMA-200)
  2. EMA-20 crosses below EMA-50 (death cross) on 1h
  3. ADX > 25 (strong trend)
  4. RSI < 50 (momentum confirms downside)

Exit conditions (first hit wins):
  - RSI < 30 → oversold, take profit
  - EMA-20 crosses back above EMA-50 → close position
  - ATR-based stop loss hit (entry + ATR_MULTIPLIER × ATR above entry)
"""

import pandas as pd
import numpy as np

import config
from indicators.technical import add_all_indicators
from signals.regime import get_regime_series, align_regime_to_df
from data.cache import load_ohlcv
from utils.logger import get_logger

logger = get_logger(__name__)

STRATEGY_NAME = "trend_following"


def generate_signals(
    symbol: str,
    start: str,
    end: str | None = None,
    testnet: bool = True,
) -> pd.DataFrame:
    """
    Generate entry/exit signals for Strategy 1.

    Returns a DataFrame with the 1h OHLCV data + indicators + signal columns:
        signal        : 1=enter short, 0=hold, -1=exit
        entry_price   : price at entry (close of signal bar)
        stop_loss     : initial stop-loss price
        take_profit   : initial take-profit price (RSI-based, added for reference)
    """
    # Load and prepare 1h data
    df = load_ohlcv(symbol, config.TREND_TIMEFRAME, start, end, testnet=testnet)
    df = add_all_indicators(df)

    # Align regime from 4h onto 1h index
    regime = get_regime_series(symbol, start, end, testnet=testnet)
    df["bear_regime"] = align_regime_to_df(regime, df)

    # -------------------------------------------------------------------------
    # Entry: all conditions must fire simultaneously
    # -------------------------------------------------------------------------
    entry_mask = (
        df["bear_regime"]                                      # 1. regime
        & df["death_cross"]                                    # 2. EMA death cross
        & (df["adx"] > config.TREND_ADX_THRESHOLD)            # 3. strong trend
        & (df["rsi"] < config.TREND_RSI_ENTRY_MAX)            # 4. momentum
    )

    # -------------------------------------------------------------------------
    # Exit conditions (any)
    # -------------------------------------------------------------------------
    exit_rsi = df["rsi"] < config.TREND_RSI_OVERSOLD
    exit_golden_cross = df["golden_cross"]
    exit_mask = exit_rsi | exit_golden_cross

    # -------------------------------------------------------------------------
    # Combine into signal column
    # -------------------------------------------------------------------------
    df["signal"] = 0
    df.loc[entry_mask, "signal"] = 1    # enter short
    df.loc[exit_mask & ~entry_mask, "signal"] = -1  # exit

    # -------------------------------------------------------------------------
    # Entry metadata (filled only on entry bars)
    # -------------------------------------------------------------------------
    df["entry_price"] = np.where(entry_mask, df["close"], np.nan)
    df["stop_loss"] = np.where(
        entry_mask,
        df["close"] + config.ATR_MULTIPLIER * df["atr"],
        np.nan,
    )
    # Take-profit is dynamic (RSI < 30), but provide 2R as reference
    df["take_profit"] = np.where(
        entry_mask,
        df["close"] - 2 * (config.ATR_MULTIPLIER * df["atr"]),
        np.nan,
    )

    n_entries = entry_mask.sum()
    logger.info(
        "[%s] %s: %d entry signals, %d exit signals",
        STRATEGY_NAME, symbol, n_entries, exit_mask.sum(),
    )
    return df


def get_latest_signal(
    symbol: str,
    testnet: bool = True,
) -> dict:
    """
    Fetch recent data and return the latest signal for live trading.

    Returns a dict with keys:
        signal (1 | 0 | -1), timestamp, entry_price, stop_loss, take_profit
    """
    from datetime import datetime, timedelta

    end = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    # Fetch enough history for indicators (EMA-200 needs 200 bars minimum)
    start = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")

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
        "rsi": last["rsi"],
        "adx": last["adx"],
    }
