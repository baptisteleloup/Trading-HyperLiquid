"""
Strategy: Trend Following Long (Bull)

Mirror of the bear trend_following strategy, but for long positions.

Entry conditions (ALL must be true):
  1. Regime is BULL (composite score >= +3)
  2. EMA-20 > EMA-50 on 1h (ongoing bullish alignment)
  3. ADX > 25 (strong trend)
  4. RSI < 65 (not overbought — room to run)

Exit conditions (first hit wins):
  - RSI > 70 → overbought, take profit
  - EMA-20 crosses below EMA-50 (death cross) → close position
  - ATR-based stop loss hit (entry − 2×ATR below entry)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

import config
from indicators.technical import add_all_indicators
from data.cache import load_ohlcv
from regime.classifier import classify_historical, Regime
from utils.logger import get_logger

logger = get_logger(__name__)

STRATEGY_NAME = "trend_bull"


def generate_signals(
    symbol: str,
    start: str,
    end: str | None = None,
    testnet: bool = True,
) -> pd.DataFrame:
    """
    Generate entry/exit signals for the bull trend strategy.

    Returns a DataFrame with signal columns:
        signal      : 1=enter long, 0=hold, -1=exit
        side        : "long" on entry bars
        entry_price : close of signal bar
        stop_loss   : entry − 2×ATR (below entry)
        take_profit : entry + 2×risk (above entry)
    """
    # Load 1h data with warmup
    warmup_start = (
        datetime.fromisoformat(start) - timedelta(days=60)
    ).strftime("%Y-%m-%d")
    df_full = load_ohlcv(symbol, config.BULL_TIMEFRAME, warmup_start, end, testnet=testnet)
    df_full = add_all_indicators(df_full)

    # Trim to requested window
    start_ts = pd.Timestamp(start, tz="UTC")
    df = df_full[df_full.index >= start_ts].copy()

    # Fetch daily + weekly data for regime classification
    regime_start = (
        datetime.fromisoformat(start) - timedelta(days=300)
    ).strftime("%Y-%m-%d")
    df_daily = load_ohlcv(symbol, "1d", regime_start, end, testnet=testnet)

    weekly_start = (
        datetime.fromisoformat(start) - timedelta(weeks=120)
    ).strftime("%Y-%m-%d")
    df_weekly = load_ohlcv(symbol, "1w", weekly_start, end, testnet=testnet)

    # Get historical regime classification
    regimes = classify_historical(symbol, df_daily, df_weekly)

    # Align daily regime to 1h bars
    regime_aligned = (
        regimes.reindex(regimes.index.union(df.index))
        .ffill()
        .reindex(df.index)
    )
    df["regime_bull"] = regime_aligned == Regime.BULL

    # ─── Entry conditions ─────────────────────────────────────────────
    ema_bullish_alignment = df["ema_fast"] > df["ema_slow"]

    entry_mask = (
        df["regime_bull"]                                      # 1. BULL regime
        & ema_bullish_alignment                                # 2. EMA-20 > EMA-50
        & (df["adx"] > config.BULL_ADX_THRESHOLD)             # 3. strong trend
        & (df["rsi"] < config.BULL_RSI_ENTRY_MAX)             # 4. RSI < 65
    )

    # ─── Exit conditions ──────────────────────────────────────────────
    exit_rsi = df["rsi"] > config.BULL_RSI_OVERBOUGHT          # RSI > 70
    exit_death_cross = df["death_cross"]                       # EMA-20 crosses below EMA-50
    exit_mask = exit_rsi | exit_death_cross

    # ─── Signal column ────────────────────────────────────────────────
    df["signal"] = 0
    df.loc[entry_mask, "signal"] = 1
    df.loc[exit_mask & ~entry_mask, "signal"] = -1

    # ─── Trade metadata ───────────────────────────────────────────────
    risk = config.ATR_MULTIPLIER_BULL * df["atr"]
    df["side"] = np.where(entry_mask, "long", "")
    df["entry_price"] = np.where(entry_mask, df["close"], np.nan)
    df["stop_loss"] = np.where(entry_mask, df["close"] - risk, np.nan)       # below entry
    df["take_profit"] = np.where(entry_mask, df["close"] + 2 * risk, np.nan)  # above entry

    n_entries = entry_mask.sum()
    logger.info(
        "[%s] %s: %d entry signals, %d exit signals",
        STRATEGY_NAME, symbol, n_entries, exit_mask.sum(),
    )

    df.attrs["strategy"] = STRATEGY_NAME
    df.attrs["side"] = "long"
    return df


def get_latest_signal(
    symbol: str,
    testnet: bool = True,
) -> dict:
    """Return the latest signal dict for live trading."""
    end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    start = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d")

    df = generate_signals(symbol, start=start, end=end, testnet=testnet)
    last = df.iloc[-1]

    return {
        "strategy": STRATEGY_NAME,
        "symbol": symbol,
        "timestamp": df.index[-1],
        "signal": int(last["signal"]),
        "side": "long",
        "entry_price": last["entry_price"],
        "stop_loss": last["stop_loss"],
        "take_profit": last["take_profit"],
        "regime_bull": bool(last["regime_bull"]),
        "rsi": last["rsi"],
        "adx": last["adx"],
    }
