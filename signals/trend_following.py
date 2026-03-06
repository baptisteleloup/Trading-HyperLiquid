"""
Strategy: Trend Following Short (v2) — Bear Market

Entry conditions (ALL must be true):
  1. Regime is BEAR (composite score <= -3)
  2. EMA-20 < EMA-50 on 1h (ongoing bearish alignment)
  3. ADX > 25 (strong trend)
  4. RSI < 60 (not in a strong bullish rebound)

Exit conditions (first hit wins):
  - RSI < 35 → oversold, take profit
  - EMA-20 crosses back above EMA-50 (golden cross) → close position
  - ATR-based stop loss hit
"""

import pandas as pd
import numpy as np

import config
from indicators.technical import add_all_indicators
from data.cache import load_ohlcv
from regime.classifier import classify_historical, Regime
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
    Generate entry/exit signals for the bear trend strategy.

    Returns a DataFrame with signal columns:
        signal      : 1=enter short, 0=hold, -1=exit
        side        : "short" on entry bars
        entry_price : close of signal bar
        stop_loss   : entry + 2×ATR (above entry)
        take_profit : entry − 2×risk (below entry)
    """
    from datetime import datetime as _dt, timedelta as _td

    # Load 1h data with warmup
    warmup_start = (_dt.fromisoformat(start) - _td(days=60)).strftime("%Y-%m-%d")
    df_full = load_ohlcv(symbol, config.TREND_TIMEFRAME, warmup_start, end, testnet=testnet)
    df_full = add_all_indicators(df_full)
    start_ts = pd.Timestamp(start, tz="UTC")
    df = df_full[df_full.index >= start_ts].copy()

    # Fetch daily + weekly for regime classification
    regime_start = (_dt.fromisoformat(start) - _td(days=300)).strftime("%Y-%m-%d")
    df_daily = load_ohlcv(symbol, "1d", regime_start, end, testnet=testnet)
    weekly_start = (_dt.fromisoformat(start) - _td(weeks=120)).strftime("%Y-%m-%d")
    df_weekly = load_ohlcv(symbol, "1w", weekly_start, end, testnet=testnet)

    # Historical regime classification
    regimes = classify_historical(symbol, df_daily, df_weekly)
    regime_aligned = (
        regimes.reindex(regimes.index.union(df.index))
        .ffill()
        .reindex(df.index)
    )
    df["regime_bear"] = regime_aligned == Regime.BEAR

    # ─── Entry conditions ─────────────────────────────────────────────
    ema_bearish_alignment = df["ema_fast"] < df["ema_slow"]

    entry_mask = (
        df["regime_bear"]                                      # 1. BEAR regime
        & ema_bearish_alignment                                # 2. EMA-20 < EMA-50
        & (df["adx"] > config.TREND_ADX_THRESHOLD)            # 3. strong trend
        & (df["rsi"] < config.TREND_RSI_ENTRY_MAX)            # 4. RSI < 60
    )

    # ─── Exit conditions ──────────────────────────────────────────────
    exit_rsi = df["rsi"] < config.TREND_RSI_OVERSOLD          # RSI < 35
    exit_golden_cross = df["golden_cross"]
    exit_mask = exit_rsi | exit_golden_cross

    # ─── Signal column (state machine: entry → hold → exit → next) ────
    raw_signal = pd.Series(0, index=df.index)
    raw_signal[entry_mask] = 1
    raw_signal[exit_mask & ~entry_mask] = -1

    df["signal"] = 0
    in_position = False
    for i in range(len(df)):
        sig = raw_signal.iloc[i]
        if not in_position and sig == 1:
            df.iloc[i, df.columns.get_loc("signal")] = 1
            in_position = True
        elif in_position and sig == -1:
            df.iloc[i, df.columns.get_loc("signal")] = -1
            in_position = False

    # ─── Trade metadata (only on actual entries) ──────────────────────
    actual_entries = df["signal"] == 1
    risk = config.ATR_MULTIPLIER * df["atr"]
    df["side"] = np.where(actual_entries, "short", "")
    df["entry_price"] = np.where(actual_entries, df["close"], np.nan)
    df["stop_loss"] = np.where(actual_entries, df["close"] + risk, np.nan)       # above entry
    df["take_profit"] = np.where(actual_entries, df["close"] - 2 * risk, np.nan)  # below entry

    n_entries = actual_entries.sum()
    logger.info(
        "[%s] %s: %d entry signals, %d exit signals",
        STRATEGY_NAME, symbol, n_entries, exit_mask.sum(),
    )

    df.attrs["strategy"] = STRATEGY_NAME
    df.attrs["side"] = "short"
    return df


def get_latest_signal(
    symbol: str,
    testnet: bool = True,
) -> dict:
    """Return the latest signal dict for live trading."""
    from datetime import datetime, timedelta, timezone

    end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    start = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d")

    df = generate_signals(symbol, start=start, end=end, testnet=testnet)
    last = df.iloc[-1]

    return {
        "strategy": STRATEGY_NAME,
        "symbol": symbol,
        "timestamp": df.index[-1],
        "signal": int(last["signal"]),
        "side": "short",
        "entry_price": last["entry_price"],
        "stop_loss": last["stop_loss"],
        "take_profit": last["take_profit"],
        "regime_bear": bool(last["regime_bear"]),
        "rsi": last["rsi"],
        "adx": last["adx"],
        "atr": last["atr"],
    }
