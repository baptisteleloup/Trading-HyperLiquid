"""
Technical indicator computation.

All indicators are added as new columns to the input DataFrame in-place.
Uses pandas-ta for the heavy lifting; custom logic where needed.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta

import config


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and attach all indicators needed by the strategies.

    Mutates a copy of df and returns it. Requires columns:
        open, high, low, close, volume

    Added columns:
        ema_fast, ema_slow, ema_200
        rsi
        macd, macd_signal, macd_hist
        adx, adx_pos, adx_neg
        atr
        bb_upper, bb_mid, bb_lower
        volume_sma
        support_level
        death_cross, golden_cross  (boolean signals)
    """
    df = df.copy()

    # -------------------------------------------------------------------------
    # Moving averages
    # -------------------------------------------------------------------------
    df["ema_fast"] = ta.ema(df["close"], length=config.TREND_EMA_FAST)
    df["ema_slow"] = ta.ema(df["close"], length=config.TREND_EMA_SLOW)
    df["ema_200"] = ta.ema(df["close"], length=config.REGIME_EMA_PERIOD)

    # -------------------------------------------------------------------------
    # RSI
    # -------------------------------------------------------------------------
    df["rsi"] = ta.rsi(df["close"], length=config.TREND_RSI_PERIOD)

    # -------------------------------------------------------------------------
    # MACD
    # -------------------------------------------------------------------------
    macd_df = ta.macd(
        df["close"],
        fast=config.BREAKDOWN_MACD_FAST,
        slow=config.BREAKDOWN_MACD_SLOW,
        signal=config.BREAKDOWN_MACD_SIGNAL,
    )
    if macd_df is not None:
        df["macd"] = macd_df.iloc[:, 0]
        df["macd_hist"] = macd_df.iloc[:, 1]
        df["macd_signal"] = macd_df.iloc[:, 2]

    # -------------------------------------------------------------------------
    # ADX
    # -------------------------------------------------------------------------
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=config.TREND_ADX_PERIOD)
    if adx_df is not None:
        df["adx"] = adx_df.iloc[:, 0]
        df["adx_pos"] = adx_df.iloc[:, 1]
        df["adx_neg"] = adx_df.iloc[:, 2]

    # -------------------------------------------------------------------------
    # ATR
    # -------------------------------------------------------------------------
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=config.ATR_PERIOD)

    # -------------------------------------------------------------------------
    # Bollinger Bands
    # -------------------------------------------------------------------------
    bb_df = ta.bbands(df["close"], length=20, std=2)
    if bb_df is not None:
        df["bb_lower"] = bb_df.iloc[:, 0]
        df["bb_mid"] = bb_df.iloc[:, 1]
        df["bb_upper"] = bb_df.iloc[:, 2]

    # -------------------------------------------------------------------------
    # Volume SMA
    # -------------------------------------------------------------------------
    df["volume_sma"] = ta.sma(df["volume"], length=config.BREAKDOWN_SUPPORT_PERIOD)

    # -------------------------------------------------------------------------
    # Rolling support level (lowest low over N periods)
    # -------------------------------------------------------------------------
    df["support_level"] = (
        df["low"].rolling(window=config.BREAKDOWN_SUPPORT_PERIOD).min()
    )

    # -------------------------------------------------------------------------
    # Death cross / golden cross signals
    # -------------------------------------------------------------------------
    df["ema_cross"] = np.sign(df["ema_fast"] - df["ema_slow"])
    df["death_cross"] = (df["ema_cross"] == -1) & (df["ema_cross"].shift(1) == 1)
    df["golden_cross"] = (df["ema_cross"] == 1) & (df["ema_cross"].shift(1) == -1)

    return df


def add_regime_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add bear market regime column to a 4h DataFrame.

    bear_regime = True when close is below EMA-200.
    """
    df = df.copy()
    df["ema_200"] = ta.ema(df["close"], length=config.REGIME_EMA_PERIOD)
    df["bear_regime"] = df["close"] < df["ema_200"]
    return df
