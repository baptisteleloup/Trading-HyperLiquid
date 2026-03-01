"""
Bear market regime filter.

Operates on the 4h timeframe. Returns a boolean Series aligned to any
lower-timeframe DataFrame via forward-fill resampling.
"""

import pandas as pd

from data.cache import load_ohlcv
from indicators.technical import add_regime_indicator
import config
from utils.logger import get_logger

logger = get_logger(__name__)


def get_regime_series(
    symbol: str,
    start: str,
    end: str | None = None,
    testnet: bool = True,
) -> pd.Series:
    """
    Return a boolean Series (bear_regime) on the 4h timeframe.

    True  → bear market regime confirmed (price below EMA-200 on 4h)
    False → regime not confirmed, no trades allowed
    """
    df_4h = load_ohlcv(
        symbol,
        config.REGIME_TIMEFRAME,
        start,
        end,
        testnet=testnet,
    )
    df_4h = add_regime_indicator(df_4h, ema_period=config.REGIME_EMA_PERIOD)
    logger.debug(
        "Regime 4h: %d/%d bars in bear regime",
        df_4h["bear_regime"].sum(),
        len(df_4h),
    )
    return df_4h["bear_regime"]


def get_daily_regime_series(
    symbol: str,
    start: str,
    end: str | None = None,
    testnet: bool = True,
) -> pd.Series:
    """
    Return a boolean Series (bear_regime) on the daily timeframe.

    True  → price below EMA-50 daily (confirmed medium-term downtrend)
    False → not in bear regime
    """
    from datetime import datetime, timedelta

    # Fetch extra history for EMA warmup (EMA-50 daily needs 50+ bars)
    start_dt = datetime.fromisoformat(start)
    warmup_start = (start_dt - timedelta(days=80)).strftime("%Y-%m-%d")

    df_1d = load_ohlcv(
        symbol,
        config.REGIME_TIMEFRAME_DAILY,
        warmup_start,
        end,
        testnet=testnet,
    )
    df_1d = add_regime_indicator(df_1d, ema_period=config.REGIME_EMA_DAILY)
    logger.debug(
        "Regime daily: %d/%d bars in bear regime",
        df_1d["bear_regime"].sum(),
        len(df_1d),
    )
    return df_1d["bear_regime"]


def align_regime_to_df(
    regime: pd.Series,
    target_df: pd.DataFrame,
) -> pd.Series:
    """
    Forward-fill the 4h regime signal onto a lower-timeframe index.

    Ensures every bar in target_df has a regime label.
    """
    aligned = (
        regime.reindex(regime.index.union(target_df.index))
        .ffill()
        .reindex(target_df.index)
    )
    aligned = aligned.fillna(False)
    return aligned.rename("bear_regime")
