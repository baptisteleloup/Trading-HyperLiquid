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
    df_4h = add_regime_indicator(df_4h)
    logger.debug(
        "Regime: %d/%d bars in bear regime",
        df_4h["bear_regime"].sum(),
        len(df_4h),
    )
    return df_4h["bear_regime"]


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
