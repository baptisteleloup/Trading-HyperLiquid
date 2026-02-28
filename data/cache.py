"""
Local CSV cache for OHLCV data.

Avoids re-fetching data on every run. Cache is keyed by symbol + timeframe.
Stale data (end date in the past) is automatically refreshed.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional

from data.fetcher import fetch_ohlcv, OHLCV_COLUMNS
from utils.logger import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"


def _cache_path(symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe_symbol}_{timeframe}.csv"


def load_ohlcv(
    symbol: str,
    timeframe: str,
    start: str,
    end: Optional[str] = None,
    testnet: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return OHLCV data, loading from local CSV cache when possible.

    Cache logic:
    - If no cache exists → fetch and save.
    - If cache exists and covers the full requested window → return cached.
    - If cache is partial (missing recent data) → fetch missing tail and append.
    - force_refresh=True → always re-fetch from API.

    Args:
        symbol:        e.g. "BTC/USDT:USDT"
        timeframe:     e.g. "1h"
        start:         ISO date string
        end:           ISO date string (exclusive). Defaults to now.
        testnet:       Use Binance testnet
        force_refresh: Bypass cache

    Returns:
        DataFrame filtered to [start, end) window.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, timeframe)

    cached: Optional[pd.DataFrame] = None

    if path.exists() and not force_refresh:
        try:
            cached = _read_cache(path)
        except Exception as exc:
            logger.warning("Failed to read cache %s: %s", path, exc)
            cached = None

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")

    if cached is not None:
        cache_start = cached.index.min()
        cache_end = cached.index.max()

        needs_head = start_ts < cache_start
        needs_tail = end_ts > cache_end + pd.Timedelta("1ms")

        if not needs_head and not needs_tail:
            logger.info("Cache hit for %s %s", symbol, timeframe)
            return _filter(cached, start_ts, end_ts)

        if needs_tail and not needs_head:
            # Fetch only the missing tail
            tail_start = cache_end.strftime("%Y-%m-%d")
            logger.info(
                "Partial cache for %s %s — fetching tail from %s",
                symbol, timeframe, tail_start,
            )
            tail = fetch_ohlcv(symbol, timeframe, tail_start, end, testnet)
            if not tail.empty:
                combined = pd.concat([cached, tail])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                _write_cache(combined, path)
                return _filter(combined, start_ts, end_ts)
            return _filter(cached, start_ts, end_ts)

    # Full fetch required
    logger.info("Cache miss for %s %s — fetching from API", symbol, timeframe)
    df = fetch_ohlcv(symbol, timeframe, start, end, testnet)
    if not df.empty:
        _write_cache(df, path)
    return df


def _read_cache(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path)
    logger.debug("Wrote %d rows to cache %s", len(df), path)


def _filter(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df.index >= start) & (df.index < end)].copy()
