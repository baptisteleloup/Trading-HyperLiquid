"""
OHLCV data fetcher with pagination support.

Fetches historical candle data from Binance USDT-M Futures via ccxt,
handling pagination automatically to fetch arbitrarily long histories.
"""

import time
import ccxt
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

import config
from utils.logger import get_logger

logger = get_logger(__name__)

# Column names matching ccxt OHLCV format
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _build_exchange(testnet: bool = True) -> ccxt.binanceusdm:
    """
    Create a ccxt exchange instance for fetching public OHLCV data.
    Always uses the real Binance API — historical market data is public
    and not available on demo/testnet environments.
    """
    exchange = ccxt.binanceusdm(
        {
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
    )
    # Do NOT use demo/testnet for market data — no historical data there
    return exchange


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    start: str,
    end: Optional[str] = None,
    testnet: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance Futures with automatic pagination.

    Args:
        symbol:    e.g. "BTC/USDT:USDT"
        timeframe: e.g. "1h", "4h", "1d"
        start:     ISO date string, e.g. "2022-06-01"
        end:       ISO date string (exclusive). Defaults to now.
        testnet:   Use Binance testnet if True.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
        Index is a DatetimeIndex (UTC).
    """
    exchange = _build_exchange(testnet)

    start_ts = int(
        datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1000
    )
    if end:
        end_ts = int(
            datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp() * 1000
        )
    else:
        end_ts = int(time.time() * 1000)

    all_candles: list[list] = []
    since = start_ts
    limit = 1000  # Binance max per request

    logger.info(
        "Fetching %s %s from %s to %s",
        symbol,
        timeframe,
        start,
        end or "now",
    )

    while since < end_ts:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=limit
            )
        except ccxt.NetworkError as exc:
            logger.warning("Network error, retrying in 5s: %s", exc)
            time.sleep(5)
            continue
        except ccxt.ExchangeError as exc:
            logger.error("Exchange error: %s", exc)
            raise

        if not candles:
            break

        # Filter candles within requested window
        candles = [c for c in candles if c[0] < end_ts]
        if not candles:
            break  # All remaining data is past end_ts
        all_candles.extend(candles)

        last_ts = candles[-1][0]
        if last_ts <= since:
            break  # No progress — avoid infinite loop
        since = last_ts + 1

        # Respect rate limit
        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        logger.warning("No data returned for %s %s", symbol, timeframe)
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = pd.DataFrame(all_candles, columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    logger.info("Fetched %d candles for %s %s", len(df), symbol, timeframe)
    return df
