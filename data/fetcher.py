"""
OHLCV data fetcher with pagination support.

Fetches historical candle data from HyperLiquid via ccxt,
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


def _build_exchange(testnet: bool = True) -> ccxt.hyperliquid:
    """
    Create a ccxt exchange instance for fetching public OHLCV data.
    Always uses HyperLiquid mainnet — testnet has no historical data.
    """
    exchange = ccxt.hyperliquid(
        {
            "enableRateLimit": True,
        }
    )
    # Do NOT use testnet for market data — no historical data there
    return exchange


def _build_binance_exchange() -> ccxt.binanceusdm:
    """Binance USDT-M futures — used as fallback for historical backtest data."""
    return ccxt.binanceusdm({"enableRateLimit": True})


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    start: str,
    end: Optional[str] = None,
    testnet: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from HyperLiquid with automatic pagination.

    Args:
        symbol:    e.g. "BTC/USDC:USDC"
        timeframe: e.g. "1h", "4h", "1d"
        start:     ISO date string, e.g. "2022-06-01"
        end:       ISO date string (exclusive). Defaults to now.
        testnet:   Ignored — always uses mainnet for historical data.

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
    limit = 5000  # HyperLiquid allows larger pages

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
        logger.warning("No data returned for %s %s — retrying via Binance", symbol, timeframe)
        return _fetch_ohlcv_binance(symbol, timeframe, start_ts, end_ts)

    df = pd.DataFrame(all_candles, columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # If HL data starts later than requested, backfill the gap from Binance
    first_ts_ms = int(df.index[0].timestamp() * 1000)
    if first_ts_ms > start_ts + 86_400_000:  # gap > 1 day
        logger.info(
            "HL data starts at %s — backfilling gap from Binance (%s → %s)",
            df.index[0], start, df.index[0].strftime("%Y-%m-%d"),
        )
        gap_df = _fetch_ohlcv_binance(symbol, timeframe, start_ts, first_ts_ms)
        if not gap_df.empty:
            df = pd.concat([gap_df, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()

    logger.info("Fetched %d candles for %s %s", len(df), symbol, timeframe)
    return df


def _fetch_ohlcv_binance(
    hl_symbol: str, timeframe: str, start_ts: int, end_ts: int
) -> pd.DataFrame:
    """Fetch OHLCV from Binance USDT-M futures as fallback for historical backtests.

    Maps HyperLiquid symbol (e.g. BTC/USDC:USDC) to Binance equivalent (BTC/USDT:USDT).
    Price action is essentially identical for BTC perps across venues.
    """
    # Map HL symbol to Binance equivalent
    binance_symbol = hl_symbol.replace("USDC", "USDT")
    exchange = _build_binance_exchange()

    logger.info("Fetching %s %s from Binance (start=%d)", binance_symbol, timeframe, start_ts)

    all_candles: list[list] = []
    since = start_ts
    limit = 1500

    while since < end_ts:
        try:
            candles = exchange.fetch_ohlcv(binance_symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as exc:
            logger.error("Binance fetch error: %s", exc)
            break

        if not candles:
            break

        candles = [c for c in candles if c[0] < end_ts]
        if not candles:
            break

        all_candles.extend(candles)
        last_ts = candles[-1][0]
        if last_ts <= since:
            break
        since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        logger.error("No data from Binance either for %s %s", binance_symbol, timeframe)
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = pd.DataFrame(all_candles, columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    logger.info("Binance fallback: fetched %d candles for %s %s", len(df), binance_symbol, timeframe)
    return df
