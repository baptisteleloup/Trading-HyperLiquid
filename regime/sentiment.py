"""
Sentiment scoring module — 3 sources, score from -3 to +3.

Sources:
  1. Fear & Greed Index (Alternative.me) → -1 to +1
  2. BTC/ETH ratio momentum → -1 to +1
  3. Funding rate (HyperLiquid perps) → -1 to +1
"""

import requests
import ccxt
import numpy as np
from dataclasses import dataclass
from typing import Optional

import config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentResult:
    score: int  # -3 to +3
    factors: dict
    raw: dict


def _fetch_fear_greed() -> int:
    """
    Fear & Greed Index → vote -1, 0, +1.
    
    Interpretation is TREND-FOLLOWING (not contrarian):
      - Extreme Fear → confirms bear → -1
      - Fear → -1 (below 35)
      - Neutral → 0
      - Greed → +1 (above 65)
      - Extreme Greed → confirms bull → +1
    """
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        value = int(data["data"][0]["value"])
        
        if value < 35:
            vote = -1
        elif value > 65:
            vote = +1
        else:
            vote = 0
        
        logger.info("Fear & Greed: %d (%s) → vote=%+d", value, data["data"][0]["value_classification"], vote)
        return vote, value
    except Exception as exc:
        logger.warning("Fear & Greed fetch failed: %s", exc)
        return 0, None


def _fetch_btc_eth_ratio(testnet: bool = False) -> int:
    """
    BTC/ETH relative strength over last 7 days.
    
    If BTC outperforms ETH → risk-off → -1 (bear)
    If ETH outperforms BTC → risk-on → +1 (bull)
    Neutral zone: < 2% difference → 0
    """
    try:
        exchange = ccxt.hyperliquid()  # public, no auth needed

        # Fetch last 7 daily candles for both
        btc_ohlcv = exchange.fetch_ohlcv("BTC/USDC:USDC", "1d", limit=8)
        eth_ohlcv = exchange.fetch_ohlcv("ETH/USDC:USDC", "1d", limit=8)
        
        if len(btc_ohlcv) < 8 or len(eth_ohlcv) < 8:
            return 0, None
        
        # 7-day returns
        btc_ret = (btc_ohlcv[-1][4] - btc_ohlcv[0][4]) / btc_ohlcv[0][4]  # close prices
        eth_ret = (eth_ohlcv[-1][4] - eth_ohlcv[0][4]) / eth_ohlcv[0][4]
        
        diff = eth_ret - btc_ret  # positive = ETH outperforms
        
        if diff > 0.02:  # ETH outperforms by >2%
            vote = +1
        elif diff < -0.02:  # BTC outperforms by >2%
            vote = -1
        else:
            vote = 0
        
        logger.info(
            "BTC/ETH ratio: BTC 7d=%.2f%% ETH 7d=%.2f%% diff=%.2f%% → vote=%+d",
            btc_ret * 100, eth_ret * 100, diff * 100, vote,
        )
        return vote, {"btc_7d": round(btc_ret * 100, 2), "eth_7d": round(eth_ret * 100, 2), "diff": round(diff * 100, 2)}
    except Exception as exc:
        logger.warning("BTC/ETH ratio fetch failed: %s", exc)
        return 0, None


def _fetch_funding_rate(symbol: str = "BTC/USDC:USDC") -> int:
    """
    Current funding rate from HyperLiquid perps.

    Funding > +0.03% → market overleveraged long → -1 (bear signal)
    Funding < -0.03% → market overleveraged short → +1 (bull signal)
    Between → 0
    """
    try:
        exchange = ccxt.hyperliquid()
        
        # Fetch funding rate
        funding = exchange.fetch_funding_rate(symbol)
        rate = funding.get("fundingRate", 0) or 0
        
        if rate > 0.0003:  # > 0.03%
            vote = -1  # overleveraged longs → bearish signal
        elif rate < -0.0003:  # < -0.03%
            vote = +1  # overleveraged shorts → bullish signal
        else:
            vote = 0
        
        logger.info("Funding rate %s: %.4f%% → vote=%+d", symbol, rate * 100, vote)
        return vote, round(rate * 100, 4)
    except Exception as exc:
        logger.warning("Funding rate fetch failed: %s", exc)
        return 0, None


def score_sentiment(symbol: str = "BTC/USDC:USDC", testnet: bool = False) -> SentimentResult:
    """
    Compute composite sentiment score from all 3 sources.
    Returns SentimentResult with score in [-3, +3].
    """
    fng_vote, fng_raw = _fetch_fear_greed()
    ratio_vote, ratio_raw = _fetch_btc_eth_ratio(testnet)
    funding_vote, funding_raw = _fetch_funding_rate(symbol)
    
    total = fng_vote + ratio_vote + funding_vote
    total = max(-3, min(3, total))  # clamp
    
    factors = {
        "fear_greed": fng_vote,
        "btc_eth_ratio": ratio_vote,
        "funding_rate": funding_vote,
    }
    
    raw = {
        "fear_greed_value": fng_raw,
        "btc_eth_data": ratio_raw,
        "funding_rate_pct": funding_raw,
    }
    
    logger.info(
        "Sentiment score: %+d | fng=%+d ratio=%+d funding=%+d",
        total, fng_vote, ratio_vote, funding_vote,
    )
    
    return SentimentResult(score=total, factors=factors, raw=raw)
