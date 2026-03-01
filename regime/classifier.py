"""
Composite market regime classifier.

Scores market conditions on a scale of -5 (extreme bear) to +5 (extreme bull)
using 5 independent factors across daily and weekly timeframes.

Decision thresholds (configurable):
    Score <= BEAR_THRESHOLD  → BEAR  (bear_trader active, bull_trader idle)
    Score >= BULL_THRESHOLD  → BULL  (bull_trader active, bear_trader idle)
    Otherwise                → NEUTRAL (both idle)

The neutral zone is intentionally wide to avoid regime misclassification.
Analogy: Neyman-Pearson — we control Type I error (wrong regime) at the cost
of fewer trades (higher Type II / lower power).

Factors:
    1. Price vs EMA-200 daily    — long-term trend
    2. EMA-50d vs EMA-200d       — structural trend (golden/death cross state)
    3. Price vs EMA-50 weekly    — medium-term trend
    4. RSI weekly                — momentum direction
    5. MACD histogram daily      — short-term momentum
"""

import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta, timezone

import config
from data.cache import load_ohlcv
from utils.logger import get_logger

logger = get_logger(__name__)


class Regime(Enum):
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"


@dataclass
class RegimeResult:
    regime: Regime
    score: int          # -5 to +5
    factors: dict       # individual factor votes
    timestamp: str


def classify(
    symbol: str,
    testnet: bool = True,
) -> RegimeResult:
    """
    Classify the current market regime for a given symbol.

    Fetches daily and weekly data, computes 5 factors, and returns
    the aggregated regime with individual factor breakdown.
    """
    now = datetime.now(timezone.utc)
    end = now.strftime("%Y-%m-%dT%H:%M:%S")

    # --- Daily data (300 days for EMA-200 warmup) ---
    daily_start = (now - timedelta(days=300)).strftime("%Y-%m-%d")
    df_daily = load_ohlcv(symbol, "1d", daily_start, end, testnet=testnet)

    # --- Weekly data (120 weeks ~2.3 years for EMA-50w warmup) ---
    weekly_start = (now - timedelta(weeks=120)).strftime("%Y-%m-%d")
    df_weekly = load_ohlcv(symbol, "1w", weekly_start, end, testnet=testnet)

    factors = {}
    score = 0

    # ── Factor 1: Price vs EMA-200 daily ──────────────────────────────
    ema_200d = ta.ema(df_daily["close"], length=200)
    if ema_200d is not None and len(ema_200d.dropna()) > 0:
        last_close = df_daily["close"].iloc[-1]
        last_ema200 = ema_200d.dropna().iloc[-1]
        if last_close < last_ema200:
            factors["price_vs_ema200d"] = -1
        else:
            factors["price_vs_ema200d"] = +1
    else:
        factors["price_vs_ema200d"] = 0

    # ── Factor 2: EMA-50d vs EMA-200d (structural cross) ─────────────
    ema_50d = ta.ema(df_daily["close"], length=50)
    if ema_50d is not None and ema_200d is not None:
        e50 = ema_50d.dropna()
        e200 = ema_200d.dropna()
        if len(e50) > 0 and len(e200) > 0:
            if e50.iloc[-1] < e200.iloc[-1]:
                factors["ema50d_vs_ema200d"] = -1   # death cross state
            else:
                factors["ema50d_vs_ema200d"] = +1   # golden cross state
        else:
            factors["ema50d_vs_ema200d"] = 0
    else:
        factors["ema50d_vs_ema200d"] = 0

    # ── Factor 3: Price vs EMA-50 weekly ──────────────────────────────
    ema_50w = ta.ema(df_weekly["close"], length=50)
    if ema_50w is not None and len(ema_50w.dropna()) > 0:
        last_close_w = df_weekly["close"].iloc[-1]
        last_ema50w = ema_50w.dropna().iloc[-1]
        if last_close_w < last_ema50w:
            factors["price_vs_ema50w"] = -1
        else:
            factors["price_vs_ema50w"] = +1
    else:
        factors["price_vs_ema50w"] = 0

    # ── Factor 4: RSI weekly ──────────────────────────────────────────
    rsi_w = ta.rsi(df_weekly["close"], length=config.REGIME_RSI_WEEKLY_PERIOD)
    if rsi_w is not None and len(rsi_w.dropna()) > 0:
        last_rsi = rsi_w.dropna().iloc[-1]
        if last_rsi < config.REGIME_RSI_BEAR_MAX:
            factors["rsi_weekly"] = -1
        elif last_rsi > config.REGIME_RSI_BULL_MIN:
            factors["rsi_weekly"] = +1
        else:
            factors["rsi_weekly"] = 0   # dead zone
    else:
        factors["rsi_weekly"] = 0

    # ── Factor 5: MACD histogram daily ────────────────────────────────
    macd_df = ta.macd(df_daily["close"], fast=12, slow=26, signal=9)
    if macd_df is not None:
        hist = macd_df.iloc[:, 1]  # histogram column
        if len(hist.dropna()) > 0:
            last_hist = hist.dropna().iloc[-1]
            if last_hist < 0:
                factors["macd_hist_daily"] = -1
            else:
                factors["macd_hist_daily"] = +1
        else:
            factors["macd_hist_daily"] = 0
    else:
        factors["macd_hist_daily"] = 0

    # ── Aggregate score ───────────────────────────────────────────────
    score = sum(factors.values())

    if score <= config.REGIME_SCORE_BEAR_THRESHOLD:
        regime = Regime.BEAR
    elif score >= config.REGIME_SCORE_BULL_THRESHOLD:
        regime = Regime.BULL
    else:
        regime = Regime.NEUTRAL

    logger.info(
        "Regime: %s (score=%d) | %s",
        regime.value, score,
        " | ".join(f"{k}={v:+d}" for k, v in factors.items()),
    )

    return RegimeResult(
        regime=regime,
        score=score,
        factors=factors,
        timestamp=end,
    )


def classify_historical(
    symbol: str,
    df_daily: pd.DataFrame,
    df_weekly: pd.DataFrame,
) -> pd.Series:
    """
    Classify regime for each daily bar in df_daily.

    Returns a Series of Regime values aligned to df_daily.index.
    Used by the backtest engine for historical regime classification.
    """
    # Compute daily indicators
    ema_200d = ta.ema(df_daily["close"], length=200)
    ema_50d = ta.ema(df_daily["close"], length=50)
    macd_df = ta.macd(df_daily["close"], fast=12, slow=26, signal=9)
    macd_hist = macd_df.iloc[:, 1] if macd_df is not None else pd.Series(0, index=df_daily.index)

    # Compute weekly indicators, then forward-fill to daily
    ema_50w = ta.ema(df_weekly["close"], length=50)
    rsi_w = ta.rsi(df_weekly["close"], length=config.REGIME_RSI_WEEKLY_PERIOD)

    # Align weekly to daily via forward-fill
    if ema_50w is not None:
        ema_50w_daily = ema_50w.reindex(
            ema_50w.index.union(df_daily.index)
        ).ffill().reindex(df_daily.index)
    else:
        ema_50w_daily = pd.Series(float("nan"), index=df_daily.index)

    if rsi_w is not None:
        rsi_w_daily = rsi_w.reindex(
            rsi_w.index.union(df_daily.index)
        ).ffill().reindex(df_daily.index)
    else:
        rsi_w_daily = pd.Series(50.0, index=df_daily.index)

    # Compute score for each daily bar
    scores = pd.Series(0, index=df_daily.index, dtype=int)

    # Factor 1: Price vs EMA-200 daily
    if ema_200d is not None:
        scores += ((df_daily["close"] < ema_200d).astype(int) * -2 + 1).fillna(0).astype(int)
    # Factor 2: EMA-50d vs EMA-200d
    if ema_50d is not None and ema_200d is not None:
        scores += ((ema_50d < ema_200d).astype(int) * -2 + 1).fillna(0).astype(int)
    # Factor 3: Price vs EMA-50 weekly
    scores += ((df_daily["close"] < ema_50w_daily).astype(int) * -2 + 1).fillna(0).astype(int)
    # Factor 4: RSI weekly
    rsi_vote = pd.Series(0, index=df_daily.index)
    rsi_vote[rsi_w_daily < config.REGIME_RSI_BEAR_MAX] = -1
    rsi_vote[rsi_w_daily > config.REGIME_RSI_BULL_MIN] = +1
    scores += rsi_vote
    # Factor 5: MACD histogram daily
    scores += ((macd_hist < 0).astype(int) * -2 + 1).fillna(0).astype(int)

    # Map scores to regimes
    regimes = pd.Series(Regime.NEUTRAL, index=df_daily.index)
    regimes[scores <= config.REGIME_SCORE_BEAR_THRESHOLD] = Regime.BEAR
    regimes[scores >= config.REGIME_SCORE_BULL_THRESHOLD] = Regime.BULL

    return regimes
