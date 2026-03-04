"""
Composite regime classifier — combines technical + sentiment scoring.

Final score range: -8 to +8
  - Technical: -5 to +5 (from classifier.py)
  - Sentiment: -3 to +3 (from sentiment.py)

Thresholds (same as before, applied to final score):
  - BEAR: ≤ -3
  - BULL: ≥ +3
  - NEUTRAL: between -2 and +2
"""

from dataclasses import dataclass
from typing import Optional

import config
from regime.classifier import classify as classify_technical, Regime, RegimeResult
from regime.sentiment import score_sentiment, SentimentResult
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CompositeResult:
    regime: Regime
    score: int  # final composite score
    technical_score: int
    sentiment_score: int
    technical_factors: dict
    sentiment_factors: dict
    sentiment_raw: dict


def classify_composite(
    symbol: str,
    testnet: bool = False,
) -> CompositeResult:
    """
    Run technical + sentiment analysis and combine into a single regime decision.
    
    Score = technical (-5 to +5) + sentiment (-3 to +3)
    Thresholds: BEAR ≤ -3, BULL ≥ +3, else NEUTRAL
    """
    # Technical analysis
    tech = classify_technical(symbol, testnet=testnet)
    
    # Sentiment analysis (use USDC pair for sentiment regardless of trading pair)
    base = symbol.split("/")[0]
    sentiment_symbol = f"{base}/USDC:USDC"
    sent = score_sentiment(sentiment_symbol, testnet=testnet)
    
    # Combine
    final_score = tech.score + sent.score
    
    # Apply thresholds
    bear_threshold = getattr(config, "REGIME_SCORE_BEAR_THRESHOLD", -3)
    bull_threshold = getattr(config, "REGIME_SCORE_BULL_THRESHOLD", 3)
    
    if final_score <= bear_threshold:
        regime = Regime.BEAR
    elif final_score >= bull_threshold:
        regime = Regime.BULL
    else:
        regime = Regime.NEUTRAL
    
    logger.info(
        "Composite regime: %s (score=%+d = tech %+d + sent %+d) | "
        "tech=%s | sent=%s",
        regime.value, final_score, tech.score, sent.score,
        tech.factors, sent.factors,
    )
    
    return CompositeResult(
        regime=regime,
        score=final_score,
        technical_score=tech.score,
        sentiment_score=sent.score,
        technical_factors=tech.factors,
        sentiment_factors=sent.factors,
        sentiment_raw=sent.raw,
    )
