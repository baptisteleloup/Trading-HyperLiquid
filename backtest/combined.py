"""
Combined regime-aware backtest.

Runs each strategy INDEPENDENTLY with its own equity pool, then merges
the results. This ensures strategies don't interfere (shared circuit
breaker, shared max_positions, etc.).

  - BEAR regime  → trend_following (short) signals
  - BULL regime  → trend_bull (long) signals
  - NEUTRAL      → no new trades
"""

import pandas as pd
import numpy as np
import os

import config
from data.cache import load_ohlcv
from regime.classifier import classify_historical, Regime
from backtest.engine import run_backtest, BacktestResult
from backtest.metrics import print_metrics, compute_metrics
from backtest.plotter import plot_equity_and_drawdown
from utils.logger import get_logger

import signals.trend_following as bear_strategy
import signals.trend_bull as bull_strategy

logger = get_logger(__name__)


def run_combined_backtest(
    symbol: str,
    start: str,
    end: str,
    capital: float,
    testnet: bool = False,
) -> BacktestResult:
    """
    Run each strategy independently, then merge equity curves additively.
    Each strategy gets the full capital and its own circuit breaker.
    Combined PnL = bear PnL + bull PnL.
    """
    logger.info("Combined backtest: %s  %s → %s  capital=%.0f", symbol, start, end, capital)

    # ── 1. Generate signals ─────────────────────────────────────────────
    logger.info("Generating BEAR (short) signals...")
    df_bear = bear_strategy.generate_signals(symbol, start, end, testnet=testnet)
    df_bear.attrs["strategy"] = "trend_following"
    df_bear.attrs["side"] = "short"

    logger.info("Generating BULL (long) signals...")
    df_bull = bull_strategy.generate_signals(symbol, start, end, testnet=testnet)
    df_bull.attrs["strategy"] = "trend_bull"
    df_bull.attrs["side"] = "long"

    # ── 2. Run each strategy independently ──────────────────────────────
    logger.info("Running BEAR backtest independently...")
    bear_result = run_backtest(signal_df=df_bear, initial_capital=capital,
                               circuit_breaker_pct=1.0)
    bear_pnl = bear_result.metrics.get("total_return_pct", 0)
    bear_n = len(bear_result.trades)
    logger.info("BEAR result: %.1f%% return, %d trades", bear_pnl, bear_n)

    logger.info("Running BULL backtest independently...")
    bull_result = run_backtest(signal_df=df_bull, initial_capital=capital,
                               circuit_breaker_pct=1.0)
    bull_pnl = bull_result.metrics.get("total_return_pct", 0)
    bull_n = len(bull_result.trades)
    logger.info("BULL result: %.1f%% return, %d trades", bull_pnl, bull_n)

    # ── 3. Merge equity curves ──────────────────────────────────────────
    # Combined equity = capital + (bear_equity - capital) + (bull_equity - capital)
    # i.e. combined PnL = bear PnL + bull PnL
    bear_eq = bear_result.equity_curve
    bull_eq = bull_result.equity_curve

    # Align to common timeline
    all_idx = bear_eq.index.union(bull_eq.index).sort_values()
    bear_aligned = bear_eq.reindex(all_idx).ffill().fillna(capital)
    bull_aligned = bull_eq.reindex(all_idx).ffill().fillna(capital)

    combined_equity = capital + (bear_aligned - capital) + (bull_aligned - capital)
    combined_equity.name = "equity"

    # ── 4. Merge trades ─────────────────────────────────────────────────
    all_trades = pd.concat([bear_result.trades, bull_result.trades], ignore_index=True)
    if not all_trades.empty and "entry_time" in all_trades.columns:
        all_trades = all_trades.sort_values("entry_time").reset_index(drop=True)

    # ── 5. Compute combined metrics ─────────────────────────────────────
    metrics = compute_metrics(combined_equity, all_trades, capital)

    logger.info(
        "COMBINED: %.1f%% return | Sharpe=%.2f | MaxDD=%.1f%% | %d trades",
        metrics.get("total_return_pct", 0),
        metrics.get("sharpe_ratio", 0),
        metrics.get("max_drawdown_pct", 0),
        len(all_trades),
    )

    # Build a dummy signal_df for compatibility
    dummy_signal = pd.DataFrame(index=all_idx)

    return BacktestResult(
        equity_curve=combined_equity,
        trades=all_trades,
        metrics=metrics,
        signal_df=dummy_signal,
    )


def print_combined_summary(result: BacktestResult, symbol: str, start: str, end: str) -> None:
    print(f"\n{'='*55}")
    print(f"  COMBINED (regime-aware) | {symbol}")
    print(f"  Period : {start}  →  {end}")
    print_metrics(result.metrics)

    if not result.trades.empty:
        bear_trades = result.trades[result.trades.get("side", pd.Series()) == "short"] if "side" in result.trades else pd.DataFrame()
        bull_trades = result.trades[result.trades.get("side", pd.Series()) == "long"]  if "side" in result.trades else pd.DataFrame()
        print(f"\n  Trade breakdown:")
        print(f"    Short (bear regime) : {len(bear_trades)} trades")
        print(f"    Long  (bull regime) : {len(bull_trades)} trades")
