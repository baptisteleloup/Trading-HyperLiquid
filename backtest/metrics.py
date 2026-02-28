"""
Performance metric calculations for backtest results.
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    initial_capital: float,
    risk_free_rate: float = 0.04,  # 4% annual
) -> dict:
    """
    Compute a comprehensive set of performance metrics.

    Args:
        equity_curve:    Time-indexed equity Series.
        trades:          DataFrame of closed trades with 'pnl' column.
        initial_capital: Starting capital in USDT.
        risk_free_rate:  Annual risk-free rate for Sharpe/Sortino.

    Returns:
        Dictionary of metric names → values.
    """
    metrics: dict = {}

    if equity_curve.empty or len(equity_curve) < 2:
        return {"error": "insufficient_data"}

    # -------------------------------------------------------------------------
    # Return metrics
    # -------------------------------------------------------------------------
    final_equity = equity_curve.iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    metrics["initial_capital"] = round(initial_capital, 2)
    metrics["final_equity"] = round(final_equity, 2)
    metrics["total_return_pct"] = round(total_return * 100, 2)

    # Annualized return (CAGR)
    start = equity_curve.index[0]
    end = equity_curve.index[-1]
    years = max((end - start).days / 365.25, 1 / 365.25)
    cagr = (final_equity / initial_capital) ** (1 / years) - 1
    metrics["annualized_return_pct"] = round(cagr * 100, 2)

    # -------------------------------------------------------------------------
    # Drawdown
    # -------------------------------------------------------------------------
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    metrics["max_drawdown_pct"] = round(drawdown.min() * 100, 2)
    metrics["avg_drawdown_pct"] = round(drawdown[drawdown < 0].mean() * 100, 2)

    # -------------------------------------------------------------------------
    # Risk-adjusted returns
    # -------------------------------------------------------------------------
    returns = equity_curve.pct_change().dropna()

    # Determine frequency scaling factor
    if len(equity_curve) > 1:
        avg_delta = (end - start) / (len(equity_curve) - 1)
        hours_per_bar = avg_delta.total_seconds() / 3600
        bars_per_year = 8760 / max(hours_per_bar, 0.01)
    else:
        bars_per_year = 252

    rf_per_bar = (1 + risk_free_rate) ** (1 / bars_per_year) - 1

    excess = returns - rf_per_bar
    std = returns.std()
    metrics["sharpe_ratio"] = round(
        (excess.mean() / std * np.sqrt(bars_per_year)) if std > 0 else 0.0, 3
    )

    downside_returns = returns[returns < rf_per_bar]
    downside_std = downside_returns.std()
    metrics["sortino_ratio"] = round(
        (
            excess.mean() / downside_std * np.sqrt(bars_per_year)
            if downside_std > 0
            else 0.0
        ),
        3,
    )

    # -------------------------------------------------------------------------
    # Trade-level metrics
    # -------------------------------------------------------------------------
    metrics["total_trades"] = 0
    metrics["win_rate_pct"] = 0.0
    metrics["profit_factor"] = 0.0
    metrics["avg_rr"] = 0.0
    metrics["avg_win_usdt"] = 0.0
    metrics["avg_loss_usdt"] = 0.0

    if not trades.empty and "pnl" in trades.columns:
        pnls = trades["pnl"].dropna()
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        metrics["total_trades"] = len(pnls)
        metrics["win_rate_pct"] = round(len(wins) / len(pnls) * 100, 1)

        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        metrics["profit_factor"] = round(
            gross_profit / gross_loss if gross_loss > 0 else float("inf"), 3
        )
        metrics["avg_win_usdt"] = round(wins.mean() if len(wins) > 0 else 0.0, 2)
        metrics["avg_loss_usdt"] = round(losses.mean() if len(losses) > 0 else 0.0, 2)

        # Average R:R (win / |loss|)
        if len(losses) > 0 and losses.mean() != 0:
            metrics["avg_rr"] = round(
                abs(wins.mean() / losses.mean()) if len(wins) > 0 else 0.0, 2
            )

    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print the metrics dictionary to stdout."""
    print("\n" + "=" * 55)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("=" * 55)

    sections = {
        "Returns": [
            "initial_capital", "final_equity",
            "total_return_pct", "annualized_return_pct",
        ],
        "Risk": [
            "sharpe_ratio", "sortino_ratio",
            "max_drawdown_pct", "avg_drawdown_pct",
        ],
        "Trades": [
            "total_trades", "win_rate_pct",
            "profit_factor", "avg_rr",
            "avg_win_usdt", "avg_loss_usdt",
        ],
    }

    for section, keys in sections.items():
        print(f"\n  {section}:")
        for key in keys:
            if key in metrics:
                label = key.replace("_", " ").title()
                value = metrics[key]
                unit = " %" if "pct" in key else (" USDT" if "usdt" in key else "")
                if "ratio" in key or "factor" in key or "rr" in key:
                    print(f"    {label:<30} {value:.3f}")
                elif "usdt" in key.lower() or key in ("initial_capital", "final_equity"):
                    print(f"    {label:<30} ${value:,.2f}")
                else:
                    print(f"    {label:<30} {value}{unit}")

    print("=" * 55 + "\n")
