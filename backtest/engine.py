"""
Vectorized backtesting engine — supports LONG and SHORT positions.

Simulates:
  - 0.04% taker commission on entry and exit
  - 5bps slippage (price moves against us on fill)
  - 0.01% funding rate every 8 hours (charged while holding)

The engine reads the `side` attribute from signal_df.attrs or the 'side' column
to determine position direction (long or short). Default: short (backward compat).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

import config
from utils.logger import get_logger

logger = get_logger(__name__)

FUNDING_PERIOD_HOURS = 8


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict
    signal_df: pd.DataFrame


def run_backtest(
    signal_df: pd.DataFrame,
    initial_capital: float = config.BACKTEST_INITIAL_CAPITAL,
    taker_fee: float = config.BACKTEST_TAKER_FEE,
    slippage: float = config.BACKTEST_SLIPPAGE,
    funding_rate: float = config.BACKTEST_FUNDING_RATE,
    leverage: int = config.LEVERAGE,
    risk_pct: float = config.RISK_PER_TRADE,
    max_positions: int = config.MAX_OPEN_POSITIONS,
) -> BacktestResult:
    """
    Simulate trading the signals in signal_df.

    The position side is determined by signal_df.attrs["side"] (default "short").
    """
    default_side = signal_df.attrs.get("side", "short")
    equity = initial_capital
    session_high = initial_capital
    halted = False

    open_trades: list[dict] = []
    closed_trades: list[dict] = []
    equity_history: list[tuple] = []

    if len(signal_df) > 1:
        delta_hours = (
            signal_df.index[1] - signal_df.index[0]
        ).total_seconds() / 3600
    else:
        delta_hours = 1.0

    for ts, row in signal_df.iterrows():
        # ── Mark-to-market and check SL/TP ────────────────────────────
        still_open = []
        for trade in open_trades:
            pnl, closed_trade = _check_exit(row, trade, taker_fee, slippage)
            if closed_trade is not None:
                equity += pnl
                closed_trade["exit_time"] = ts
                closed_trade["pnl"] = pnl
                closed_trades.append(closed_trade)
            else:
                if delta_hours > 0:
                    funding_events = delta_hours / FUNDING_PERIOD_HOURS
                    funding_cost = (
                        trade["quantity"]
                        * row["close"]
                        * funding_rate
                        * funding_events
                    )
                    trade["total_funding"] = trade.get("total_funding", 0) + funding_cost
                    equity -= funding_cost
                still_open.append(trade)
        open_trades = still_open

        # ── Circuit breaker ───────────────────────────────────────────
        if equity > session_high:
            session_high = equity
        drawdown = (session_high - equity) / session_high if session_high > 0 else 0
        if drawdown >= config.CIRCUIT_BREAKER_DRAWDOWN:
            halted = True
        if halted:
            equity_history.append((ts, equity))
            continue

        # ── New entry ─────────────────────────────────────────────────
        if (
            row["signal"] == 1
            and len(open_trades) < max_positions
            and not np.isnan(row.get("entry_price", float("nan")))
        ):
            # Determine side from row or default
            side = row.get("side", default_side)
            if not side or (isinstance(side, float) and np.isnan(side)):
                side = default_side

            if side == "long":
                entry_px = row["entry_price"] * (1 + slippage)  # slippage against long
            else:
                entry_px = row["entry_price"] * (1 - slippage)  # slippage against short

            stop_px = row["stop_loss"]
            tp_px = row["take_profit"]

            risk_per_unit = abs(stop_px - entry_px)
            if risk_per_unit <= 0:
                equity_history.append((ts, equity))
                continue

            risk_amount = equity * risk_pct
            quantity = risk_amount / risk_per_unit
            notional = quantity * entry_px
            if notional < config.MIN_ORDER_USDT:
                equity_history.append((ts, equity))
                continue

            commission = notional * taker_fee
            equity -= commission

            trade = {
                "entry_time": ts,
                "entry_price": entry_px,
                "stop_loss": stop_px,
                "take_profit": tp_px,
                "quantity": quantity,
                "risk_amount": risk_amount,
                "commission_entry": commission,
                "total_funding": 0.0,
                "strategy": signal_df.attrs.get("strategy", "unknown"),
                "side": side,
            }
            open_trades.append(trade)

        equity_history.append((ts, equity))

    # ── Force-close remaining positions ───────────────────────────────
    last_row = signal_df.iloc[-1]
    last_ts = signal_df.index[-1]
    for trade in open_trades:
        side = trade.get("side", default_side)
        if side == "long":
            exit_px = last_row["close"] * (1 - slippage)
        else:
            exit_px = last_row["close"] * (1 + slippage)
        pnl, closed = _force_close(trade, exit_px, taker_fee, last_ts)
        equity += pnl
        closed_trades.append(closed)
    equity_history.append((last_ts, equity))

    # ── Build outputs ─────────────────────────────────────────────────
    equity_curve = pd.Series(
        {ts: eq for ts, eq in equity_history}, name="equity"
    )
    equity_curve.index = pd.DatetimeIndex(equity_curve.index)

    trades_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()

    from backtest.metrics import compute_metrics
    metrics = compute_metrics(equity_curve, trades_df, initial_capital)

    logger.info(
        "Backtest complete: %.1f%% return | Sharpe=%.2f | MaxDD=%.1f%% | %d trades",
        metrics.get("total_return_pct", 0),
        metrics.get("sharpe_ratio", 0),
        metrics.get("max_drawdown_pct", 0),
        len(trades_df),
    )

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades_df,
        metrics=metrics,
        signal_df=signal_df,
    )


def _check_exit(
    row: pd.Series,
    trade: dict,
    taker_fee: float,
    slippage: float,
) -> tuple[float, Optional[dict]]:
    """
    Check whether a trade's SL or TP was hit in this bar.
    Handles both long and short positions.
    """
    sl = trade["stop_loss"]
    tp = trade["take_profit"]
    entry = trade["entry_price"]
    qty = trade["quantity"]
    side = trade.get("side", "short")

    exit_price = None
    exit_reason = None

    if side == "short":
        # SL hit: price went UP through SL
        if row["high"] >= sl:
            exit_price = sl * (1 + slippage)
            exit_reason = "stop_loss"
        # TP hit: price went DOWN through TP
        elif row["low"] <= tp:
            exit_price = tp * (1 - slippage)
            exit_reason = "take_profit"
        # Signal exit
        elif row["signal"] == -1:
            exit_price = row["close"] * (1 + slippage)
            exit_reason = "signal_exit"
    else:  # long
        # SL hit: price went DOWN through SL
        if row["low"] <= sl:
            exit_price = sl * (1 - slippage)
            exit_reason = "stop_loss"
        # TP hit: price went UP through TP
        elif row["high"] >= tp:
            exit_price = tp * (1 + slippage)
            exit_reason = "take_profit"
        # Signal exit
        elif row["signal"] == -1:
            exit_price = row["close"] * (1 - slippage)
            exit_reason = "signal_exit"

    if exit_price is None:
        return 0.0, None

    # PnL depends on side
    if side == "short":
        gross_pnl = (entry - exit_price) * qty
    else:
        gross_pnl = (exit_price - entry) * qty

    commission = exit_price * qty * taker_fee
    net_pnl = gross_pnl - commission - trade.get("total_funding", 0)

    closed = {**trade, "exit_price": exit_price, "exit_reason": exit_reason}
    return net_pnl, closed


def _force_close(
    trade: dict,
    exit_price: float,
    taker_fee: float,
    ts: pd.Timestamp,
) -> tuple[float, dict]:
    qty = trade["quantity"]
    side = trade.get("side", "short")

    if side == "short":
        gross_pnl = (trade["entry_price"] - exit_price) * qty
    else:
        gross_pnl = (exit_price - trade["entry_price"]) * qty

    commission = exit_price * qty * taker_fee
    net_pnl = gross_pnl - commission - trade.get("total_funding", 0)
    closed = {
        **trade,
        "exit_price": exit_price,
        "exit_reason": "forced_close",
        "exit_time": ts,
        "pnl": net_pnl,
    }
    return net_pnl, closed
