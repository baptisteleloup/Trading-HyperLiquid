"""
Backtest visualization: equity curve + drawdown chart.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

RESULTS_DIR = Path("backtest/results")


def plot_equity_and_drawdown(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    symbol: str,
    strategy: str,
    save: bool = True,
    show: bool = False,
) -> str:
    """
    Plot equity curve (top) and drawdown (bottom).

    Returns the saved file path or empty string if save=False.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(
        f"Backtest: {strategy.replace('_', ' ').title()} | {symbol}",
        fontsize=14, fontweight="bold",
    )

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------
    ax1.plot(equity_curve.index, equity_curve.values, color="#2196F3", linewidth=1.5)
    ax1.fill_between(equity_curve.index, equity_curve.values,
                     equity_curve.values.min(), alpha=0.1, color="#2196F3")
    ax1.axhline(equity_curve.iloc[0], color="gray", linestyle="--", linewidth=0.8,
                label="Initial capital")
    ax1.set_ylabel("Equity (USDT)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Mark trade entries and exits
    if not trades.empty and "entry_time" in trades.columns:
        _mark_trades(ax1, trades, equity_curve)

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100

    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color="#F44336", alpha=0.6, label="Drawdown")
    ax2.plot(drawdown.index, drawdown.values, color="#F44336", linewidth=0.8)
    ax2.axhline(-15, color="darkred", linestyle="--", linewidth=0.8,
                label="Circuit breaker (-15%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()

    filepath = ""
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        filename = f"{strategy}_{safe_symbol}_equity.png"
        filepath = str(RESULTS_DIR / filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Chart saved: {filepath}")

    if show:
        plt.show()

    plt.close(fig)
    return filepath


def _mark_trades(ax: plt.Axes, trades: pd.DataFrame, equity: pd.Series) -> None:
    """Overlay trade entry/exit markers on the equity chart."""
    try:
        for _, trade in trades.iterrows():
            entry_t = trade.get("entry_time")
            exit_t = trade.get("exit_time")
            pnl = trade.get("pnl", 0)
            color = "green" if pnl > 0 else "red"

            if entry_t and entry_t in equity.index:
                ax.axvline(entry_t, color=color, alpha=0.3, linewidth=0.7)
    except Exception:
        pass  # Don't let plotting errors crash the backtest
