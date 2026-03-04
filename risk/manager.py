"""
Risk management module.

Responsibilities:
  - Position sizing (risk-based)
  - Stop-loss / take-profit calculation
  - Circuit breaker (halt on session drawdown)
  - Max open position enforcement
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeParams:
    """All parameters needed to place a trade."""
    symbol: str
    side: str                  # "short" (only short trades in this system)
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float            # Base asset quantity
    leverage: int
    risk_amount: float         # USDT at risk
    strategy: str = ""


@dataclass
class RiskManager:
    """
    Stateful risk manager — tracks equity and open positions.

    Call update_equity() after each trade or PnL mark to keep state current.
    """
    initial_equity: float = config.BACKTEST_INITIAL_CAPITAL
    equity: float = field(init=False)
    session_high: float = field(init=False)
    open_positions: int = 0
    halted: bool = False

    def __post_init__(self) -> None:
        self.equity = self.initial_equity
        self.session_high = self.initial_equity

    # -------------------------------------------------------------------------
    # Core sizing
    # -------------------------------------------------------------------------

    def calculate_trade(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        strategy: str = "",
    ) -> Optional[TradeParams]:
        """
        Compute position size and validate risk rules.

        Returns TradeParams or None if the trade should be rejected.
        """
        if self.halted:
            logger.warning("Trading halted by circuit breaker — rejecting trade")
            return None

        if self.open_positions >= config.MAX_OPEN_POSITIONS:
            logger.warning(
                "Max open positions (%d) reached — rejecting trade",
                config.MAX_OPEN_POSITIONS,
            )
            return None

        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            logger.error("Invalid SL: entry=%s stop=%s", entry_price, stop_loss)
            return None

        risk_amount = self.equity * config.RISK_PER_TRADE
        raw_quantity = risk_amount / risk_per_unit

        # Apply leverage — we need less notional collateral
        # Notional = quantity × entry_price; margin = notional / leverage
        notional = raw_quantity * entry_price
        margin_required = notional / config.LEVERAGE

        if margin_required > self.equity:
            logger.warning(
                "Insufficient equity (%.2f) for margin (%.2f) — capping size",
                self.equity, margin_required,
            )
            raw_quantity = (self.equity * config.LEVERAGE) / entry_price

        # Minimum order check
        notional_value = raw_quantity * entry_price
        if notional_value < config.MIN_ORDER_SIZE:
            logger.warning(
                "Order too small (%.4f USDC) — minimum is %.2f USDC",
                notional_value, config.MIN_ORDER_SIZE,
            )
            return None

        # Round down to avoid over-ordering (exchange typically needs precision)
        quantity = _floor_to_precision(raw_quantity, decimals=3)

        params = TradeParams(
            symbol=symbol,
            side="short",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            leverage=config.LEVERAGE,
            risk_amount=risk_amount,
            strategy=strategy,
        )

        logger.info(
            "Trade approved: %s short %.4f @ %.4f | SL=%.4f TP=%.4f risk=%.2f USDT",
            symbol, quantity, entry_price, stop_loss, take_profit, risk_amount,
        )
        return params

    # -------------------------------------------------------------------------
    # Position tracking
    # -------------------------------------------------------------------------

    def on_position_opened(self) -> None:
        self.open_positions += 1
        logger.debug("Position opened. Open: %d", self.open_positions)

    def on_position_closed(self, pnl: float) -> None:
        self.open_positions = max(0, self.open_positions - 1)
        self.update_equity(self.equity + pnl)
        logger.info(
            "Position closed. PnL=%.2f USDT. New equity=%.2f. Open=%d",
            pnl, self.equity, self.open_positions,
        )

    def update_equity(self, new_equity: float) -> None:
        self.equity = new_equity
        if new_equity > self.session_high:
            self.session_high = new_equity
        self._check_circuit_breaker()

    # -------------------------------------------------------------------------
    # Circuit breaker
    # -------------------------------------------------------------------------

    def _check_circuit_breaker(self) -> None:
        drawdown = (self.session_high - self.equity) / self.session_high
        if drawdown >= config.CIRCUIT_BREAKER_DRAWDOWN:
            if not self.halted:
                logger.critical(
                    "CIRCUIT BREAKER TRIGGERED: drawdown=%.1f%% "
                    "(equity=%.2f, session_high=%.2f)",
                    drawdown * 100, self.equity, self.session_high,
                )
            self.halted = True

    def reset_circuit_breaker(self) -> None:
        """Manually reset — requires human intervention."""
        self.halted = False
        self.session_high = self.equity
        logger.warning("Circuit breaker manually reset.")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @property
    def drawdown_pct(self) -> float:
        if self.session_high == 0:
            return 0.0
        return (self.session_high - self.equity) / self.session_high

    def status(self) -> dict:
        return {
            "equity": self.equity,
            "session_high": self.session_high,
            "drawdown_pct": round(self.drawdown_pct * 100, 2),
            "open_positions": self.open_positions,
            "halted": self.halted,
        }


def _floor_to_precision(value: float, decimals: int = 3) -> float:
    factor = 10 ** decimals
    return math.floor(value * factor) / factor
