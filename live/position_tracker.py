"""
Real-time position state tracker.

Queries the exchange for current open positions and reconciles them
with the bot's internal state to detect fills, SL/TP hits, etc.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from live.exchange import BinanceFuturesExchange
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    symbol: str
    side: str              # "short"
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy: str
    entry_time: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    unrealized_pnl: float = 0.0
    is_open: bool = True


class PositionTracker:
    """
    Maintains the current portfolio state by syncing with the exchange.

    Internal positions dict: symbol → Position
    """

    def __init__(self, exchange: BinanceFuturesExchange) -> None:
        self._ex = exchange
        self._positions: dict[str, Position] = {}

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def sync(self) -> None:
        """Refresh position state from exchange."""
        try:
            raw_positions = self._ex.fetch_positions()
        except Exception as exc:
            logger.error("Failed to fetch positions: %s", exc)
            return

        active_symbols = set()
        for pos in raw_positions:
            notional = abs(float(pos.get("notional", 0) or 0))
            if notional < 1.0:
                continue  # skip dust

            symbol = pos["symbol"]
            active_symbols.add(symbol)
            unrealized = float(pos.get("unrealizedPnl", 0) or 0)

            if symbol in self._positions:
                self._positions[symbol].unrealized_pnl = unrealized
            else:
                # Position opened outside bot (e.g., manually) — track it
                side_raw = pos.get("side", "short").lower()
                qty = abs(float(pos.get("contracts", 0) or 0))
                entry_px = float(pos.get("entryPrice", 0) or 0)
                self._positions[symbol] = Position(
                    symbol=symbol,
                    side=side_raw,
                    quantity=qty,
                    entry_price=entry_px,
                    stop_loss=0.0,
                    take_profit=0.0,
                    strategy="external",
                    unrealized_pnl=unrealized,
                )
                logger.warning(
                    "External position detected: %s %s %.4f @ %.4f",
                    side_raw, symbol, qty, entry_px,
                )

        # Mark positions closed if they no longer appear on exchange
        for symbol in list(self._positions.keys()):
            if symbol not in active_symbols:
                pos = self._positions.pop(symbol)
                logger.info(
                    "Position closed (detected via sync): %s. Was open since %s",
                    symbol, pos.entry_time,
                )

    def add_position(self, position: Position) -> None:
        self._positions[position.symbol] = position
        logger.info(
            "Tracking new position: %s short %.4f @ %.4f",
            position.symbol, position.quantity, position.entry_price,
        )

    def remove_position(self, symbol: str) -> Optional[Position]:
        return self._positions.pop(symbol, None)

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def open_count(self) -> int:
        return len(self._positions)

    def all_positions(self) -> list[Position]:
        return list(self._positions.values())

    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())

    def summary(self) -> list[dict]:
        return [
            {
                "symbol": p.symbol,
                "side": p.side,
                "quantity": p.quantity,
                "entry_price": p.entry_price,
                "unrealized_pnl": round(p.unrealized_pnl, 2),
                "strategy": p.strategy,
            }
            for p in self._positions.values()
        ]
