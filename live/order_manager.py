"""
Order lifecycle management for live trading.

Responsibilities:
  - Place limit entry orders
  - Monitor fill status; fall back to market if not filled in time
  - Place SL and TP orders after entry fill
  - Cancel stale orders
"""

import time
from typing import Optional

import config
from live.exchange import BinanceFuturesExchange
from utils.logger import get_logger

logger = get_logger(__name__)


class OrderManager:
    def __init__(self, exchange: BinanceFuturesExchange) -> None:
        self._ex = exchange

    def enter_short(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        strategy: str = "",
    ) -> Optional[dict]:
        """
        Enter a short position.

        1. Place limit sell order 2bps inside best bid.
        2. Wait up to ORDER_TIMEOUT_SECONDS for fill.
        3. If not filled, cancel and retry as market order.
        4. Once filled, place OCO-like SL and TP orders.

        Returns the filled order dict or None on failure.
        """
        self._ex.set_leverage(symbol, config.LEVERAGE)

        # --- Try limit entry ---
        best_bid, _ = self._ex.get_best_bid_ask(symbol)
        limit_price = round(best_bid * (1 - config.LIMIT_ORDER_OFFSET_BPS), 4)

        logger.info(
            "[%s] Entering short: qty=%.4f limit=%.4f SL=%.4f TP=%.4f",
            symbol, quantity, limit_price, stop_loss, take_profit,
        )

        try:
            order = self._ex.place_limit_order(symbol, "sell", quantity, limit_price)
            order_id = order["id"]
        except Exception as exc:
            logger.error("Failed to place limit entry: %s", exc)
            return None

        filled_order = self._wait_for_fill(order_id, symbol)

        if filled_order is None:
            # Cancel and fall back to market
            logger.info("[%s] Limit order unfilled, cancelling → market", symbol)
            try:
                self._ex.cancel_order(order_id, symbol)
            except Exception as exc:
                logger.warning("Cancel failed: %s", exc)

            try:
                filled_order = self._ex.place_market_order(symbol, "sell", quantity)
            except Exception as exc:
                logger.error("Market fallback failed: %s", exc)
                return None

        # --- Place protective orders ---
        self._place_sl_tp(symbol, quantity, stop_loss, take_profit)

        logger.info(
            "[%s] Short entered. Fill price ≈ %.4f", symbol,
            filled_order.get("average") or filled_order.get("price", 0),
        )
        return filled_order

    def close_position(
        self,
        symbol: str,
        quantity: float,
        reason: str = "signal_exit",
    ) -> Optional[dict]:
        """
        Close a short position with a market buy.
        Also attempts to cancel any open SL/TP orders for the symbol.
        """
        logger.info("[%s] Closing short (%.4f) — reason: %s", symbol, quantity, reason)
        try:
            order = self._ex.place_market_order(
                symbol, "buy", quantity, params={"reduceOnly": True}
            )
            return order
        except Exception as exc:
            logger.error("Failed to close position: %s", exc)
            return None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _wait_for_fill(
        self, order_id: str, symbol: str
    ) -> Optional[dict]:
        """Poll order status until filled or timeout."""
        deadline = time.time() + config.ORDER_TIMEOUT_SECONDS
        while time.time() < deadline:
            try:
                order = self._ex.fetch_order(order_id, symbol)
                status = order.get("status", "")
                if status == "closed":
                    return order
                if status in ("canceled", "rejected", "expired"):
                    return None
            except Exception as exc:
                logger.warning("Error fetching order %s: %s", order_id, exc)
            time.sleep(1)
        return None  # timed out

    def _place_sl_tp(
        self,
        symbol: str,
        quantity: float,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        try:
            self._ex.place_stop_loss_order(symbol, "buy", quantity, stop_loss)
        except Exception as exc:
            logger.error("SL order failed: %s", exc)

        try:
            self._ex.place_take_profit_order(symbol, "buy", quantity, take_profit)
        except Exception as exc:
            logger.error("TP order failed: %s", exc)
