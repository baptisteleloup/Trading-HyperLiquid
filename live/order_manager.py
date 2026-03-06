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
from live.exchange import HyperLiquidExchange
from utils.logger import get_logger

logger = get_logger(__name__)


class OrderManager:
    def __init__(self, exchange: HyperLiquidExchange) -> None:
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

    def enter_long(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        strategy: str = "",
    ) -> Optional[dict]:
        """Enter a long position (mirror of enter_short)."""
        self._ex.set_leverage(symbol, config.LEVERAGE)
        _, best_ask = self._ex.get_best_bid_ask(symbol)
        limit_price = round(best_ask * (1 + config.LIMIT_ORDER_OFFSET_BPS), 4)
        logger.info(
            "[%s] Entering long: qty=%.4f limit=%.4f SL=%.4f TP=%.4f",
            symbol, quantity, limit_price, stop_loss, take_profit,
        )
        try:
            order = self._ex.place_limit_order(symbol, "buy", quantity, limit_price)
            order_id = order["id"]
        except Exception as exc:
            logger.error("Failed to place limit entry: %s", exc)
            return None
        filled_order = self._wait_for_fill(order_id, symbol)
        if filled_order is None:
            logger.info("[%s] Limit order unfilled, cancelling -> market", symbol)
            try:
                self._ex.cancel_order(order_id, symbol)
            except Exception as exc:
                logger.warning("Cancel failed: %s", exc)
            try:
                filled_order = self._ex.place_market_order(symbol, "buy", quantity)
            except Exception as exc:
                logger.error("Market fallback failed: %s", exc)
                return None
        try:
            self._ex.place_stop_loss_order(symbol, "sell", quantity, stop_loss)
        except Exception as exc:
            logger.error("SL order failed: %s", exc)
        try:
            self._ex.place_take_profit_order(symbol, "sell", quantity, take_profit)
        except Exception as exc:
            logger.error("TP order failed: %s", exc)
        logger.info(
            "[%s] Long entered. Fill price ~ %.4f", symbol,
            filled_order.get("average") or filled_order.get("price", 0),
        )
        return filled_order

    def close_position(
        self,
        symbol: str,
        quantity: float,
        side: str = "short",
        reason: str = "signal_exit",
    ) -> Optional[dict]:
        """
        Close a position with a market order + cancel remaining SL/TP orders.
        side: the position side ('short' or 'long').
        """
        close_side = "buy" if side == "short" else "sell"
        logger.info("[%s] Closing %s (%.4f) — reason: %s", symbol, side, quantity, reason)

        # Cancel SL/TP orders first to avoid them triggering during close
        try:
            self._ex.cancel_all_orders(symbol)
        except Exception as exc:
            logger.warning("Failed to cancel orders before close: %s", exc)

        try:
            order = self._ex.place_market_order(
                symbol, close_side, quantity, params={"reduceOnly": True}
            )
            return order
        except Exception as exc:
            logger.error("Failed to close position: %s", exc)
            return None

    def update_stop_loss(
        self,
        symbol: str,
        quantity: float,
        new_sl: float,
        sl_side: str,
    ) -> bool:
        """
        Cancel all open orders for the symbol and place a new SL only.
        Used when trailing stop activates (replaces both fixed SL and TP).
        sl_side: 'buy' for shorts, 'sell' for longs.
        """
        try:
            self._ex.cancel_all_orders(symbol)
        except Exception as exc:
            logger.warning("Failed to cancel orders for trailing update on %s: %s", symbol, exc)

        try:
            self._ex.place_stop_loss_order(symbol, sl_side, quantity, new_sl)
            return True
        except Exception as exc:
            logger.error("Failed to place updated trailing SL for %s: %s", symbol, exc)
            return False

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
