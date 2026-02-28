"""
ccxt binanceusdm exchange wrapper.

Provides a thin, retry-safe interface to Binance USDT-M Futures.
All order placement goes through this module.
"""

import time
import ccxt
from typing import Optional

import config
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2.0


class BinanceFuturesExchange:
    """Thin wrapper around ccxt.binanceusdm with retry logic."""

    def __init__(self, testnet: bool = config.BINANCE_TESTNET) -> None:
        self._exchange = ccxt.binanceusdm(
            {
                "apiKey": config.BINANCE_API_KEY,
                "secret": config.BINANCE_API_SECRET,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )
        if testnet:
            self._exchange.set_sandbox_mode(True)
            logger.info("Exchange initialised in TESTNET mode")
        else:
            logger.warning("Exchange initialised in LIVE mode — real money!")

        self._markets: dict = {}

    # -------------------------------------------------------------------------
    # Market data
    # -------------------------------------------------------------------------

    def load_markets(self) -> dict:
        self._markets = self._retry(self._exchange.load_markets)
        return self._markets

    def fetch_ticker(self, symbol: str) -> dict:
        return self._retry(self._exchange.fetch_ticker, symbol)

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int, limit: int = 1000
    ) -> list:
        return self._retry(
            self._exchange.fetch_ohlcv, symbol, timeframe, since, limit
        )

    def get_best_bid_ask(self, symbol: str) -> tuple[float, float]:
        """Return (best_bid, best_ask)."""
        order_book = self._retry(self._exchange.fetch_order_book, symbol, 5)
        best_bid = order_book["bids"][0][0] if order_book["bids"] else 0.0
        best_ask = order_book["asks"][0][0] if order_book["asks"] else 0.0
        return best_bid, best_ask

    # -------------------------------------------------------------------------
    # Account
    # -------------------------------------------------------------------------

    def fetch_balance(self) -> dict:
        return self._retry(self._exchange.fetch_balance)

    def get_usdt_balance(self) -> float:
        balance = self.fetch_balance()
        return float(balance.get("USDT", {}).get("free", 0.0))

    def fetch_positions(self) -> list[dict]:
        return self._retry(self._exchange.fetch_positions)

    def set_leverage(self, symbol: str, leverage: int = config.LEVERAGE) -> None:
        try:
            self._exchange.set_leverage(leverage, symbol)
            logger.debug("Leverage set to %d× for %s", leverage, symbol)
        except ccxt.ExchangeError as exc:
            logger.warning("Could not set leverage: %s", exc)

    # -------------------------------------------------------------------------
    # Orders
    # -------------------------------------------------------------------------

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        params: Optional[dict] = None,
    ) -> dict:
        """Place a limit order. side: 'buy' or 'sell'."""
        params = params or {}
        order = self._retry(
            self._exchange.create_limit_order,
            symbol, side, amount, price, params,
        )
        logger.info(
            "Limit order placed: %s %s %.4f @ %.4f (id=%s)",
            side, symbol, amount, price, order.get("id"),
        )
        return order

    def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: Optional[dict] = None,
    ) -> dict:
        params = params or {}
        order = self._retry(
            self._exchange.create_market_order,
            symbol, side, amount, params,
        )
        logger.info(
            "Market order placed: %s %s %.4f (id=%s)",
            side, symbol, amount, order.get("id"),
        )
        return order

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        return self._retry(self._exchange.cancel_order, order_id, symbol)

    def fetch_order(self, order_id: str, symbol: str) -> dict:
        return self._retry(self._exchange.fetch_order, order_id, symbol)

    def place_stop_loss_order(
        self, symbol: str, side: str, amount: float, stop_price: float
    ) -> dict:
        """Place a stop-market order for SL."""
        params = {
            "stopPrice": stop_price,
            "type": "STOP_MARKET",
            "reduceOnly": True,
        }
        order = self._retry(
            self._exchange.create_order,
            symbol, "STOP_MARKET", side, amount, stop_price, params,
        )
        logger.info(
            "Stop-loss order placed: %s %s %.4f @ stop=%.4f (id=%s)",
            side, symbol, amount, stop_price, order.get("id"),
        )
        return order

    def place_take_profit_order(
        self, symbol: str, side: str, amount: float, tp_price: float
    ) -> dict:
        """Place a take-profit-market order."""
        params = {
            "stopPrice": tp_price,
            "type": "TAKE_PROFIT_MARKET",
            "reduceOnly": True,
        }
        order = self._retry(
            self._exchange.create_order,
            symbol, "TAKE_PROFIT_MARKET", side, amount, tp_price, params,
        )
        logger.info(
            "TP order placed: %s %s %.4f @ tp=%.4f (id=%s)",
            side, symbol, amount, tp_price, order.get("id"),
        )
        return order

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _retry(self, fn, *args, **kwargs):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except ccxt.NetworkError as exc:
                if attempt == MAX_RETRIES:
                    raise
                logger.warning("Network error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
                time.sleep(RETRY_DELAY * attempt)
            except ccxt.RateLimitExceeded as exc:
                logger.warning("Rate limit hit, sleeping 10s: %s", exc)
                time.sleep(10)
            except ccxt.ExchangeError:
                raise
