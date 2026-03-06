"""
ccxt HyperLiquid exchange wrapper.

Provides a thin, retry-safe interface to HyperLiquid perpetual futures.
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


class HyperLiquidExchange:
    """Thin wrapper around ccxt.hyperliquid with retry logic."""

    def __init__(self, testnet: bool = config.HL_TESTNET) -> None:
        self._exchange = ccxt.hyperliquid(
            {
                "walletAddress": config.HL_WALLET_ADDRESS,
                "privateKey": config.HL_PRIVATE_KEY,
                "enableRateLimit": True,
                "timeout": 15000,  # 15s timeout on all API calls
            }
        )
        if testnet:
            self._exchange.set_sandbox_mode(True)
            logger.info("Exchange initialised in TESTNET mode (HyperLiquid testnet)")
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
        self, symbol: str, timeframe: str, since: int, limit: int = 5000
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

    def get_balance(self) -> float:
        """Return account equity (margin + unrealized PnL + free cash) via HL API.

        Uses `clearinghouseState` for perp equity first (accountValue), then
        falls back to `spotClearinghouseState` for unified/spot-only accounts.
        This avoids returning only the *free* balance when margin is in use.
        """
        import requests as _req

        addr = config.HL_WALLET_ADDRESS
        base_url = "https://api.hyperliquid-testnet.xyz/info" if config.HL_TESTNET else "https://api.hyperliquid.xyz/info"

        # Primary: perp clearinghouse → accountValue includes margin + unrealised PnL
        try:
            resp = _req.post(
                base_url,
                json={"type": "clearinghouseState", "user": addr},
                timeout=10,
            )
            data = resp.json()
            account_value = float(data.get("marginSummary", {}).get("accountValue", 0.0))
            if account_value > 0:
                logger.info("Balance (perp equity): %.2f USDC", account_value)
                return account_value
        except Exception as exc:
            logger.warning("clearinghouseState fetch failed: %s", exc)

        # Fallback: spot clearinghouse (unified accounts with no open perp positions)
        try:
            resp = _req.post(
                base_url,
                json={"type": "spotClearinghouseState", "user": addr},
                timeout=10,
            )
            data = resp.json()
            available = dict(data.get("tokenToAvailableAfterMaintenance", []))
            spot_usdc = float(available.get(0, 0.0))
            if spot_usdc > 0:
                logger.info("Balance (unified spot): %.2f USDC", spot_usdc)
            return spot_usdc
        except Exception as exc:
            logger.warning("Balance fetch via native API failed: %s", exc)
            return 0.0

    def transfer_spot_to_perp(self, amount: float) -> bool:
        """Transfer USDC from spot wallet to perp margin (needed before trading perps)."""
        try:
            self._exchange.transfer("USDC", amount, "spot", "swap")
            logger.info("Transferred %.2f USDC from spot to perp", amount)
            return True
        except Exception as exc:
            if "unified account" in str(exc).lower() or "action disabled" in str(exc).lower():
                logger.info("Unified account — spot/perp are merged, no transfer needed.")
            else:
                logger.error("Spot→Perp transfer failed: %s", exc)
            return False

    def fetch_positions(self) -> list[dict]:
        return self._retry(self._exchange.fetch_positions)

    def set_leverage(self, symbol: str, leverage: int = config.LEVERAGE) -> None:
        try:
            self._exchange.set_margin_mode(
                "cross", symbol, {"leverage": leverage}
            )
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
        """Place a market order. Uses ticker price as price hint for HyperLiquid."""
        params = params or {}
        # HyperLiquid requires a price hint for market orders
        ticker = self._retry(self._exchange.fetch_ticker, symbol)
        price = ticker.get("last") or ticker.get("close", 0)
        order = self._retry(
            self._exchange.create_market_order,
            symbol, side, amount, price, params,
        )
        logger.info(
            "Market order placed: %s %s %.4f (id=%s)",
            side, symbol, amount, order.get("id"),
        )
        return order

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        return self._retry(self._exchange.cancel_order, order_id, symbol)

    def fetch_open_orders(self, symbol: str) -> list[dict]:
        return self._retry(self._exchange.fetch_open_orders, symbol)

    def cancel_all_orders(self, symbol: str) -> None:
        """Cancel all open orders for a symbol."""
        orders = self.fetch_open_orders(symbol)
        for order in orders:
            try:
                self.cancel_order(order["id"], symbol)
            except Exception as exc:
                logger.warning("Failed to cancel order %s: %s", order["id"], exc)

    def fetch_order(self, order_id: str, symbol: str) -> dict:
        return self._retry(self._exchange.fetch_order, order_id, symbol)

    def place_stop_loss_order(
        self, symbol: str, side: str, amount: float, stop_price: float
    ) -> dict:
        """Place a stop-loss trigger order on HyperLiquid.

        Uses `stopLossPrice` (not `triggerPrice`) so CCXT sends tpsl='sl'
        to HL, which triggers when price moves *against* the position
        (i.e. above the SL for a short, below for a long).
        """
        params = {
            "stopLossPrice": stop_price,
            "reduceOnly": True,
        }
        order = self._retry(
            self._exchange.create_order,
            symbol, "market", side, amount, stop_price, params,
        )
        logger.info(
            "Stop-loss order placed: %s %s %.4f @ stop=%.4f (id=%s)",
            side, symbol, amount, stop_price, order.get("id"),
        )
        return order

    def place_take_profit_order(
        self, symbol: str, side: str, amount: float, tp_price: float
    ) -> dict:
        """Place a take-profit trigger order on HyperLiquid.

        Uses `takeProfitPrice` so CCXT sends tpsl='tp' to HL, which
        triggers when price moves *in favour* of the position
        (i.e. below the TP for a short, above for a long).
        """
        params = {
            "takeProfitPrice": tp_price,
            "reduceOnly": True,
        }
        order = self._retry(
            self._exchange.create_order,
            symbol, "market", side, amount, tp_price, params,
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
