"""
Live / dry-run trading bot main loop.

Polls every POLL_INTERVAL_SECONDS for new signal conditions.
In dry-run mode, logs all signals without placing any orders.
"""

import time
import traceback
from datetime import datetime, timezone

import config
import signals.trend_following as tf_strategy
import signals.breakdown as bd_strategy
from live.exchange import BinanceFuturesExchange
from live.order_manager import OrderManager
from live.position_tracker import Position, PositionTracker
from risk.manager import RiskManager
from utils.logger import get_logger, log_trade

logger = get_logger(__name__)


class TradingBot:
    """
    Main polling loop for live/dryrun modes.

    Args:
        symbols:    List of symbols to trade, e.g. ["BTC/USDT:USDT"]
        strategies: List of "trend" and/or "breakdown"
        mode:       "live" or "dryrun"
        testnet:    Use Binance testnet
    """

    def __init__(
        self,
        symbols: list[str],
        strategies: list[str],
        mode: str = "dryrun",
        testnet: bool = config.BINANCE_TESTNET,
    ) -> None:
        self.symbols = symbols
        self.strategies = strategies
        self.mode = mode
        self.testnet = testnet
        self.dryrun = mode == "dryrun"

        self._exchange = BinanceFuturesExchange(testnet=testnet)
        self._order_mgr = OrderManager(self._exchange)
        self._position_tracker = PositionTracker(self._exchange)
        self._risk_mgr = RiskManager(
            initial_equity=self._get_account_equity()
        )

        logger.info(
            "Bot started | mode=%s testnet=%s symbols=%s strategies=%s",
            mode, testnet, symbols, strategies,
        )

    def run(self) -> None:
        """Start the main polling loop (runs until interrupted)."""
        logger.info("Entering main loop (poll interval: %ds)", config.POLL_INTERVAL_SECONDS)
        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("Bot stopped by user (KeyboardInterrupt)")
                break
            except Exception as exc:
                logger.error("Unexpected error in tick: %s\n%s", exc, traceback.format_exc())

            time.sleep(config.POLL_INTERVAL_SECONDS)

    # -------------------------------------------------------------------------
    # Core tick
    # -------------------------------------------------------------------------

    def _tick(self) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        logger.debug("Tick at %s", now)

        # Refresh position state from exchange
        self._position_tracker.sync()

        # Update equity from exchange
        equity = self._get_account_equity()
        self._risk_mgr.update_equity(equity)

        if self._risk_mgr.halted:
            logger.critical("CIRCUIT BREAKER ACTIVE — no new trades")
            return

        for symbol in self.symbols:
            for strategy_name in self.strategies:
                self._process_signal(symbol, strategy_name)

    def _process_signal(self, symbol: str, strategy_name: str) -> None:
        """Fetch latest signal and act on it."""
        try:
            if strategy_name == "trend":
                signal_data = tf_strategy.get_latest_signal(symbol, self.testnet)
            elif strategy_name == "breakdown":
                signal_data = bd_strategy.get_latest_signal(symbol, self.testnet)
            else:
                logger.warning("Unknown strategy: %s", strategy_name)
                return
        except Exception as exc:
            logger.error("Signal fetch failed for %s/%s: %s", symbol, strategy_name, exc)
            return

        sig = signal_data.get("signal", 0)

        # ----- EXIT signal -----
        if sig == -1 and self._position_tracker.has_position(symbol):
            pos = self._position_tracker.get_position(symbol)
            if self.dryrun:
                logger.info(
                    "[DRYRUN] EXIT %s (strategy=%s) qty=%.4f",
                    symbol, strategy_name, pos.quantity,
                )
            else:
                result = self._order_mgr.close_position(
                    symbol, pos.quantity, reason="signal_exit"
                )
                if result:
                    self._position_tracker.remove_position(symbol)
                    self._risk_mgr.on_position_closed(pnl=0)  # PnL updated via sync

            log_trade({**signal_data, "action": "exit", "dryrun": self.dryrun})
            return

        # ----- ENTRY signal -----
        if sig != 1:
            return

        if self._position_tracker.has_position(symbol):
            logger.debug("[%s/%s] Already in position — skipping entry", symbol, strategy_name)
            return

        entry_price = signal_data.get("entry_price")
        stop_loss = signal_data.get("stop_loss")
        take_profit = signal_data.get("take_profit")

        if not all([entry_price, stop_loss, take_profit]):
            logger.warning("[%s/%s] Incomplete signal data — skipping", symbol, strategy_name)
            return

        trade_params = self._risk_mgr.calculate_trade(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=strategy_name,
        )

        if trade_params is None:
            return

        if self.dryrun:
            logger.info(
                "[DRYRUN] ENTER short %s | qty=%.4f entry=%.4f SL=%.4f TP=%.4f",
                symbol, trade_params.quantity, entry_price, stop_loss, take_profit,
            )
        else:
            result = self._order_mgr.enter_short(
                symbol=symbol,
                quantity=trade_params.quantity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=strategy_name,
            )
            if result:
                self._position_tracker.add_position(
                    Position(
                        symbol=symbol,
                        side="short",
                        quantity=trade_params.quantity,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        strategy=strategy_name,
                    )
                )
                self._risk_mgr.on_position_opened()

        log_trade({**signal_data, "action": "entry", "dryrun": self.dryrun,
                   "quantity": trade_params.quantity})

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_account_equity(self) -> float:
        if self.dryrun:
            return self._risk_mgr.equity if hasattr(self._risk_mgr, "equity") else config.BACKTEST_INITIAL_CAPITAL
        try:
            return self._exchange.get_usdt_balance()
        except Exception as exc:
            logger.warning("Could not fetch account equity: %s", exc)
            return self._risk_mgr.equity
