"""
Live / dry-run trading bot with regime-based orchestration.

Architecture:
  - Each tick, the regime classifier determines BULL / NEUTRAL / BEAR
  - BULL  -> bull strategy signals, enter long positions
  - BEAR  -> bear strategy signals, enter short positions
  - NEUTRAL -> no new positions (existing ones hit SL/TP naturally)

Only one strategy is active at a time.
"""

import time
import traceback
from datetime import datetime, timezone

import config
import signals.trend_following as bear_strategy
import signals.trend_bull as bull_strategy
from regime.composite import classify_composite, CompositeResult
from regime.classifier import Regime
from live.exchange import HyperLiquidExchange
from live.order_manager import OrderManager
from live.position_tracker import Position, PositionTracker
from risk.manager import RiskManager
from utils.logger import get_logger, log_trade

logger = get_logger(__name__)


class TradingBot:
    def __init__(
        self,
        symbols: list[str],
        strategies: list[str],
        mode: str = "dryrun",
        testnet: bool = config.HL_TESTNET,
    ) -> None:
        self.symbols = symbols
        self.strategies = strategies
        self.mode = mode
        self.testnet = testnet
        self.dryrun = mode == "dryrun"

        self._exchange = HyperLiquidExchange(testnet=testnet)
        self._order_mgr = OrderManager(self._exchange)
        self._position_tracker = PositionTracker(self._exchange)
        self._risk_mgr = RiskManager(
            initial_equity=self._get_account_equity()
        )

        self._current_regime = Regime.NEUTRAL
        self._regime_check_interval = 6  # check regime every N ticks
        self._tick_count = 0

        logger.info(
            "Bot started | mode=%s testnet=%s symbols=%s strategies=%s",
            mode, testnet, symbols, strategies,
        )

    def run(self) -> None:
        logger.info("Entering main loop (poll interval: %ds)", config.POLL_INTERVAL_SECONDS)
        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as exc:
                logger.error("Unexpected error in tick: %s\n%s", exc, traceback.format_exc())
            time.sleep(config.POLL_INTERVAL_SECONDS)

    def _tick(self) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self._tick_count += 1

        self._position_tracker.sync()
        equity = self._get_account_equity()
        self._risk_mgr.update_equity(equity)

        if self._risk_mgr.halted:
            logger.critical("CIRCUIT BREAKER ACTIVE")
            return

        # Check regime periodically
        if self._tick_count % self._regime_check_interval == 1:
            try:
                result = classify_composite(self.symbols[0], testnet=self.testnet)
                self._current_regime = result.regime
                logger.info(
                    "Regime: %s (score=%+d = tech %+d + sent %+d) at %s",
                    result.regime.value, result.score,
                    result.technical_score, result.sentiment_score, now,
                )
            except Exception as exc:
                logger.error("Regime classification failed: %s", exc)

        regime = self._current_regime

        for symbol in self.symbols:
            if regime == Regime.BEAR:
                self._process_signal(symbol, "bear")
            elif regime == Regime.BULL:
                self._process_signal(symbol, "bull")
            else:
                logger.debug("NEUTRAL — no trades for %s", symbol)

    def _process_signal(self, symbol: str, direction: str) -> None:
        try:
            if direction == "bear":
                signal_data = bear_strategy.get_latest_signal(symbol, self.testnet)
                side = "short"
            else:
                signal_data = bull_strategy.get_latest_signal(symbol, self.testnet)
                side = "long"
        except Exception as exc:
            logger.error("%s signal failed for %s: %s", direction, symbol, exc)
            return

        sig = signal_data.get("signal", 0)

        # EXIT
        if sig == -1 and self._position_tracker.has_position(symbol):
            pos = self._position_tracker.get_position(symbol)
            if self.dryrun:
                logger.info("[DRYRUN] EXIT %s %s qty=%.4f", side, symbol, pos.quantity)
            else:
                result = self._order_mgr.close_position(symbol, pos.quantity, reason="signal_exit")
                if result:
                    self._position_tracker.remove_position(symbol)
                    self._risk_mgr.on_position_closed(pnl=0)
            log_trade({**signal_data, "action": "exit", "side": side, "dryrun": self.dryrun})
            return

        # ENTRY
        if sig != 1 or self._position_tracker.has_position(symbol):
            return

        entry_price = signal_data.get("entry_price")
        stop_loss = signal_data.get("stop_loss")
        take_profit = signal_data.get("take_profit")
        if not all([entry_price, stop_loss, take_profit]):
            return

        strategy_name = signal_data.get("strategy", direction)
        trade_params = self._risk_mgr.calculate_trade(
            symbol=symbol, entry_price=entry_price, stop_loss=stop_loss,
            take_profit=take_profit, strategy=strategy_name,
        )
        if trade_params is None:
            return

        if self.dryrun:
            logger.info(
                "[DRYRUN] ENTER %s %s | qty=%.4f entry=%.4f SL=%.4f TP=%.4f",
                side, symbol, trade_params.quantity, entry_price, stop_loss, take_profit,
            )
        else:
            if side == "short":
                result = self._order_mgr.enter_short(
                    symbol=symbol, quantity=trade_params.quantity,
                    entry_price=entry_price, stop_loss=stop_loss,
                    take_profit=take_profit, strategy=strategy_name,
                )
            else:
                result = self._order_mgr.enter_long(
                    symbol=symbol, quantity=trade_params.quantity,
                    entry_price=entry_price, stop_loss=stop_loss,
                    take_profit=take_profit, strategy=strategy_name,
                )
            if result:
                self._position_tracker.add_position(Position(
                    symbol=symbol, side=side, quantity=trade_params.quantity,
                    entry_price=entry_price, stop_loss=stop_loss,
                    take_profit=take_profit, strategy=strategy_name,
                ))
                self._risk_mgr.on_position_opened()

        log_trade({**signal_data, "action": "entry", "side": side,
                   "dryrun": self.dryrun, "quantity": trade_params.quantity})

    def _get_account_equity(self) -> float:
        if self.dryrun:
            risk_mgr = getattr(self, "_risk_mgr", None)
            return risk_mgr.equity if risk_mgr is not None else config.BACKTEST_INITIAL_CAPITAL
        try:
            return self._exchange.get_balance()
        except Exception as exc:
            logger.warning("Could not fetch account equity: %s", exc)
            risk_mgr = getattr(self, "_risk_mgr", None)
            return risk_mgr.equity if risk_mgr is not None else config.BACKTEST_INITIAL_CAPITAL
