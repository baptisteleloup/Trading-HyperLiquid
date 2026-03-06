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

import numpy as np

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
from utils.notifier import notify_trade_entry, notify_trade_exit

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
            initial_equity=max(self._get_account_equity(), 1.0)  # avoid zero division
        )

        self._current_regime = Regime.NEUTRAL
        self._regime_check_interval = 6  # check regime every N ticks
        self._tick_count = 0

        # Auto-transfer spot USDC to perp if needed
        self._ensure_perp_funded()

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

        # Snapshot positions before sync to detect SL/TP closures
        positions_before = set(
            p.symbol for p in self._position_tracker.all_positions()
        )
        self._position_tracker.sync()
        positions_after = set(
            p.symbol for p in self._position_tracker.all_positions()
        )

        # Notify risk manager of positions closed by exchange (SL/TP)
        for closed_symbol in positions_before - positions_after:
            self._risk_mgr.on_position_closed(pnl=0)
            logger.info("Exchange closed %s (SL/TP hit) — risk manager updated", closed_symbol)

        self._update_trailing_stops()
        equity = self._get_account_equity()
        if equity <= 0:
            logger.warning("Balance = 0, funds not yet deposited on HyperLiquid — waiting...")
            return
        self._risk_mgr.update_equity(equity)

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
            # Always check exits for open positions, regardless of regime
            # (mirrors backtest: each strategy's exits fire independently)
            self._check_exit_signals(symbol)

            # New entries only in the active regime
            if regime == Regime.BEAR:
                self._process_signal(symbol, "bear")
            elif regime == Regime.BULL:
                self._process_signal(symbol, "bull")
            else:
                logger.debug("NEUTRAL — no trades for %s", symbol)

    def _check_exit_signals(self, symbol: str) -> None:
        """Check exit signals for open positions, regardless of current regime.

        Mirrors backtest behavior: each strategy's exits (RSI, golden cross,
        death cross) fire independently of the regime classifier.
        """
        pos = self._position_tracker.get_position(symbol)
        if pos is None:
            return

        # Determine which strategy to check based on position side
        try:
            if pos.side == "short":
                signal_data = bear_strategy.get_latest_signal(symbol, self.testnet)
            else:
                signal_data = bull_strategy.get_latest_signal(symbol, self.testnet)
        except Exception as exc:
            logger.error("Exit signal check failed for %s: %s", symbol, exc)
            return

        if signal_data.get("signal") != -1:
            return

        side = pos.side
        if self.dryrun:
            logger.info("[DRYRUN] EXIT %s %s qty=%.4f (signal)", side, symbol, pos.quantity)
        else:
            result = self._order_mgr.close_position(symbol, pos.quantity, side=side, reason="signal_exit")
            if result:
                self._position_tracker.remove_position(symbol)
                self._risk_mgr.on_position_closed(pnl=0)
                notify_trade_exit(
                    symbol=symbol, side=side,
                    price=signal_data.get("entry_price", 0),
                    reason="signal_exit", dryrun=self.dryrun,
                )
        log_trade({**signal_data, "action": "exit", "side": side, "dryrun": self.dryrun})

    def _process_signal(self, symbol: str, direction: str) -> None:
        """Process entry signals only (exits handled by _check_exit_signals)."""
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

        # ENTRY only
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
                atr_value = signal_data.get("atr", 0.0)
                if atr_value is None or (isinstance(atr_value, float) and np.isnan(atr_value)):
                    atr_value = 0.0
                self._position_tracker.add_position(Position(
                    symbol=symbol, side=side, quantity=trade_params.quantity,
                    entry_price=entry_price, stop_loss=stop_loss,
                    take_profit=take_profit, strategy=strategy_name,
                    atr_at_entry=atr_value,
                    hwm=entry_price,
                ))
                self._risk_mgr.on_position_opened()
                notify_trade_entry(
                    symbol=symbol, side=side, price=entry_price,
                    quantity=trade_params.quantity, stop_loss=stop_loss,
                    take_profit=take_profit,
                    regime=self._current_regime.name,
                    dryrun=self.dryrun,
                )

        log_trade({**signal_data, "action": "entry", "side": side,
                   "dryrun": self.dryrun, "quantity": trade_params.quantity})

    def _update_trailing_stops(self) -> None:
        """Update trailing stops for all open positions (mirrors backtest logic)."""
        if not config.USE_TRAILING_STOP:
            return

        for pos in self._position_tracker.all_positions():
            if pos.atr_at_entry <= 0:
                continue

            try:
                ticker = self._exchange.fetch_ticker(pos.symbol)
                current_price = ticker.get("last", 0)
                if current_price <= 0:
                    continue
            except Exception as exc:
                logger.warning("Trailing: failed to fetch ticker for %s: %s", pos.symbol, exc)
                continue

            atr = pos.atr_at_entry

            if pos.side == "long":
                activation_dist = config.TRAILING_ACTIVATION_ATR_LONG * atr
                trail_dist = config.TRAILING_ATR_MULTIPLIER_LONG * atr

                # Update high water mark
                if current_price > pos.hwm:
                    pos.hwm = current_price

                # Check activation
                if (pos.hwm - pos.entry_price) < activation_dist:
                    continue

                first_activation = not pos.trailing_activated
                pos.trailing_activated = True
                if first_activation:
                    logger.info(
                        "[%s] Trailing ACTIVATED (long) | hwm=%.2f entry=%.2f atr=%.2f",
                        pos.symbol, pos.hwm, pos.entry_price, atr,
                    )

                trailing_sl = pos.hwm - trail_dist

                # Only tighten (move SL up for long)
                if trailing_sl > pos.stop_loss:
                    old_sl = pos.stop_loss
                    pos.stop_loss = trailing_sl
                    if not self.dryrun:
                        self._order_mgr.update_stop_loss(
                            pos.symbol, pos.quantity, trailing_sl, "sell",
                        )
                    logger.info(
                        "[%s] Trailing SL tightened: %.2f -> %.2f (hwm=%.2f)",
                        pos.symbol, old_sl, trailing_sl, pos.hwm,
                    )

            else:  # short
                activation_dist = config.TRAILING_ACTIVATION_ATR_SHORT * atr
                trail_dist = config.TRAILING_ATR_MULTIPLIER_SHORT * atr

                # Update low water mark
                if current_price < pos.hwm:
                    pos.hwm = current_price

                # Check activation
                if (pos.entry_price - pos.hwm) < activation_dist:
                    continue

                first_activation = not pos.trailing_activated
                pos.trailing_activated = True
                if first_activation:
                    logger.info(
                        "[%s] Trailing ACTIVATED (short) | hwm=%.2f entry=%.2f atr=%.2f",
                        pos.symbol, pos.hwm, pos.entry_price, atr,
                    )

                trailing_sl = pos.hwm + trail_dist

                # Only tighten (move SL down for short)
                if trailing_sl < pos.stop_loss:
                    old_sl = pos.stop_loss
                    pos.stop_loss = trailing_sl
                    if not self.dryrun:
                        self._order_mgr.update_stop_loss(
                            pos.symbol, pos.quantity, trailing_sl, "buy",
                        )
                    logger.info(
                        "[%s] Trailing SL tightened: %.2f -> %.2f (hwm=%.2f)",
                        pos.symbol, old_sl, trailing_sl, pos.hwm,
                    )

    def _ensure_perp_funded(self) -> None:
        """Transfer all spot USDC to perp margin if perp balance is low.
        
        On HyperLiquid unified accounts, spot and perp are merged — no transfer needed.
        """
        if self.dryrun:
            return
        try:
            spot = float(self._exchange._exchange.fetch_balance({"type": "spot"}).get("USDC", {}).get("free", 0.0))
            perp = float(self._exchange._exchange.fetch_balance({"type": "swap"}).get("USDC", {}).get("free", 0.0))
            if spot > 1.0 and perp < spot:
                logger.info("Auto-transferring %.2f USDC from spot to perp...", spot)
                self._exchange.transfer_spot_to_perp(spot)
        except Exception as exc:
            if "unified account" in str(exc).lower() or "action disabled" in str(exc).lower():
                logger.info("Unified account detected — spot/perp balances are merged, skipping transfer.")
            else:
                logger.warning("Auto-transfer check failed: %s", exc)

    def _get_account_equity(self) -> float:
        if self.dryrun:
            risk_mgr = getattr(self, "_risk_mgr", None)
            return risk_mgr.equity if risk_mgr is not None else config.BACKTEST_INITIAL_CAPITAL
        try:
            bal = self._exchange.get_balance()
            if bal <= 0:
                logger.warning("Balance is 0 — funds not yet available on HL, skipping tick")
            return bal
        except Exception as exc:
            logger.warning("Could not fetch account equity: %s", exc)
            risk_mgr = getattr(self, "_risk_mgr", None)
            return risk_mgr.equity if risk_mgr is not None else config.BACKTEST_INITIAL_CAPITAL
